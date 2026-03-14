import numpy
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rl.buffer import Batch
from rl.utils.loss_fn import categorical_td_loss_torch
import diablo_state
import random
import json


class EpsilonScheduler:
    def __init__(self, eps_start=1.0, eps_end=0.01, decay_steps=10000):
        self.eps_start = eps_start  # starting epsilon
        self.eps_end = eps_end  # minimum epsilon
        self.decay_steps = decay_steps  # how many steps to decay over
        self.step = 0

    def get_epsilon(self):
        epsilon = self.eps_start - (self.step / self.decay_steps) * (
            self.eps_start - self.eps_end
        )
        epsilon = max(self.eps_end, epsilon)
        return epsilon

    def update(self):
        self.step += 1


class RecurrentDQN:
    def __init__(
        self,
        n_actions,
        epsilon_start,
        epsilon_end,
        epsilon_decay_step,
        model,
        target_model,
        discount,
        lr,
        recurrence_length,
        tau,
        target_net_update_freq,
        burn_in_phase,
        device="cuda",
    ):
        self.model = model
        self.target_model = target_model
        self.discount = discount
        self.recurrence_length = recurrence_length
        self.burn_in_phase = burn_in_phase
        self.eps_scheduler = EpsilonScheduler(
            eps_start=epsilon_start, eps_end=epsilon_end, decay_steps=epsilon_decay_step
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        # self.update_fast = torch.compile(self.update_fast, mode="reduce-overhead")

        self.device = device
        self.n_actions = n_actions
        self.update_steps = 0
        self.mse = False

    @torch.no_grad()
    def soft_update(self, tau: float = 0.005):
        for target_param, online_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * online_param.data)

    @torch.no_grad()
    def sample(self, obs, game_features, memory, masks):
        # CrossQ
        masks = torch.tensor(masks)
        self.model.q.eval()

        n_envs = obs.shape[0]
        current_epsilon = self.eps_scheduler.get_epsilon()

        values, _, memory = self.model(obs, game_features, memory)

        # Mask Q-values for greedy selection
        masked_values = values.clone()
        masked_values[masks == True] = -float("inf")

        actions = torch.empty(n_envs, dtype=torch.long, device=values.device)

        for i in range(n_envs):
            if random.random() < current_epsilon:
                # Randomly choose among allowed actions
                allowed_actions = (
                    torch.nonzero(~masks[i], as_tuple=False).squeeze(-1).tolist()
                )
                actions[i] = random.choice(allowed_actions)  # <- correct!
            else:
                # Greedy action
                actions[i] = torch.argmax(masked_values[i])

        self.eps_scheduler.update()

        return actions.cpu().numpy(), memory

    @torch.no_grad()
    def greedy_actions(self, obs, game_features, memory):
        # CrossQ
        self.model.q.eval()

        obs = obs.to(self.device)
        values, _, memory = self.model(obs, game_features, memory)
        greedy_actions = torch.argmax(values, dim=-1)
        return greedy_actions.cpu().numpy(), memory

    @staticmethod
    def unroll_sequence(
        model, obs, game_features, seq_lengths, burn_in, recurrent_states
    ):
        # Split memory into h and c
        h0, c0 = (
            recurrent_states[:, :64],
            recurrent_states[:, 64:],
        )  # adjust hidden_size
        h0 = h0.unsqueeze(0)  # shape (num_layers=1, B, hidden_size)
        c0 = c0.unsqueeze(0)
        h, c = h0, c0
        B, S = obs.shape[:2]

        # Flatten batch x seq for feature extraction
        flattened_obs = obs.reshape(-1, *obs.shape[2:])
        conv_features = model.features(flattened_obs)
        conv_features = conv_features.reshape(B, S, -1).contiguous()
        features = torch.concatenate((conv_features, game_features), dim=-1)
        h = h.contiguous()
        c = c.contiguous()

        # Burn-in phase (no gradient)
        if burn_in > 0:
            with torch.no_grad():
                _, (h, c) = model.lstm(features[:, :burn_in], (h, c))

        # Adjust sequence lengths after burn-in
        # seq_lengths_adj = seq_lengths - burn_in

        # Pack the remaining sequence
        recurrent_input = pack_padded_sequence(
            features[:, burn_in:],
            seq_lengths.int().cpu().squeeze(),
            batch_first=True,
            enforce_sorted=False,
        )

        recurrent_output, *_ = model.lstm(recurrent_input, (h, c))
        recurrent_output, _ = pad_packed_sequence(recurrent_output, batch_first=True)
        q_values, infos = model.q(recurrent_output)
        return q_values, infos

    def update(self, batch: Batch):
        # CrossQ
        self.model.q.train()
        self.target_model.q.train()

        B = batch.obs.shape[0]
        memory = torch.zeros((B, 128), device=batch.obs.device)
        target_memory = torch.zeros((B, 128), device=batch.obs.device)

        q_values, q_infos = self.unroll_sequence(
            model=self.model,
            obs=batch.obs,
            game_features=batch.game_features,
            seq_lengths=batch.seq_lengths,
            burn_in=self.burn_in_phase,
            recurrent_states=memory,
        )
        next_q_values, target_infos = self.unroll_sequence(
            model=self.target_model,
            obs=batch.next_obs,
            game_features=batch.next_game_features,
            seq_lengths=batch.seq_lengths,
            burn_in=self.burn_in_phase,
            recurrent_states=target_memory,
        )

        if self.mse:
            target_q = batch.rewards + self.discount * next_q_values.max(-1)[0] * (
                1 - batch.dones.float()
            )
            chosen_q = q_values.gather(-1, batch.actions[..., None]).squeeze()
            td_error = chosen_q - target_q.detach()
            loss = (td_error**2)[batch.loss_mask]
        else:
            q_log_probs = q_infos["log_probs"]
            chosen_q_log_probs = torch.gather(
                q_log_probs,
                dim=1,
                index=batch.actions.reshape(-1)[:, None, None].expand(-1, 1, 101),
            ).squeeze(1)
            target_log_probs = target_infos["log_probs"]
            max_indices = torch.argmax(next_q_values, dim=-1)
            target_log_probs = torch.gather(
                target_log_probs,
                dim=1,
                index=max_indices[:, None, None].expand(-1, 1, 101),
            ).squeeze(1)
            max_q_values = torch.gather(
                next_q_values, dim=-1, index=max_indices[:, None]
            )
            loss, cat_loss_infos = categorical_td_loss_torch(
                pred_log_probs=chosen_q_log_probs,
                target_log_probs=target_log_probs,
                reward=batch.rewards,
                next_values=max_q_values,
                mask=~batch.dones,
                gamma=self.discount,
            )

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1

        if self.update_steps % self.target_net_update_freq == 0:
            self.soft_update(tau=self.tau)

        return loss, q_values, torch.zeros(())

    # @torch.jit.script
    # def update(self, batch):
    # B, seq_length = batch.obs.shape[:2]
    # device = batch.obs.device
    # total_loss = 0.0
    #
    # memory = torch.zeros((B, 128), device=device)
    # target_memory = torch.zeros((B, 128), device=device)
    #
    # with torch.no_grad():
    # for j in range(self.burn_in_phase):
    # obs = batch.obs[:, j]
    # next_obs = batch.next_obs[:, j]
    # mask = batch.loss_mask[:, j]
    #
    # _, memory = self.model(obs, memory * mask)
    # _, target_memory = self.target_model(next_obs, target_memory * mask)
    #
    #
    # for i in range(self.recurrence_length):
    # obs = batch.obs[:, i + self.burn_in_phase]
    # next_obs = batch.next_obs[:, i + self.burn_in_phase]
    # mask = batch.loss_mask[:, i + self.burn_in_phase]
    #
    # q_values, memory = self.model(obs, memory * mask)
    #
    # with torch.no_grad():
    # next_q_values, target_memory = self.target_model(next_obs, target_memory * mask)
    # target_q = (
    # batch.rewards[:, i]
    # + self.discount
    # * next_q_values.max(dim=-1)[0]
    # * (1 - batch.dones[:, i].float())
    # )
    #
    # chosen_q = q_values.gather(-1, batch.actions[:, i]).squeeze(-1)
    # td_error = chosen_q - target_q
    # loss = (td_error ** 2) * mask.squeeze()
    # total_loss += loss.mean()
    #
    #
    # total_loss /= (seq_length - self.burn_in_phase)
    # self.optimizer.zero_grad()
    # total_loss.backward()
    # self.optimizer.step()
    #
    # self.update_steps += 1
    #
    # if self.update_steps % self.target_net_update_freq == 0:
    # self.soft_update(tau=self.tau)
    #
    #
    # return total_loss, q_values, target_q
    #

    def run_recurrent_loop(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        masks: torch.Tensor,
        memory: torch.Tensor,
        target_memory: torch.Tensor,
        discount: float,
    ):
        with torch.no_grad():
            for j in range(self.burn_in_phase):
                ob = obs[:, j]
                next_ob = next_obs[:, j]
                mask = masks[:, j]

                _, memory = self.model(ob, memory * mask)
                _, target_memory = self.target_model(next_ob, target_memory * mask)

        total_loss = torch.zeros(1, device=obs.device)
        for i in range(self.recurrence_length):
            ob = obs[:, i + self.burn_in_phase]
            next_ob = next_obs[:, i + self.burn_in_phase]
            mask = masks[:, i]
            done = dones[:, i]
            reward = rewards[:, i]
            action = actions[:, i]

            q_values, memory = self.model(ob, memory * mask)

            with torch.no_grad():
                next_q_values, target_memory = self.target_model(
                    next_ob, target_memory * mask
                )
                max_next_q = next_q_values.max(dim=-1)[0]
                target_q = reward + discount * max_next_q * (1.0 - done.squeeze(-1))

            chosen_q = q_values.gather(-1, action).squeeze()
            step_loss = torch.pow(chosen_q - target_q, 2) * mask
            total_loss += step_loss.mean()

        return total_loss, q_values, target_q

    def update_fast(self, batch):
        B = batch.obs.shape[0]
        device = batch.obs.device
        memory = torch.zeros((B, 128), device=device)
        target_memory = torch.zeros((B, 128), device=device)
        return self.run_recurrent_loop(
            batch.obs,
            batch.next_obs,
            batch.actions,
            batch.rewards,
            batch.dones,
            batch.loss_mask,
            memory,
            target_memory,
            self.discount,
        )

    def save(self, path: str):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "eps_scheduler_step": self.eps_scheduler.step,
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.eps_scheduler.step = checkpoint.get("eps_scheduler_step", 0)
