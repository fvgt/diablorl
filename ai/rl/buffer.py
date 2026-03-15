# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Taken from https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

import numpy as np
import torch
from typing import NamedTuple
import os


class Batch(NamedTuple):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor
    loss_mask: torch.Tensor
    game_features: torch.Tensor
    next_game_features: torch.Tensor
    seq_lengths: torch.Tensor
    next_action_mask_seq: torch.Tensor

    @staticmethod
    def combine(batch1, batch2):
        combined = tuple(
            torch.concatenate([b1, b2], axis=0).to(b1.device)
            for b1, b2 in zip(batch1, batch2)
        )
        return Batch(*combined)


class ReplayBuffer:
    def __init__(
        self,
        n_envs,
        capacity,
        obs_shape,
        n_actions,
        n_additional_feat,
        recurrence_length,
        burn_in_phase,
    ):
        self.capacity = capacity
        self.insert_index = 0
        self.size = 0
        self.n_envs = n_envs
        self.recurrence_length = recurrence_length
        self.burn_in_phase = burn_in_phase

        self.obs = np.zeros((n_envs, capacity, *obs_shape), dtype=np.int32)
        # self.next_obs = np.zeros((n_envs, capacity, *obs_shape), dtype=np.int32)
        self.actions = np.zeros((n_envs, capacity), dtype=np.int64)
        self.next_action_masks = np.ones((n_envs, capacity, n_actions), dtype=np.int8)
        self.rewards = np.zeros((n_envs, capacity), dtype=np.float32)
        self.dones = np.zeros((n_envs, capacity), dtype=np.bool)
        self.game_features = np.zeros(
            (n_envs, capacity, n_additional_feat), dtype=np.int32
        )
        self.next_game_features = np.zeros(
            (n_envs, capacity, n_additional_feat), dtype=np.int32
        )
        self.episode_lengths = np.zeros((n_envs, 1), dtype=np.int32)
        self.remaining_episode_lenghts = np.full(
            (n_envs, capacity), dtype=np.int32, fill_value=-1
        )  # we have to differentiate between transitions that just ended (remaining episode length == 0)
        # and transitions that are not in the buffer yet

    def __len__(self):
        return self.size

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        trunc,
        game_features,
        next_game_features,
        next_action_mask,
    ):
        self.obs[:, self.insert_index] = obs
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.dones[:, self.insert_index] = done
        self.game_features[:, self.insert_index] = game_features
        self.next_game_features[:, self.insert_index] = next_game_features
        self.next_action_masks[:, self.insert_index] = next_action_mask
        self.episode_lengths += 1

        episode_over_idxs = np.where(np.logical_or(done, trunc))[0]
        if len(episode_over_idxs) > 0:
            for idx in episode_over_idxs:
                episode_length = self.episode_lengths[idx].item()
                for t in range(episode_length):
                    pos = (self.insert_index - t) % self.capacity
                    self.remaining_episode_lenghts[idx, pos] = (
                        t + 1
                    )  # the transition with the done state should still be included, hence t + 1
                # next episode must start with 0
                self.episode_lengths[idx] = 0

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def is_valid_idx(self, idx):
        # remove idxs that do not have enough burn in transitions
        seq_lengths = self.remaining_episode_lenghts[:, idx]
        ongoing_mask = seq_lengths == -1
        seq_lengths = np.where(ongoing_mask, self.episode_lengths, seq_lengths)
        valid = np.where(seq_lengths >= self.burn_in_phase, True, False)
        return np.all(valid)

    def sample(self, batch_size):
        B = batch_size // self.n_envs
        window = self.recurrence_length + self.burn_in_phase

        idxs = np.random.randint(0, self.size, size=B)
        for i in range(B):
            while not self.is_valid_idx(idxs[i]):
                idxs[i] = np.random.randint(0, self.size, size=1)
                # print('invalid')

        lstm_seq_idxs = idxs[:, None] + np.arange(window)[None, :]
        lstm_seq_idxs = lstm_seq_idxs % self.capacity
        default_seq_idxs = lstm_seq_idxs[:, self.burn_in_phase :]

        obs_seq = self.obs[:, lstm_seq_idxs]
        next_obs_seq = self.obs[:, (lstm_seq_idxs + 1) % self.capacity]
        game_feat_seq = self.game_features[:, lstm_seq_idxs]
        next_game_feat_seq = self.next_game_features[:, lstm_seq_idxs]

        actions_seq = self.actions[:, default_seq_idxs]
        next_action_mask_seq = self.next_action_masks[:, default_seq_idxs]
        rewards_seq = self.rewards[:, default_seq_idxs]
        dones_seq = self.dones[:, default_seq_idxs]

        # This is to pad sequences for LSTM using torch functions
        # true episode length of the sampled episodes
        seq_lengths = self.remaining_episode_lenghts[:, idxs + self.burn_in_phase]
        # for ongoing episodes, we have to compute it manually
        ongoing_mask = seq_lengths == -1
        seq_lengths = np.where(ongoing_mask, self.episode_lengths, seq_lengths)
        seq_lengths = np.where(
            seq_lengths > self.recurrence_length, self.recurrence_length, seq_lengths
        )

        # only include transitions in the loss that were valid
        recurrence = np.arange(self.recurrence_length)
        loss_mask = recurrence < seq_lengths[..., np.newaxis]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        return Batch(
            obs=torch.from_numpy(obs_seq.reshape(-1, *obs_seq.shape[2:])).to(device),
            actions=torch.from_numpy(
                actions_seq.reshape(-1, *actions_seq.shape[2:])
            ).to(device),
            rewards=torch.from_numpy(
                rewards_seq.reshape(-1, *rewards_seq.shape[2:])
            ).to(device),
            dones=torch.from_numpy(dones_seq.reshape(-1, *dones_seq.shape[2:])).to(
                device
            ),
            next_obs=torch.from_numpy(
                next_obs_seq.reshape(-1, *next_obs_seq.shape[2:])
            ).to(device),
            game_features=torch.from_numpy(
                game_feat_seq.reshape(-1, *game_feat_seq.shape[2:])
            ).to(device),
            next_game_features=torch.from_numpy(
                next_game_feat_seq.reshape(-1, *next_game_feat_seq.shape[2:])
            ).to(device),
            loss_mask=torch.from_numpy(loss_mask.reshape(-1, *loss_mask.shape[2:])).to(
                device
            ),
            seq_lengths=torch.from_numpy(seq_lengths.reshape(-1)).to(device),
            next_action_mask_seq=torch.from_numpy(
                next_action_mask_seq.reshape(-1, *next_action_mask_seq.shape[2:])
            ).to(device),
        )


class OfflineBuffer(ReplayBuffer):
    def __init__(
        self,
        n_envs,
        capacity,
        obs_shape,
        n_additional_feat,
        recurrence_length,
        burn_in_phase,
        data_dir,
        n_actions,
    ):
        # Initialize the parent class first
        super().__init__(
            n_envs=n_envs,
            capacity=capacity,
            obs_shape=obs_shape,
            n_additional_feat=n_additional_feat,
            recurrence_length=recurrence_length,
            burn_in_phase=burn_in_phase,
            n_actions=n_actions,
        )

        obs = []
        actions = []
        rewards = []
        dones = []
        game_features = []
        next_game_features = []
        remaining_episode_lengths = []

        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
        ]

        for file in files:
            episode = np.load(file)
            obs.append(episode["obs"].squeeze())
            actions.append(episode["actions"])
            rewards.append(episode["rewards"])
            game_features.append(episode["game_features"].squeeze())
            next_game_features.append(episode["next_game_features"].squeeze())
            dones.append(episode["dones"])

            assert episode["dones"][-1] == True
            episode_length = episode["dones"].shape[0]
            remaining = [episode_length - t for t in range(episode_length)]
            remaining_episode_lengths.append(remaining)

        n_envs = 1
        self.obs = np.concatenate(obs, axis=0, dtype=self.obs.dtype)[None, ...]
        self.actions = np.concatenate(actions, axis=0, dtype=self.actions.dtype)[
            None, ...
        ]
        self.next_action_masks = np.ones(
            shape=(n_envs, self.actions.shape[1], n_actions)
        )
        self.rewards = np.concatenate(rewards, axis=0, dtype=self.rewards.dtype)[
            None, ...
        ].squeeze(-1)
        self.game_features = np.concatenate(
            game_features, axis=0, dtype=self.game_features.dtype
        )[None, ...]
        self.next_game_features = np.concatenate(
            next_game_features, axis=0, dtype=self.next_game_features.dtype
        )[None, ...]
        self.dones = np.concatenate(dones, axis=0, dtype=self.dones.dtype)[None, ...]
        self.remaining_episode_lenghts = np.concatenate(
            remaining_episode_lengths,
            axis=0,
            dtype=self.remaining_episode_lenghts.dtype,
        )[None, ...]
        self.episode_lengths = np.zeros((n_envs, 1), dtype=np.int32)
        self.size = self.obs.shape[1]
        self.capacity = self.size
        self.n_envs = n_envs

    def __len__(self):
        return self.size

    def is_valid_idx(self, idx):
        seq_length = self.remaining_episode_lenghts[0, idx]
        if seq_length >= self.burn_in_phase:
            return True
        else:
            return False

    def sample(self, batch_size):
        return super().sample(batch_size=batch_size)
