import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import time
import diablo_state
import numpy as np
from rl.buffer import (
    ReplayBuffer,
    Batch,
    OfflineBuffer,
)
import wandb
import json
from rl.env.parallel_env import ParallelEnv
import devilutionx as dx

from diablo_env import DiabloEnv, ActionEnum
from rl.models.recurrent import RecurrentModel
from rl.agent.dqn import RecurrentDQN
import torch
from tqdm import tqdm
from rl.utils.evaluate import evaluate
from rl.agent.agent_wrapper import InputNormalizationWrapper
from rl.utils.normalization import RewardNormalizer
from rl.utils.save_utils import save_to_pkl
from rl.utils.load_utils import load_from_pkl
from rl.utils.action_mask import build_action_mask_fn

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def train(args, gameconfig):

    agent = load_from_pkl(path="saved_models/check_potential_bug")
    save_to_pkl(path="saved_models/cpu_agent", obj=agent, move_to_cpu=True)

    exit()
    # Init wandb
    args_dict = vars(args)
    wandb_init_kwargs = dict(entity="fvgt", project="rl_diablo", config=args_dict)

    # fix it for now
    seeds = [1 for _ in range(args.n_envs)]

    with wandb.init(**wandb_init_kwargs) as wandb_run:
        train_envs = ParallelEnv(
            env_name=args.env,
            gameconfig=gameconfig,
            n_envs=args.n_envs,
            seeds=seeds,
        )
        eval_envs = ParallelEnv(
            env_name=args.env,
            gameconfig=gameconfig,
            n_envs=1,
            seeds=seeds,
        )

        print(f"Environments created!")
        n_actions = train_envs.action_space.n

        # Create replay buffer
        obs_shape = train_envs.observation_space["env"].shape[1:]
        n_additional_feat = train_envs.observation_space["env_status"].shape[-1]

        replay_buffer = ReplayBuffer(
            n_envs=args.n_envs,
            capacity=args.buffer_size,
            obs_shape=obs_shape,
            n_additional_feat=n_additional_feat,
            recurrence_length=args.recurrence_length,
            burn_in_phase=args.burn_in_phase,
            n_actions=n_actions,
        )

        offline_buffer = OfflineBuffer(
            n_envs=args.n_envs,
            capacity=args.buffer_size,
            obs_shape=obs_shape,
            n_additional_feat=n_additional_feat,
            recurrence_length=args.recurrence_length,
            burn_in_phase=args.burn_in_phase,
            n_actions=n_actions,
            data_dir="/home/"
            "setsailforfail/Desktop/DevilutionX-AI/ai/data/offline_demonstrations/seed_1",
        )

        model = RecurrentModel(
            obs_space=train_envs.observation_space,
            n_actions=n_actions,
            mlp_hidden_dim=args.mlp_hidden_dim,
            lstm_hidden_dim=args.lstm_hidden_dim,
            additional_features=n_additional_feat,
        )
        target_model = RecurrentModel(
            obs_space=train_envs.observation_space,
            n_actions=n_actions,
            mlp_hidden_dim=args.mlp_hidden_dim,
            lstm_hidden_dim=args.lstm_hidden_dim,
            additional_features=n_additional_feat,
        )
        target_model.load_state_dict(model.state_dict())

        model.to(args.device)
        target_model.to(args.device)

        agent = RecurrentDQN(
            n_actions=n_actions,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_step=args.epsilon_decay_steps,
            model=model,
            target_model=target_model,
            discount=args.discount,
            recurrence_length=args.recurrence_length,
            burn_in_phase=args.burn_in_phase,
            tau=args.tau,
            target_net_update_freq=args.target_net_update_freq,
            lr=args.lr,
        )
        # agent.load(
        # "/home/setsailforfail/Desktop/DevilutionX-AI/ai/saved_models/agent/recurrent_dqn_final.pth"
        # )

        agent = InputNormalizationWrapper(
            agent=agent, obs_space=train_envs.observation_space, device=args.device
        )

        reward_normalization = RewardNormalizer(
            n_envs=args.n_envs, max_v=5, gamma=args.discount
        )

        memory = torch.zeros(args.n_envs, args.lstm_hidden_dim, device=args.device)
        obs, infos = train_envs.reset(seeds=seeds)

        env_steps = 0
        frames = []

        get_action_mask = build_action_mask_fn(
            gameconfig=gameconfig, n_actions=n_actions
        )
        action_mask = get_action_mask(obs)

        with tqdm(total=args.max_env_steps, desc="Environment Steps") as pbar:
            while env_steps < args.max_env_steps:
                current_game_features = infos["game_features"]

                actions, memory = agent.sample(
                    obs=obs,
                    game_features=current_game_features,
                    memory=memory,
                    masks=action_mask,
                )
                if env_steps < args.start_training:
                    actions = np.random.randint(n_actions, size=(args.n_envs,))

                next_obs, rewards, terminated, truncated, infos = train_envs.step(
                    actions
                )
                next_action_mask = get_action_mask(obs)
                reward_normalization.update(rewards, terminated, truncated)

                frame = train_envs.render(mode="image")
                frames.append(frame)

                replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=actions,
                    reward=rewards,
                    done=terminated,
                    trunc=truncated,
                    game_features=current_game_features,
                    next_game_features=infos["game_features"],
                    next_action_mask=next_action_mask,
                )

                if np.logical_or(terminated[0], truncated[0]):
                    frames[0].save(
                        f"train_episode.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=500,
                        loop=0,
                    )
                    frames = []

                obs = next_obs
                obs, infos = train_envs.reset_where_done(
                    obs, infos, terminated, truncated, seeds=seeds
                )
                action_mask = next_action_mask

                if env_steps > args.start_training:
                    online_batch = replay_buffer.sample(batch_size=args.batch_size // 2)
                    offline_batch = offline_buffer.sample(
                        batch_size=args.batch_size // 2
                    )
                    batch = Batch.combine(online_batch, offline_batch)

                    unnormalized_rewards = batch.rewards
                    normalized_rewards = reward_normalization.normalize(
                        unnormalized_rewards
                    )
                    batch = batch._replace(rewards=normalized_rewards)

                    # batch = create()
                    # loss, q_values, target_q = agent.update(batch)
                    loss, q_values, target_q = agent.update(batch)

                    loss = loss.detach().cpu().numpy().mean()
                    q_values = q_values.detach().cpu().numpy().mean()
                    target_q = target_q.detach().cpu().numpy().mean()

                if (
                    env_steps > args.start_training
                    and env_steps % (args.log_interval // args.n_envs) == 0
                ):
                    stats = {
                        "train/loss": loss,  # numeric, not formatted string
                        "train/q_mean": q_values,
                        "train/target_q": target_q,
                        "train/epsilon": agent.agent.eps_scheduler.get_epsilon(),
                        "train/batch_reward": batch.rewards.cpu().numpy().mean(),
                        "train/seq_lengths": batch.seq_lengths.cpu().numpy().mean(),
                        "train/G_rms_var": reward_normalization.G_rms.var.mean(),
                        "train/unnormalized_rewards": unnormalized_rewards.mean(),
                    }

                    # Format stats nicely for tqdm
                    formatted_stats = {
                        k: f"{v:.3f}" if isinstance(v, float) else v
                        for k, v in stats.items()
                    }

                    # Update the progress bar
                    pbar.set_postfix(formatted_stats)

                    # Log only training stats to wandb
                    wandb_run.log({**stats, "step": env_steps})

                if env_steps % args.eval_interval == 0:
                    eval_reward, eval_success, gif_path = evaluate(
                        env=eval_envs,
                        agent=agent,
                        _mem=memory,
                        n_episodes=1,
                        action_mask_fn=get_action_mask,
                    )
                    wandb_run.log(
                        {
                            "eval/reward": eval_reward,
                            "eval/success": eval_success,
                            "step": env_steps,  # optional for proper x-axis tracking
                            "episode_gif": wandb.Video(gif_path, format="gif"),
                        }
                    )

                # Increment the counter and update tqdm
                env_steps += args.n_envs
                pbar.update(args.n_envs)

            save_dir = "./saved_models/agent"
            os.makedirs(save_dir, exist_ok=True)
            save_to_pkl(path=os.path.join(save_dir, "check_potential_bug"), obj=agent)
