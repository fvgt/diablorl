import copy
import diablo_state
import diablo_bot
import os
import numpy as np
from rl.env.parallel_env import ParallelEnv


def collect_experience(args, gameconfig):
    bot_constructor = diablo_bot.get_bot_constructor(args.bot)
    env_config = copy.deepcopy(gameconfig)

    seed = env_config["seed"]
    save_dir = f"data/offline_demonstrations/seed_{seed}/"
    os.makedirs(save_dir, exist_ok=True)

    n_episodes = 1000
    successful_episodes = 0
    total_episodes = 0

    while successful_episodes < n_episodes:
        # NOTE for now we are collecting only experience from one seed for testing
        game = diablo_state.DiabloGame.run_or_attach(env_config)
        bot = bot_constructor(
            game, args, view_radius=env_config["view-radius"], controlled_by_env=True
        )
        env = ParallelEnv(args.env, env_config, n_envs=1, seeds=[seed])
        obs, info = env.reset(seeds=[seed])
        game_features = info["game_features"]
        bot.reset(d=env.penv_pool.envs[0].game.state)
        done = False
        trunc = False

        episodes_obs = []
        episodes_game_feat = []
        episodes_next_game_feat = []
        episodes_next_obs = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_truncs = []

        while not done and not trunc:
            done, action = bot.step(
                obs.squeeze(), d=env.penv_pool.envs[0].game.safe_state
            )
            next_obs, reward, _, trunc, info = env.step([action])
            next_game_features = info["game_features"]

            episodes_obs.append(obs)
            episodes_next_obs.append(next_obs)
            episodes_game_feat.append(game_features)
            episodes_next_game_feat.append(next_game_features)
            episode_actions.append(action)
            episode_dones.append(done)
            episode_truncs.append(trunc)
            episode_rewards.append(reward)

            obs = next_obs
            game_features = next_game_features

        total_episodes += 1

        if not bot.discard_run:
            episode_data = {
                "obs": np.array(episodes_obs),
                "next_obs": np.array(episodes_next_obs),
                "game_features": np.array(episodes_game_feat),
                "next_game_features": np.array(episodes_next_game_feat),
                "actions": np.array(episode_actions),
                "rewards": np.array(episode_rewards),
                "dones": np.array(episode_dones),
                "truncs": np.array(episode_truncs),
            }
            np.savez(
                os.path.join(save_dir, f"episode_{successful_episodes}.npz"),
                **episode_data,
            )
            successful_episodes += 1
        print(
            f"Episode successfully saved! {successful_episodes} saved episodes so far!"
        )

    print(
        f"Expert Demons created. Total amount of episodes run {total_episodes}. Episodes saved: {successful_episodes}"
    )
