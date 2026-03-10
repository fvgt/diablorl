import numpy as np
import torch
import collections


class EventsQueue:
    queue = None
    # Use Braille patterns for representing progress,
    # see here: https://www.unicode.org/charts/nameslist/c_2800.html
    progress = [0x2826, 0x2816, 0x2832, 0x2834]
    progress_cnt = 0

    def __init__(self):
        self.queue = collections.deque(maxlen=10)


def evaluate(env, agent, _mem):
    n_episodes = 1
    n_envs = 1
    total_rewards = np.zeros(shape=(n_envs, n_episodes))
    total_success = np.zeros(shape=(n_envs, n_episodes))
    for i in range(n_episodes):
        obs, infos = env.reset(seeds=[1])
        memory = torch.zeros_like(_mem)[:1]
        ongoing_dones = np.zeros(shape=(n_envs))
        ongoing_truncs = np.zeros(shape=(n_envs))
        frames = []
        while not np.all(np.logical_or(ongoing_dones, ongoing_truncs)):
            current_game_features = infos["game_features"]
            actions, memory = agent.greedy_actions(
                obs=obs,
                game_features=current_game_features,
                memory=memory,
            )
            next_obs, rewards, done, trunc, infos = env.step(actions)
            total_rewards[:, i] += rewards * (1 - ongoing_dones)
            total_success[:, i] = infos["success"] * (1 - ongoing_dones)
            ongoing_dones = np.maximum(ongoing_dones, done)
            ongoing_truncs = np.maximum(ongoing_truncs, trunc)
            obs = next_obs

            frame = env.penv_pool.envs[0].env.render("image")
            frames.append(frame)

        gif_path = f"episode.gif"
        if frames:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,
                loop=0,
            )

    avg_rewards = total_rewards.mean()
    avg_success = total_success.mean()

    return avg_rewards, avg_success, gif_path
