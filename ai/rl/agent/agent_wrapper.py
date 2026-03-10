from rl.agent.dqn import RecurrentDQN
from rl.buffer import Batch
import numpy as np
import torch


def make_input_normalizer(env_space):
    env_status_space_high = env_space["env_status"].high.max()
    env_space_high = env_space["env"].high.max()

    def normalize(obs, env_info):
        normalized_obs = obs / env_space_high
        normalized_feat = env_info["additional_features"] / env_status_space_high
        return normalized_obs, normalized_feat

    return normalize


class InputNormalizationWrapper:
    def __init__(self, agent: RecurrentDQN, obs_space, device="cuda"):
        self.agent = agent
        self.env_status_space_high = obs_space["env_status"].high.max()
        self.env_space_high = obs_space["env"].high.max()
        self.device = device

    def process_obs(self, obs):
        if isinstance(obs, np.ndarray):
            normalized_obs = obs.astype(np.float32) / self.env_space_high
            normalized_obs = torch.from_numpy(normalized_obs).to(self.device)
        else:
            normalized_obs = obs.float() / self.env_space_high
        normalized_obs = normalized_obs.unsqueeze(-1)
        return normalized_obs

    def process_game_features(self, features):
        if isinstance(features, np.ndarray):
            normalized_feat = features.astype(np.float32) / self.env_status_space_high
            normalized_feat = torch.from_numpy(normalized_feat).to(self.device)
        else:
            normalized_feat = features.float() / self.env_status_space_high

        return normalized_feat

    def sample(self, obs, game_features, memory, masks):
        normalized_obs = self.process_obs(obs)
        normalized_features = self.process_game_features(game_features)
        return self.agent.sample(normalized_obs, normalized_features, memory, masks)

    def greedy_actions(self, obs, game_features, memory):
        normalized_obs = self.process_obs(obs)
        normalized_features = self.process_game_features(game_features)
        return self.agent.greedy_actions(normalized_obs, normalized_features, memory)

    def update(self, batch: Batch):
        obs = self.process_obs(batch.obs)
        next_obs = self.process_obs(batch.next_obs)
        game_features = self.process_game_features(batch.game_features)
        next_game_features = self.process_game_features(batch.next_game_features)
        batch = batch._replace(
            obs=obs,
            next_obs=next_obs,
            game_features=game_features,
            next_game_features=next_game_features,
        )
        return self.agent.update(batch)

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent.load(path)
