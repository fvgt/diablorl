import numpy as np
import torch
import torch.nn as nn
import numpy as np


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(
        self, epsilon=1e-8, shape=(), dtype=np.float64, max_v=-np.inf, min_v=np.inf
    ):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.mean_squared = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.max = np.ones(shape, dtype=dtype) * -np.inf
        self.min = np.ones(shape, dtype=dtype) * np.inf
        self.count = np.zeros(shape, dtype=dtype)
        self.epsilon = epsilon

    def update(self, x, idxs=None):
        """Updates the mean, var and count from a batch of samples."""
        # if the batch size is 1, as the rewards that are collected from the env, it is often collapsed
        if x.shape == self.mean.shape:
            x = x[None]
        # update all seeds

        batch_mean = np.mean(x, axis=0)
        batch_mean_squared = np.mean(np.square(x), axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_mean_squared, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_mean_sq, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.mean_squared, self.var, self.count = (
            update_mean_var_count_from_moments(
                self.mean,
                self.mean_squared,
                self.var,
                self.count,
                batch_mean,
                batch_mean_sq,
                batch_var,
                batch_count,
            )
        )

    def reset_max(self):
        self.running_max = np.ones_like(self.running_max, dtype=np.float32) * -np.inf


def update_mean_var_count_from_moments(
    mean, mean_sq, var, count, batch_mean, batch_mean_sq, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    delta_sq = batch_mean_sq - mean_sq

    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    new_mean_sq = mean_sq + delta_sq * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_mean_sq, new_var, new_count


class RewardNormalizer:
    def __init__(self, n_envs, gamma, max_v):
        self.gamma = gamma
        self.g_max = max_v
        self.G = np.zeros((n_envs,))
        self.G_rms = RunningMeanStd(shape=(n_envs,))
        self.epsilon = 1e-8

    def update(self, reward, terminated, truncated):
        done = np.logical_or(terminated, truncated)
        self.G = self.gamma * (1 - done) * self.G + reward
        self.G_rms.update(self.G)

    def normalize(self, rewards):
        return rewards / (
            torch.sqrt(torch.tensor(self.G_rms.var.mean(), device=rewards.device))
            + self.epsilon
        )


def normalize_network(model, eps=1e-4):
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data
            norm = torch.linalg.vector_norm(weights, dim=-1, keepdim=True)
            normalized_weights = weights / norm
            layer.weight.data.copy_(normalized_weights)
