import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def categorical_td_loss_torch(
    pred_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    reward: torch.Tensor,
    next_values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    num_bins: int = 101,
    min_v: float = -5,
    max_v: float = 5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    reward = reward.reshape(-1, 1)
    next_values = next_values.reshape(-1, 1)
    mask = mask.reshape(-1, 1).float()
    target_log_probs = target_log_probs.reshape(-1, target_log_probs.shape[-1])
    pred_log_probs = pred_log_probs.reshape(-1, pred_log_probs.shape[-1])

    n = reward.size(0)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=pred_log_probs.device
    ).reshape(1, -1)
    target_bin_values = reward + gamma * bin_values * mask 
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)
    clipped_mask = (target_bin_values == min_v) | (target_bin_values == max_v)
    clip_percentage = torch.mean(clipped_mask.float())
    delta_z = (max_v - min_v) / (num_bins - 1)
    b = (target_bin_values - min_v) / delta_z

    l = torch.floor(b).long()
    u = torch.ceil(b).long()
    l = torch.clamp(l, 0, num_bins - 1)
    u = torch.clamp(u, 0, num_bins - 1)
    l_mask = F.one_hot(l.reshape(-1), num_bins).reshape(n, num_bins, num_bins).float()
    u_mask = F.one_hot(u.reshape(-1), num_bins).reshape(n, num_bins, num_bins).float()
    _target_probs = torch.exp(target_log_probs)
    l_equals_u_float = (l == u).float()
    m_l = (_target_probs * (u.float() + l_equals_u_float - b)).unsqueeze(-1)
    m_u = (_target_probs * (b - l.float())).unsqueeze(-1)

    projected_probs = torch.sum(m_l * l_mask + m_u * u_mask, dim=1)
    target_probs = projected_probs.detach()

    loss = -torch.mean(torch.sum(target_probs * pred_log_probs, dim=1))

    infos = {
        "clip_percentage": clip_percentage,
    }

    return loss, infos
