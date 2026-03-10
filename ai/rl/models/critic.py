import torch.nn as nn
import torch
from rl.utils.normalization import normalize_network
import torch.nn.functional as F


class DistributionalCritic(nn.Module):
    def __init__(
        self,
        embedding_size,
        use_batch_norm,
        use_weight_norm,
        hidden_dims,
        n_actions,
        n_atoms=101,
        min_v=-5,
        max_v=5,
    ):
        super().__init__()

        layers = []

        input_dim = embedding_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, n_atoms * n_actions))

        self.model = nn.Sequential(*layers)
        self.use_weight_norm = use_weight_norm
        self.n_atoms = n_atoms
        self.min_v = min_v
        self.max_v = max_v
        self.n_actions = n_actions

    def forward(self, x):
        if self.use_weight_norm:
            with torch.no_grad():
                normalize_network(self.model)
        # for BN we have to reshape the input
        orig_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, orig_shape[-1])
            logits = self.model(x)
            logits = logits.reshape(*orig_shape[:2], self.n_actions * self.n_atoms)
        else:
            logits = self.model(x)

        logits = logits.reshape(x.shape[0], self.n_actions, self.n_atoms)

        bin_values = torch.linspace(
            start=self.min_v, end=self.max_v, steps=self.n_atoms, device=x.device
        )

        log_probs = F.log_softmax(logits, dim=-1)
        values = torch.sum(torch.exp(log_probs) * bin_values, dim=-1)

        return values, {"log_probs": log_probs}
