import torch.nn as nn
from rl.models.cnn import CNN
from rl.models.critic import DistributionalCritic
import torch


class RecurrentModel(nn.Module):
    def __init__(
        self,
        obs_space,
        n_actions,
        mlp_hidden_dim,
        lstm_hidden_dim,
        additional_features,
        mlp_n_layers=2,
    ):
        super().__init__()
        obs_shape = obs_space["env"].shape
        in_channels = 1
        self.n_actions = n_actions
        self.lstm_hidden_dim = lstm_hidden_dim

        self.features = CNN(in_channels=in_channels, output_dim=lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_dim + additional_features,
            hidden_size=self.semi_lstm_hidden_dim,
            batch_first=True,
        )
        self.embedding_size = self.semi_lstm_hidden_dim
        mlp_hidden_dims = [mlp_hidden_dim for _ in range(mlp_n_layers)]
        self.q = DistributionalCritic(
            embedding_size=self.embedding_size,
            hidden_dims=mlp_hidden_dims,
            use_batch_norm=True,
            use_weight_norm=True,
            n_actions=n_actions,
        )

    @property
    def semi_lstm_hidden_dim(self):
        return self.lstm_hidden_dim // 2

    def forward(self, obs, game_features, memory):
        features = self.features(obs)  # (B, H)
        if isinstance(memory, tuple):
            memory = torch.concatenate(memory, 1)

        h = (
            memory[..., : self.semi_lstm_hidden_dim].unsqueeze(0).contiguous()
        )  # (1, N, H//2) -> D = 1, not bidirectional
        c = (
            memory[..., self.semi_lstm_hidden_dim :].unsqueeze(0).contiguous()
        )  # (1, N, H//2) -> D = 1, not bidirectional

        features = torch.concatenate((features, game_features), -1)
        features = features.unsqueeze(1)  # (B, 1, H) -> L = 1
        _, (h, c) = self.lstm(features, (h, c))
        h = h.squeeze(0)  # (B, H//2)
        c = c.squeeze(0)  # (B, H//2)

        memory = torch.cat((h, c), dim=1)  # (B, H)
        values, infos = self.q(h)  # (B, A)
        log_probs = infos["log_probs"]
        return values, log_probs, memory
