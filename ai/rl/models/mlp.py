import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        use_layer_norm=False,
        activate_final=False,
        normalize_networks=False,
    ):
        super().__init__()

        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Final output layer (assuming regression with 1 output)
        layers.append(nn.Linear(input_dim, output_dim))
        if activate_final:
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self.normalize_networks = normalize_networks

    def forward(self, x):
        if self.normalize_networks:
            with torch.no_grad():
                normalize_network(self.model)
        return self.model(x)
