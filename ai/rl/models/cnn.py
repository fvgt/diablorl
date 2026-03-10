import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=16, output_dim=512):
        super(CNN, self).__init__()

        self.network = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # This layer doubles the channels (64->128) and halves the grid size (stride=2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsamples
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # This layer doubles the channels (128->256) and halves the grid size (stride=2)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsamples
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # This layer doubles the channels (256->512) and halves the grid size (stride=2)
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Downsamples
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Head Part (untouched)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_dim),
        )

    def forward(self, obs):
        if obs.ndim == 3:
            # (H, W, C) -> (1, C, H, W)
            obs = obs.permute(2, 0, 1).unsqueeze(0)

        elif obs.ndim == 4:
            # (B, H, W, C) -> (B, C, H, W)
            obs = obs.permute(0, 3, 1, 2)

        elif obs.ndim == 5:
            # (B, T, H, W, C) -> (B, T, C, H, W)
            obs = obs.permute(0, 1, 4, 2, 3)

        else:
            raise ValueError(f"Unexpected input shape: {obs.shape}")

        features = self.network(obs)
        features = features.reshape(obs.shape[0], -1)
        return features
