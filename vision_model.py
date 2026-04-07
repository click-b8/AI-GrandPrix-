"""
Standalone DroneVisionExtractor for competition deployment.
Extracted from train_vision.py to avoid MuJoCo dependency.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DroneVisionExtractor(BaseFeaturesExtractor):
    """Coarse-to-Fine CNN + SE(3)-aware State MLP feature extractor.

    Inspired by event-sharp-nerf-drones (Zou et al. 2026):
    - Coarse-to-fine with feature concatenation (VoxelNeRF pattern)
    - Smaller per-stage networks = faster forward pass
    - State branch receives 6D rotation (continuous representation)

    Coarse CNN:
        Conv2d(in_channels, 16, 5, stride=2) -> ReLU
        Conv2d(16, 32, 3, stride=2) -> ReLU -> Flatten -> Linear(N, 64) -> ReLU

    Fine CNN (processes coarse feature maps):
        Conv2d(32, 48, 3, stride=1, pad=1) -> ReLU
        Conv2d(48, 64, 3, stride=1) -> ReLU -> Flatten -> Linear(N, 128) -> ReLU

    State MLP:
        Linear(19, 64) -> ReLU -> Linear(64, 64) -> ReLU

    Output: cat(coarse_64, fine_128, state_64) = 256 features
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        img_space = observation_space["image"]
        n_channels = img_space.shape[0]
        img_h, img_w = img_space.shape[1], img_space.shape[2]

        # Coarse CNN: fast gate detection
        self.coarse_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # Fine CNN: refines coarse features (concatenation pattern from voxnerf.py)
        self.fine_cnn = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute feature map sizes
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, img_h, img_w)
            coarse_out = self.coarse_cnn(dummy)
            coarse_flat = coarse_out.flatten(1).shape[1]
            fine_out = self.fine_cnn(coarse_out)
            fine_flat = fine_out.flatten(1).shape[1]

        self.coarse_fc = nn.Sequential(
            nn.Linear(coarse_flat, 64),
            nn.ReLU(),
        )
        self.fine_fc = nn.Sequential(
            nn.Linear(fine_flat, 128),
            nn.ReLU(),
        )

        # State MLP (receives 6D rotation + VIO estimates)
        state_dim = observation_space["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, observations):
        img = observations["image"].float() / 255.0

        # Coarse-to-fine with feature concatenation
        coarse_maps = self.coarse_cnn(img)
        coarse_features = self.coarse_fc(coarse_maps.flatten(1))
        fine_maps = self.fine_cnn(coarse_maps)
        fine_features = self.fine_fc(fine_maps.flatten(1))

        state_features = self.state_mlp(observations["state"].float())

        return torch.cat([coarse_features, fine_features, state_features], dim=1)
