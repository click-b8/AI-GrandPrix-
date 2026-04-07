"""
SCUBA LAB - DCL AI Grand Prix Adapter (Direct Weight Loading)
Bridges trained distilled vision model to DCL competition API

This version loads weights directly instead of using PPO.load() to avoid
version compatibility issues with SB3.
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from vision_model import DroneVisionExtractor


class PolicyNet(nn.Module):
    """Minimal policy network matching the trained model architecture."""

    def __init__(self):
        super().__init__()
        # Features from DroneVisionExtractor output (256D)
        # + State (19D) = 275D input
        self.pi_net = nn.Sequential(
            nn.Linear(275, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 continuous actions: [throttle, roll, pitch, yaw]
            nn.Tanh(),
        )

    def forward(self, features, state):
        x = torch.cat([features, state], dim=-1)
        return self.pi_net(x)


class SCUBALabAdapter:
    """Adapter for DCL AI Grand Prix competition with direct weight loading"""

    def __init__(self, model_path: str, weights_path: str = None):
        """
        Initialize SCUBA Lab adapter with trained distilled model

        Args:
            model_path: Path to trained aigp_distill_final.zip (for historical reference)
            weights_path: Path to policy.pth weights file
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[SCUBA Lab] Device: {self.device}")

        # Create observation space for feature extractor
        obs_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(14, 48, 48), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32),
        })

        # Initialize feature extractor and policy
        print(f"[SCUBA Lab] Loading model: {model_path}")
        self.feature_extractor = DroneVisionExtractor(obs_space).to(self.device)
        self.policy = PolicyNet().to(self.device)

        # Load pre-trained weights
        if weights_path is None:
            import zipfile
            import tempfile
            import os
            import shutil

            # Extract policy.pth from zip if not provided separately
            tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(model_path, 'r') as z:
                    z.extract('policy.pth', tmpdir)
                weights_path = os.path.join(tmpdir, 'policy.pth')
                self._load_weights(weights_path)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            self._load_weights(weights_path)

        print(f"[SCUBA Lab] ✓ Model loaded successfully")

    def _load_weights(self, weights_path: str):
        """Load weights from saved file"""
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Filter weights: only load those that match current architecture
        feature_dict = {k: v for k, v in checkpoint.items() if k.startswith('features_extractor.')}
        policy_dict = {k: v for k, v in checkpoint.items() if k.startswith('pi_net.')}

        # Rename keys to remove prefix
        feature_dict = {k.replace('features_extractor.', ''): v for k, v in feature_dict.items()}
        policy_dict = {k.replace('pi_net.', ''): v for k, v in policy_dict.items()}

        # Load with partial matching
        try:
            self.feature_extractor.load_state_dict(feature_dict, strict=False)
            print(f"[SCUBA Lab] Loaded feature extractor weights")
        except Exception as e:
            print(f"[SCUBA Lab] Warning: Could not load all feature extractor weights: {e}")

        try:
            self.policy.load_state_dict(policy_dict, strict=False)
            print(f"[SCUBA Lab] Loaded policy weights")
        except Exception as e:
            print(f"[SCUBA Lab] Warning: Could not load all policy weights: {e}")

        self.feature_extractor.eval()
        self.policy.eval()

    def process_observation(self, telemetry: dict, visual_data: np.ndarray) -> tuple:
        """
        Convert DCL inputs to model observation format

        Args:
            telemetry: Dict with keys: position, velocity, orientation
            visual_data: Visual stream from FPV camera (H, W, C)

        Returns:
            Tuple of (features_tensor, state_tensor) ready for policy
        """
        # Resize to model input size (48x48) if needed
        if visual_data.shape[:2] != (48, 48):
            from PIL import Image
            pil_img = Image.fromarray(visual_data)
            pil_img = pil_img.resize((48, 48), Image.BILINEAR)
            visual_data = np.array(pil_img)

        # Convert to (C, H, W) for PyTorch if needed
        if visual_data.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
            image = np.transpose(visual_data, (2, 0, 1))
        else:
            image = visual_data

        # Model trained with event camera: 14 channels total
        # Structure: (RGB 3ch + Events 4ch) stacked 2x = 14 channels
        # DCL provides RGB only, so pad with zeros for event channels
        rgb_frame = image.astype(np.float32)  # Keep as 0-255 uint8 range
        event_channels = np.zeros((4, 48, 48), dtype=np.float32)

        # Stack: first frame (RGB + zero events), second frame (RGB + zero events)
        frame_with_events = np.concatenate([rgb_frame, event_channels], axis=0)  # 7 channels
        image_14ch = np.concatenate([frame_with_events, frame_with_events], axis=0)  # 14 channels

        # Convert to uint8 for model
        image_14ch = image_14ch.astype(np.uint8)

        # Build state vector from telemetry (19D state)
        position = np.array(telemetry.get('position', [0, 0, 0]), dtype=np.float32)
        velocity = np.array(telemetry.get('velocity', [0, 0, 0]), dtype=np.float32)
        orientation = np.array(telemetry.get('orientation', [0, 0, 0]), dtype=np.float32)

        # Concatenate and pad to 19D
        state = np.concatenate([position, velocity, orientation])
        state = np.pad(state, (0, max(0, 19 - len(state))), mode='constant')[:19]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_14ch).float().unsqueeze(0).to(self.device)  # (1, 14, 48, 48)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # (1, 19)

        return image_tensor, state_tensor

    def predict_action(self, image_tensor: torch.Tensor, state_tensor: torch.Tensor) -> np.ndarray:
        """
        Get drone command from model

        Args:
            image_tensor: Processed image tensor (1, 14, 48, 48)
            state_tensor: Processed state tensor (1, 19)

        Returns:
            Action array: [throttle, roll, pitch, yaw]
        """
        with torch.no_grad():
            # Extract features
            obs = {"image": image_tensor, "state": state_tensor}
            features = self.feature_extractor(obs)

            # Get action from policy (features is (batch, 256), state_tensor is (batch, 19))
            action = self.policy(features, state_tensor)

        # Convert to numpy and clamp to valid ranges
        action = action.squeeze(0).cpu().numpy()

        # Clamp: throttle [0, 1], roll/pitch/yaw [-1, 1]
        action[0] = np.clip(action[0], 0.0, 1.0)  # throttle
        action[1:] = np.clip(action[1:], -1.0, 1.0)  # roll, pitch, yaw

        return action

    def action_to_dcl_command(self, action: np.ndarray) -> dict:
        """
        Convert model action to DCL drone command format

        Args:
            action: Model output [throttle, roll, pitch, yaw]

        Returns:
            Dict with drone control: {throttle, roll, pitch, yaw}
        """
        return {
            'throttle': float(np.clip(action[0], 0.0, 1.0)),
            'roll': float(np.clip(action[1], -1.0, 1.0)),
            'pitch': float(np.clip(action[2], -1.0, 1.0)),
            'yaw': float(np.clip(action[3], -1.0, 1.0)),
        }

    def step(self, telemetry: dict, visual_data: np.ndarray) -> dict:
        """
        End-to-end inference: telemetry + visual -> drone command

        Args:
            telemetry: Dict with position, velocity, orientation
            visual_data: FPV camera frame (H, W, 3)

        Returns:
            Dict with throttle, roll, pitch, yaw commands
        """
        image_tensor, state_tensor = self.process_observation(telemetry, visual_data)
        action = self.predict_action(image_tensor, state_tensor)
        command = self.action_to_dcl_command(action)

        return command
