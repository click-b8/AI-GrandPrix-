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
    """Policy head mirroring SB3 MultiInputPolicy's mlp_extractor.policy_net + action_net.

    Topology matches the trained checkpoint (trained_distilled/policy.pth):
        mlp_extractor.policy_net.0.weight  (128, 256)  ─► Linear(256, 128)
                                                           Tanh      (SB3 default activation_fn)
        mlp_extractor.policy_net.2.weight  (64, 128)   ─► Linear(128, 64)
                                                           Tanh
        action_net.weight                  (4, 64)     ─► Linear(64, 4)  (bare linear;
                                                           Gaussian mean — no squashing here)

    Input is the 256D features extractor output directly. State is NOT re-concatenated:
    the state MLP is already inside DroneVisionExtractor (64D state features are part of
    the 256D output). Squashing to action ranges is handled downstream in
    action_to_dcl_command().
    """

    def __init__(self):
        super().__init__()
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
            ),
        })
        self.action_net = nn.Linear(64, 4)

    def forward(self, features):
        latent_pi = self.mlp_extractor['policy_net'](features)
        return self.action_net(latent_pi)


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

        # Only reached if _load_weights succeeded (strict=True raises on any mismatch).
        print(f"[SCUBA Lab] [OK] Feature extractor + policy head loaded (strict=True)")

    def _load_weights(self, weights_path: str):
        """Load weights from the trained policy.pth with strict=True.

        The checkpoint is SB3 MultiInputPolicy state: feature-extractor trees are
        stored under the 'features_extractor.' prefix; the policy head is under
        'mlp_extractor.policy_net.' and 'action_net.'. We map those into this
        adapter's two modules (DroneVisionExtractor, PolicyNet) by renaming
        prefixes, then load each with strict=True so any drift between this
        adapter's topology and the saved checkpoint fails loudly instead of
        silently leaving layers at Kaiming init.

        History: the previous implementation filtered keys with a 'pi_net.' prefix
        that matched zero checkpoint keys, and called load_state_dict with
        strict=False inside try/except. Both the filter and the handwritten
        PolicyNet topology were wrong; the result was a policy head running on
        PyTorch default random init while the adapter printed a success message.
        See obsidian/fragilities.md.
        """
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Feature extractor: SB3 stores shared + per-head copies; all three are
        # identical in our training, so we load from the shared 'features_extractor.*'.
        feature_dict = {
            k.removeprefix('features_extractor.'): v
            for k, v in checkpoint.items()
            if k.startswith('features_extractor.')
        }
        self.feature_extractor.load_state_dict(feature_dict, strict=True)
        self.feature_extractor.eval()

        # Policy head: combines SB3's mlp_extractor.policy_net (two Linear layers
        # with Tanh activation between) and action_net (final Linear).
        policy_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('mlp_extractor.policy_net.'):
                # 'mlp_extractor.policy_net.0.weight' -> 'mlp_extractor.policy_net.0.weight'
                # Our PolicyNet uses nn.ModuleDict with key 'policy_net', so the
                # target key matches after stripping no prefix. We keep the same
                # leading name for symmetry with the checkpoint layout.
                policy_dict[k] = v
            elif k.startswith('action_net.'):
                policy_dict[k] = v
        self.policy.load_state_dict(policy_dict, strict=True)
        self.policy.eval()

        self._log_weight_statistics()

    def _log_weight_statistics(self):
        """Log std/mean of each loaded layer after weight load.

        Kaiming-uniform init at fan-in N has std ~= sqrt(2/N) / sqrt(3) ~= sqrt(2/(3N)):
          fan-in 256 -> ~0.051
          fan-in 128 -> ~0.072
          fan-in 64  -> ~0.102

        Trained weights typically deviate meaningfully from these values (either
        larger, from learning-signal accumulation, or with a clearly non-uniform
        distribution). If you see post-load stds that land exactly on these
        numbers, assume the weights did not load (even though we now use
        strict=True, logging is still useful for spotting future drift).
        """
        def stats(t: torch.Tensor) -> str:
            t = t.detach().float()
            return (f"shape={tuple(t.shape)} mean={t.mean().item():+.5f} "
                    f"std={t.std().item():.5f} |max|={t.abs().max().item():.5f}")

        print(f"[SCUBA Lab] Post-load weight statistics:")
        print(f"  features_extractor.coarse_cnn.0.weight : {stats(self.feature_extractor.coarse_cnn[0].weight)}")
        print(f"  features_extractor.fine_cnn.0.weight   : {stats(self.feature_extractor.fine_cnn[0].weight)}")
        print(f"  features_extractor.state_mlp.0.weight  : {stats(self.feature_extractor.state_mlp[0].weight)}")
        print(f"  policy.mlp_extractor.policy_net.0      : {stats(self.policy.mlp_extractor['policy_net'][0].weight)}")
        print(f"  policy.mlp_extractor.policy_net.2      : {stats(self.policy.mlp_extractor['policy_net'][2].weight)}")
        print(f"  policy.action_net.weight               : {stats(self.policy.action_net.weight)}")

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

        # Build image input for model.
        #
        # Original distill model (aigp_distill_final): EVENT_CAMERA_ENABLED=True
        #   Input shape: (14, 48, 48) — 2 stacked frames × (3 RGB + 4 event channels)
        #   Zero-pad event channels at inference since DCL provides RGB only.
        #
        # Fine-tuned model (aigp_finetune_tilt_final): EVENT_CAMERA_ENABLED=False
        #   Input shape: (6, 48, 48) — 2 stacked frames × 3 RGB channels only.
        #   No zero-padding needed.
        #
        # We detect which model is loaded by checking the feature extractor's
        # expected input channels (first conv layer in_channels).
        try:
            expected_channels = self.features_extractor.coarse_cnn[0].in_channels
        except AttributeError:
            expected_channels = 14  # default to original model

        rgb_frame = image.astype(np.float32)  # (3, 48, 48)

        if expected_channels == 14:
            # Original model: pad 4 zero event channels per frame, stack 2 frames
            event_channels = np.zeros((4, 48, 48), dtype=np.float32)
            frame_with_events = np.concatenate([rgb_frame, event_channels], axis=0)  # 7ch
            image_out = np.concatenate([frame_with_events, frame_with_events], axis=0)  # 14ch
        elif expected_channels == 6:
            # Fine-tuned model: RGB only, stack 2 frames
            image_out = np.concatenate([rgb_frame, rgb_frame], axis=0)  # 6ch
        else:
            # Unknown — stack RGB frames and warn
            import warnings
            warnings.warn(f"Unexpected CNN in_channels={expected_channels}; defaulting to RGB stack")
            image_out = np.concatenate([rgb_frame, rgb_frame], axis=0)

        image_14ch = image_out.astype(np.uint8)

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
            # State is already embedded inside the 256D feature vector via the
            # feature extractor's state_mlp branch; the policy head takes features
            # alone. See PolicyNet docstring.
            obs = {"image": image_tensor, "state": state_tensor}
            features = self.feature_extractor(obs)
            action = self.policy(features)

        # Convert to numpy and clamp to valid ranges. The trained policy outputs
        # the Gaussian mean (no Tanh) — clipping here enforces the competition
        # action-space bounds as a safety net.
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
