"""
SCUBA LAB - DCL AI Grand Prix Adapter (Direct Weight Loading)
Bridges trained distilled vision model to DCL competition API

This version loads weights directly instead of using PPO.load() to avoid
version compatibility issues with SB3.
"""

import logging
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from vision_model import DroneVisionExtractor
from config import FPV_FRAME_STACK, GRAVITY

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation helpers — copied from drone_race_env._rotmat_to_6d / _quat_to_rotmat.
# Must stay in sync with the training env.
# ---------------------------------------------------------------------------

def _euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles (rad) -> 3x3 body-to-world rotation matrix.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Matches _quat_to_rotmat() in the training env (verified against it for
    pure roll, pitch, yaw cases).  See obsidian/state-vector-fix.md.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr            ],
    ], dtype=np.float32)


def _rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened -> 6D continuous rotation (Zhou CVPR 2019).

    Identical to _rotmat_to_6d in drone_race_env.py.
    """
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


# NED -> training-frame basis change.
#
# The DCL sim reports attitude in NED: z-down world, FRD body (Forward-Right-
# Down). The training env (MuJoCo) is z-up world, FLU body (Forward-Left-Up).
# The two differ by a 180 deg rotation about the forward (x) axis:
#     C = diag(1, -1, -1) = Rx(pi)
# It keeps the forward axis (heading), flips the lateral (right<->left) and
# vertical (down<->up) axes — exactly the FRD<->FLU / z-down<->z-up relabeling.
#
# Applied to the rotation matrix as a TWO-SIDED similarity:
#     R_train = C @ R_ned @ C
# (C is symmetric and its own inverse, so C == C.T == C^-1). This keeps the
# result a proper rotation (det +1). A naive element/row flip of R_ned would
# be a reflection (det -1) and corrupt the orientation — that is the trap we
# are avoiding. After this transform a nose-up pitch yields a forward axis with
# POSITIVE world-z (up), matching what the z-up-trained policy expects, instead
# of the inverted forward-down that drove the full-thrust tumble.
_NED_TO_TRAIN = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _ned_to_train_rotmat(R_ned: np.ndarray) -> np.ndarray:
    """Re-express a body->NED rotation as a body->training-world rotation.

    R_train = C @ R_ned @ C, with C = diag(1, -1, -1). Stays a proper rotation;
    cross-checked against the training env's _quat_to_rotmat in the test suite.
    """
    return (_NED_TO_TRAIN @ R_ned @ _NED_TO_TRAIN).astype(np.float32)


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

    def __init__(self, model_path: str, weights_path: str = None, device: str = None):
        """
        Initialize SCUBA Lab adapter with trained distilled model

        Args:
            model_path:   Path to trained aigp_distill_final.zip
            weights_path: Path to policy.pth weights file (extracted from zip if None)
            device:       'cuda', 'mps', or 'cpu'.  None = auto-select: CUDA > MPS > CPU.
        """
        import os
        import shutil
        import tempfile
        import zipfile

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"[SCUBA Lab] Device: {self.device}")
        print(f"[SCUBA Lab] Loading model: {model_path}")

        # Extract policy.pth from zip and load into memory; clean up temp files immediately.
        if weights_path is None:
            _tmpdir = tempfile.mkdtemp()
            try:
                with zipfile.ZipFile(model_path, 'r') as z:
                    z.extract('policy.pth', _tmpdir)
                checkpoint = torch.load(
                    os.path.join(_tmpdir, 'policy.pth'),
                    map_location=self.device,
                    weights_only=False,
                )
            finally:
                shutil.rmtree(_tmpdir, ignore_errors=True)
        else:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Detect input channel count from the first conv weight before building the model.
        # Finetune (EVENT_CAMERA_ENABLED=False): 6ch = 2 RGB frames × 3ch.
        # Distill  (EVENT_CAMERA_ENABLED=True):  14ch = 2 frames × (3 RGB + 4 event) ch.
        # Building the model with the wrong channel count causes strict=True to fail.
        first_weight = checkpoint.get('features_extractor.coarse_cnn.0.weight')
        n_channels = int(first_weight.shape[1]) if first_weight is not None else 14
        expected_ch = 3 * FPV_FRAME_STACK   # 9 for N=3: RGB-only 10-D vision policy
        print(f"[SCUBA Lab] Checkpoint input channels: {n_channels} (expected {expected_ch})")
        assert n_channels == expected_ch, (
            f"10-D vision policy expects {expected_ch} image channels "
            f"(3 RGB x {FPV_FRAME_STACK} frames); checkpoint has {n_channels}. "
            f"Point CANONICAL_MODEL_PATH at a 10-D/{expected_ch}-ch checkpoint."
        )

        obs_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(n_channels, 48, 48), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
        })
        self.n_channels = n_channels
        self.feature_extractor = DroneVisionExtractor(obs_space).to(self.device)
        self.policy = PolicyNet().to(self.device)

        self._load_weights(checkpoint)
        print(f"[SCUBA Lab] [OK] Feature extractor + policy head loaded (strict=True)")
        self._last_state_log = 0.0
        self._prev_action = np.zeros(4, dtype=np.float32)  # [throttle, roll, pitch, yaw]
        self._frame_deque = deque(maxlen=FPV_FRAME_STACK)  # real oldest->newest frame stack

    def _load_weights(self, checkpoint: dict):
        """Load weights from a pre-loaded checkpoint dict with strict=True.

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
        feature_dict = {
            k.removeprefix('features_extractor.'): v
            for k, v in checkpoint.items()
            if k.startswith('features_extractor.')
        }
        self.feature_extractor.load_state_dict(feature_dict, strict=True)
        self.feature_extractor.eval()

        policy_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('mlp_extractor.policy_net.'):
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

    def reset_observation(self):
        """Clear per-race observation state (frame deque + prev_action). Called on
        the GO edge. This is the ONLY method that empties _frame_deque — the
        _stack_frames() warmup contract (fill-on-empty) depends on that."""
        self._frame_deque.clear()
        self._prev_action = np.zeros(4, dtype=np.float32)

    @staticmethod
    def _build_state_10d(gravity_frd, gyro_frd, prev_action):
        """Pure 10-D state builder (no model/torch), matching drone_race_env
        _get_minimal_state: gravity_unit(3, FLU) + body_rates(3, FLU) + prev_action(4).
        gravity_frd (norm~=g) and gyro_frd are FRD; mapped to FLU via C=_NED_TO_TRAIN."""
        g = np.asarray(gravity_frd, dtype=np.float32)
        g_unit_flu = _NED_TO_TRAIN @ (g / (np.linalg.norm(g) + 1e-9))        # FRD->FLU, unit
        body_rates = _NED_TO_TRAIN @ np.asarray(gyro_frd, dtype=np.float32)  # FRD->FLU
        return np.concatenate(
            [g_unit_flu, body_rates, np.asarray(prev_action, dtype=np.float32)]
        ).astype(np.float32)

    def _stack_frames(self, rgb_hwc):
        """Append a 48x48x3 frame; stack oldest->newest along channels -> (9,48,48)
        CHW. Warmup fills the deque with the first frame (matches env reset). The
        deque is empty only at construction and after reset_observation()."""
        if not self._frame_deque:                       # empty only at init / after reset
            for _ in range(self._frame_deque.maxlen):
                self._frame_deque.append(rgb_hwc.copy())
        else:
            self._frame_deque.append(rgb_hwc.copy())
        stacked_hwc = np.concatenate(list(self._frame_deque), axis=2)   # (48,48,9)
        return np.transpose(stacked_hwc, (2, 0, 1)).astype(np.uint8)    # (9,48,48)

    def process_observation(self, telemetry: dict, visual_data: np.ndarray) -> tuple:
        """Convert DCL inputs to the 10-D vision+IMU observation.

        image: last FPV_FRAME_STACK RGB frames, 48x48, stacked oldest->newest
               along channels -> (3*N, 48, 48) CHW uint8 (matches the env).
        state: gravity_unit(3, FLU) + body_rates(3, FLU) + prev_action(4) = 10-D.
               gravity from the A2 GravityEstimator (run in the run_vq1 RX loop,
               published as telemetry['gravity_frd'], FRD); body rates from
               HIGHRES_IMU gyro (telemetry['velocity'], FRD). Both FRD->FLU via C.
        """
        # Resize to 48x48 RGB if needed, then build the oldest->newest frame stack.
        if visual_data.shape[:2] != (48, 48):
            from PIL import Image
            visual_data = np.array(
                Image.fromarray(visual_data).resize((48, 48), Image.BILINEAR)
            )
        rgb_hwc = visual_data[..., :3].astype(np.uint8)          # (48,48,3)
        image_input = self._stack_frames(rgb_hwc)                # (9,48,48) CHW uint8

        # Build the 10-D state: gravity_unit(3, FLU) + body_rates(3, FLU) + prev_action(4).
        # gravity_frd comes from the A2 GravityEstimator (run_vq1 RX loop); the old
        # rot_6d/lin_vel/position dims are gone — the sim blocks ATTITUDE/
        # LOCAL_POSITION_NED, so those telemetry sources no longer exist.
        state = self._build_state_10d(
            telemetry.get("gravity_frd", (0.0, 0.0, GRAVITY)),
            telemetry.get("velocity", (0.0, 0.0, 0.0)),
            self._prev_action,
        )

        now = time.time()
        if now - self._last_state_log >= 1.0:
            self._last_state_log = now
            _log.info(
                "[State 10D fed to model]\n"
                "  d0-2  gravity_unit FLU  = [%+.3f %+.3f %+.3f]\n"
                "  d3-5  body_rates FLU    = [%+.3f %+.3f %+.3f]\n"
                "  d6-9  prev_action       = [%+.3f %+.3f %+.3f %+.3f]",
                state[0], state[1], state[2],
                state[3], state[4], state[5],
                state[6], state[7], state[8], state[9],
            )

        image_tensor = torch.from_numpy(image_input).float().unsqueeze(0).to(self.device)  # (1,9,48,48)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)        # (1,10)
        return image_tensor, state_tensor

    def predict_action(self, image_tensor: torch.Tensor, state_tensor: torch.Tensor) -> np.ndarray:
        """
        Get drone command from model

        Args:
            image_tensor: Processed image tensor (1, n_channels, 48, 48)
            state_tensor: Processed state tensor (1, 10)

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

        # Store for next step's prev_action dim (d6-9 of the 10D state).
        self._prev_action = action.copy()

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
