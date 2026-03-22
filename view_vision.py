"""View trained vision policy — FPV camera + event camera (what the model sees)."""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from gymnasium import spaces

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from drone_race_env import DroneRaceEnv
from config import (
    FPV_RESOLUTION, FPV_FRAME_STACK, VISION_STATE_DIM,
    EVENT_CAMERA_ENABLED, EVENT_FRAME_BINS,
)


class DroneVisionExtractor(BaseFeaturesExtractor):
    """Must match the training architecture exactly."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        img_space = observation_space["image"]
        n_channels = img_space.shape[0]
        img_h, img_w = img_space.shape[1], img_space.shape[2]

        self.coarse_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.fine_cnn = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, img_h, img_w)
            coarse_out = self.coarse_cnn(dummy)
            coarse_flat = coarse_out.flatten(1).shape[1]
            fine_out = self.fine_cnn(coarse_out)
            fine_flat = fine_out.flatten(1).shape[1]

        self.coarse_fc = nn.Sequential(nn.Linear(coarse_flat, 64), nn.ReLU())
        self.fine_fc = nn.Sequential(nn.Linear(fine_flat, 128), nn.ReLU())

        state_dim = observation_space["state"].shape[0]
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

    def forward(self, observations):
        img = observations["image"].float() / 255.0
        coarse_maps = self.coarse_cnn(img)
        coarse_features = self.coarse_fc(coarse_maps.flatten(1))
        fine_maps = self.fine_cnn(coarse_maps)
        fine_features = self.fine_fc(fine_maps.flatten(1))
        state_features = self.state_mlp(observations["state"].float())
        return torch.cat([coarse_features, fine_features, state_features], dim=1)


def main():
    model_path = './trained_vision/best_model/best_model.zip'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print(f'Loading model: {model_path}')

    policy_kwargs = dict(
        features_extractor_class=DroneVisionExtractor,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # No render_mode="human" — we render FPV ourselves
    env = DroneRaceEnv(
        render_mode=None,
        domain_rand=False,
        difficulty=1.0,
        vision_mode=True,
    )

    model = PPO.load(model_path, env=env, custom_objects={
        'policy_kwargs': policy_kwargs,
    })

    print('Running policy with FPV camera view (what the model sees).')
    print('Press Q to quit, N for next episode.')

    display_scale = 8  # Scale up 48x48 -> 384x384 for visibility
    window_name = 'Drone FPV - Model View'
    event_window = 'Event Camera' if EVENT_CAMERA_ENABLED else None
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FPV_RESOLUTION * display_scale,
                     FPV_RESOLUTION * display_scale)
    if event_window:
        cv2.namedWindow(event_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(event_window, FPV_RESOLUTION * display_scale,
                         FPV_RESOLUTION * display_scale)

    for episode in range(10):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Get the current FPV frame (what the CNN sees)
            fpv_frame = env._render_fpv()  # (48, 48, 3) RGB

            # Scale up for display
            display = cv2.resize(fpv_frame, None, fx=display_scale,
                                 fy=display_scale,
                                 interpolation=cv2.INTER_NEAREST)

            # Add HUD overlay
            gates = info.get('gates_passed', 0)
            total_gates = info.get('total_gates', 0)
            speed = info.get('speed', 0)
            hud = f'Gates: {gates}/{total_gates}  Speed: {speed:.1f}m/s  R: {total_reward:.0f}'
            cv2.putText(display, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # Convert RGB -> BGR for OpenCV display
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_bgr)

            # Visualize event camera output
            if event_window and env._event_sensor is not None:
                event_frame = env._event_sensor.process_frame(fpv_frame)
                # Sum all bins, show pos=green, neg=red
                pos_total = np.zeros((FPV_RESOLUTION, FPV_RESOLUTION), dtype=np.float32)
                neg_total = np.zeros((FPV_RESOLUTION, FPV_RESOLUTION), dtype=np.float32)
                for b in range(EVENT_FRAME_BINS):
                    pos_total += event_frame[:, :, 2 * b].astype(np.float32)
                    neg_total += event_frame[:, :, 2 * b + 1].astype(np.float32)
                event_vis = np.zeros((FPV_RESOLUTION, FPV_RESOLUTION, 3), dtype=np.uint8)
                event_vis[:, :, 1] = np.clip(pos_total * 80, 0, 255).astype(np.uint8)  # green=ON
                event_vis[:, :, 2] = np.clip(neg_total * 80, 0, 255).astype(np.uint8)  # red=OFF
                event_display = cv2.resize(event_vis, None, fx=display_scale,
                                           fy=display_scale,
                                           interpolation=cv2.INTER_NEAREST)
                cv2.imshow(event_window, event_display)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return
            if key == ord('n'):
                break

            if terminated or truncated:
                print(f'  Episode {episode+1}: reward={total_reward:.1f}, '
                      f'steps={steps}, gates={gates}/{total_gates}, '
                      f'speed={speed:.1f} m/s')
                cv2.waitKey(1000)  # Pause 1s between episodes
                break

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
