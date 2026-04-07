"""
SCUBA Lab DCL Adapter - Test Suite
Validates adapter before DCL simulator integration
"""

import numpy as np
import os
from dcl_adapter import SCUBALabAdapter


def test_adapter():
    """Test the DCL adapter with mock DCL data"""

    print("=" * 60)
    print("SCUBA LAB - DCL ADAPTER TEST SUITE")
    print("=" * 60)

    # Load adapter (model is in the AI GrandPrix directory)
    model_path = os.path.expanduser('~/Desktop') + '/AI GrandPrix/AI_GrandPrix_Models/aigp_distill_final.zip'

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Download the model from HPC first!")
        return False

    adapter = SCUBALabAdapter(model_path)

    # Test 1: Mock DCL input
    print("\n[Test 1] Processing mock DCL inputs...")
    telemetry = {
        'position': [0.0, 0.0, 1.0],          # x, y, z
        'velocity': [1.0, 0.0, 0.0],          # vx, vy, vz
        'orientation': [0.0, 0.0, 0.0],       # roll, pitch, yaw
    }

    # Mock visual data (48x48x3 image, 0-255)
    visual_data = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)

    command = adapter.step(telemetry, visual_data)
    print(f"✓ Input processed successfully")
    print(f"  Throttle: {command['throttle']:.3f}")
    print(f"  Roll:     {command['roll']:.3f}")
    print(f"  Pitch:    {command['pitch']:.3f}")
    print(f"  Yaw:      {command['yaw']:.3f}")

    # Test 2: Multiple steps (simulating continuous flight)
    print("\n[Test 2] Simulating continuous flight (10 steps)...")
    for step in range(10):
        # Update telemetry (simulate moving forward)
        telemetry['position'][0] += command['throttle'] * 0.1
        visual_data = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)

        command = adapter.step(telemetry, visual_data)
        print(f"  Step {step+1}: throttle={command['throttle']:.2f}, "
              f"roll={command['roll']:.2f}, pitch={command['pitch']:.2f}, "
              f"yaw={command['yaw']:.2f}")

    print("\n✓ All tests passed!")
    print("=" * 60)
    print("SCUBA LAB ADAPTER READY FOR DCL COMPETITION")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = test_adapter()
    exit(0 if success else 1)
