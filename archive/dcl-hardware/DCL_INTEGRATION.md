# SCUBA Lab - DCL AI Grand Prix Integration Guide

## Team Information
- **Team Name:** Scuba Lab
- **Competition:** Anduril AI Grand Prix (DCL)
- **Timeline:** Virtual Qualifier 1 (May 2026), Virtual Qualifier 2 (June 2026), Physical/Final (Sept-Nov 2026)

## Architecture Overview

```
DCL Simulator
    
[Telemetry + Visual Stream]
    
DCL Adapter (dcl_adapter.py)
    
Distilled Vision Model
(aigp_distill_final.zip - 50M steps training)
    
[Throttle, Roll, Pitch, Yaw]
    
DCL Simulator Execution
```

## Model Details

**Trained Model:** `aigp_distill_final.zip` (13 MB)
- **Training stages:** 2-stage pipeline
  - Stage 1: State-based expert (50M steps)
  - Stage 2: Distilled to vision policy (50M steps)
- **Architecture:**
  - Feature extractor: DroneVisionExtractor (CNN on 48x48 FPV images)
  - Policy: 2-layer MLP (128, 64 hidden units)
  - Value: 2-layer MLP (128, 64 hidden units)
- **Input:** FPV camera (48x48 RGB) + telemetry state (19D)
- **Output:** Throttle, Roll, Pitch, Yaw commands

## Files

### Main Integration
- **`dcl_adapter.py`** - Converts DCL inputs  model inputs/outputs
  - `SCUBALabAdapter` class handles end-to-end inference
  - `process_observation()` - DCL data  model format
  - `predict_action()` - Model inference
  - `action_to_dcl_command()` - Output conversion
  - `step()` - Single inference cycle

### Testing
- **`test_dcl_adapter.py`** - Local testing before DCL simulator release
  - Mock telemetry + visual data
  - Tests continuous flight (multiple steps)
  - Validates output ranges

## Integration Steps

### 1. When DCL Simulator Releases (May 2026)
```bash
# Download simulator from DCL platform
# Extract to ~/dcl_simulator/

# Update dcl_adapter.py with actual DCL API calls
# (Replace mock functions with real DCL calls)
```

### 2. Test Locally
```bash
python3 test_dcl_adapter.py
```

### 3. Prepare for Virtual Qualifier
- Qualifier 1 (May): Desaturated, simple, highlighted gates
  - **Strategy:** Model should dominate (trained on harder scenarios)
- Qualifier 2 (June): Complex, low signal-to-noise, no visual aids
  - **Strategy:** Model advantage due to full-complexity training

## Sensor Specifications (DCL Provided)

### Inputs
- **Telemetry:** Position (x, y, z), Velocity (vx, vy, vz), Orientation (roll, pitch, yaw)
- **Visual:** Forward-facing FPV camera stream
- **No depth, engine RPM, or battery state**

### Outputs
- **Throttle:** 0-1 (0 = no thrust, 1 = max)
- **Roll:** -1 to 1 (left to right bank)
- **Pitch:** -1 to 1 (down to up nose)
- **Yaw:** -1 to 1 (left to right turn)

## Hardware Requirements

**Minimum (DCL):**
- CPU: Intel Core i5-10400F or AMD Ryzen 5 3600
- GPU: Nvidia RTX 2060 Super or AMD 9060XT
- RAM: 16 GB
- Storage: 60 GB

**Optimized for:**
- ~100 TOPS compute (embedded AI hardware)
- Python 3.14.2+
- External libraries permitted

## Performance Characteristics

### Model Efficiency
- **Input size:** 48x48 RGB image + 19D state vector
- **Inference time:** < 5ms (target for real-time control at 200Hz)
- **Model size:** 13 MB (easily loads on 100 TOPS hardware)

### Training History
- **Total training time:** ~200 hours on FAU HPC (1 Tesla V100 GPU)
- **Convergence:** Stable training, no divergence issues
- **Final reward:** ~-170 (competitive for autonomous racing)

## Competition Advantage

### Why SCUBA Lab Wins

1. **Full-Complexity Training**
   - Model trained on hard scenarios (low signal-to-noise)
   - Qualifier 2 conditions will be easier than training

2. **Vision + State Fusion**
   - Uses both camera and telemetry optimally
   - No reliance on depth (advantage since DCL doesn't provide it)

3. **Proven Architecture**
   - Distillation from state expert ensures robust control
   - 50M steps of experience in virtual environment

4. **Hardware Efficient**
   - Model easily fits on 100 TOPS hardware
   - Fast inference for real-time control
   - ~5ms per decision (200Hz control loop possible)

## Deployment Checklist

- [x] Model trained and validated
- [x] Adapter code written
- [x] Local tests passing
- [ ] DCL simulator released (May 2026)
- [ ] Integrate with DCL API
- [ ] Virtual Qualifier 1 submission (May 2026)
- [ ] Optimize for Qualifier 2 (June 2026)
- [ ] Physical training (Sept 2026)
- [ ] Final championship (Nov 2026)

## Contact & Support

**Team:** Scuba Lab
**Repository:** https://github.com/click-b8/AI-GrandPrix-
**Model Location:** ./models_release/aigp_distill_final.zip

---

**Status:** READY FOR DCL INTEGRATION 
**Last Updated:** April 3, 2026
