# May Qualifier - Pure Simulation Strategy

**STATUS:** No real drone access. All work done via pure simulation.

---

## 🎯 Revised Approach

Since we're not pursuing real drone testing, we optimize through:

1. **Aggressive domain randomization** - Make models robust to hardware variations
2. **Extensive local benchmarking** - Validate model performance on diverse conditions
3. **Vision-based primary** - `aigp_distill_final.zip` more resilient to sim-to-real gap
4. **Statistical validation** - 100+ test runs to ensure reliability

---

## 📊 Model Selection (Pure Sim)

### **PRIMARY: `aigp_distill_final.zip` (Vision Policy)**
- ✅ CNN-based, perception-grounded (more robust)
- ✅ 13MB (manageable size)
- ✅ Trained on FPV + event camera (realistic sensor)
- ✅ Better sim-to-real transfer properties
- **Risk:** Slightly slower inference (mitigated by distillation)

### **BACKUP: `aigp_racer_final.zip` (Swift Racer)**
- ✅ State-based, 100M steps (well-trained)
- ✅ Proven fast on HPC
- ✅ Small model (504KB)
- **Risk:** Pure state-based (sensitive to sim accuracy)

### **FALLBACK: `aigp_8gates_final.zip` (8-Gate Racer)**
- ✅ Aggressive banking (120° roll)
- ✅ Specialized for gate racing
- **Risk:** Less training time, potential overfitting

---

## 📋 Immediate Implementation (Week 1)

### **LOCAL WORK (This Week)**

1. **Measure inference latency**
   ```bash
   python3 measure_inference_time.py
   ```
   - Ensures <50ms per decision
   - Validates target hardware compatibility

2. **Create domain-robust variants**
   ```bash
   python3 train_domain_robust.py
   ```
   - Fine-tune with ±25% mass, ±30% drag variations
   - Add motor latency (+10ms)
   - 10M steps on each model

3. **Validate robustness locally**
   ```bash
   python3 benchmark_hpc_models.py
   ```
   - Run each model 10x locally
   - Quick validation before HPC
   - Check gate completion rates

### **HPC WORK (This Week - Parallel)**

1. **Submit benchmarking job**
   ```bash
   sbatch submit_benchmark.sh
   ```
   - Runs `benchmark_hpc_final.py`
   - 20 laps per model
   - Generates comprehensive statistics
   - Results → Model selection decision

2. **Expected output:**
   - Success rates (target: >95%)
   - Lap time statistics
   - Consistency metrics
   - Failure mode analysis

---

## 🔄 Week 2-3 Plan (April)

### **Week 2: Optimization**
- [ ] Analyze HPC benchmark results
- [ ] Select primary model based on success rate
- [ ] If success <95%: Continue domain randomization training
- [ ] If success ≥95%: Prepare for submission packaging

### **Week 3: Final Validation**
- [ ] Run 100x trials on selected model (local + HPC)
- [ ] Verify 100% gate completion
- [ ] Test inference performance on target hardware
- [ ] Prepare submission package

---

## 💪 Robustness Strategy (No Real Drone)

Since we can't test on real hardware, we maximize sim robustness:

### Domain Randomization Parameters
```python
mass_scale: (0.75, 1.25)           # ±25% mass variation
drag_scale: (0.70, 1.30)           # ±30% drag coefficient
motor_latency: 10ms                # Add control latency
obs_noise_scale: 0.05              # 5% sensor noise
init_pos_noise: 0.1                # Random start positions
init_yaw_noise: 0.3                # Random orientations
```

### Testing with Variations
- Run models with domain_rand=True 20+ times
- Measure consistency across variations
- Identify failure modes under perturbations
- Ensure >95% completion rate

---

## 📊 Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| Gate completion | 100% (8/8) | 100 trial runs |
| Consistency | >95% success | Domain randomization tests |
| Lap time | Beat competitors | HPC benchmarks |
| Inference latency | <50ms | Measure on target hardware |
| Robustness | Handles variations | Domain rand testing |

---

## 🚀 Implementation Timeline

```
TODAY (Mar 26)
    ├─ Inference latency measurement ✓
    ├─ Domain randomization training (parallel with HPC)
    └─ Submit HPC benchmarking job
            ↓
MARCH 27-28 (HPC running + local training)
    ├─ Domain-robust model training (~30-60min each)
    ├─ Local quick validation
    └─ Monitor HPC job progress
            ↓
MARCH 29-30 (HPC results analysis)
    ├─ Review benchmark statistics
    ├─ Model selection decision
    └─ Prepare final model
            ↓
MARCH 31 (Final validation)
    ├─ 100x test runs on selected model
    └─ Submission package ready
            ↓
APRIL 1-15 (Buffer + additional testing)
    └─ Any needed fine-tuning
            ↓
MAY (Virtual Qualifier)
    └─ 🏆 DEPLOY & WIN
```

---

## 🎯 Model Selection Decision Tree

**After HPC Benchmarking (March 29):**

```
IF aigp_distill_final success >= 95%:
    ✅ USE AS PRIMARY
    └─ Vision-based robust to sim-to-real gap
ELSE IF aigp_racer_final success >= 95%:
    ✅ USE AS PRIMARY
    └─ State-based, well-trained, fast
ELSE:
    → Continue domain randomization training
    → Re-benchmark after 10M steps
    → Select best performer
```

---

## ✅ Submission Checklist

- [ ] Primary model selected (success rate ≥95%)
- [ ] Backup model ready (different type)
- [ ] Inference latency <50ms verified
- [ ] 100x validation runs completed
- [ ] Domain randomization testing passed
- [ ] Code packaged and documented
- [ ] Submission file ready to deploy

---

## 📝 Files Ready

✅ `benchmark_hpc_final.py` - HPC benchmarking (20x each model)
✅ `train_domain_robust.py` - Domain randomization fine-tuning
✅ `measure_inference_time.py` - Local latency measurement
✅ `submit_benchmark.sh` - HPC SLURM submission

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Sim-to-real gap unknown | Aggressive domain randomization |
| Model too slow | Quantization or distilled version |
| Low completion rate | Continue training with curriculum |
| Last-minute failures | Have 3 backup models ready |

---

**Strategy Owner:** Claude (AI Assistant)
**Last Updated:** 2026-03-26
**Status:** READY FOR EXECUTION
