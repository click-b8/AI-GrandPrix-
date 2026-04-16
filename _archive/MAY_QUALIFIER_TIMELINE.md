# AI Grand Prix May Virtual Qualifier
## 6-Week Countdown to Competition

---

## 📅 TIMELINE

### **WEEK 1-2 (Now - Early April)**
**Goal: Establish baseline, get real drone if possible**

- [ ] **Day 1-2: Real drone access**
  - Contact Anduril/FAU for drone time
  - If available: Setup hardware testing lab
  - If unavailable: Prepare pure sim pipeline

- [ ] **Day 3-5: Benchmark on HPC**
  - Run `aigp_racer_final.zip` 20x (timing + completion)
  - Run `aigp_8gates_final.zip` 20x
  - Run `aigp_distill_final.zip` 20x
  - Document baseline performance

- [ ] **Day 6-7: Real drone testing (if available)**
  - Deploy `aigp_distill_final.zip` on hardware
  - Measure real lap time
  - Identify sim-to-real gap
  - Collect domain randomization parameters

---

### **WEEK 3 (Mid-April)**
**Goal: Optimize fastest model**

- [ ] **If real drone showed gap:**
  - Fine-tune best model with real hardware parameters
  - Add actual sensor noise to simulation
  - Test 10 runs on hybrid sim+real

- [ ] **If pure simulation:**
  - Aggressive domain randomization
  - Stress test for edge cases

- [ ] **Speed optimization:**
  - Profile inference time
  - Quantize if needed (<10ms per decision)

- [ ] **Reliability hardening:**
  - 100x trial runs on HPC
  - Verify 0 crashes, 100% completion

---

### **WEEK 4 (Late April - 1 week before)**
**Goal: Final model selection & submission prep**

- [ ] **Model selection**
  - Choose final submission model
  - Prepare 2-3 backup models
  - Create inference wrapper

- [ ] **Documentation**
  - Model specs (training steps, params, performance)
  - Deployment instructions
  - Hardware requirements

- [ ] **Final validation**
  - 50x test runs on competition environment (if available)
  - Edge case testing
  - Stress testing

---

### **WEEK 5 (April - Week of qualifier)**
**Goal: Ready to deploy**

- [ ] **Submission package**
  - Model files
  - Code (inference + environment)
  - Documentation
  - Logs from testing

- [ ] **Pre-flight checks**
  - All systems operational
  - Model loads correctly
  - Inference works end-to-end

---

### **WEEK 6 (May - Qualifier week)**
**Goal: WIN 🏆**

- [ ] Deploy model
- [ ] Run qualifier
- [ ] Post-mortem & improvements for finals

---

## 🎯 MODEL SELECTION FOR MAY

### **Best Bet: `aigp_racer_final.zip`**
- 100M steps of championship tuning
- Proven fast on HPC
- Well-tested baseline
- **Risk:** Pure state-based (needs simulation accuracy)

### **If Real Drone Available: `aigp_distill_final.zip`**
- Vision-based (camera input)
- Better sim-to-real potential
- Fine-tune with real hardware data
- **Risk:** Larger (13MB), slower inference

### **If Need Speed: `aigp_8gates_final.zip`**
- Aggressive banking (120° roll)
- Optimized for speed
- Smaller model
- **Risk:** Less training time, might be overfit

---

## 🔧 CRITICAL PATH

```
NOW ─→ REAL DRONE TEST ─→ FINE-TUNE ─→ VALIDATE ─→ MAY QUALIFIER ─→ WIN
      (Week 1-2)        (Week 3)      (Week 4)      (Week 5-6)
```

**Blocker:** Real drone access
- **If available:** Use it (high priority)
- **If unavailable:** Pure sim (feasible but riskier)

---

## 📊 SUCCESS METRICS FOR MAY

| Metric | Target | Current Status |
|--------|--------|-----------------|
| Lap time | Beat competitors | Unknown (need baseline) |
| Gate completion | 100% (8/8) | Simulated: ~90% |
| Reliability | 0 crashes | Unknown |
| Inference time | <50ms | Unknown (measure Week 1) |
| Robustness | Works on real hardware | Unknown (need drone) |

---

## 🚨 RISKS & MITIGATIONS

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Real drone unavailable | Medium | Aggressive domain randomization |
| Simulation inaccurate | Low-Medium | Hardware param validation |
| Model too slow | Low | Pre-quantize, measure inference |
| Low completion rate | Low | Ensemble voting backup |
| Last-minute issues | Medium | Have 3 backup models ready |

---

## 💡 WINNING STRATEGY

### **Scenario A: Real Drone Available**
1. ✅ Test `aigp_distill_final` on hardware (Week 1-2)
2. ✅ Fine-tune with real data (Week 3)
3. ✅ Validate on drone 20x (Week 4)
4. ✅ Submit refined model to qualifier (Week 5-6)
→ **Best chance to win** 🏆

### **Scenario B: Pure Simulation**
1. ✅ Baseline all 3 models (Week 1-2)
2. ✅ Select fastest + most reliable (Week 2)
3. ✅ Aggressive domain randomization (Week 3)
4. ✅ 100x validation runs (Week 4)
5. ✅ Submit hardest model (Week 5-6)
→ **Good chance, less confident** 🥈

---

## 📋 IMMEDIATE ACTIONS (This Week)

**MUST DO:**
- [ ] Contact Anduril/FAU about real drone access
- [ ] Run HPC benchmarks (20x per model)
- [ ] If drone available: Schedule testing session
- [ ] Measure inference time on target hardware

**SHOULD DO:**
- [ ] Review May qualifier rules/format
- [ ] Research competitor approaches
- [ ] Document domain randomization parameters

---

## 🎯 RECOMMENDATION

**Go all-in on real drone testing this week.**

That single data point (real lap time vs sim) will tell you everything:
- If gap is small (<5%): Use `aigp_racer_final` (faster)
- If gap is large (>10%): Use `aigp_distill_final` (vision, adaptable)

**Without real drone data, you're guessing.** With 6 weeks, you have time to optimize.

---

## ✅ CHECKLIST FOR MAY SUBMISSION

- [ ] Model selected & tested 50+ times
- [ ] Inference latency <50ms
- [ ] Gate completion rate 100%
- [ ] Robustness to domain variations verified
- [ ] Code packaged & tested
- [ ] Documentation complete
- [ ] Backup models ready
- [ ] All systems operational
- [ ] Ready to deploy 🚀

---

**GOAL:** Win AI Grand Prix May Virtual Qualifier
**CONFIDENCE:** High (if real drone testing done)
**NEXT STEP:** Get real drone this week
