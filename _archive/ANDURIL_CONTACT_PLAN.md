# Anduril Real Drone Access - Contact & Coordination

## Objective
Secure real drone hardware access to test `aigp_distill_final.zip` and measure the simulation-to-reality gap for the May virtual qualifier.

**Timeline:** This week (by March 31, 2026) to allow time for testing in Week 2

---

## Contact Information Needed

### Primary Contacts
- **Anduril Industries**
  - Competition coordinator: [TO BE FOUND]
  - Email: [TO BE FOUND]
  - Phone: [TO BE FOUND]

- **Florida Atlantic University (FAU)**
  - Robotics/Drone Lab: [TO BE FOUND]
  - Contact: [TO BE FOUND]

### Questions to Ask
1. Is drone hardware available for testing during March 27-31?
2. What is the exact hardware configuration (model, battery, sensors)?
3. Can we perform 5-10 test flights of our vision model?
4. Can we log sensor data (camera frames, IMU, control outputs)?
5. What safety/insurance requirements are needed?
6. Location and scheduling availability?

---

## Testing Plan (if drone available)

### Setup (1-2 hours)
- Deploy `aigp_distill_final.zip` on drone
- Validate camera input matches simulation (48×48 RGB + event camera simulation)
- Set up safety zone and monitoring

### Testing Phase (2-3 hours)
- Run model 10 times on actual track
- Measure:
  - Actual lap time vs simulated
  - Gate completion rate
  - Real sensor data (camera frames, control latency)
  - Crash modes and failure conditions
- Log telemetry for post-analysis

### Data Collection
- Raw camera frames from real flights
- IMU/accelerometer data
- Motor command timing
- Actual vs simulated trajectory comparison
- Crash/failure logs

---

## Fallback Strategy (if drone unavailable)

If real drone access not available by March 31:

1. **Aggressive Domain Randomization**
   - Increase mass variation: ±20%
   - Add sensor noise: camera blur, latency
   - Motor response lag: +5-10ms
   - Run 100x trials with domain rand enabled

2. **Use Vision Model as Primary**
   - `aigp_distill_final.zip` is more robust to perception variations
   - Better suited for real hardware even with sim-to-real gap

3. **Hardware Parameter Estimation**
   - Research Anduril drone specs online
   - Adjust MuJoCo simulation to match hardware specs
   - Fine-tune with estimated parameters

---

## Timeline

| Date | Task | Status |
|------|------|--------|
| 2026-03-26 | Contact Anduril/FAU | ⏳ TODO |
| 2026-03-27 | Confirm availability & scheduling | ⏳ TODO |
| 2026-03-28 | Test flights (if available) | ⏳ TODO |
| 2026-03-29 | Analyze data & adjust models | ⏳ TODO |
| 2026-03-31 | Finalize model selection | ⏳ TODO |

---

## Expected Outcomes

### Best Case (Real Drone Available)
- ✅ Measure actual sim-to-real gap
- ✅ Collect real sensor data for fine-tuning
- ✅ Validate gate detection on real hardware
- ✅ Confidence boost for model selection

### Acceptable Case (Real Drone Unavailable)
- ✅ Aggressive domain randomization applied
- ✅ Fallback model (`aigp_distill_final`) selected
- ✅ Proceed with pure simulation approach
- ⚠️ Lower confidence in real hardware performance

---

## Success Criteria

**Drone Testing Success:**
- [ ] 8/8 gates completed in at least 7/10 flights
- [ ] Actual lap time within 10% of simulation
- [ ] Sensor data collected for analysis
- [ ] No critical failures/crashes

**Model Ready for Qualifier:**
- [ ] sim-to-real gap quantified
- [ ] Domain randomization parameters documented
- [ ] Final model selected with confidence
- [ ] Backup model prepared

---

## Next Steps

1. **IMMEDIATE:** Find contact information for Anduril/FAU competition team
2. **Send inquiry:** Request drone access for March 28-29
3. **Prepare deployment:** Have inference script ready for real hardware
4. **Backup plan:** Queue aggressive domain randomization training if needed

---

**Status:** BLOCKED - Awaiting contact information
**Priority:** CRITICAL - Determines model selection strategy
