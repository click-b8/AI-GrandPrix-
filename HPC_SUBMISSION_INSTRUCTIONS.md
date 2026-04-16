# HPC Job Submission Instructions

## Situation
SSH authentication to HPC is currently blocked ("Too many authentication failures"). You'll need to submit the benchmarking job manually.

## Files Ready for Submission

Located in: `/Users/mpcrmini2/Desktop/AI GrandPrix/drone-race-sim/`

### Files to copy to HPC:
1. **submit_benchmark.sh** - SLURM job script (ready to go)
2. **benchmark_hpc_final.py** - Benchmarking script (already on HPC from git)

## Option 1: Manual SSH Submission (If SSH works for you)

```bash
# 1. Copy script to HPC
scp submit_benchmark.sh nbrande2020@athene-login.hpc.fau.edu:/home/nbrande2020/

# 2. SSH into HPC
ssh nbrande2020@athene-login.hpc.fau.edu

# 3. Navigate to drone-race-sim directory
cd /home/nbrande2020/AI-GrandPrix/drone-race-sim

# 4. Submit the job
sbatch /home/nbrande2020/submit_benchmark.sh

# 5. Check job status
squeue --user=nbrande2020
```

## Option 2: Upload via Browser/File Transfer

If SSH is blocked, use your HPC cluster's web portal or file transfer interface:
1. Upload `submit_benchmark.sh` to `/home/nbrande2020/`
2. SSH in and run: `sbatch submit_benchmark.sh`

## Option 3: Clear SSH Auth and Retry

The "Too many authentication failures" can sometimes be reset by waiting 5-10 minutes or reconnecting from a fresh terminal.

```bash
# Fresh terminal - try SSH again
ssh -i ~/.ssh/id_ed25519 nbrande2020@athene-login.hpc.fau.edu
```

## What the Job Does

```bash
#!/bin/bash
#SBATCH --job-name=aigp-bench-may
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Runs: python3 benchmark_hpc_final.py
# This tests aigp_8gates_final.zip, aigp_racer_final.zip, aigp_distill_final.zip
# 20 laps each
# Results saved to ./benchmark_results/
# Estimated time: 2-4 hours
```

## Expected Output

After job completes (1-2 days):
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS.json` - Comprehensive results
- Success rates for each model
- Lap time statistics
- Consistency metrics
- Recommendation for best model

## Job Status Commands (on HPC)

```bash
# Check job status
squeue --user=nbrande2020

# Check job logs (while running)
tail -f benchmark_*.log

# Check after completion
cat benchmark_*.log
```

## Troubleshooting

**"command not found: sbatch"**
- Load module: `module load slurm`

**"No such file or directory"**
- Verify paths: `ls -la drone-race-sim/benchmark_hpc_final.py`

**"Too many authentication failures"**
- Wait 5-10 minutes, try again from new terminal
- Or use different authentication method (VPN, web portal, etc)

## Next Steps After Job Completes

1. Download `benchmark_results/benchmark_*.json`
2. Analyze which model has highest success rate
3. If success ≥95%: Use that model for May qualifier
4. If <95%: Fine-tune with domain randomization
5. Final submission by April 15

---

**Status:** Ready to submit
**Blocker:** SSH auth - needs manual submission
**Timeline:** Submit this week, results in 1-2 days
