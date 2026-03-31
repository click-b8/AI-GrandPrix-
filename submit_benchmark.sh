#!/bin/bash
#SBATCH --job-name=aigp-bench-may
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -e

echo "=========================================="
echo "AI GRAND PRIX - MAY QUALIFIER BENCHMARKING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Starting: $(date)"
echo ""

# Set working directory
WORK_DIR="/home/$(whoami)/AI-GrandPrix/drone-race-sim"
MODEL_DIR="/home/$(whoami)/AI-GrandPrix/models_release"

echo "Working directory: $WORK_DIR"
echo "Model directory: $MODEL_DIR"
echo ""

# Load modules
module load python/3.11 2>/dev/null || module load python/3.9 2>/dev/null || echo "Python loaded via system"

# Navigate to work directory
cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No venv found, using system Python"
fi

echo ""
echo "Running benchmark_hpc_final.py..."
echo "Testing 3 models × 20 laps each"
echo "Estimated time: 2-4 hours"
echo "=========================================="
echo ""

# Run benchmarking script
python3 benchmark_hpc_final.py

RESULT=$?

echo ""
echo "=========================================="
if [ $RESULT -eq 0 ]; then
    echo "✅ BENCHMARKING COMPLETED SUCCESSFULLY"
else
    echo "❌ BENCHMARKING FAILED (exit code: $RESULT)"
fi
echo "Finished: $(date)"
echo "=========================================="

exit $RESULT
