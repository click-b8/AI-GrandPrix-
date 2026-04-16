#!/bin/bash
cd "/Users/mpcrmini2/Desktop/AI GrandPrix/drone-race-sim"
echo "Uploading code + expert (best model only)..."
scp config.py drone_race_env.py train_state.py train_distill.py train_hpc.slurm view_expert.py train_vision.py track.py athene:~/drone-race-sim/
ssh athene "mkdir -p ~/drone-race-sim/trained_state_expert/best_model"
scp trained_state_expert/best_model/best_model.zip trained_state_expert/aigp_state_final.zip athene:~/drone-race-sim/trained_state_expert/best_model/
echo "Done! Now SSH in, scancel, and sbatch."
