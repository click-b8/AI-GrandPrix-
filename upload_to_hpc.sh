#!/bin/bash
rsync -av \
  --exclude='venv' \
  --exclude='.venv' \
  --exclude='trained*' \
  --exclude='*.mp4' \
  --exclude='*.png' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='MUJOCO_LOG.TXT' \
  "/Users/mpcrmini2/Desktop/AI GrandPrix/drone-race-sim" \
  nbrande2020@athene-login.hpc.fau.edu:~/
