"""Wrapper to run train_distill.py with tilt-fix settings.

Overrides module-level constants before main() runs.
NOTE: trained_finetune/aigp_finetune_final.zip is state-based (MlpPolicy)
and cannot be used as a vision warmstart. Training starts from
trained_vision_events if available, otherwise from scratch.
15M steps is enough for tilt adaptation from a vision warmstart.
"""

import train_distill as td

td.SAVE_DIR = './trained_finetune_tilt'
td.TOTAL_TIMESTEPS = 15_000_000
td.EXPERT_PATHS = [
    './trained_state_expert/aigp_state_final.zip',
    './trained_8gates/best_model/best_model.zip',
]
td.VISION_WARMSTART_PATHS = []

if __name__ == '__main__':
    td.main()
