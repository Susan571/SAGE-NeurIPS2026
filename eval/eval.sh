#!/bin/bash
# SAGE evaluation script
# Usage: bash eval.sh <checkpoint_path> [output_dir] [num_episodes]

set -e

CHECKPOINT_PATH=${1:-"out/sage_checkpoint.pt"}
OUTPUT_DIR=${2:-"./eval_outputs"}
NUM_EPISODES=${3:-100}
STATES_TYPE=${4:-"all"}

echo "SAGE Evaluation"
echo "==============="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Number of episodes: $NUM_EPISODES"
echo "States type: $STATES_TYPE"
echo ""

python -m sage.eval.evaluate \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$NUM_EPISODES" \
    --states_type "$STATES_TYPE" \
    --use_sage_guidance \
    --save_trajectories \
    --save_failed \
    --seed 42

echo ""
echo "Evaluation complete!"
