#!/bin/bash
# Helper script to evaluate linear models on CIFAR-100-C

# Default paths
CIFAR100C_PATH="./datasets/CIFAR-100-C"
BATCH_SIZE=256
NUM_WORKERS=4
DEVICE="cuda"

# Example usage:
# ./eval_cifar100c.sh <checkpoint_path> [args_json_path] [config_path] [output_path]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [args_json_path] [config_path] [output_path]"
    echo ""
    echo "Examples:"
    echo "  $0 trained_models/linear/simclr/checkpoint.ckpt"
    echo "  $0 trained_models/linear/simclr/checkpoint.ckpt path/to/args.json"
    echo "  $0 trained_models/linear/simclr/checkpoint.ckpt path/to/args.json scripts/linear/cifar-100/simclr.yaml"
    echo "  $0 trained_models/linear/simclr/checkpoint.ckpt path/to/args.json scripts/linear/cifar-100/simclr.yaml results.json"
    exit 1
fi

CHECKPOINT=$1
ARGS_JSON=${2:-""}
CONFIG=${3:-""}
OUTPUT=${4:-""}

# Build command
CMD="python eval_cifar100c.py --checkpoint $CHECKPOINT --cifar100c-path $CIFAR100C_PATH --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --device $DEVICE"

if [ -n "$ARGS_JSON" ]; then
    CMD="$CMD --args-json $ARGS_JSON"
fi

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

echo "Running CIFAR-100-C evaluation..."
echo "Checkpoint: $CHECKPOINT"
echo "Config: ${CONFIG:-'from checkpoint'}"
echo "Output: ${OUTPUT:-'auto-generated'}"
echo ""

eval $CMD
