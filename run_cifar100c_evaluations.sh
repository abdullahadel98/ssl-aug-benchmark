#!/bin/bash

# Automated script to run CIFAR-100-C evaluations on all linear models
# This script iterates through all linear evaluation checkpoints and runs CIFAR-100-C evaluation sequentially

EXPERIMENTS_DIR="$HOME/my_work/code/experiments/linear"
SSL_AUG_DIR="$HOME/my_work/code/ssl-aug-benchmark/learning/solo-learn"
RESULTS_DIR="$HOME/my_work/code/ssl-aug-benchmark/results/cifar100c"
LOG_DIR="$HOME/my_work/code/ssl-aug-benchmark/logs/cifar100c"
CIFAR100C_PATH="$HOME/my_work/code/ssl-aug-benchmark/learning/solo-learn/datasets/CIFAR-100-C"

# Default parameters
BATCH_SIZE=256
NUM_WORKERS=4
DEVICE="cuda"

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Change to the solo-learn directory
cd "$SSL_AUG_DIR"

echo "Starting CIFAR-100-C evaluations at $(date)"
echo "========================================"
echo "CIFAR-100-C Dataset: $CIFAR100C_PATH"
echo "Results Directory: $RESULTS_DIR"
echo "Logs Directory: $LOG_DIR"
echo "========================================"

# Verify CIFAR-100-C dataset exists
if [[ ! -d "$CIFAR100C_PATH" ]]; then
    echo "ERROR: CIFAR-100-C dataset not found at: $CIFAR100C_PATH"
    echo "Please download CIFAR-100-C dataset first."
    exit 1
fi

# Function to run CIFAR-100-C evaluation for a single experiment
run_cifar100c_eval() {
    local exp_dir=$1
    local exp_name=$(basename "$exp_dir")
    
    # Skip if not a directory
    if [[ ! -d "$exp_dir" ]]; then
        return
    fi
    
    # Find the linear subdirectory
    linear_dir="$exp_dir/linear"
    if [[ ! -d "$linear_dir" ]]; then
        echo "Warning: linear directory not found in $exp_dir"
        return
    fi
    
    # Find the experiment code subdirectory (usually has random name like gcx6op5k)
    experiment_code=$(ls -1 "$linear_dir" | head -n 1)
    
    if [[ -z "$experiment_code" ]]; then
        echo "Warning: No experiment code found in $linear_dir"
        return
    fi
    
    experiment_code_dir="$linear_dir/$experiment_code"
    if [[ ! -d "$experiment_code_dir" ]]; then
        echo "Warning: Experiment code directory not found: $experiment_code_dir"
        return
    fi
    
    # Find the checkpoint file (.ckpt)
    checkpoint_file=$(ls -1 "$experiment_code_dir"/*.ckpt 2>/dev/null | head -n 1)
    
    # Find the args.json file
    args_json_file=$(ls -1 "$experiment_code_dir"/args.json 2>/dev/null | head -n 1)
    
    if [[ -z "$checkpoint_file" ]]; then
        echo "Warning: No checkpoint file found in $experiment_code_dir"
        return
    fi
    
    if [[ ! -f "$checkpoint_file" ]]; then
        echo "Warning: Checkpoint file not found: $checkpoint_file"
        return
    fi
    
    if [[ ! -f "$args_json_file" ]]; then
        echo "Warning: args.json file not found in $experiment_code_dir"
        return
    fi
    
    # Check if already evaluated (results file exists)
    results_file="$RESULTS_DIR/cifar100c_${exp_name}_${experiment_code}.json"
    if [[ -f "$results_file" ]]; then
        echo "Skipping already evaluated experiment: $exp_name (results exist)"
        return
    fi
    
    # Extract method and experiment name from parent directory
    parent_dir=$(basename "$exp_dir")
    
    # Parse experiment name: <method>_<dataset>_<experiment>
    if [[ "$parent_dir" =~ ^(byol|simclr|dino)_cifar_(.+)$ ]]; then
        method="${BASH_REMATCH[1]}"
        experiment="${BASH_REMATCH[2]}"
    else
        echo "Warning: Unexpected naming convention: $parent_dir"
        return
    fi
    
    log_file="$LOG_DIR/cifar100c_${method}_${experiment}.log"
    
    echo ""
    echo "========================================"
    echo "Processing: $parent_dir"
    echo "Method: $method"
    echo "Experiment: $experiment"
    echo "Experiment Code: $experiment_code"
    echo "Checkpoint: $checkpoint_file"
    echo "Args JSON: $args_json_file"
    echo "Results File: $results_file"
    echo "Log File: $log_file"
    echo "Started at: $(date)"
    echo "========================================"
    
    # Run the CIFAR-100-C evaluation
    python eval_cifar100c.py \
        --checkpoint "$checkpoint_file" \
        --args-json "$args_json_file" \
        --cifar100c-path "$CIFAR100C_PATH" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --output "$results_file" \
        > "$log_file" 2>&1
    
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✓ Successfully completed: $parent_dir"
        echo "Finished at: $(date)"
    else
        echo "✗ Failed: $parent_dir (exit code: $exit_code)"
        echo "Check log file: $log_file"
        # Continue to next experiment even if this one fails
    fi
}

# Get all experiment directories and sort them
experiments=($(ls -1d "$EXPERIMENTS_DIR"/*/ | sort))

# Counter for tracking progress
total=${#experiments[@]}
current=0

# Iterate through all experiments
for exp_dir in "${experiments[@]}"; do
    current=$((current + 1))
    echo ""
    echo "Progress: $current/$total"
    run_cifar100c_eval "$exp_dir"
done

echo ""
echo "========================================"
echo "All CIFAR-100-C evaluations completed at $(date)"
echo "========================================"
echo "Results are available in: $RESULTS_DIR"
echo "Logs are available in: $LOG_DIR"

# Generate summary report
echo ""
echo "Generating summary report..."

summary_file="$RESULTS_DIR/cifar100c_summary.txt"
cat > "$summary_file" << 'EOF'
CIFAR-100-C Evaluation Summary Report
====================================

EOF

# Extract summary metrics from all results
for results_file in "$RESULTS_DIR"/cifar100c_*.json; do
    if [[ -f "$results_file" ]]; then
        exp_name=$(basename "$results_file" .json | sed 's/^cifar100c_//')
        echo "Processing: $exp_name"
        
        # Extract mean accuracy using python
        python3 << PYTHON_EOF
import json
import sys

try:
    with open("$results_file") as f:
        data = json.load(f)
        if "aggregates" in data:
            mean_acc = data["aggregates"].get("mean_accuracy", "N/A")
            mean_cal = data["aggregates"].get("mean_calibration_error", "N/A")
            print(f"$exp_name: Mean Accuracy = {mean_acc}%, Mean ECE = {mean_cal}")
except Exception as e:
    print(f"Error processing $results_file: {e}")
PYTHON_EOF
    fi
done >> "$summary_file"

echo "Summary report saved to: $summary_file"
