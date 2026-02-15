#!/bin/bash

# Automated script to run linear evaluations for all CIFAR experiments
# This script iterates through all experiment directories and runs linear evaluation sequentially

set -e  # Exit on error

EXPERIMENTS_DIR="$HOME/my_work/code/experiments"
SSL_AUG_DIR="$HOME/my_work/code/ssl-aug-benchmark/learning/solo-learn"
LOG_DIR="$HOME/my_work/code/ssl-aug-benchmark/logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Change to the solo-learn directory
cd "$SSL_AUG_DIR"

echo "Starting linear evaluations at $(date)"
echo "========================================"

# Function to run linear evaluation for a single experiment
run_linear_eval() {
    local exp_dir=$1
    local exp_name=$(basename "$exp_dir")
    
    # Skip if not a directory
    if [[ ! -d "$exp_dir" ]]; then
        return
    fi
    
    # Skip linear directory
    if [[ "$exp_name" == "linear" ]]; then
        echo "Skipping linear directory"
        return
    fi
    
    # Parse experiment name early to check if already run
    if [[ "$exp_name" =~ ^(byol|simclr|dino)_cifar_(.+)$ ]]; then
        method="${BASH_REMATCH[1]}"
        experiment="${BASH_REMATCH[2]}"
    else
        # Will skip in next check
        method=""
        experiment=""
    fi
    
    # Check if already run (log file exists)
    if [[ -n "$method" ]]; then
        log_file="$LOG_DIR/test_${method}_${experiment}.log"
        if [[ -f "$log_file" ]]; then
            echo "Skipping already completed experiment: $exp_name (log exists)"
            return
        fi
    fi
    
    # Only process CIFAR experiments (exclude mvtec, draem, etc.)
    if [[ ! "$exp_name" =~ cifar ]]; then
        echo "Skipping non-CIFAR experiment: $exp_name"
        return
    fi
    
    # Check if method was already extracted and validated above
    if [[ -z "$method" ]]; then
        echo "Skipping experiment with unexpected naming: $exp_name"
        return
    fi
    
    dataset="cifar"
    
    # Find the experiment code directory
    method_dir="$exp_dir/$method"
    if [[ ! -d "$method_dir" ]]; then
        echo "Warning: Method directory not found: $method_dir"
        return
    fi
    
    # Get the experiment code (first subdirectory in method folder)
    experiment_code=$(ls -1 "$method_dir" | head -n 1)
    
    if [[ -z "$experiment_code" ]]; then
        echo "Warning: No experiment code found in $method_dir"
        return
    fi
    
    # Verify the experiment code directory exists
    if [[ ! -d "$method_dir/$experiment_code" ]]; then
        echo "Warning: Experiment code directory not found: $method_dir/$experiment_code"
        return
    fi
    
    # Construct command parameters
    pretrained_path="$EXPERIMENTS_DIR/${method}_${dataset}_${experiment}/${method}/${experiment_code}/"
    run_name="${method}-${experiment}-${dataset}100"
    checkpoint_dir="$EXPERIMENTS_DIR/linear/${method}_${dataset}_${experiment}"
    log_file="$LOG_DIR/test_${method}_${experiment}.log"
    
    echo ""
    echo "========================================"
    echo "Processing: $exp_name"
    echo "Method: $method"
    echo "Experiment: $experiment"
    echo "Experiment Code: $experiment_code"
    echo "Pretrained Path: $pretrained_path"
    echo "Log File: $log_file"
    echo "Started at: $(date)"
    echo "========================================"
    
    # Run the linear evaluation (synchronously, not with nohup)
    python main_linear.py \
        --config-path scripts/linear/cifar-100 \
        --config-name "$method" \
        ++pretrained_feature_extractor="$pretrained_path" \
        ++name="$run_name" \
        ++checkpoint.dir="$checkpoint_dir" \
        ++devices=[1] \
        > "$log_file" 2>&1
    
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✓ Successfully completed: $exp_name"
        echo "Finished at: $(date)"
    else
        echo "✗ Failed: $exp_name (exit code: $exit_code)"
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
    run_linear_eval "$exp_dir"
done

echo ""
echo "========================================"
echo "All linear evaluations completed at $(date)"
echo "========================================"
echo "Logs are available in: $LOG_DIR"
