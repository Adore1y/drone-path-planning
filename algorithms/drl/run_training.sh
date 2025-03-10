#!/bin/bash

# DRL Training Runner Script
# This script makes it easy to start DRL training with different configurations

# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Default values
ALGO="ppo"
TIMESTEPS=100000
HEADLESS=false
ENV_CONFIG="${SCRIPT_DIR}/configs/env_config.json"
OUTPUT_DIR="training_results"
ENVIRONMENT="standard"

# Help message
show_help() {
    echo "Usage: ./run_training.sh [options]"
    echo ""
    echo "Options:"
    echo "  -a, --algorithm ALGO   Specify algorithm to use (ppo, dqn, td3, or gat_td3)"
    echo "  -t, --timesteps NUM    Specify number of timesteps to train"
    echo "  -e, --env CONFIG       Specify environment config file"
    echo "  -o, --output DIR       Specify output directory"
    echo "  -h, --headless         Run in headless mode (no visualization)"
    echo "  -u, --urban            Use urban environment instead of standard"
    echo "  --help                 Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run_training.sh -a ppo -t 500000 -h"
    echo "  ./run_training.sh -a td3 -u -t 100000"
    echo "  ./run_training.sh -a gat_td3 -u -t 50000"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -a|--algorithm)
            ALGO="$2"
            shift 2
            ;;
        -t|--timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -e|--env)
            ENV_CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--headless)
            HEADLESS=true
            shift
            ;;
        -u|--urban)
            ENVIRONMENT="urban"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check algorithm
if [[ "$ALGO" != "ppo" && "$ALGO" != "dqn" && "$ALGO" != "td3" && "$ALGO" != "gat_td3" ]]; then
    echo "Error: Algorithm must be 'ppo', 'dqn', 'td3', or 'gat_td3'"
    exit 1
fi

# Confirm environment config exists
if [[ ! -f "$ENV_CONFIG" ]]; then
    echo "Warning: Environment config file '$ENV_CONFIG' not found. Using default parameters."
    ENV_CONFIG=""
fi

# 进入脚本所在目录
cd "$SCRIPT_DIR"

# Construct command
COMMAND="python train.py --algo $ALGO --timesteps $TIMESTEPS --output_dir $OUTPUT_DIR --environment $ENVIRONMENT"

# Add environment config if provided
if [[ -n "$ENV_CONFIG" ]]; then
    COMMAND="$COMMAND --env_config $ENV_CONFIG"
fi

# Add algorithm config
ALGO_CONFIG="configs/${ALGO}_config.json"
if [[ -f "$ALGO_CONFIG" ]]; then
    COMMAND="$COMMAND --algo_config $ALGO_CONFIG"
fi

# Add headless flag if needed
if [[ "$HEADLESS" = true ]]; then
    COMMAND="$COMMAND --headless"
fi

# Print training information
echo "========================================"
echo "Starting Drone DRL Training"
echo "========================================"
echo "Algorithm:      $ALGO"
echo "Environment:    $ENVIRONMENT"
echo "Timesteps:      $TIMESTEPS"
echo "Env Config:     $ENV_CONFIG"
echo "Output Dir:     $OUTPUT_DIR"
echo "Headless Mode:  $HEADLESS"
echo "Working Dir:    $SCRIPT_DIR"
echo "========================================"
echo "Running command: $COMMAND"
echo "========================================"

# Execute command
eval $COMMAND

# Show completion message
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "========================================"
else
    echo "========================================"
    echo "Training failed with error code $?"
    echo "========================================"
fi 