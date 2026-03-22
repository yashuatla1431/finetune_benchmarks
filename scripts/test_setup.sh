#!/bin/bash

# Quick test script - runs in ~5 minutes to verify everything works
# Usage: bash scripts/test_setup.sh

echo "========================================="
echo "Testing Training Setup"
echo "========================================="
echo ""
echo "This will run 50 training steps (~5 min)"
echo "to verify everything works correctly."
echo ""

# Check if data exists
if [ ! -f "finetune_shards/shard_train.npy" ]; then
    echo "ERROR: finetune_shards/shard_train.npy not found!"
    echo "Please create training data first using ft_dataset.py"
    exit 1
fi

if [ ! -f "finetune_shards/shard_val.npy" ]; then
    echo "ERROR: finetune_shards/shard_val.npy not found!"
    echo "Please create validation data first using ft_dataset.py"
    exit 1
fi

echo "Data files found ✓"
echo ""

# Run test
echo "Starting test training..."
echo "Started at: $(date)"
echo ""

torchrun --nproc_per_node=1 scripts/train.py --config configs/test.yaml

echo ""
echo "Finished at: $(date)"
echo ""

# Check if checkpoint was created
if [ -d "experiments/test_run" ]; then
    echo "========================================="
    echo "✓ TEST PASSED!"
    echo "========================================="
    echo ""
    echo "Setup is working correctly."
    echo "You can now run the full experiments:"
    echo "  bash scripts/run_all_experiments.sh"
    echo ""
else
    echo "========================================="
    echo "✗ TEST FAILED"
    echo "========================================="
    echo ""
    echo "No output directory created."
    echo "Check the error messages above."
    echo ""
fi
