#!/bin/bash

# Sequential experiment runner - runs all 3 methods overnight
# Usage: bash scripts/run_all_experiments.sh

set -e  # Exit on error

echo "========================================="
echo "Starting LLM Finetuning Comparison Suite"
echo "========================================="
echo ""
echo "This will run 3 experiments sequentially:"
echo "1. Custom LoRA (fp16 baseline)"
echo "2. Custom 8-bit LoRA"
echo "3. QLoRA (4-bit NF4)"
echo ""
echo "Estimated time: ~15-20 hours total"
echo "========================================="
echo ""

# Experiment 1: Custom LoRA
echo "[1/3] Starting Custom LoRA experiment..."
echo "Started at: $(date)"
python scripts/train.py --config configs/lora.yaml 2>&1 | tee experiments/lora_custom/training.log
echo "Finished at: $(date)"
echo ""

# Experiment 2: 8-bit LoRA
echo "[2/3] Starting 8-bit LoRA experiment..."
echo "Started at: $(date)"
python scripts/train.py --config configs/lora_8bit.yaml 2>&1 | tee experiments/lora_8bit_custom/training.log
echo "Finished at: $(date)"
echo ""

# Experiment 3: QLoRA
echo "[3/3] Starting QLoRA experiment..."
echo "Started at: $(date)"
python scripts/train.py --config configs/qlora.yaml 2>&1 | tee experiments/qlora_nf4/training.log
echo "Finished at: $(date)"
echo ""

echo "========================================="
echo "All experiments completed!"
echo "Total time: $(date)"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  - experiments/lora_custom/"
echo "  - experiments/lora_8bit_custom/"
echo "  - experiments/qlora_nf4/"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/aggregate_results.py"
echo "  2. Check: results/comparison_report.md"
