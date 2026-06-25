#!/bin/bash
# =============================================================================
# Run All Experiments for LoRA vs DAPT Legal QA Project
# =============================================================================
# This script runs all experiments using 2 GPUs in parallel:
#   - GPU 0: DAPT experiments (500, 1k, 3k, 10k)
#   - GPU 1: Baseline and LoRA experiments
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh
#
# To run in background:
#   nohup ./scripts/run_all_experiments.sh > logs/all_experiments.log 2>&1 &
# =============================================================================

set -e  # Exit on error

# Configuration — uses current directory by default
PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs

echo "========================================"
echo "LoRA vs DAPT Legal QA Experiments"
echo "Project directory: $PROJECT_DIR"
echo "Started: $(date)"
echo "========================================"

# =============================================================================
# GPU 0 Pipeline: Zero-shot + DAPT Experiments
# (Yewon Joung)
# =============================================================================
run_gpu0() {
    export CUDA_VISIBLE_DEVICES=0

    echo ""
    echo "[GPU 0] Starting Zero-shot evaluation at $(date)"
    python3 scripts/eval_zeroshot.py \
        --output_dir results/zeroshot \
        2>&1 | tee logs/zeroshot.log

    echo ""
    echo "[GPU 0] Starting DAPT 500 at $(date)"
    python3 scripts/train_dapt.py \
        --unlabeled_corpus data/unlabeled/legal_corpus_500.txt \
        --output_dir results/dapt_500 \
        --dapt_epochs 1 --ft_epochs 3 \
        2>&1 | tee logs/dapt_500.log

    echo ""
    echo "[GPU 0] Starting DAPT 1000 at $(date)"
    python3 scripts/train_dapt.py \
        --unlabeled_corpus data/unlabeled/legal_corpus_1000.txt \
        --output_dir results/dapt_1000 \
        --dapt_epochs 1 --ft_epochs 3 \
        2>&1 | tee logs/dapt_1000.log

    echo ""
    echo "[GPU 0] Starting DAPT 3000 at $(date)"
    python3 scripts/train_dapt.py \
        --unlabeled_corpus data/unlabeled/legal_corpus_3000.txt \
        --output_dir results/dapt_3000 \
        --dapt_epochs 1 --ft_epochs 3 \
        2>&1 | tee logs/dapt_3000.log

    echo ""
    echo "[GPU 0] Starting DAPT 10000 at $(date)"
    python3 scripts/train_dapt.py \
        --unlabeled_corpus data/unlabeled/legal_corpus_10000.txt \
        --output_dir results/dapt_10000 \
        --dapt_epochs 1 --ft_epochs 3 \
        2>&1 | tee logs/dapt_10000.log

    echo "[GPU 0] All DAPT experiments complete at $(date)"
}

# =============================================================================
# GPU 1 Pipeline: Baseline and LoRA Experiments
# (Yewon Joung: Baseline / Grace Wang: LoRA)
# =============================================================================
run_gpu1() {
    export CUDA_VISIBLE_DEVICES=1

    echo ""
    echo "[GPU 1] Starting Baseline FT (seed 42) at $(date)"
    python3 scripts/train_baseline.py \
        --output_dir results/baseline \
        --epochs 3 --batch_size 8 --seed 42 \
        2>&1 | tee logs/baseline.log

    echo ""
    echo "[GPU 1] Starting Baseline FT (seed 123) at $(date)"
    python3 scripts/train_baseline.py \
        --output_dir results/baseline_seed123 \
        --epochs 3 --batch_size 8 --seed 123 \
        2>&1 | tee logs/baseline_seed123.log

    echo ""
    echo "[GPU 1] Starting LoRA FT (seed 42) at $(date)"
    python3 scripts/train_lora.py \
        --output_dir results/lora \
        --epochs 3 --batch_size 8 --seed 42 \
        2>&1 | tee logs/lora.log

    echo ""
    echo "[GPU 1] Starting LoRA FT (seed 123) at $(date)"
    python3 scripts/train_lora.py \
        --output_dir results/lora_seed123 \
        --epochs 3 --batch_size 8 --seed 123 \
        2>&1 | tee logs/lora_seed123.log

    echo "[GPU 1] All Baseline/LoRA experiments complete at $(date)"
}

# =============================================================================
# Run Both GPUs in Parallel
# =============================================================================
run_gpu0 &
PID0=$!

run_gpu1 &
PID1=$!

echo ""
echo "Experiments running in parallel:"
echo "  GPU 0 (PID: $PID0): Zero-shot, DAPT 500/1k/3k/10k"
echo "  GPU 1 (PID: $PID1): Baseline (2 seeds), LoRA (2 seeds)"
echo ""

# Wait for both to complete
wait $PID0
wait $PID1

# =============================================================================
# Compare Results
# =============================================================================
echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "========================================"

echo ""
echo "Running results comparison..."
python3 scripts/compare_results.py --results_dir results

echo ""
echo "Results saved in: $PROJECT_DIR/results/"
echo "Logs saved in: $PROJECT_DIR/logs/"
