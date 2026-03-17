#!/usr/bin/env bash
# Run MAPPO training on VAMP across 3 seeds sequentially.
set -euo pipefail

SEEDS=(42) # 123 456)
RESULTS_DIR="results"
CUDA_VISIBLE_DEVICES=0,1

mkdir -p "${RESULTS_DIR}/plots"

for SEED in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Starting training with seed=${SEED}"
    echo "============================================"

    python -m torch.distributed.run --standalone --nproc_per_node=2 run_madt_vamp.py \
        --seed "${SEED}" \
        --F_size 8 \
        --n_agents 2 \
        --max_timestep 50 \
        --online_buffer_size 8 \
        --online_epochs 300 \
        --online_ppo_epochs 5 \
        --online_lr 5e-4 \
        --eval_episodes 8 \
        --online_eval_interval 5 \
        --target_rtgs 2.0 \
        --n_embd 64 \
        --n_layer 2 \
        --n_head 2 \
        --context_length 1 \
        --log_dir "${RESULTS_DIR}/logs/seed_${SEED}/" \
        --save_dir "${RESULTS_DIR}/checkpoints/seed_${SEED}/" \
        2>&1 | tee "${RESULTS_DIR}/train_seed_${SEED}.log"

    echo "Seed ${SEED} done."
    echo ""
done

echo "All seeds complete."
