#!/usr/bin/env bash
# Run MAPPO training on VAMP across 3 seeds sequentially.
set -euo pipefail

SEEDS=(42) # 123 456)
RESULTS_DIR="results"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    NPROC="${NPROC_PER_NODE}"
else
    IFS=',' read -r -a _VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${#_VISIBLE_GPUS[@]}"
fi

if (( NPROC > 1 )); then
    LAUNCHER=(python -m torch.distributed.run --standalone --nproc_per_node="${NPROC}")
    DIST_ARGS=(--distributed)
else
    LAUNCHER=(python)
    DIST_ARGS=()
fi

mkdir -p "${RESULTS_DIR}/plots"

for SEED in "${SEEDS[@]}"; do
    echo "============================================"
    echo "Starting training with seed=${SEED}"
    echo "============================================"

    "${LAUNCHER[@]}" run_madt_vamp.py \
        "${DIST_ARGS[@]}" \
        --seed "${SEED}" \
        --num_theorems 4 \
        --n_agents 2 \
        --max_timestep 100 \
        --online_buffer_size 32 \
        --online_epochs 200 \
        --online_ppo_epochs 8 \
        --online_lr 3e-4 \
        --online_batch_size 4096 \
        --eval_episodes 8 \
        --online_eval_interval 5 \
        --target_rtgs 5.0 \
        --n_embd 128 \
        --n_layer 3 \
        --n_head 4 \
        --context_length 20 \
        --initial_public_concrete_prob 0.5 \
        --publish_resolution_bonus 0.15 \
        --operation_gas_fee 0.01 \
        --proof_success_bonus 0.05 \
        --bounty_quantity 50 \
        --log_dir "${RESULTS_DIR}/logs/seed_${SEED}/" \
        --save_dir "${RESULTS_DIR}/checkpoints/seed_${SEED}/" \
        2>&1 | tee "${RESULTS_DIR}/train_seed_${SEED}.log"

    echo "Seed ${SEED} done."
    echo ""
done

echo "All seeds complete."
