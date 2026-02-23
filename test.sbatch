#!/bin/bash

#SBATCH --partition=h200
#SBATCH --job-name=tabicl_test    # Job name
#SBATCH --output=job.%x.%j.out # Name of stdout output file (%x expands to job name and %j expands to %jobId)
#SBATCH --time=2-00:00:00         # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1            # Number of GPUs to be used
#SBATCH --qos=gpu-h200         # QOS to be used

echo "==============================="
echo "Job started"
echo "JobID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==============================="

# Descobre raiz do projeto automaticamente
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. && pwd )"

PYTHON="$PROJECT_ROOT/venv/bin/python"
TORCHRUN="$PROJECT_ROOT/venv/bin/torchrun"

# Start timer
START_TIME=$(date +%s)

# Rodar código Python
torchrun --standalone --nproc_per_node=1 ./src/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir ./wandb/ \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 100000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_type mix_scm \
            --prior_device cuda \
            --batch_size_per_gp 4 \
            --min_features 2 \
            --max_features 100 \
            --max_classes 10 \
            --min_seq_len 2 \
            --max_seq_len 1024 \
            --min_train_size 0.1 \
            --max_train_size 0.9 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ./checkpoints/stage1 \
            --save_temp_every 5 \
            --save_perm_every 5000


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
python ./src/tabicl/prior/genload.py \
    --save_dir ./prior/stage1 \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 100000 \
    --resume_from 0 \
    --batch_size 512 \
    --batch_size_per_gp 4 \
    --prior_type mix_scm \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --min_seq_len 2 \
    --max_seq_len 1024 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --n_jobs -1 \
    --num_threads_per_generate 1 \
    --device cuda

# Loading from disk and training
torchrun --standalone --nproc_per_node=1 ./src/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir ./wandb/ \
            --wandb_mode online \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 100000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --lr 1e-4 \
            --scheduler cosine_warmup \
            --warmup_proportion 0.02 \
            --gradient_clipping 1.0 \
            --prior_dir ./prior/stage1 \
            --load_prior_start 0 \
            --delete_after_load False \
            --prior_device cuda \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir ./checkpoints/stage1 \
            --save_temp_every 5 \
            --save_perm_every 5000

# End timer
END_TIME=$(date +%s)

ELAPSED=$((END_TIME-START_TIME))

echo "==============================="
echo "JOB FINISHED"
echo "==============================="

echo "Elapsed time (seconds): $ELAPSED"
echo "Elapsed time (minutes): $(($ELAPSED/60))"

echo ""
echo "===== SLURM RESOURCE USAGE ====="

sacct -j $SLURM_JOB_ID \
--format=JobID,State,Elapsed,MaxRSS,AllocCPUS

echo "==============================="