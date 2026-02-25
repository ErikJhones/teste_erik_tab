# This script is used to train TabICL for the first stage of the curriculum learning

#!/bin/bash

# Descobre raiz do projeto automaticamente
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. && pwd )"

PYTHON="$PROJECT_ROOT/venv/bin/python"
TORCHRUN="$PROJECT_ROOT/venv/bin/torchrun"

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

torchrun --standalone --nproc_per_node=1 $PROJECT_ROOT/tabiclv2/src/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir $PROJECT_ROOT/tabiclv2/wandb/ \
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
            --checkpoint_dir $PROJECT_ROOT/tabiclv2/checkpoints/stage1 \
            --save_temp_every 50 \
            --save_perm_every 5000


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------

# Saving to disk
python $PROJECT_ROOT/tabiclv2/src/tabicl/prior/genload.py \
    --save_dir $PROJECT_ROOT/tabiclv2/prior/stage1 \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 10 \
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
torchrun --standalone --nproc_per_node=1 $PROJECT_ROOT/tabiclv2/src/tabicl/train/run.py \
            --wandb_log True \
            --wandb_project TabICL \
            --wandb_name Stage1 \
            --wandb_dir $PROJECT_ROOT/tabiclv2/wandb/ \
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
            --prior_dir $PROJECT_ROOT/tabiclv2/prior/stage1 \
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
            --checkpoint_dir $PROJECT_ROOT/tabiclv2/checkpoints/stage1 \
            --save_temp_every 50 \
            --save_perm_every 5000