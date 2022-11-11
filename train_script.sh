#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 # partition
#SBATCH --mem=8Gb # RAM memory pool for each core (4GB)
#SBATCH -t 1-0:00 # time (D-HH:MM)
#SBATCH -c 8 # number of CPU cores you need
#SBATCH --gres=gpu:2 # indicate that we need 1 gpu (maximum 8)
#SBATCH -D /home/faridk/pixel-nerf # the directory where the job starts
#SBATCH -o logs/%x.%N.%j.out # log STDOUT to a file
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)


python train/train.py \
    -n srn_block_insert_exp_v5 \
    --conf conf/exp/ravens.conf \
    -D /work/dlclarge2/faridk-nerf_rl/srn_block_insertion/block_insertion \
    --logs_path /work/dlclarge2/faridk-nerf_rl/logs \
    --gpu_id='0 1' \
    --wandb_name basic_nerf