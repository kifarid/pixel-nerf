#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 # partition
#SBATCH --array=1 #-3
#SBATCH --nodes=1
#SBATCH --mem=16Gb # RAM memory pool for each core (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 16 # number of CPU cores you need
#SBATCH --gres=gpu:6 # indicate that we need 1 gpu (maximum 8)
#SBATCH -D /home/faridk/pixel-nerf # the directory where the job starts
#SBATCH -o logs/%x.%N.%j.%a.%A.out # log STDOUT to a file
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)

##'0 1' \

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME and $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
ulimit -n

export EXP_NAME=block_insert_global_image_fin_no_normalize'_'$SLURM_ARRAY_TASK_ID
python3.8 -m main \
-n $EXP_NAME \
--base configs/default_mv_lgn.yaml configs/exp/ravens_global_image_multi.yaml \
-t \
-l /work/dlclarge2/faridk-nerf_rl/logs/backbone \
-s $SLURM_ARRAY_TASK_ID

# export EXP_NAME=/work/dlclarge2/faridk-nerf_rl/logs/backbone/2022-12-14T17-52-11_block_insert_global_image_mask_2
# python3.8 -m main -r $EXP_NAME -t \
# #--base configs/default_mv_lgn.yaml configs/exp/ravens_global_image.yaml \
# #-t \
# -l /work/dlclarge2/faridk-nerf_rl/logs/backbone \
# #-s $SLURM_ARRAY_TASK_ID
# -r /work/dlclarge2/faridk-nerf_rl/logs/backbone/2023-01-17T16-42-18_block_insert_global_image_fin_1 -t
