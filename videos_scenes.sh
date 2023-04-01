#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 # partition
#SBATCH --array=1
#SBATCH --mem=8Gb # RAM memory pool for each core (4GB)
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -c 8 # number of CPU cores you need
#SBATCH --gres=gpu:1 # indicate that we need 1 gpu (maximum 8)
#SBATCH -D /home/faridk/pixel-nerf # the directory where the job starts
#SBATCH -o logs/%x.%N.%j.%a.%A.out # log STDOUT to a file
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME and $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source ~/.bashrc
conda activate nerf_rl
echo "activated env"

python -m eval.gen_video  \
-n rw \
--gpu_id 0 \
-P '0 1' \
-S 7 \
--num_views 10 \
--radius 0.08 \
-D /work/dlclarge2/faridk-nerf_il/data/expert_data/train \
-F rw \
--checkpoint /work/dlclarge2/faridk-nerf_il/logs/backbones/2023-03-25T00-17-41_rw_spatial_1/checkpoints/model.ckpt \
-c /work/dlclarge2/faridk-nerf_il/logs/backbones/2023-03-25T00-17-41_rw_spatial_1/configs/2023-03-25T00-17-41-project.yaml