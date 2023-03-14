#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 # partition
#SBATCH --array=1-3
#SBATCH --mem=8Gb # RAM memory pool for each core (4GB)
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -c 8 # number of CPU cores you need
#SBATCH --gres=gpu:1 # indicate that we need 1 gpu (maximum 8)
#SBATCH -D /home/faridk/pixel-nerf # the directory where the job starts
#SBATCH -o logs_rw/%x.%N.%j.%a.%A.out # log STDOUT to a file
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)

##'0 1' \
AR=(0 3407 1337 47)
export SEED=${AR[$SLURM_ARRAY_TASK_ID]}

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME and $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION and seed $SEED;" ;
#export EXP_NAME=bc_vanilla_res_20_vps_1'_'$SLURM_ARRAY_TASK_ID
source ~/.bashrc                                                                                                                                                                         
conda activate nerf_rl
echo "activated env"

python -m main -t --base configs/bc/bc_realworld.yaml -s $SEED -l /work/dlclarge2/faridk-nerf_il/logs/bc_rw \


