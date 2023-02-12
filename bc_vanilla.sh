#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 # partition
#SBATCH --array=1
#SBATCH --mem=8Gb # RAM memory pool for each core (4GB)
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -c 8 # number of CPU cores you need
#SBATCH --gres=gpu:1 # indicate that we need 1 gpu (maximum 8)
#SBATCH -D /home/faridk/pixel-nerf # the directory where the job starts
#SBATCH -o logs_20_bc/%x.%N.%j.%a.%A.out # log STDOUT to a file
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)

##'0 1' \
AR=(0 3407 1337 47)
export SEED=${AR[$SLURM_ARRAY_TASK_ID]}

echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME and $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION and seed $SEED;" ;
#export EXP_NAME=bc_vanilla_res_20_vps_1'_'$SLURM_ARRAY_TASK_ID
python3.8 -m main -t \
--base configs/bc/vanilla_bc/vanilla_bc_vps_1_lr_6-4_res_false.yaml \
-l /work/dlclarge2/faridk-nerf_rl/logs/bc \
-s $SEED 



#-r /work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-28T16-51-49_vanilla_bc \

#/work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-28T19-26-22_vanilla_bc/
#/work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-28T16-42-58_vanilla_bc
#/work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-28T16-51-49_vanilla_bc

#-r /work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-26T13-27-36_vanilla_bc 
# -r /work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-25T13-44-02_vanilla_bc
#--resume_from_checkpoint /work/dlclarge2/faridk-nerf_rl/logs/bc/2022-12-22T11-50-24_vanilla_bc/checkpoints/model.ckpt \
