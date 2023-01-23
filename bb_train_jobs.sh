#!/bin/bash
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# shellcheck disable=SC1091
source ~/.bashrc                                                                                                                                                                         
conda activate nerf_rl
#sbatch train_bb_block_insert_spatial.sh
#sbatch train_bb_block_insert_field.sh
sbatch train_bb_block_insert_global_image.sh
#sbatch train_bb_block_insert_global_image_frozen.sh