#!/bin/bash
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# shellcheck disable=SC1091
source ~/.bashrc                                                                                                                                                                         
conda activate pixelnerf
sbatch train_script.sh
    

# print information about the end-time
echo "DONE";
echo "Finished at $(date)";
