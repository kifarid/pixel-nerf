#!/bin/bash
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# shellcheck disable=SC1091
source ~/.bashrc                                                                                                                                                                         
conda activate pixelnerf
# sbatch train_script_default.sh
# sbatch train_script_masks_no_vps_3.sh
# sbatch train_script_masks_no_vps_5.sh
# sbatch train_script_masks_no.sh
# sbatch train_script_v_2_vps_3.sh
# sbatch train_script_v_2_vps_5.sh
# sbatch train_script_v_2.sh
# sbatch train_script_vps_3.sh
# sbatch train_script_vps_7.sh


sbatch train_script_v1_vps_2.sh
sbatch train_script_v1_vps_3.sh
sbatch train_script_v2_vps_3.sh
sbatch train_script_v1_vps_4.sh
sbatch train_script_v2_vps_4.sh
#sbatch train_script_v3_vps_4.sh

sbatch train_script_v2_vps_3_bg.sh
sbatch train_script_v2_vps_4_bg.sh


sbatch train_script_v1_vps_2_masks_no.sh
sbatch train_script_v1_vps_3_masks_no.sh
sbatch train_script_v2_vps_3_masks_no.sh
#sbatch train_script_v1_vps_4_masks_no.sh
#sbatch train_script_v2_vps_4_masks_no.sh
#sbatch train_script_v3_vps_4_masks_no.sh

# sbatch train_script_default.sh
# sbatch train_script_masks_no.sh
# sbatch train_script_v_2_vps_3_old.sh
# sbatch train_script_v_2_vps_5_old.sh
# sbatch train_script_v_2_vps_5_old_bg.sh


# print information about the end-time
echo "DONE";
echo "Finished at $(date)";
