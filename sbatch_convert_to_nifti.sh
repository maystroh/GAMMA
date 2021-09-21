#!/bin/bash

#SBATCH -p 1080GPU
#SBATCH -J nifti
#SBATCH --output=convert_2_nifti_%j.out
#SBATCH --gres=gpu
#SBATCH -c 6


srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/hassan/Containers/torch-latest-modified.simg python octip-convert-2-nifti.py -r octip_models --depth 64 --height 224 --width 224 --input_dirs ../Brest_data/OCT_Images ../Brest_data/OCT_Non_Interprete