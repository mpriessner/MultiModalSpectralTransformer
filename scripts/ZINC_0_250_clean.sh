#!/bin/bash
#SBATCH --job-name=script_ZINC_250_350
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=4-1:00:00
#SBATCH --constraints=volta
# USER CHANGE REQUIRED: Update output path to your actual project location
#SBATCH --output=/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/script_ZINC_250_350.txt

echo "SBATCH";
module purge
echo $(which nvcc)

# Load Anaconda module
# Activate the desired conda environment
echo "ACTIVATE";
# USER CHANGE REQUIRED: Update these paths to your actual conda installation and environment
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate  /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator 
module load CUDA/11.3.1
echo $(which python)

echo "nvidia-smi";    
nvidia-smi
# USER CHANGE REQUIRED: Update this path to your actual miniconda installation
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/

echo "python";
# USER CHANGE REQUIRED: Update this path to your actual script location
# This should point to your cleaned script in the MultiModalSpectralTransformer_cleaned project
python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/script_ZINC_250_350_clean.py

