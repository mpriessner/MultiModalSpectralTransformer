#!/bin/bash
#SBATCH --job-name=PC_250_350
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=4-1:00:00
#SBATCH --output=/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/PC_250_350.txt

echo "SBATCH";
module purge
echo $(which nvcc)

# Load Anaconda module
# Activate the desired conda environment
echo "ACTIVATE";
source /projects/cc/knlr326/miniconda_all/bin/activate  /projects/cc/knlr326/miniconda_all/envs/NMR_Structure_Elucidator 
module load CUDA/11.1.0
echo $(which python)

echo "nvidia-smi";    
nvidia-smi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/knlr326/miniconda_all/lib/

echo "python";
python /projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/script_PC_250_350.py
