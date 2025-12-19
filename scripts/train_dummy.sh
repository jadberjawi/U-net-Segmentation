#!/bin/bash
#SBATCH --job-name=Dummy_test
#SBATCH --output=logs/Dummy_test_log_%j.log
#SBATCH --error=logs/Dummy_test_log_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-A100
#SBATCH --mem=32G  # Request memory (adjust as needed)
#SBATCH --cpus-per-task=4  # Request CPU cores

# Load the correct CUDA module
module load /softs/modules/cuda/12.8.1_570.124.06

# Activate the correct environment
source ~/phd_v1_env/bin/activate

# Navigate to the script directory
cd ~/EVADIAB_clinical_ML/

# ---- W&B storage location ----
export WANDB_DIR="$(pwd)/outputs"
mkdir -p outputs

# Confirm correct Python environment
echo "✅ Current Python environment: $(which python)"

# Run dataset verification script
python src/train.py --config configs/config.yaml
