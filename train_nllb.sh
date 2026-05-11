#!/bin/bash
#SBATCH --job-name=nllb_frs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=aihg.p
#SBATCH --time=2-00:00
# logs/ must exist before submitting: mkdir -p logs
#SBATCH --output=logs/nllb_frs_%j.log
#SBATCH --error=logs/nllb_frs_%j.log

set -euo pipefail

# Run from your submitted repository root.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_NAME="pytorch"

module load hpc-env/13.1 CUDA Anaconda3 git GCC/13.1.0

# Initialise conda shell integration and activate the training environment
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
	echo "ERROR: conda environment '${ENV_NAME}' not found."
	echo "Set up the environment first."
	exit 1
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Working dir: ${REPO_DIR}"
echo "Conda env:   ${ENV_NAME}"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "Python:      $(which python)"
echo "Started at:  $(date)"
echo "=========================================="

cd "${REPO_DIR}"

echo "--- Training ---"
python nllb_model/trainer.py

echo "--- Validating ---"
python nllb_model/validator.py

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
