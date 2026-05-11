#!/bin/bash
#SBATCH --job-name=nllb_frs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=aihg.p
#SBATCH --time=7-00:00
#SBATCH --output=/dev/null

set -euo pipefail

# Run from your submitted repository root.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="${REPO_DIR}/logs"
ENV_NAME="pytorch"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/nllb_train_${SLURM_JOB_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

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
echo "Node:        ${SLURMD_NODENAME}"
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
