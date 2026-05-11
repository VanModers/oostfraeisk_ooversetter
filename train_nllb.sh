#!/bin/bash
#SBATCH --job-name=nllb_frs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=aihg.p
#SBATCH --time=2-00:00
#SBATCH --output=/dev/null

set -euo pipefail

# Run from your submitted repository root.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

module load hpc-env/13.1 CUDA Anaconda3 git GCC/13.1.0

mkdir -p "${LOG_DIR}"
VENV_DIR="${REPO_DIR}/../oostfraeisk_llm/venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: venv not found at ${VENV_DIR}"
  echo "Expected the venv from the oostfraeisk_llm project at that path."
  exit 1
fi
source "${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME:-unknown}"
echo "Working dir: ${REPO_DIR}"
echo "Venv dir:    ${VENV_DIR}"
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
