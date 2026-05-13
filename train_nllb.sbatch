#!/bin/bash
#SBATCH --job-name=nllb_frs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=aihg.p
#SBATCH --time=5-00:00
#SBATCH --output=/dev/null

set -euo pipefail

module load hpc-env/13.1 CUDA Anaconda3 git GCC/13.1.0

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
VENV_DIR="${REPO_DIR}/../oostfraeisk_llm/venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: venv not found at ${VENV_DIR}"
  echo "Expected the venv from the oostfraeisk_llm project at that path."
  exit 1
fi
source "${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE="${LOG_DIR}/nllb_frs_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "Job ID   : ${SLURM_JOB_ID:-manual}"
echo "Node     : ${SLURMD_NODENAME:-unknown}"
echo "Repo dir : ${REPO_DIR}"
echo "Venv     : ${VENV_DIR}"
echo "Python   : $(which python)"
echo "Started  : $(date)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU(s):"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "=========================================="

cd "${REPO_DIR}"

echo "--- Training ---"
python nllb_model/trainer.py

echo "--- Validating ---"
python nllb_model/validator.py

echo "=========================================="
echo "Finished : $(date)"
echo "Log file : ${LOG_FILE}"
echo "=========================================="
