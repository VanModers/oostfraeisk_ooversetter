#!/bin/bash
#SBATCH --job-name=nllb_frs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=nllb-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Cluster environment.
module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$HOME/venvs/gronings-mt/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_CACHE="/scratch/$USER/huggingface/hub"
export HF_XET_CACHE="/scratch/$USER/huggingface/xet"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"

# Pass these paths explicitly to every stage, even if CLI options override them.
TRAIN_ARGS=("$@")
OUTPUT_DIR="${NLLB_OUTPUT_DIR:-./nllb_frs_model}"
VALIDATION_PATH="${NLLB_VALIDATION_PATH:-$PWD/validation data}"
for ((i=0; i<${#TRAIN_ARGS[@]}; i++)); do
  case "${TRAIN_ARGS[i]}" in
    --output-dir) OUTPUT_DIR="${TRAIN_ARGS[i+1]:?Missing value for --output-dir}" ;;
    --output-dir=*) OUTPUT_DIR="${TRAIN_ARGS[i]#*=}" ;;
    --validation-path) VALIDATION_PATH="${TRAIN_ARGS[i+1]:?Missing value for --validation-path}" ;;
    --validation-path=*) VALIDATION_PATH="${TRAIN_ARGS[i]#*=}" ;;
  esac
done
export NLLB_OUTPUT_DIR="$OUTPUT_DIR"
export NLLB_VALIDATION_PATH="$VALIDATION_PATH"

printf 'Job: %s\nNode: %s\nPython: %s\nOutput: %s\nBenchmark: %s\n' \
  "${SLURM_JOB_ID:-manual}" "${SLURMD_NODENAME:-unknown}" \
  "$(command -v python)" "$OUTPUT_DIR" "$VALIDATION_PATH"

python nllb_model/check_training_env.py --require-cuda --require-bf16

python nllb_model/validator.py \
  --model-path VanModers114/East_Frisian_NLLB_Model \
  --validation-path "$VALIDATION_PATH" \
  --results-path "nllb-validation-${SLURM_JOB_ID:-manual}.txt" \
  --model-label "Published model before training"

python nllb_model/trainer.py "${TRAIN_ARGS[@]}"

python nllb_model/validator.py \
  --model-path "$OUTPUT_DIR" \
  --validation-path "$VALIDATION_PATH" \
  --results-path "nllb-validation-${SLURM_JOB_ID:-manual}.txt" \
  --model-label "New model after training"

printf 'Finished: %s\nModel: %s\n' "$(date)" "$OUTPUT_DIR"
