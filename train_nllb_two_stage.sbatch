#!/bin/bash
#SBATCH --job-name=nllb_frs_2stage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=nllb-two-stage-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$HOME/venvs/gronings-mt/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_CACHE="/scratch/$USER/huggingface/hub"
export HF_XET_CACHE="/scratch/$USER/huggingface/xet"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"

TRAIN_ARGS=("$@")
OUTPUT_DIR="./nllb_frs_two_stage"
CHECKPOINT_DIR=""
VALIDATION_PATH="${NLLB_VALIDATION_PATH:-$PWD/validation data}"
for ((i=0; i<${#TRAIN_ARGS[@]}; i++)); do
  case "${TRAIN_ARGS[i]}" in
    --output-dir) OUTPUT_DIR="${TRAIN_ARGS[i+1]:?Missing --output-dir value}" ;;
    --output-dir=*) OUTPUT_DIR="${TRAIN_ARGS[i]#*=}" ;;
    --checkpoint-dir) CHECKPOINT_DIR="${TRAIN_ARGS[i+1]:?Missing --checkpoint-dir value}" ;;
    --checkpoint-dir=*) CHECKPOINT_DIR="${TRAIN_ARGS[i]#*=}" ;;
    --validation-path) VALIDATION_PATH="${TRAIN_ARGS[i+1]:?Missing --validation-path value}" ;;
    --validation-path=*) VALIDATION_PATH="${TRAIN_ARGS[i]#*=}" ;;
  esac
done
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/$USER/oostfraeisk_ooversetter_training/$(basename -- "$OUTPUT_DIR")-${SLURM_JOB_ID:-manual}}"
RESULTS="nllb-two-stage-validation-${SLURM_JOB_ID:-manual}.txt"
printf 'Models: %s\nCheckpoints: %s\n' "$OUTPUT_DIR" "$CHECKPOINT_DIR"
python nllb_model/check_training_env.py --require-cuda --require-bf16
python nllb_model/trainer_two_stage.py "${TRAIN_ARGS[@]}" \
  --checkpoint-dir "$CHECKPOINT_DIR" --validation-path "$VALIDATION_PATH"

for MODEL in VanModers114/East_Frisian_NLLB_Model "$OUTPUT_DIR/stage1" "$OUTPUT_DIR/stage2"; do
  python nllb_model/validator.py --model-path "$MODEL" \
    --validation-path "$VALIDATION_PATH" --results-path "$RESULTS" --model-label "$MODEL"
done
printf 'Finished: %s\nResults: %s\n' "$(date)" "$RESULTS"
