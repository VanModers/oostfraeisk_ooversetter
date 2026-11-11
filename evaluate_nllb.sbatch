#!/bin/bash
#SBATCH --job-name=nllb_compare
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=nllb-compare-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$HOME/venvs/gronings-mt/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE="/scratch/$USER/huggingface/hub"
export HF_XET_CACHE="/scratch/$USER/huggingface/xet"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4

MODEL_DIR="${1:-./nllb_frs_model}"
RESULTS="nllb-comparison-${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULTS"
python nllb_model/check_training_env.py --require-cuda --require-bf16
MODELS=(VanModers114/East_Frisian_NLLB_Model "$MODEL_DIR" "$MODEL_DIR/last-epoch")
LABELS=(published best-loss epoch40)
for i in 0 1 2; do
  python nllb_model/validator.py --model-path "${MODELS[i]}" \
    --validation-path "$PWD/validation data" --direction both --num-beams 1 \
    --results-path "$RESULTS/scores.txt" \
    --predictions-path "$RESULTS/${LABELS[i]}.json"
done
printf 'Comparison files: %s\n' "$RESULTS"
