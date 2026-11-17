#!/bin/bash
#SBATCH --job-name=name-that-llm
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
python -u train.py --data "${1:?Pass the extracted competition data directory}" \
    --model "${2:?Pass the local Qwen3 checkpoint directory}" --offline \
    --output "runs/$SLURM_JOB_ID" "${@:3}"
