#!/bin/bash
#SBATCH --job-name=fjsp_test_10010_20010
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

srun nvidia-smi || true

exec srun "${PYTHON}" run_test_suite.py
