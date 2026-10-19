#!/bin/bash
#SBATCH -J jaxknn-gpu
#SBATCH -p mit_normal_gpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# GPU type from the submit line: sbatch -G h200:1 job.sh  /  sbatch -G l40s:1 job.sh
set -uo pipefail
module load miniforge
ENV=$HOME/envs/jaxknn-bench
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Fail fast on a node whose CUDA driver cannot initialise (node3006 returned
# CUDA_ERROR_UNKNOWN from cuInit even though nvidia-smi worked).
"$ENV/bin/python" -c "import ctypes, sys; r = ctypes.CDLL('libcuda.so.1').cuInit(0); print('cuInit ->', r); sys.exit(r != 0)" \
  || { echo "CUDA driver failed to initialise on $(hostname)"; exit 3; }
# jaxkd's optional CUDA extension is a source build; try it on the GPU node.
# Built per job into a job-local dir (two jobs share the venv), against the
# venv's jax (no isolation) and without its unpinned jax[cuda12] dependency.
JKC=$(mktemp -d)
(module load cuda/12.9.1 && PATH=$ENV/bin:$PATH "$ENV/bin/pip" install -q \
   --no-build-isolation --no-deps --target "$JKC" jaxkd-cuda) \
  && echo "jaxkd-cuda: built" || echo "jaxkd-cuda: build failed, benchmarking without it"
export PYTHONPATH="$JKC${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
"$ENV/bin/python" bench_gpu.py "results_${GPU}_${SLURM_JOB_ID}.json"
