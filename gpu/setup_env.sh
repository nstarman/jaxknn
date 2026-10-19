#!/bin/bash
# Run on the login node: create the benchmark venv (installs only, no compute).
set -euo pipefail
module load miniforge
ENV=$HOME/envs/jaxknn-bench
[ -d "$ENV" ] || python -m venv "$ENV"
"$ENV/bin/pip" install -q --upgrade pip
"$ENV/bin/pip" install -q "jax[cuda12]==0.8.2" "jaxkd==0.1.2" scipy numpy matplotlib \
  "jaxknn @ git+https://github.com/nstarman/jaxknn@f093366" \
  scikit-build-core cmake ninja  # to build jaxkd-cuda in the job, without isolation
"$ENV/bin/python" -c "import jax, jaxkd, jaxknn, scipy; print(jax.__version__, jaxknn.__version__, scipy.__version__)"
