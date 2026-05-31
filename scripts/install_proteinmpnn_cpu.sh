#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MIRA_MODEL_DIR:-/data/mira/models}"
COMPOSE_SERVICE="${MIRA_COMPOSE_SERVICE:-mira}"
REPO_URL="${MIRA_PROTEINMPNN_REPO_URL:-https://github.com/dauparas/ProteinMPNN.git}"

if command -v install >/dev/null 2>&1; then
  install -d -m 0755 "$MODEL_DIR"
else
  mkdir -p "$MODEL_DIR"
fi

if [ "$(id -u)" = "0" ]; then
  chown -R 10001:10001 "$MODEL_DIR"
fi

docker compose exec -T "$COMPOSE_SERVICE" /bin/sh -lc "
set -eu
cd /data/mira/models
if [ ! -d ProteinMPNN/.git ]; then
  git clone --depth 1 '$REPO_URL' ProteinMPNN
else
  git -C ProteinMPNN pull --ff-only
fi
python -m venv proteinmpnn-venv
. proteinmpnn-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy biopython
python - <<'PY'
from pathlib import Path
root = Path('/data/mira/models/ProteinMPNN')
weights = root / 'vanilla_model_weights'
script = root / 'protein_mpnn_run.py'
if not script.exists():
    raise SystemExit('ProteinMPNN script missing: protein_mpnn_run.py')
if not weights.exists():
    raise SystemExit('ProteinMPNN weights missing: vanilla_model_weights')
print('ProteinMPNN CPU backend installed at /data/mira/models/ProteinMPNN')
PY
"
