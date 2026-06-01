#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MIRA_MODEL_DIR:-/data/mira/models}"
COMPOSE_SERVICE="${MIRA_COMPOSE_SERVICE:-mira}"
REPO_URL="${MIRA_FOLDINGDIFF_REPO_URL:-https://github.com/microsoft/foldingdiff.git}"

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
if [ ! -d FoldingDiff/.git ]; then
  git clone --depth 1 '$REPO_URL' FoldingDiff
else
  git -C FoldingDiff pull --ff-only
fi
python -m venv foldingdiff-venv
. foldingdiff-venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scipy scikit-learn requests 'transformers==4.30.2' 'huggingface_hub<1.0' pytorch-lightning biotite seaborn gitpython astropy mpl-scatter-density
pip install -e FoldingDiff
python - <<'PY'
from pathlib import Path

path = Path('/data/mira/models/FoldingDiff/bin/train.py')
text = path.read_text()
old = 'assert torch.cuda.is_available(), "Requires CUDA to train"'
new = (
    'if not torch.cuda.is_available() and Path(sys.argv[0]).name != "sample.py":\n'
    '    raise AssertionError("Requires CUDA to train")'
)
if old in text:
    path.write_text(text.replace(old, new))
PY
python - <<'PY'
from pathlib import Path
root = Path('/data/mira/models/FoldingDiff')
script = root / 'bin' / 'sample.py'
if not script.exists():
    raise SystemExit('FoldingDiff script missing: bin/sample.py')
print('FoldingDiff CPU backend installed at /data/mira/models/FoldingDiff')
print('Set MIRA_FOLDINGDIFF_REPO=/data/mira/models/FoldingDiff')
print('Set MIRA_FOLDINGDIFF_PYTHON=/data/mira/models/foldingdiff-venv/bin/python')
PY
"
