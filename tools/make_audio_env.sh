#!/usr/bin/env bash
set -o pipefail
ENV=/mnt/workspace/venvs/jaume-audio
rm -rf "$ENV"
python -m venv "$ENV"
"$ENV/bin/pip" install --disable-pip-version-check 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/pip" install --disable-pip-version-check 'torchaudio==2.7.1' --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/pip" install --disable-pip-version-check soundfile 'funasr>=1.3.26' 'transformers>=4.40'
"$ENV/bin/python" -c "import torch, torchaudio, funasr; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'funasr', funasr.__version__)" 2>&1 | tail -5