#!/bin/bash
date +%H:%M:%S
uptime
npu-smi info 2>&1 | sed -n '6,8p'
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)" 2>&1 | tail -2
python -c "import torchvision; print('torchvision', torchvision.__version__)" 2>&1 | tail -1
python -c "import timm; print('timm', timm.__version__)" 2>&1 | tail -1
df -h /mnt/workspace | tail -1