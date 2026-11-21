#!/bin/bash
python -c "import timm; print('timm', timm.__version__)" 2>&1 | tail -1
python -c "from torchvision import datasets; print('tv datasets OK')" 2>&1 | tail -1
python -c "import torchvision.models as m; m.resnet18(); print('resnet18 OK')" 2>&1 | tail -1
ls /home/developer/.cache/torch 2>/dev/null | head -3
df -h /mnt/workspace | tail -1