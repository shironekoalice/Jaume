#!/bin/bash
cd /mnt/workspace
timeout 300 python -c "from torchvision.datasets import CIFAR10; d=CIFAR10(root=\"/mnt/workspace/data/cifar\", train=True, download=True); print(\"CIFAR10 train\", len(d)); t=CIFAR10(root=\"/mnt/workspace/data/cifar\", train=False, download=True); print(\"test\", len(t))" 2>&1 | tail -8
