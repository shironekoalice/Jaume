#!/bin/bash
pkill -f CIFAR10 2>/dev/null
pkill -f "torchvision.datasets" 2>/dev/null
echo killed
ls -la /mnt/workspace/data/cifar/cifar-10-batches-py 2>/dev/null | head -3
