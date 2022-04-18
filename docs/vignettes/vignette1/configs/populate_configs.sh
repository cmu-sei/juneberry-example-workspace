#! /usr/bin/env bash

if [ $# -lt 1 ]; then
  echo "This script requires one argument, the path to the juneberry directory"
  exit -1
fi

# Find out where this script is
cd ${1}

# Now make the dirs and copy things into the right place
mkdir -p data_sets/torchvision
cp docs/vignettes/vignette1/configs/cifar10.json data_sets/torchvision/.

mkdir -p "models/cifar_R20"
cp docs/vignettes/vignette1/configs/config.json models/cifar_R2

cp docs/vignettes/vignette1/configs/resnet_simple.py juneberry/architectures/pytoch/.

mkdir experiments/
