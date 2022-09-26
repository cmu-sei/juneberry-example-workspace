README
==========

# Introduction

This document is intended to walk a user through the necessary steps to run object detection on the [mnistod](https://github.com/cmu-sei/mnistod) dataset with Juneberry.

# Setup 
Install opencv, Juneberry, and pull down the mnistod repo
`
git clone https://github.com/cmu-sei/mnistod.git
`
# Create your datasets
The mnistod repository has a script that allows a user to create reproducible datasets to test.

1. We first need to generate our train/test/val datasets. 

For this example we will do a 1000/250/250 split respectively.

Generate train/test/val datasets and annotations:
`
python scripts/mnistod.py --data_root /dataroot --images_path /dataroot/mnistod/images/ --train_split 1000 --test_split 250 --val_split 250 --seed 123`


# Run training and evaluation
`jb_train -w /juneberry-example-workspace mnistod_R50 `
<br>
`jb_evaluate -w /juneberry-example-workspace mnistod_R50 data_sets/mnistod_test.json`

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.

