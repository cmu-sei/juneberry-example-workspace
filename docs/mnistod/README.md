README
==========

# Introduction

This document is intended to walk a user through the necessary steps to run object detection on the [mnistod](https://github.com/cmu-sei/mnistod) dataset with Juneberry.

# Create your datasets
The Juneberry repository has a script that allows a user to create reproducible datasets to test.

1. We first need to generate our train/test/val datasets. 

For this example we will do a 1000/250/250 split respectively.

Generate train/test/val datasets and annotations:
`
python scripts/mnistod.py /dataroot --train_split 1000 --test_split 250 --val_split 250 --seed 123
`

- `--data_root`: defines the root base directory for creating relative pathing
- `--train_split` `--test_split` `--val_split`: defines the number of images to generate for each split respectively
- `--seed`: defines the random seed to use for repeatable testing


# Run training and evaluation
The data can be used by Detectron2 or MMDetection sample models.  Assuming you have a properly
set up Juneberry with this workspace (juneberry-example-workspace) as your workspaceL

For Detectron2:
<br>
`jb_train mnistod/dt2`
<br>
`jb_evaluate mnistod/dt2 data_sets/mnistod_test.json`

For MMDetection:
<br>
`jb_train mnistod/mmd`
<br>
`jb_evaluate mnistod/mmd data_sets/mnistod_test.json`

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.

