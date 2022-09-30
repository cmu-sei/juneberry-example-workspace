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
python scripts/mnistod.py /dataroot --train_split 1000 --test_split 250 --val_split 250 --seed 123
`

- `--data_root`: defines the root base directory for creating relative pathing
- `--images_path`: defines where the generated mnist .jpg imges will be saved
- `--annotations_path`: defines the directory to save the `annotation_train.json` `annotation_test.json` `annotation_val.json`
- `--train_split` `--test_split` `--val_split`: defines the number of images to generate for each split respectively
- `--seed`: defines the random seed to use for repeatable testing


# Run training and evaluation
`jb_train -w /juneberry-example-workspace mnistod_R50 `
<br>
`jb_evaluate -w /juneberry-example-workspace mnistod_R50 data_sets/mnistod_test.json`

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.

