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

For this example we will do a 80/10/10 split respectively.

Training Set:
`
python mnistod/make_data.py --data_root /dataroot --images_num 80 --images_path mnistod/Images80_train/ --coco_json mnistod/annotations_train.json --seed 123
`

Validation Set:
`
python mnistod/make_data.py --data_root /dataroot --images_num 10 --images_path mnistod/Images10_val/ --coco_json mnistod/annotations_val.json --seed 456
`

Test Set:
`
python mnistod/make_data.py --data_root /dataroot --images_num 10 --images_path mnistod/Images10_test/ --coco_json mnistod/annotations_test.json --seed 789
`

2. Update paths for newly created datasets

`data_sets/mnistod_train.json`

`"image_data": {
        "sources": [
            {
                "directory": "mnistod/Images80_train.json"
            }
        ]
}
`

`data_sets/mnistod_val.json`

`"image_data": {
        "sources": [
            {
                "directory": "mnistod/Images10_val.json"
            }
        ]
}
`

`data_sets/mnistod_test.json`

`"image_data": {
        "sources": [
            {
                "directory": "mnistod/Images10_test.json"
            }
        ]
}
`

# Run training and evaluation
`jb_train -w /juneberry-example-workspace mnistod_R50 `
<br>
`jb_evaluate -w /juneberry-example-workspace mnistod_R50 data_sets/mnistod_test.json`

# Copyright

Copyright 2022 Carnegie Mellon University.  See LICENSE.txt file for license terms.

