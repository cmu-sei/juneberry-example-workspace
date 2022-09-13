README
==========

# Introduction

This document is intended to walk a user through the necessary steps to run object detection on the [yymnsit](https://github.com/cmu-sei/yymnist) dataset with Juneberry.

# Setup 
Install opencv, Juneberry, and pull down the yymnsit repo
<br>
`git config --global http.proxy http://cloudproxy.sei.cmu.edu:80/`
`
git clone https://github.com/cmu-sei/yymnist.git
`
# Create your datasets
The yymnist repository has a script that allows a user to create reproducible datasets to test.

1. We first need to generate our train/test/val datasets. 

For this example we will do a 80/10/10 split respectively.

Training Set:
`
python yymnist/make_data.py --images_num 80 --images_path /dataroot/yymnist/Images80_train/ --labels_txt /dataroot/yymnist/Images80_train.txt --seed 123
`

Validation Set:
`
python yymnist/make_data.py --images_num 10 --images_path /dataroot/yymnist/Images10_val/ --labels_txt /dataroot/yymnist/Images10_val.txt --seed 456
`

Test Set:
`
python yymnist/make_data.py --images_num 10 --images_path /dataroot/yymnist/Images10_test/ --labels_txt /dataroot/yymnist/Images10_test.txt --seed 789
`

2. Convert your labels.txt to coco format so that it can be consumable in Juneberry. <br>

Training JSON:
`scripts/yymnist_txtdetection2coco.py /dataroot/yymnist/Images80_train/ /dataroot/yymnist/Images80_train.txt /dataroot/yymnist/Images80_train.json`

Validation JSON:
`scripts/yymnist_txtdetection2coco.py /dataroot/yymnist/Images10_val/ /dataroot/yymnist/Images10_val.txt /dataroot/yymnist/Images10_val.json`

Testing JSON:
`scripts/yymnist_txtdetection2coco.py /dataroot/yymnist/Images10_test/ /dataroot/yymnist/Images10_test.txt /dataroot/yymnist/Images10_test.json`

3. Update paths for newly created datasets

`data_sets/yymnist_train.json`

`"image_data": {
        "sources": [
            {
                "directory": "yymnist/Images80_train.json"
            }
        ]
}
`

`data_sets/yymnist_val.json`

`"image_data": {
        "sources": [
            {
                "directory": "yymnist/Images10_val.json"
            }
        ]
}
`

`data_sets/yymnist_test.json`

`"image_data": {
        "sources": [
            {
                "directory": "yymnist/Images10_test.json"
            }
        ]
}
`

# Run training and evaluation
`jb_train -w /juneberry-example-workspace yymnist_R50 `
<br>
`jb_evaluate -w /juneberry-example-workspace yymnist_R50 data_sets/yymnist_test.json`

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.

