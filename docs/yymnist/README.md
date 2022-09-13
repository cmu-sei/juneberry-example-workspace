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
<br >
<br >
1. We first need to generate our train/test/val datasets. 
<br>
For this example we will do a 80/10/10 split respectively.
<br>
<br >
Training Set:
`
python yymnist/make_data.py --images_num 80 --images_path /dataroot/yymnist/Images80_train/ --labels_txt /dataroot/yymnist/Images80_train.txt --seed 123
`
<br >
<br>
Validation Set:
`
python yymnist/make_data.py --images_num 10 --images_path /dataroot/yymnist/Images10_val/ --labels_txt /dataroot/yymnist/Images10_val.txt --seed 456
`
<br >
<br>
Test Set:
`
python yymnist/make_data.py --images_num 10 --images_path /dataroot/yymnist/Images10_test/ --labels_txt /dataroot/yymnist/Images10_test.txt --seed 789
`
<br >
<br >
2. Convert your labels.txt to coco format so that it can be consumable in Juneberry. <br>
<br >
Training JSON:
`scripts/yymnist_txtdetection2coco.py /dataroot/yymnist/Images80_train/ /dataroot/yymnist/Images80_train.txt /dataroot/yymnist/Images80_train.json`
<br >
<br>
Validation JSON:
`scripts/yymnist_txtdetection2coco.py /dataroot/yymnist/Images10_val/ /dataroot/yymnist/Images10_val.txt /dataroot/yymnist/Images10_val.json`
<br >
<br>
Testing JSON:
`scripts/yymnist_txtdetection2coco.py /dataroot/yymnist/Images10_test/ /dataroot/yymnist/Images10_test.txt /dataroot/yymnist/Images10_test.json`
<br>
<br>
3. Update paths for newly created datasets
<br>
<br>
`data_sets/yymnist_train.json`
<br>
`"image_data": {
        "sources": [
            {
                "directory": "yymnist/Images80_train.json"
            }
        ]
}
`
<br>
<br>
`data_sets/yymnist_val.json`
<br>
`"image_data": {
        "sources": [
            {
                "directory": "yymnist/Images10_val.json"
            }
        ]
}
`
<br>
<br>
`data_sets/yymnist_test.json`
<br>
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

