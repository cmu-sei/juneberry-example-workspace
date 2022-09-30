#! /usr/bin/env python

# ======================================================================================================================
# Juneberry - General Release
#
# Copyright 2021 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software subject to its own license.
#
# DM21-0884
#
# ======================================================================================================================

import argparse
from collections import defaultdict
import json
from pathlib import Path
from PIL import Image, ImageOps
import random
import sys

import torchvision

# We use a global randomizer instance instead of random
randomizer = random.Random()


# function that generates the mnist dataset and their train/test/val annotation files for Juneberry
def labels_and_images(data_root, all_items, train_split,
                      test_split, val_split, sizes, image_size, max_iou):
    # pre-defined annotation file names
    annotations = ["annotations_train.json",
                   "annotations_test.json",
                   "annotations_val.json"]

    # user defined splits
    splits = [train_split,
              test_split,
              val_split]

    # mnist category list
    categories = [
        {"id": 0, "name": 'Zero'},
        {"id": 1, "name": 'One'},
        {"id": 2, "name": 'Two'},
        {"id": 3, "name": 'Three'},
        {"id": 4, "name": 'Four'},
        {"id": 5, "name": 'Five'},
        {"id": 6, "name": 'Six'},
        {"id": 7, "name": 'Seven'},
        {"id": 8, "name": 'Eight'},
        {"id": 9, "name": 'Nine'}]

    data_rt = Path(data_root)
    anno_path = data_rt / "mnistod"
    images_path = anno_path / "images"

    # Make sure the directories are there
    images_path.mkdir(parents=True, exist_ok=True)

    # We have a sequential set of produced images in the output directory
    image_file_num = 0

    # iterate over our train/test/val split annotation files
    for i in range(len(annotations)):
        print(f"Generating: {annotations[i]}")

        # Each annotation file has sequential image ids and bbox ids
        # We could use one unique id across all.
        image_id = 0
        bbox_id = 0

        # coco.json annotation format
        result = {'annotations': [], 'categories': categories, 'images': []}
        # iterate over the length of each split for train/test/val
        while image_id < splits[i]:
            # Make a new grayscape image as that is what they come in as
            data = Image.new('L', (image_size, image_size))

            # A list of bounding boxes to check for collistions  inthe image
            bboxes = [[0, 0, 1, 1]]

            # Where to save the image for the images spec
            new_image_path = images_path / f"{image_file_num + 1:06d}.jpg"

            # Make the image annotation for the file
            img_obj = {
                "id": image_id + 1,
                "file_name": str(new_image_path.relative_to(data_rt)),
                "height": image_size,
                "width": image_size
            }
            result["images"].append(img_obj)

            # small object
            ratios = [0.5, 0.8]
            N = randomizer.randint(0, sizes[0])
            for _ in range(N):
                ratio = randomizer.choice(ratios)
                insert_number(result, data, bboxes, data_rt, all_items, image_size, image_id, bbox_id,
                              max_iou, ratio)
                bbox_id += 1

            # medium object
            ratios = [1., 1.5, 2.]
            N = randomizer.randint(0, sizes[1])
            for _ in range(N):
                ratio = randomizer.choice(ratios)
                insert_number(result, data, bboxes, data_rt, all_items, image_size, image_id, bbox_id,
                              max_iou, ratio)
                bbox_id += 1

            # big object
            ratios = [3., 4.]
            N = randomizer.randint(0, sizes[2])
            for _ in range(N):
                ratio = randomizer.choice(ratios)
                insert_number(result, data, bboxes, data_rt, all_items, image_size, image_id, bbox_id,
                              max_iou, ratio)
                bbox_id += 1

            # Invert so it is black on white and prettier
            data = ImageOps.invert(data)

            # save the new image using PIL
            data.save(new_image_path)

            # Increment the image file name and the image id within the anno file
            image_file_num += 1
            image_id += 1

        # generate annotation file
        with open(str(anno_path / annotations[i]), "w+") as wf:
            json.dump(result, wf, indent=4)
    print("Completed image and annotation generation")


def setup_randomizer(seed):
    global randomizer
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        if seed > 2 ** 32 - 1:
            print("The random seed must be and unsigned 32 bit integer. Exiting")
            sys.exit(-1)

    print(f"Using random seed: {seed}")
    randomizer.seed(seed)


def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""

    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return ((xmax - xmin) * (ymax - ymin)) / (A1 + A2)


def insert_number(result, data, boxes, data_rt, all_items,
                  image_size, image_num, bbox_id, max_iou, ratio: float = 1.0):
    # Select and image from out label -> [ images ] dict.
    label = randomizer.randint(0, 9)
    idx = randomizer.randint(0, len(all_items[label]) - 1)
    image = all_items[label][idx]

    new_size = (int(28 * ratio), int(28 * ratio))
    image.resize(new_size)

    # TODO: Add in configurable augmentations/rotations/normalizations

    w, h = image.size

    # coco.json annotation format
    labels_obj = {
        "id": bbox_id,
        "image_id": image_num + 1,
        "category_id": label,
        "area": 0,
        "bbox": [],
        "is_crowd": 0
    }

    while True:
        # generate random bbox
        xmin = randomizer.randint(0, image_size - w)
        ymin = randomizer.randint(0, image_size - h)
        xmax = xmin + w
        ymax = ymin + h

        # bbox format to calculate IOU with other objects in image
        iou_box = [xmin, ymin, xmax, ymax]

        # calculate bbox in coco.json format
        width = (xmax - xmin)
        height = (ymax - ymin)
        area = height * width
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        # [x,y,width,height] x,y are the center coordinates)
        bbox = [int(x_center), int(y_center), width, height]

        iou = [compute_iou(iou_box, b) for b in boxes]

        # Verify that this new addition to the image is under our IOU limit
        if max(iou) < max_iou:
            labels_obj["bbox"] = bbox
            boxes.append(iou_box)
            labels_obj["area"] = area
            # append the annotation in coco format
            result["annotations"].append(labels_obj)
            break

    # Paste the new number on the image
    data.paste(image, (xmin, ymin))


def main():
    parser = argparse.ArgumentParser(
        description="This tool generates an object detection dataset based on mnist. Each images has"
                    "some number of numbers randomly placed around inside it. Images will not be"
                    "overalapped based on a minimum IOU number")

    parser.add_argument("data_root", type=str, default="/dataroot",
                        help="Where to create the image directory and to place the annotations file.")
    parser.add_argument("--image_size", type=int, default=416,
                        help="Size in pixels. Default is 416")
    parser.add_argument("--max_iou", type=float, default=0.02,
                        help="IOU used to detect overlap, default is 0.02")
    parser.add_argument("--small", type=int, default=3,
                        help="Maximum number of small placement. Default is 3.")
    parser.add_argument("--medium", type=int, default=6,
                        help="Maximum number of medium placements. Default is 6")
    parser.add_argument("--big", type=int, default=3,
                        help="Maximum number of large placements. Default is 2")
    parser.add_argument("--train_split", type=int, default=1000,
                        help="Number of images in training split. Default is 80")
    parser.add_argument("--val_split", type=int, default=250,
                        help="Number of images in validation split. Default is 10")
    parser.add_argument("--test_split", type=int, default=250,
                        help="Number of images in test split. Default is 10")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to use for randomization. Must be 32 bit unsigned for numpy. "
                             "By default, one will be made for you.")

    args = parser.parse_args()

    setup_randomizer(args.seed)

    image_size = args.image_size
    sizes = [args.small, args.medium, args.big]

    # Download MNIST dataset from torchvision
    mn = torchvision.datasets.MNIST('./mnist-raw', download=True)
    all_items = defaultdict(list)
    for item in mn:
        all_items[item[1]].append(item[0])

    # Generate random images and their corresponding annotation for train/test/val
    print(f"Creating content in {args.data_root}")
    labels_and_images(args.data_root, all_items, args.train_split, args.test_split, args.val_split,
                      sizes, image_size, args.max_iou)


if __name__ == "__main__":
    main()
