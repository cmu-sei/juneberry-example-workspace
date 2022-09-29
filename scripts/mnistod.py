#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Editor      : VIM
#   File name   : mnistod.py
#   Author      : Hayden Moore
#   Created date: 2022-09-20
#
# ================================================================

import os
import PIL
import shutil
import random
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import torchvision

# ================================================================
# Changelog
# sei-hmmoore 2022-09-15 - Wrapped the parser into a main() function,
#                          Wrapped the logic into function outside of main,
#                          Added in ability to output coco.json format,
#                          Generates train/test/val splits in one call,
#                          Added more configurations
#
# ================================================================


# We use a global randomizer instance instead of random
randomizer = random.Random()


# function that generates the mnist dataset and their train/test/val annotation files for Juneberry
def labels_and_images(data_root, all_items, images_path, anno_path, train_split,
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

    bboxes_num = 0
    # iterate over our train/test/val split annotation files
    for i in range(len(annotations)):
        print(f"Generating: {annotations[i]}")
        image_num = 0
        # coco.json annotation format
        result = {'annotations': [], 'categories': categories, 'images': []}
        # iterate over the length of each split for train/test/val
        while image_num < splits[i]:
            data = PIL.Image.new('RGB', (image_size, image_size))
            bboxes = [[0, 0, 1, 1]]
            img_pth = Path(images_path)
            new_image_path = os.path.realpath(os.path.join(data_rt, img_pth,
                                                           "%06d.jpg" % (image_num + 1)))

            # small object
            ratios = [0.5, 0.8]
            N = randomizer.randint(0, sizes[0])
            if N != 0: bboxes_num += 1
            for _ in range(N):
                ratio = randomizer.choice(ratios)
                make_image(result, data, bboxes, data_rt, all_items, new_image_path, image_size, image_num, bboxes_num,
                           max_iou, ratio)

            # medium object
            ratios = [1., 1.5, 2.]
            N = randomizer.randint(0, sizes[1])
            if N != 0: bboxes_num += 1
            for _ in range(N):
                ratio = randomizer.choice(ratios)
                make_image(result, data, bboxes, data_rt, all_items, new_image_path, image_size, image_num, bboxes_num,
                           max_iou, ratio)

            # big object
            ratios = [3., 4.]
            N = randomizer.randint(0, sizes[2])
            if N != 0: bboxes_num += 1
            for _ in range(N):
                ratio = randomizer.choice(ratios)
                make_image(result, data, bboxes, data_rt, all_items, new_image_path, image_size, image_num, bboxes_num,
                           max_iou, ratio)

            if bboxes_num == 0: continue

            # save the new image using PIL
            data.save(new_image_path)

            image_num += 1

        # generate annotation file
        with open(os.path.join(anno_path, annotations[i]), "w+") as wf:
            json_object = json.dumps(result)
            wf.write(json_object)
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


def make_image(result, data, boxes, data_rt, all_items, new_image_path,
               image_size, image_num, bbox_num, max_iou, ratio:float=1.0):

    label = randomizer.randint(0, 9)
    idx = randomizer.randint(0, len(all_items[label]) - 1)
    image = all_items[label][idx]

    new_size = (int(28 * ratio), int(28 * ratio))
    image.resize(new_size)

    # TODO: Add in configurable augmentations/rotations/normalizations

    w, h = image.size
    rel_img_path = Path(new_image_path)
    rel_img_path = rel_img_path.relative_to(data_rt)

    # coco.json annotation format
    img_obj = {
        "id": image_num + 1,
        "file_name": str(rel_img_path),
        "height": image_size,
        "width": image_size
    }

    labels_obj = {
        "id": bbox_num,
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
            # append the image path if it is not already defined
            if img_obj not in result["images"]:
                result["images"].append(img_obj)
            # append the annotation in coco format
            result["annotations"].append(labels_obj)
            break
    # Paste the new number on the image
    data.paste(image, (xmin, ymin))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/dataroot")
    parser.add_argument("--annotations_path", type=str, default="./mnistod/")
    parser.add_argument("--images_path", type=str, default="./mnistod/images/")
    parser.add_argument("--images_num", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--max_iou", type=float, default=0.02)
    parser.add_argument("--small", type=int, default=3)
    parser.add_argument("--medium", type=int, default=6)
    parser.add_argument("--big", type=int, default=3)
    parser.add_argument("--train_split", type=int, default=80)
    parser.add_argument("--val_split", type=int, default=10)
    parser.add_argument("--test_split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to use for randomization. Must be 32 bit unsigned for numpy. "
                             "By default, one will be made for you.")

    flags = parser.parse_args()

    setup_randomizer(flags.seed)

    image_size = flags.image_size
    sizes = [flags.small, flags.medium, flags.big]

    if os.path.exists(flags.images_path): shutil.rmtree(flags.images_path)
    os.mkdir(flags.images_path)

    # Download MNIST dataset from torchvision
    mn = torchvision.datasets.MNIST('./mnistod', download=True)
    all_items = defaultdict(list)
    for item in mn:
        all_items[item[1]].append(item[0])

    # Generate random images and their corresponding annotation for train/test/val
    labels_and_images(flags.data_root, all_items, flags.images_path, flags.annotations_path, flags.train_split,
                      flags.test_split, flags.val_split, sizes, image_size, flags.max_iou)

    print(f"Images saved to this location: {flags.images_path}")
    print(f"Annotation files saved to this location: {flags.annotations_path}")

if __name__ == "__main__":
    main()
