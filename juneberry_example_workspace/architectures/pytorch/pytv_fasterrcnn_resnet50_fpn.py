#! /usr/bin/env python3

"""
Basic torchvision pretrained FasterRCNN resnet50 for the pedestrian example.
"""

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

import logging

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logger = logging.getLogger(__name__)


class FasterRCNNResnet50Reference:
    def __call__(self, num_classes):
        """
        Constructs a pretrained (on COCO) FasterRCNN with box predictor 
        :param num_classes: The number of classes in the model.
        :return: A FasterRCNN model.
        """
        # load an instance segmentation model pre-trained on COCO
        logger.info("Constructing FasterRCNNResnet50Reference...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        logger.info(f"...with BOX predictor FastRCNNPredictor with {in_features} features")

        return model
