#! /usr/bin/env python3

"""
Basic torchvision pretrained MaskRCNN resnet50 for the pedestrian example.
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
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

logger = logging.getLogger(__name__)


class MaskRCNNResnet50Reference:
    def __call__(self, num_classes, hidden_layer=256):
        """
        Constructs a pretrained (on COCO) MaskRCNN with a FastRCNNPredictor box predictor and a
        MaskRCNNPredictor mask predictor.
        :param num_classes: The number of classes in the model.
        :param hidden_layer: The size of the hidden layer in the mask predictor.
        :return: A MaskRCNN model.

        Additional args from MaskRCNNResnet50 docs that we could optionally expose.
        Arguments:
            pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet.
            trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        """
        # load an instance segmentation model pre-trained on COCO
        logger.info("Constructing MaskRCNNResnet50Reference...")
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        # NOTE: The MaskRCNN docs say this:
        #   "If box_predictor is specified, num_classes should be None."
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        logger.info(f"...with BOX predictor FastRCNNPredictor with {in_features} features")

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # and replace the mask predictor with a new one
        # TODO: What does hidden_layers (dim_reduced) mean?
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        logger.info(f"...with MASK predictor MaskRCNNPredictor with {in_features_mask} features "
                     f"and {hidden_layer} hidden layers")

        return model
