#! /usr/bin/env python3

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
import sys

import torch.nn as nn

logger = logging.getLogger(__name__)


class BinaryNet(nn.Module):
    def __init__(self, num_features=2):
        super(BinaryNet, self).__init__()
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_out = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu(self.layer_1(input))
        x = self.sigmoid(self.layer_out(x))

        return x


class BinaryModel:
    def __call__(self, num_classes):
        if num_classes != 2:
            logger.error("This model only works with binary tabular datasets.")
            sys.exit(-1)

        return BinaryNet()


class MultiClassNet(nn.Module):
    def __init__(self, num_features=2):
        super(MultiClassNet, self).__init__()
        self.layer_1 = nn.Linear(num_features, 4)
        self.layer_out = nn.Linear(4, 3)

        self.relu = nn.ReLU()
        self.activation = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.relu(self.layer_1(input))
        x = self.activation(self.layer_out(x))

        return x


class MultiClassModel:
    def __call__(self, num_classes):
        if num_classes != 3:
            logger.error("This model is intended to work with 3-class tabular datasets.")
            sys.exit(-1)

        return MultiClassNet()


class XORClassifier(nn.Module):
    """ 
    Simple MLP for XOR tests. Roughly follows implementation of 
       https://courses.cs.washington.edu/courses/cse446/18wi/sections/section8/XOR-Pytorch.html

    Includes a nonlinear boolean parameter to flip the Sigmoid activation function between the
    two linear layers on or off.
    """

    def __init__(self, in_features, num_classes, hidden_features, nonlinear=True):
        super(XORClassifier, self).__init__()

        self.linear1 = nn.Linear(in_features, hidden_features)

        if nonlinear:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.linear2 = nn.Linear(hidden_features, num_classes)

        # Override default initialization with a N(0,1) distribution
        self.linear1.weight.data.normal_(0, 1)
        self.linear2.weight.data.normal_(0, 1)

        # Log the initialized values for testing reproducibility
        logger.info("Initialized two linear layers with the following weights:")
        logger.info(self.linear1.weight)
        logger.info(self.linear2.weight)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x) 
        x = self.linear2(x)
        return x


class XORModel:
    """
    Juneberry wrapper class for the XOR tests
    """

    def __call__(self, in_features=2, num_classes=2, hidden_features=2, nonlinear=True):

        # Juneberry requires classes to be natural counts, while the pytorch
        # binary loss functions, BCELoss and MSELoss, require a single output node.
        # 
        # Consequently, decrease num_class when it is set to two. 
        if num_classes == 2:
            num_classes = 1

        return XORClassifier(in_features, num_classes, hidden_features, nonlinear)
