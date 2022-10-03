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

import csv


def add_file(filename, csv_writer, label):
    with open(filename) as in_file:
        csv_reader = csv.reader(in_file, delimiter=',')
        for row in csv_reader:
            row.append(label)
            csv_writer.writerow(row)


def concat_files(prefix):
    with open(prefix + '_data.csv', "w") as out_file:
        csv_writer = csv.writer(out_file, delimiter=',')
        csv_writer.writerow(['x', 'y', 'label'])

        add_file(prefix + '_label_0.csv', csv_writer, '0')
        add_file(prefix + '_label_1.csv', csv_writer, '1')
        add_file(prefix + '_label_2.csv', csv_writer, '2')


concat_files('train')
concat_files('test')
