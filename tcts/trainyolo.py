import os
import sys
import cv2
import argparse

import yolointerface as yoloint
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Trains the terminal YOLOv3 network' +\
        'on the provided 2D object detection dataset, inferring on the dataset with' +\
        'an SfMLearner network to create an augmented, 4 channel YOLOv3.')
    parser.add_argument('--use-tfds', nargs=1, default=None, required=False, type=str, \
        help='Instead of trying to load from an on disk dataset, load the dataset from' +\
            'tfds-nightly with the specified name.')
    parser.add_argument('--batchSize', nargs=1, default=[8], required=False, type=int, \
        help='The batch size while training.')
    parser.add_argument('--maxEpochs', nargs=1, default=[3], required=False, type=int, \
        help='The maximum number of iterations over the data (can be stopped early).')
    parser.add_argument('--checkPoint', nargs=1, default=None, required=False, type=str, \
        help='The previous trained version of the YOLO network.')
    args = parser.parse_args()

    yoloint.load(8, pretrained=True)
    

if __name__=='__main__':
    main()
