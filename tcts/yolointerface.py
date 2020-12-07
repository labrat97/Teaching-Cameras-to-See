"""
Interacts with the YOLOv3 network for easy interfacing with the main publisher
subscriber nodes.
"""

import git
import os
import sys
import shutil
import wget

import tensorflow as tf
import numpy as np

YOLO_REPOSITORY_URL = 'https://github.com/labrat97/yolov3-tf2.git'
YOLO_REPOSITORY_PATH = os.path.dirname(__file__) + os.sep + 'yololib'
YOLO_PRETRAINED_WEIGHT_URL = 'https://pjreddie.com/media/files/yolov3.weights'
YOLO_PRETRAINED_FILENAME_RAW = 'yolov3.weights'
YOLO_PRETRAINED_FILENAME_CONV = 'yolov3.tf'
YOLO_TINY_ENABLE = False
YOLO_PRETRAINED_WEIGHT_PATH = YOLO_REPOSITORY_PATH + os.sep + 'data'
sys.path.append(os.path.dirname(__file__))
sys.path.append(YOLO_REPOSITORY_PATH)
sys.path.append(YOLO_REPOSITORY_PATH+os.sep+'yolov3_tf2')

def rawWeightsAvailable():
    return os.path.exists(YOLO_PRETRAINED_WEIGHT_PATH+os.sep+YOLO_PRETRAINED_FILENAME_RAW)

def downloadPretrainedWeights():
    wget.download(YOLO_PRETRAINED_WEIGHT_URL, 
        out=YOLO_PRETRAINED_WEIGHT_PATH+os.sep+YOLO_PRETRAINED_FILENAME_RAW)

def libraryDownloaded():
    try:
        _ = git.Repo(YOLO_REPOSITORY_PATH).git_dir
        return True
    except git.exc.NoSuchPathError or git.InvalidGitRepositoryError:
        return False

def downloadLibrary():
    # Clean redownload the library
    if libraryDownloaded():
        shutil.rmtree(YOLO_REPOSITORY_PATH)
    repoClone = git.Repo.clone_from(YOLO_REPOSITORY_URL, YOLO_REPOSITORY_PATH)

    # Download the weights, convert later during network load
    downloadPretrainedWeights()

# Only import post download
if not libraryDownloaded():
    downloadLibrary()
if not rawWeightsAvailable():
    downloadPretrainedWeights()

import yololib
import yololib.yolov3_tf2.models
from yololib import yolov3_tf2 as yolov3
from yololib.yolov3_tf2.utils import load_darknet_weights

def load(classCount, weights=None, tiny=YOLO_TINY_ENABLE, pretrained=False):
    # If available, setup GPU memory
    tfDevices = tf.config.experimental.list_physical_devices('GPU')
    if len(tfDevices) > 0:
        tf.config.experimental.set_memory_growth(tfDevices[0], True)
    
    # Open the network
    yolo = None
    if tiny:
        yolo = yolov3.models.YoloV3Tiny(channels=4, classes=classCount, training=True)
    else:
        yolo = yolov3.models.YoloV3(channels=4, classes=classCount, training=True)
        

    # Get the pretrained weights
    if pretrained:
        if not rawWeightsAvailable():
            downloadPretrainedWeights()
    load_darknet_weights(yolo, YOLO_PRETRAINED_WEIGHT_PATH+os.sep+YOLO_PRETRAINED_FILENAME_RAW, tiny=YOLO_TINY_ENABLE)

def evaluate(image):
    print("Not yet implemented.")
    exit(-1)
