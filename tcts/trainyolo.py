import os
import sys
import cv2
import argparse

# Note: For some reason, sfminterface must be imported before yolointerface. Fix later
import sfminterface as sfmint
import yolointerface as yoloint
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import PIL as pil

def deepenDataframe(sfmSession, depthNetwork, dataframe):
    images = dataframe['image']

    for i in range(len(images)):
        image = pil.Image.fromarray(images[i])
        image = image.resize((sfmint.SFM_DEFAULT_IMG_WIDTH, sfmint.SFM_DEFAULT_IMG_HEIGHT), pil.Image.ANTIALIAS)
        image = np.array(image).reshape(1, sfmint.SFM_DEFAULT_IMG_HEIGHT, sfmint.SFM_DEFAULT_IMG_WIDTH, 3)

        depth = depthNetwork.inference(image, sfmSession, mode='depth')['depth']
        images[i] = np.concatenate((image, depth), axis=3)

        if i % 100 == 0: print(str(i) + '/' + str(len(images)) + ' images deepened...')
    
    dataframe['image'] = images
    return dataframe


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Trains the terminal YOLOv3 network' +\
        'on the provided 2D object detection dataset, inferring on the dataset with' +\
        'an SfMLearner network to create an augmented, 4 channel YOLOv3.')
    #parser.add_argument('--use-tfds', nargs=1, default=None, required=False, type=str, \
    #    help='Instead of trying to load from an on disk dataset, load the dataset from' +\
    #        'tfds-nightly with the specified name.')
    parser.add_argument('--batchSize', nargs=1, default=[8], required=False, type=int, \
        help='The batch size while training.')
    parser.add_argument('--maxEpochs', nargs=1, default=[3], required=False, type=int, \
        help='The maximum number of iterations over the data (can be stopped early).')
    parser.add_argument('--checkPoint', nargs=1, default=None, required=False, type=str, \
        help='The previous trained version of the YOLO network.')
    args = parser.parse_args()

    # Load networks
    sfmSession, depthModel, _ = sfmint.load(loadPose=False)
    print('Depth model loaded...')

    yoloSession, yolo = yoloint.load(classCount=8)
    print('YOLOv3 model loaded...')

    # Load KITTI dataset
    dataset, dsInfo = tfds.load('kitti', split=['train', \
        'validation', 'test'], shuffle_files=False, download=True, with_info=True)
    dataframe = []
    for i in range(len(dataset)): dataframe.append(tfds.as_dataframe(dataset[i], dsInfo))
    print('Dataset loaded...')

    for i in range(len(dataset)):
        dataframe[i] = deepenDataframe(sfmSession, depthModel, dataframe[i])
    (dsTrain, dsVal, dsTest) = dataframe
    print('Dataset modified...')
    
    return (dsTrain, dsVal, dsTest), dsInfo, sfmSession, depthModel, yolo

if __name__=='__main__':
    main()
