"""
Creates a set of publishers and subscribers to make a subjective view of the
world through the use of more objective viewed cameras.
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import argparse

import rospy
from std_msgs import msg

import sfminterface
import yolointerface
from constants import *

def main():
    # Get command data
    parser = argparse.ArgumentParser(prog="SubjectiveCamera",
        description="Takes in camera input through ROS2's DDS then, using " \
        + "machine learning, computes depth, objects, and pose without explicit camera calibration.")
    # TODO - This
    return 0

if __name__ == '__main__':
    main()
