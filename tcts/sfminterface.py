"""
Interacts with the SFM learner network for easy interfacing with the main publisher
subscriber nodes.
"""

import git
import os
import sys
import shutil
import tensorflow.compat.v1 as tf

DEFAULT_DEPTH_ENABLE = True
DEFAULT_POSE_ENABLE = False

SFM_REPOSITORY_URL = 'https://github.com/labrat97/SfMLearner-Tensorflow2.git'
SFM_REPOSITORY_PATH = os.path.dirname(__file__) + os.sep + 'sfm'
SFM_DEPTH_DOWNLOAD_SCRIPT = SFM_REPOSITORY_PATH + os.sep + 'models' + os.sep + 'download_depth_model.sh'
SFM_POSE_DOWNLOAD_SCRIPT = SFM_REPOSITORY_PATH + os.sep + 'models' + os.sep + 'download_pose_model.sh'
# TODO - Clean this, it's a little messy in terms of the runtime
sys.path.append(os.path.dirname(__file__))
sys.path.append(SFM_REPOSITORY_PATH)
from sfm.SfMLearner import SfMLearner

SFM_DEFAULT_DEPTH_MODEL = SFM_REPOSITORY_PATH + os.sep + 'models/model-190532'
SFM_DEFAULT_IMG_HEIGHT = 128
SFM_DEFAULT_IMG_WIDTH = 416

def libraryDownloaded():
    try:
        _ = git.Repo(SFM_REPOSITORY_PATH).git_dir
        return True
    except git.InvalidGitRepositoryError:
        return False

def downloadLibrary(downloadDepth=DEFAULT_DEPTH_ENABLE, downloadPose=DEFAULT_POSE_ENABLE):
    # Clean redownload the library
    if libraryDownloaded():
        shutil.rmtree(SFM_REPOSITORY_PATH)
    repoClone = git.Repo.clone_from(SFM_REPOSITORY_URL, SFM_REPOSITORY_PATH)
    
    # Download the depth and pose models optionally
    if downloadDepth or downloadPose:
        cwd = os.getcwd()
        os.chdir(SFM_REPOSITORY_PATH)

        if downloadDepth:
            os.system('sh ' + SFM_DEPTH_DOWNLOAD_SCRIPT)
        if downloadPose:
            os.system('sh ' + SFM_POSE_DOWNLOAD_SCRIPT)

        os.chdir(cwd)
    
    return repoClone

def load(path=None, loadDepth=DEFAULT_DEPTH_ENABLE, loadPose=DEFAULT_POSE_ENABLE):
    # Running data
    depthModel = None
    poseModel = None
    session = None

    # Quick exit
    if not loadDepth and not loadPose:
        return None, None, None
    
    # Load a pre-trained model
    if path is None: 
        path = SFM_DEFAULT_DEPTH_MODEL

    # Open an SfMLearner
    def loadSubModel(mode): 
        sfm = SfMLearner()
        sfm.setup_inference(SFM_DEFAULT_IMG_HEIGHT, 
            SFM_DEFAULT_IMG_WIDTH, 
            mode=str(mode))
        return sfm
    if loadDepth:
        depthModel = loadSubModel('depth')
    if loadPose:
        poseModel = loadSubModel('pose')

    # Open session and export for use with sfm.interface()
    saver = tf.train.Saver([var for var in tf.model_variables()])
    session = tf.Session()
    saver.restore(session, path)
    
    return session, depthModel, poseModel

def train(video):
    print("Not yet implemented.");
    exit(-1)

def predict(image, session, model, modelMode):
    return model.inference(image[None,:,:,:], session, mode=modelMode)
