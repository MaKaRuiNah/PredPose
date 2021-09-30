#!/usr/bin/python
# -*- coding:utf8 -*-


from .process import *

# human pose topology
from .zoo.posetrack import *

# dataset zoo
from .zoo.build import build_train_loader, build_eval_loader, get_dataset_name

# datasets (Required for DATASET_REGISTRY)
from .zoo.posetrack.PoseTrack import PoseTrack
from .zoo.posetrack.PoseTrackPred1 import PoseTrackPred1
from .zoo.human36.Human36M import Human36M
from .zoo.human36.Human36M2 import Human36M2
