import numpy as np
from typing import List

class SceneData():
    def __init__(self, pklgrasp_dir: List[str], obj_poses: List[np.ndarray], grasps_tf: np.ndarray, grasps_score: np.ndarray):
        self.pklgrasp_dir = pklgrasp_dir
        self.obj_poses = obj_poses
        self.grasps_tf = grasps_tf
        self.grasps_score = grasps_score