import numpy as np
from typing import List

class SceneData():
    def __init__(self, obj_file_list: List[str], obj_poses: List[np.ndarray], grasps_tf: np.ndarray, grasps_score: np.ndarray):
        self.obj_file_list = obj_file_list
        self.obj_poses = obj_poses
        self.grasps_tf = grasps_tf
        self.grasps_score = grasps_score