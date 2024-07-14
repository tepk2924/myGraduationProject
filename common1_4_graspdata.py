import numpy as np
import trimesh

class GraspData():
    def __init__(self, obj_path: str, grasp_info: dict, meta_data: dict):
        self.obj_path = obj_path
        self.grasp_info = grasp_info
        self.meta_data = meta_data