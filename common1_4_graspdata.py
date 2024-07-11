import trimesh

class GraspData():
    def __init__(self, mesh: trimesh.base.Trimesh, grasp_info: dict, meta_data: dict):
        self.mesh = mesh
        self.grasp_info = grasp_info
        self.meta_data = meta_data