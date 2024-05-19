import blenderproc as bproc
import argparse
import bpy
import numpy as np
import random
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str)
parser.add_argument('--target_folder', type=str)
parser.add_argument('--texture_folder', type=str)
args = parser.parse_args()

filepath = args.filepath
target_folder = args.target_folder
texture_folder = args.texture_folder

bproc.init() # 이 줄이 없으면 249장을 추가로 렌더링하게 됨.

objs = bproc.loader.load_obj(filepath)
with np.load(filepath.replace(".obj", ".npz")) as grasp_data:
    grasps_tf = grasp_data["scene_grasps_tf"]
    grasps_scores = grasp_data["scene_grasps_scores"]

obj_tags = []
for obj in objs:
    obj_name = obj.get_name()
    if obj_name == "table":
        obj_tag = "background"
    else:
        obj_tag = random.choice(["invalid", "valid"])
    if obj_tag == "background":
        obj.set_cp("category_id", 0)
    elif obj_tag == "invalid":
        obj.set_cp("category_id", 1)
    else:
        obj.set_cp("category_id", 2)
    texture_folder_tagged = os.path.join("/home/tepk2924/tepk2924Works/myGraduationProject/texture_dataset", obj_tag)
    texture_filename = random.choice(os.listdir(texture_folder_tagged))
    mat = bproc.material.create_material_from_texture(os.path.join(texture_folder_tagged, texture_filename), texture_filename)
    obj.add_uv_mapping(projection="cube")
    obj.add_material(mat)

deg = math.pi/180
lights = []

#광원 배치 : 1개부터 5개까지 랜덤
for _ in range(random.randint(1, 5)):
    theta = 2*math.pi*random.random()
    phi = 70*deg + (15*deg)*random.random()
    dist = 1.8 + 2.5*random.random()
    lights.append(bproc.types.Light())
    lights[-1].set_type("POINT")
    lights[-1].set_location([dist*math.cos(phi)*math.cos(theta), dist*math.cos(phi)*math.sin(theta), 0.3 + dist*math.sin(phi)])
    lights[-1].set_energy(250 + 250*random.random())

K = np.array([[616.36529541, 0, 310.25881958],
            [0, 616.20294189, 236.59980774],
            [0, 0, 1]])
bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)

deg = math.pi/180
theta = 2*math.pi*random.random()
phi = 50*deg + (30*deg)*random.random()
dist = 1.2 + 0.4*random.random()

camera_extrinsic = np.eye(4, dtype=float)

x_ = np.array([-math.sin(theta), math.cos(theta), 0], dtype=float)
z_ = np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)], dtype=float)
y_ = np.cross(z_, x_)

camera_translation = np.array([dist*math.cos(phi)*math.cos(theta), dist*math.cos(phi)*math.sin(theta), 0.3 + dist*math.sin(phi)], dtype=float).T

camera_extrinsic[:3, 0] = x_.T
camera_extrinsic[:3, 1] = y_.T
camera_extrinsic[:3, 2] = z_.T
camera_extrinsic[:3, 3] = camera_translation
bproc.camera.add_camera_pose(camera_extrinsic)
bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

depth = bproc.camera.depth_via_raytracing(bvh_tree)

points = bproc.camera.pointcloud_from_depth(depth)
points = points.reshape(-1, 3)
points = np.float32(points)


grasps_tf[:, :3, 3] /= 1000
rotation = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]], dtype=float)
grasps_tf[:, :3, :] = np.matmul(rotation, grasps_tf[:, :3, :])

bproc.renderer.enable_segmentation_output(map_by=["category_id"])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
data = bproc.renderer.render()
data["pc"] = [points]
data["grasps_tf"] = [grasps_tf]
data["grasps_scores"] = [grasps_scores]

bproc.writer.write_hdf5(target_folder, data, True)
bproc.clean_up(True)