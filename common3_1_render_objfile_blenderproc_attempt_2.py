import blenderproc as bproc
from blenderproc.python.types import MeshObjectUtility, MaterialUtility
from typing import List
import argparse
import bpy
import numpy as np
import random
import math
import os
import pickle
import sys
sys.path.append(os.path.dirname(__file__))

from common2_1_scenedata import SceneData

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str)
parser.add_argument('--target_folder', type=str)
parser.add_argument('--texture_folder', type=str)
args = parser.parse_args()

filepath = args.filepath
target_folder = args.target_folder
texture_folder = args.texture_folder

with open(filepath, "rb") as f:
    scenedata:SceneData = pickle.load(f)

grasps_tf = scenedata.grasps_tf
grasps_scores = scenedata.grasps_score
obj_file_list = scenedata.obj_file_list
obj_poses = scenedata.obj_poses

obj_poses = np.concatenate((obj_poses, np.array([[[1, 0, 0, 0],
                                                  [0, 1, 0, 0],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]]])), axis=0)
obj_poses = np.concatenate((obj_poses, np.array([[[1, 0, 0, 0],
                                                  [0, 1, 0, 0],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]]])), axis=0)

bproc.init() # 이 줄이 없으면 249장을 추가로 렌더링하게 됨.

objs: List[MeshObjectUtility.MeshObject] = []
for obj_file_path in obj_file_list:
    objs += bproc.loader.load_obj(obj_file_path)

objs += bproc.loader.load_obj(os.path.join(os.path.dirname(__file__), "table_attempt_2.obj"))
objs += bproc.loader.load_obj(os.path.join(os.path.dirname(__file__), "surrounding_attempt_2.obj"))

obj_tags = []
for obj, pose in zip(objs, obj_poses):
    obj_name = obj.get_name()
    print(f"{obj.get_name() = }")
    obj.apply_T(np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]]))
    obj.apply_T(pose)
    if obj_name in ["table", "surrounding"]:
        obj_tag = "background"
    else:
        obj_tag = random.choice(["invalid", "valid"])
    if obj_tag == "background":
        obj.set_cp("category_id", 0)
        texture_folder_tagged = os.path.join(texture_folder, "background")
        texture_filename = random.choice(os.listdir(texture_folder_tagged))
        obj.add_material(bproc.material.create_material_from_texture(os.path.join(texture_folder_tagged, texture_filename), texture_filename))
    elif obj_tag == "invalid":
        obj.set_cp("category_id", 1)
        texture_folder_tagged = os.path.join(texture_folder, "invalid")
        texture_filename = random.choice(os.listdir(texture_folder_tagged))
        mat = bproc.material.create_material_from_texture(os.path.join(texture_folder_tagged, texture_filename), texture_filename)
        mat.set_displacement_from_principled_shader_value("Base Color", multiply_factor=-2.0)
        obj.set_material(0, mat)
    else: #valid
        obj.set_cp("category_id", 2)
        if np.random.uniform(0, 1) > 0.7:
            mat:MaterialUtility.Material = obj.get_materials()[0]
            mat.set_principled_shader_value("Base Color", np.random.uniform([0, 0, 0, 1],
                                                                            [1, 1, 1, 1]))

floor_fluc = 0.2*random.random() - 0.1
objs[-1].apply_T(np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, floor_fluc],
                           [0, 0, 0, 1]]))

deg = math.pi/180
lights:List[bproc.types.Light] = []

#Light Source: 1-5 light sources
for _ in range(light_num := random.randint(1, 5)):
    theta = 2*math.pi*random.random()
    phi = 20*deg + (70*deg)*random.random()
    dist = 1.8
    lights.append(bproc.types.Light())
    lights[-1].set_type("POINT")
    lights[-1].set_location([dist*math.cos(phi)*math.cos(theta), dist*math.cos(phi)*math.sin(theta), 0.3 + dist*math.sin(phi)])
    lights[-1].set_energy(250/light_num + 250*random.random()/light_num)

CAMERA = "ZED"

if CAMERA == "RealSense":
    K = np.array([[616.36529541, 0, 310.25881958],
                [0, 616.20294189, 236.59980774],
                [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480) #Intel RealSense
elif CAMERA == "ZED":
    # K = np.array([[676.5935668945312, 0, 609.4650268554688],
    #             [0, 676.5935668945312, 366.338134765625],
    #             [0, 0, 1]])
    K = np.array([[676.5935668945312, 0, 640],
                [0, 676.5935668945312, 360],
                [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, 1280, 720) #ZED
else:
    raise Exception("What is this Camera?")

camera_loc = np.random.uniform([-0.5, -0.5, 1.0], [0.5, 0.5, 1.6])
poi = np.random.uniform([-0.2, -0.2, 0.3], [0.2, 0.2, 0.3])

camera_rot = bproc.camera.rotation_from_forward_vec(poi - camera_loc, inplane_rot=np.random.uniform(-np.pi, np.pi))

camera_extrinsic = bproc.math.build_transformation_mat(camera_loc, camera_rot)
bproc.camera.add_camera_pose(camera_extrinsic)
bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

depth = bproc.camera.depth_via_raytracing(bvh_tree)

points = bproc.camera.pointcloud_from_depth(depth)
points = points.reshape(-1, 3)
points = np.float32(points)

bproc.renderer.enable_segmentation_output(map_by=["category_id"])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
data = bproc.renderer.render()
unnoised_color = data["colors"][0]
if CAMERA == "RealSense":
    noise = np.round(np.random.normal(0, 1.5, (480, 640, 3))).astype(np.int32)
elif CAMERA == "ZED":
    noise = np.round(np.random.normal(0, 1.5, (720, 1280, 3))).astype(np.int32)
noised_color = np.clip(unnoised_color.astype(np.int32) + noise, 0, 255).astype(np.uint8)
data["colors"] = [noised_color]
data["pc"] = [points]
data["grasps_tf"] = [grasps_tf]
data["grasps_scores"] = [grasps_scores]
data["extrinsic"] = [camera_extrinsic]
data["original_obj_paths"] = [np.array(obj_file_list, dtype=np.string_)]
data["obj_poses"] = [obj_poses[:-1, :, :]]

bproc.writer.write_hdf5(target_folder, data, True)
bproc.clean_up(True)