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

filepath = "/home/tepk2924/tepk2924Works/myGraduationProject/DataSet/train/32770_Thingi10K/32770.obj"

#Initializing the blenderproc is must.
bproc.init()

objs: List[MeshObjectUtility.MeshObject] = bproc.loader.load_obj(filepath)

objs += bproc.loader.load_obj("/home/tepk2924/tepk2924Works/myGraduationProject/surrounding_attempt_2.obj")

mat = bproc.material.create_material_from_texture("/home/tepk2924/tepk2924Works/myGraduationProject/DataSet/texture_dataset/train/background/interlaced_0121.jpg", "surr")
objs[1].add_material(mat)

for obj in objs:
    print(obj.get_name())
    obj.apply_T(np.array([[0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1]]))

mat:List[MaterialUtility.Material] = objs[0].get_materials()
mat[0].set_principled_shader_value("Roughness", 0)
mat[0].set_principled_shader_value("Specular", 1)

# lightobj = bproc.filter.by_attr(objs, "name", "37786.001")
# print(lightobj)

# bproc.lighting.light_surface(lightobj,
#                              emission_strength=1000,
#                              emission_color=[1, 0, 0, 1])

deg = math.pi/180
lights = []

#Light Source: 1-5 light sources
for i in range(5):
    theta = 2*i*math.pi/5
    phi = 50*deg
    dist = 1.8
    lights.append(bproc.types.Light())
    lights[-1].set_type("POINT")
    lights[-1].set_location([dist*math.cos(phi)*math.cos(theta), dist*math.cos(phi)*math.sin(theta), 0.3 + dist*math.sin(phi)])
    lights[-1].set_energy(250)

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

# deg = math.pi/180
# theta = math.pi/4
# phi = 50*deg
# dist = 0.6

# camera_extrinsic = np.eye(4, dtype=float)

# x_ = np.array([-math.sin(theta), math.cos(theta), 0], dtype=float)
# z_ = np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)], dtype=float)
# y_ = np.cross(z_, x_)

# camera_translation = np.array([dist*math.cos(phi)*math.cos(theta), dist*math.cos(phi)*math.sin(theta), dist*math.sin(phi)], dtype=float).T

# camera_extrinsic[:3, 0] = x_.T
# camera_extrinsic[:3, 1] = y_.T
# camera_extrinsic[:3, 2] = z_.T
# camera_extrinsic[:3, 3] = camera_translation
poi = np.array([0, 0, 0])
# camera_loc = np.random.uniform([-0.3, -0.3, 0.5], [0.3, 0.3, 1])
camera_loc = np.array([0.1, 0.1, 0.2])
inplane = np.random.uniform(0, 0)
rot = bproc.camera.rotation_from_forward_vec(poi - camera_loc, inplane_rot=inplane)

camera_extrinsic = bproc.math.build_transformation_mat(camera_loc, rot)

print(f"{camera_loc = }")
print(f"{inplane = }")

bproc.camera.add_camera_pose(camera_extrinsic)

data = bproc.renderer.render()
unnoised_color = data["colors"][0]
if CAMERA == "RealSense":
    noise = np.round(np.random.normal(0, 1.5, (480, 640, 3))).astype(np.int32)
elif CAMERA == "ZED":
    noise = np.round(np.random.normal(0, 1.5, (720, 1280, 3))).astype(np.int32)
noised_color = np.clip(unnoised_color.astype(np.int32) + noise, 0, 255).astype(np.uint8)
data["colors"] = [noised_color]

bproc.writer.write_hdf5("/home/tepk2924/tepk2924Works/myGraduationProject/blenderproc_test/result", data, False)
bproc.clean_up(True)