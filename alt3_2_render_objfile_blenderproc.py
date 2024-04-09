import blenderproc as bproc
import argparse
import bpy
import numpy as np
import random
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument('--obj_file', type=str)
args = parser.parse_args()

bproc.init() # 이 줄이 없으면 249장을 추가로 렌더링하게 됨.

objs = bproc.loader.load_obj(args.obj_file)
obj_tags = []
for obj in objs:
    obj_name = obj.get_name()
    if obj_name == "table":
        obj_tag = "table"
    else:
        obj_tag = random.choice(["invalid", "valid"])
    if obj_tag == "table":
        obj.set_cp("category_id", 0)
    elif obj_tag == "invalid":
        obj.set_cp("category_id", 1)
    else:
        obj.set_cp("category_id", 2)
    texture_folder = os.path.join("/home/tepk2924/tepk2924Works/myGraduationProject/texture_dataset", obj_tag)
    texture_filename = random.choice(os.listdir(texture_folder))
    mat = bproc.material.create_material_from_texture(os.path.join(texture_folder, texture_filename), texture_filename)
    obj.add_uv_mapping(projection="cube")
    obj.add_material(mat)

deg = math.pi/180
theta = 2*math.pi*random.random()
phi = 40*deg + (40*deg)*random.random()
dist = 2.5 + 2.5*random.random()

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([dist*math.cos(phi)*math.cos(theta), dist*math.cos(phi)*math.sin(theta), 0.3 + dist*math.sin(phi)])
light.set_energy(500 * 2**(4*random.random()))

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

bproc.renderer.enable_segmentation_output(map_by=["category_id"])

data = bproc.renderer.render()
data["pc"] = [points]
print(data)

bproc.writer.write_hdf5("./hdf5output", data)