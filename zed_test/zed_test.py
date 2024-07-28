import pyzed.sl as sl
import numpy as np
import os
from PIL import Image

target_folder = os.path.dirname(__file__)

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER

runtime_params = sl.RuntimeParameters()
runtime_params.enable_fill_mode = True

# Open the camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
    print("Cannot Open the Camera")
    exit(1)

depth = sl.Mat()
rgb_left = sl.Mat()
pc = sl.Mat()

zed.grab(runtime_params)
zed.retrieve_image(rgb_left, sl.VIEW.LEFT)
zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
zed.retrieve_measure(pc, sl.MEASURE.XYZRGBA)

rgb_left_np = rgb_left.get_data()[:, :, :3]
rgb_left_np[:, :, [0, 2]] = rgb_left_np[:, :, [2, 0]]
depth_np = depth.get_data()
point_cloud_np = pc.get_data()

Image.fromarray(rgb_left_np, "RGB").save(os.path.join(target_folder, "Image.png"))
with open(os.path.join(target_folder, "depth_binary.npy"), "wb") as f:
    np.save(f, depth_np)
with open(os.path.join(target_folder, "point_cloud.npy"), "wb") as f:
    np.save(f, point_cloud_np)
with open(os.path.join(target_folder, "rgb.npy"), "wb") as f:
    np.save(f, rgb_left_np)