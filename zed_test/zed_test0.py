import pyzed.sl as sl
import numpy as np

zed = sl.Camera()

print(type(zed))

init_params = sl.InitParameters()

zed.open(init_params)

zed_info = zed.get_camera_information()
left_cam_info = zed_info.camera_configuration.calibration_parameters.left_cam

print(f"{zed_info.camera_model = }")
print(f"{left_cam_info.cx = }")
print(f"{left_cam_info.cy = }")
print(f"{left_cam_info.fx = }")
print(f"{left_cam_info.fy = }")
print(f"{left_cam_info.disto = }")
print(f"{left_cam_info.d_fov = }")
print(f"{left_cam_info.h_fov = }")


print(dir(left_cam_info))