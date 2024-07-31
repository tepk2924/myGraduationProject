import pyzed.sl as sl
import numpy as np

zed = sl.Camera()

print(type(zed))

init_params = sl.InitParameters()

zed.open(init_params)

zed_serial = zed.get_camera_information()

print(f"{zed_serial.camera_model = }")