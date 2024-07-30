#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import MainCamera, MainCameraResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import rospy
import numpy as np
import pyzed.sl as sl

zed = sl.Camera()
# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use QUALITY depth mode
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

# Open the camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
    print("Cannot Open the Camera")
    exit(1)

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.enable_fill_mode = True

depth = sl.Mat()
rgb_left = sl.Mat()
point_cloud = sl.Mat()
bridge = CvBridge()

def callback(req):
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        zed.retrieve_image(rgb_left, sl.VIEW.LEFT)
        rgb_left_np = rgb_left.get_data()[:, :, :3]
        rgb_left_np[:, :, [0, 2]] = rgb_left_np[:, :, [2, 0]]
        depth_np:np.ndarray = depth.get_data()
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_np.reshape((-1)).tolist()
        image_msg: Image = bridge.cv2_to_imgmsg(rgb_left_np, "rgb8")
        pc_np:np.ndarray = point_cloud.get_data()[:, :, :3]
        print(pc_np)
        pc_msg = Float32MultiArray()
        pc_msg.data = pc_np.reshape((-1)).tolist()
    return MainCameraResponse(image_msg, depth_msg, pc_msg)

rospy.init_node("main")
service = rospy.Service("main_camera", MainCamera, callback)
rospy.spin()