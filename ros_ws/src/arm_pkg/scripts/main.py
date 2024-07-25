#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import robot_main, robot_mainResponse
from arm_pkg.srv import main_camera, main_cameraRequest
from arm_pkg.srv import main_unet, main_unetRequest
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as pilimage
import numpy as np
import rospy

pub = rospy.Publisher('chatter', Image, queue_size=2)
bridge = CvBridge()

def callback(req):
    rospy.wait_for_service("/main_camera")
    service_req_image_depth = rospy.ServiceProxy("/main_camera", main_camera)
    resp1 = service_req_image_depth()
    image = resp1.img
    depth = resp1.depth
    service_req_unet_segmap = rospy.ServiceProxy("/main_unet", main_unet)
    resp2 = service_req_unet_segmap(image, depth)
    segmap_np: np.ndarray = bridge.imgmsg_to_cv2(resp2, "rgb8")
    print(segmap_np.shape)
    return robot_mainResponse()

rospy.init_node("main")
service_as_server = rospy.Service("/robot_main_service", robot_main, callback)
rospy.spin()