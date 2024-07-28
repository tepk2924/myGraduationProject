#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import RobotMain, RobotMainResponse
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from arm_pkg.srv import MainUnet, MainUnetRequest, MainUnetResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as pilimage
import numpy as np
import rospy

pub = rospy.Publisher('chatter', Image, queue_size=2)
bridge = CvBridge()

def callback(req):
    rospy.wait_for_service("main_camera")
    service_req_image_depth = rospy.ServiceProxy("main_camera", MainCamera)
    resp1:MainCameraResponse = service_req_image_depth()
    image = resp1.img
    depth = resp1.depth
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")
    pilimage.fromarray(img_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Image.png"))
    service_req_unet_segmap = rospy.ServiceProxy("main_unet", MainUnet)
    resp2:MainUnetResponse = service_req_unet_segmap(image, depth)
    segmap_np: np.ndarray = bridge.imgmsg_to_cv2(resp2.segmap, "rgb8")
    pilimage.fromarray(segmap_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Segmap.png"))
    return RobotMainResponse()

rospy.init_node("main")
service_as_server = rospy.Service("robot_main_service", RobotMain, callback)
rospy.spin()