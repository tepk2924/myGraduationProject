#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import RobotMain, RobotMainResponse
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from arm_pkg.srv import MainUnet, MainUnetRequest, MainUnetResponse
from arm_pkg.srv import MainSgnet, MainSgnetRequest, MainSgnetResponse
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
    pc = resp1.pointcloud
    pc_np = np.array(pc).reshape((-1, 3))
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")
    # pilimage.fromarray(img_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Image.png"))
    service_req_unet_segmap = rospy.ServiceProxy("main_unet", MainUnet)
    resp2:MainUnetResponse = service_req_unet_segmap(image, depth)
    segmap_np: np.ndarray = bridge.imgmsg_to_cv2(resp2.segmap, "rgb8")
    # pilimage.fromarray(segmap_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Segmap.png"))
    service_req_sgnet = rospy.ServiceProxy("main_sgnet", MainSgnet)
    resp3:MainSgnetResponse = service_req_sgnet(pc)
    pc_result_np = np.array(resp3.pointcloudresult.data).reshape((-1, 3))
    scores_np = np.array(resp3.scores.data).reshape((-1))
    approaches_np = np.array(resp3.scores.data).reshape((-1, 3))

    THRESHOLD = np.quantile(scores_np, 0.9)
    selected_idx = np.where(scores_np >= THRESHOLD)[0]
    pc_selected = pc_result_np[selected_idx]
    scores_selected = scores_np[selected_idx]
    approaches_selected = approaches_np[selected_idx]

    return RobotMainResponse()

rospy.init_node("main")
service_as_server = rospy.Service("robot_main_service", RobotMain, callback)
rospy.spin()