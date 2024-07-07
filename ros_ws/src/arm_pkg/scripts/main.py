#!/usr/bin/env python
from arm_pkg.srv import main_camera, main_cameraRequest
from arm_pkg.srv import main_unet, main_unetRequest
from arm_pkg.srv import robot_main, robot_mainResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import rospy

pub = rospy.Publisher('chatter', Image, queue_size=2)

def callback(req):
    rospy.wait_for_service("/main_camera")
    service_req_image_depth = rospy.ServiceProxy("/main_camera", main_camera)
    resp1 = service_req_image_depth()
    image = resp1.img
    depth = resp1.depth
    service_req_unet_segmap = rospy.ServiceProxy("/main_unet", main_unet)
    resp2 = service_req_unet_segmap(image, depth)
    pub.publish(resp2.segmap)
    return robot_mainResponse()

rospy.init_node("main")
service_as_server = rospy.Service("/robot_main_service", robot_main, callback)
rospy.spin()