#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import rospkg
import moveit_commander
import tf
import cv2
import numpy as np
import os
import pyzed.sl as sl
import ros_numpy

from PIL import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseStamped
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from arm_pkg.srv import RobotMain, RobotMainResponse
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
import tf.transformations

bridge = CvBridge()
pointcloud_pub = rospy.Publisher('pointcloud', PointCloud2, queue_size=10)

def callback(req):
    rospy.wait_for_service("main_camera")
    service_req_image_depth = rospy.ServiceProxy("main_camera", MainCamera)
    resp:MainCameraResponse = service_req_image_depth()
    image = resp.img
    K = np.array(resp.intrinsic.data)
    K = K.reshape((3, 3))
    img_np:np.ndarray = bridge.imgmsg_to_cv2(image, "rgb8")
    r = img_np[:, :, 0].reshape((-1))
    g = img_np[:, :, 1].reshape((-1))
    b = img_np[:, :, 2].reshape((-1))
    xyz = np.array(resp.pointcloud.data).reshape((-1, 3))
    x = xyz[:, 0].reshape((-1))
    y = xyz[:, 1].reshape((-1))
    z = xyz[:, 2].reshape((-1))

    data = np.zeros(len(r), dtype=[('x', np.float32),
                                   ('y', np.float32),
                                   ('z', np.float32),
                                   ('r', np.uint8),
                                   ('g', np.uint8),
                                   ('b', np.uint8)])

    data['x'] = x
    data['y'] = y
    data['z'] = z
    data['r'] = r
    data['g'] = g
    data['b'] = b

    # pointcloud_xyzrgb_msg = ros_numpy.msgify(PointCloud2, data)
    pointcloud_xyzrgb_msg = ros_numpy.point_cloud2.array_to_pointcloud2(data,
                                                                        None,
                                                                        "frame1")
    pointcloud_pub.publish(pointcloud_xyzrgb_msg)

    pc_msg = Float32MultiArray()
    scores_msg = Float32MultiArray()
    app_msg = Float32MultiArray()

    return RobotMainResponse(pc_msg,
                             scores_msg,
                             app_msg)

if __name__ == "__main__":
    rospy.init_node("getPointcloud_visualize")
    service_as_server = rospy.Service("robot_main_service", RobotMain, callback)
    rospy.spin()