#!/usr/bin/env python
import pyzed.sl as sl
import numpy as np
import cv2
import rospy
import inspect
import h5py
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

def zed_camera():
    with h5py.File("/home/tepk2924/tepk2924Works/myGraduationProject/hdf5scenes_generated/ZED/test/7.hdf5", "r") as f:
        rgb_left_np = np.array(f["colors"])
        depth_np = np.array(f["depth"])
    pub_rgb = rospy.Publisher('zed_rgb_frame', Image, queue_size=10)
    pub_depth = rospy.Publisher('zed_depth_frame', Float32MultiArray, queue_size=10)
    rospy.init_node('camera')
    bridge = CvBridge()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_np.reshape((-1)).tolist()
        pub_rgb.publish(bridge.cv2_to_imgmsg(rgb_left_np, "rgb8"))
        pub_depth.publish(depth_msg)
        rate.sleep()

    # Close the camera
    cv2.destroyAllWindows()

if __name__ == "__main__":
    zed_camera()