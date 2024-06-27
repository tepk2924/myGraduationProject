#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def camera():
    pub = rospy.Publisher('frame', Image, queue_size=10)
    rospy.init_node('camera', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    bridge = CvBridge()
    cap = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        ret, cv_image = cap.read()
        if not ret:
            break
        # cv2.imshow('frame', cv_image)
        # cv2.waitKey(1)
        pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        rate.sleep()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera()