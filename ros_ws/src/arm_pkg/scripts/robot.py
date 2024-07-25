#!/usr/bin/env python
import sys
import os
import copy
import rospy
import numpy as np
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
from arm_pkg.srv import robot_main, robot_mainResponse

if __name__ == "__main__":
    rospy.wait_for_service("/robot_main_service")
    call = rospy.ServiceProxy("/robot_main_service", robot_main)
    resp = call()