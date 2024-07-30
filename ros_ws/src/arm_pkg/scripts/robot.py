#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import copy
import rospy
import numpy as np
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
from arm_pkg.srv import RobotMain, RobotMainResponse

if __name__ == "__main__":
    rospy.wait_for_service("robot_main_service")
    call = rospy.ServiceProxy("robot_main_service", RobotMain)
    resp = call()