#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import rospy
from arm_pkg.srv import Execution

if __name__ == "__main__":
    rospy.wait_for_service("execution")
    call = rospy.ServiceProxy("execution", Execution)
    resp = call()
    print("finished.")