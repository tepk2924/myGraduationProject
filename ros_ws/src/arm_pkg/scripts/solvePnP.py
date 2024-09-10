#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import rospkg
import moveit_commander
import tf
import cv2
import numpy as np
import time
import pyzed.sl as sl

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseStamped
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse

bridge = CvBridge()

if __name__ == '__main__':
    rospy.init_node("solvePnP")
    rospy.wait_for_service("main_camera")
    service_req_image_depth = rospy.ServiceProxy("main_camera", MainCamera)
    resp:MainCameraResponse = service_req_image_depth()
    image = resp.img
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_img, (6, 5), None)
    if ret == True:
        corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1),
                                        criteria)
    else:
        print("No chessboard found")
        exit(1)
    
    irb120_mvc = moveit_commander.MoveGroupCommander("irb120_arm")
    tlistener = tf.TransformListener()
    while True:
        try:
            x = np.linspace(50, -50, 6, endpoint=True) / 1000
            y = np.linspace(-40, 40, 5, endpoint=True) / 1000
            x, y = np.meshgrid(x, y, indexing='xy')
            x = x.flatten()
            y = y.flatten()

            # make pose list
            obj_points = []
            for i in range(len(x)):
                # make pose msg
                pose_i = Pose()
                pose_i.position.x = x[i]
                pose_i.position.y = y[i]
                pose_i.position.z = 0
                pose_i.orientation.x = 0
                pose_i.orientation.y = 0
                pose_i.orientation.z = 0
                pose_i.orientation.w = 1

                # get chessboard frame id
                frame_id = irb120_mvc.get_end_effector_link()

                # make pose stamped msg
                pose_stamped_i = PoseStamped()
                pose_stamped_i.header.frame_id = frame_id
                pose_stamped_i.pose = pose_i

                # convert to `pose_reference frame`
                pose_stamped_i = tlistener.transformPose(irb120_mvc.get_pose_reference_frame(), pose_stamped_i)

                # convert to
                obj_points.append([
                    pose_stamped_i.pose.position.x, pose_stamped_i.pose.position.y,
                    pose_stamped_i.pose.position.z
                ])

            obj_points = np.array(obj_points)
        except tf.LookupException:
            pass
        except tf.ConnectivityException:
            pass
        except AttributeError:
            pass
        else:
            break
    flags = cv2.SOLVEPNP_ITERATIVE
    K = np.array([[670.829833984375, 0, 609.4638671875],
                  [0, 670.829833984375, 366.3249816894531],
                  [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1))
    retval, r_vector, t_vector = cv2.solvePnP(obj_points,
                                      corners,
                                      K,
                                      dist_coeffs,
                                      flags=flags)
    
    r_matrix = np.zeros((3, 3))
    cv2.Rodrigues(r_vector, r_matrix)
    r_inv = np.linalg.inv(r_matrix)
    t_vector = np.dot(-r_inv, t_vector)

    proj_matrix = np.zeros(shape=(4, 4))
    proj_matrix[0:3, 0:3] = r_inv
    proj_matrix[0:3, 3:] = t_vector
    proj_matrix[3, 3] = 1

    print(proj_matrix)