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

from PIL import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from moveit_commander import PlanningSceneInterface
import tf.transformations

bridge = CvBridge()
scene = PlanningSceneInterface()
markerarray_pub = rospy.Publisher('/Chess_Points', MarkerArray, queue_size=10)
print(dir(scene))

if __name__ == '__main__':
    rospy.init_node("solvePnP")
    rospy.wait_for_service("main_camera")
    service_req_image_depth = rospy.ServiceProxy("main_camera", MainCamera)
    resp:MainCameraResponse = service_req_image_depth()
    image = resp.img
    K = np.array(resp.intrinsic.data)
    K = K.reshape((3, 3))
    print(K)
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")
    img_np_copy = bridge.imgmsg_to_cv2(image, "rgb8")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_img, (6, 5), None)
    if ret == True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        corners = np.squeeze(corners)
        for idx, point in enumerate(corners):
            img_np_copy = cv2.circle(img_np_copy, tuple(point),
                                     10,
                                     (0, 0, 255),
                                     -1)
            img_np_copy = cv2.putText(img_np_copy,
                                      f"{idx:02d}",
                                      tuple(point),
                                      font,
                                      0.5,
                                      (255, 0, 0))
        Image.fromarray(img_np_copy).save(os.path.join(os.path.dirname(__file__), "Founded_ChessPoint.png"))
            
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

            obj_points_np = np.array(obj_points)
        except tf.LookupException:
            pass
        except tf.ConnectivityException:
            pass
        except AttributeError:
            pass
        else:
            break
    flags = cv2.SOLVEPNP_ITERATIVE
    dist_coeffs = np.zeros((4, 1))
    retval, r_vector, t_vector = cv2.solvePnP(obj_points_np,
                                            corners,
                                            K,
                                            dist_coeffs,
                                            flags=flags)
    
    r_matrix = np.zeros((3, 3))
    cv2.Rodrigues(r_vector, r_matrix)
    r_inv = np.linalg.inv(r_matrix)
    t_vector = np.dot(-r_inv, t_vector)

    cam_trans = np.zeros(shape=(4, 4))
    cam_trans[0:3, 0:3] = r_inv
    cam_trans[0:3, 3:] = t_vector
    cam_trans[3, 3] = 1
    cam_trans[:, 1] *= -1
    cam_trans[:, 2] *= -1

    arr = cam_trans.tolist()

    with open(os.path.join(os.path.dirname(__file__), "camera_tf.txt"), "a") as f:
        f.write("="*30 + "\n")
        for row in arr:
            f.write(" ".join(map(str, row)) + "\n")

    broadcaster = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    quat = tf.transformations.quaternion_from_matrix(cam_trans)
    trans = tf.transformations.translation_from_matrix(cam_trans)

    while not rospy.is_shutdown():
        markerarray = MarkerArray()
        for idx, pos in enumerate(obj_points):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.id = idx
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = tuple(pos)
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.scale.x = 0.001
            marker.scale.y = 0.001
            marker.scale.z = 0.001
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            markerarray.markers.append(marker)
        markerarray_pub.publish(markerarray)
        broadcaster.sendTransform(translation=trans,
                                  rotation=quat,
                                  time=rospy.Time.now(),
                                  child="camera",
                                  parent="base_link"
                                  )
        rate.sleep()

#Maybr I can utilize cv2.calbrateCamera which spits out
#retval, cameraMatrix (camera intrinsic is one of output as well), distCoeffs, rvecs, tvecs
#Without input of camera intrinsics.