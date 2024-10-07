#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import copy
import rospy
import moveit_commander
import numpy as np
from moveit_commander import RobotCommander
from moveit_commander import PlanningSceneInterface
from moveit_commander import MoveGroupCommander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from moveit_commander.conversions import pose_to_list
from scipy.spatial.transform import Rotation
from arm_pkg.srv import MainRobot, MainRobotResponse, MainRobotRequest
from geometry_msgs.msg import Point, Quaternion, Vector3

def init():
    global APPROACH_DIST
    APPROACH_DIST = 0.05 #Metre
    moveit_commander.roscpp_initialize(sys.argv)
    global robot
    global scene
    robot = RobotCommander()
    scene = PlanningSceneInterface()
    group_name = "irb120_arm"
    global move_group
    move_group = MoveGroupCommander(group_name)
    display_trajectory_pub = rospy.Publisher('/move_group/display_planned_path',
                                            moveit_msgs.msg.DisplayTrajectory,
                                            queue_size=20)
    global bounding_box_pub
    bounding_box_pub = rospy.Publisher('bounding_box',
                                       Marker,
                                       queue_size=10)
    global filtered_grasps_pub
    filtered_grasps_pub = rospy.Publisher('filtered_grasps',
                                          MarkerArray,
                                          queue_size=10)
    global selected_grasp_pub
    selected_grasp_pub = rospy.Publisher('selected_grasp',
                                         Marker,
                                         queue_size=10)
    #The robot's home position, the figures is the angles of robot joints.
    global HOME_JOINT
    HOME_JOINT = [0.0, 0.0, 0.0, 0.0, pi/2, 0]
    global K_VEC
    K_VEC = np.array([[0, 0, 1]])
    bbox_minX = rospy.get_param("bbox_minX")
    bbox_maxX = rospy.get_param("bbox_maxX")
    bbox_minY = rospy.get_param("bbox_minY")
    bbox_maxY = rospy.get_param("bbox_maxY")
    bbox_minZ = rospy.get_param("bbox_minZ")
    bbox_maxZ = rospy.get_param("bbox_maxZ")
    global minXYZ
    global maxXYZ
    minXYZ = np.array([bbox_minX, bbox_minY, bbox_minZ])
    maxXYZ = np.array([bbox_maxX, bbox_maxY, bbox_maxZ])
    global bounding_box_center
    global bounding_box_scale
    bounding_box_center = (minXYZ + maxXYZ)/2
    bounding_box_scale = maxXYZ - minXYZ


def abb_go_home():
    '''
    Make the robot(abb) go to its home position defined at init stage.
    '''
    move_group.go(HOME_JOINT, wait=True)
    move_group.stop()

def move_abb(point,
             approach) -> bool:
    '''
    Plan & Move abb robot
    ---
    - point: np.ndarray or list [3] [x, y, z]
    - approach: np.ndarray or list [3], direction: outward from object surface
    '''
    if isinstance(point, list):
        point = np.array(point)
    if isinstance(approach, list):
        approach = np.array(approach)
    approach = approach / np.linalg.norm(approach)
    pose_goal = geometry_msgs.msg.Pose()
    rot:Rotation = Rotation.align_vectors(np.expand_dims(-approach, axis=0), K_VEC)[0]
    pose_goal.orientation = Quaternion(*(rot.as_quat()))
    pose_goal.position = Point(*point)
    move_group.set_pose_target(pose_goal)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    return plan

def arrowmarker(color: list,
                framename: str,
                position: np.ndarray,
                direction: np.ndarray,
                arrow_radius: float,
                arrow_tip_radius: float,
                arrow_length: float,
                idx: int) -> Marker:
    marker = Marker()
    marker.type = marker.ARROW
    marker.color = ColorRGBA(*color)
    marker.header.frame_id = framename
    marker.pose.position = Point(*position)
    marker.pose.orientation = Quaternion(0, 0, 0, 1)
    marker.points = [Point(0, 0, 0), Point(*(arrow_length*direction))]
    marker.scale = Vector3(arrow_radius, arrow_tip_radius, 0)
    marker.id = idx
    return marker

def callback(req:MainRobotRequest):
    '''
    The main callback part of the node
    ---
    - 0. The robot node tries to get service from main node.
    - 1. The node filters unappropriate (out of bounding box) grasps aquired as service.
    - 2. The node decides the robot's best grasp position & pose
    - 3. The node actually make the robot(abb) move
    '''
    #STEP 0: The node gets service
    points = np.array(req.pc_filtered_msg.data).reshape((-1, 3))
    approaches = np.array(req.approaches_filtered_msg.data).reshape((-1, 3))
    scores = np.array(req.scores_filtered_msg.data).reshape((-1))

    #STEP 1: Filter Out the grasp points out of range
    #STEP 1-1: Visualizing the bounding box representing the range of possible grasp points
    
    bounding_box = Marker()
    #Bounding box is transparent blue
    bounding_box.color = ColorRGBA(0, 0, 1, 0.3)
    
    bounding_box.header.frame_id = "base_link"
    bounding_box.type = bounding_box.CUBE
    bounding_box.pose.position = Point(*bounding_box_center)
    bounding_box.pose.orientation = Quaternion(0, 0, 0, 1)
    bounding_box.scale = Vector3(*bounding_box_scale)
    bounding_box.id = 0

    bounding_box_pub.publish(bounding_box)

    #STEP 1-2: Actually filtering out grasps
    mask = np.all((points >= minXYZ) & (points <= maxXYZ), axis=1)
    filtered_points = points[mask]
    filtered_approaches = approaches[mask]
    filtered_scores = scores[mask]

    #STEP 1-3: Visualize the filtered grasps
    filtered_grasps = MarkerArray()
    filtered_grasps.markers = [arrowmarker(color=[0, 0, 1, 1],
                                           framename="base_link",
                                           position=filtered_point,
                                           direction=filtered_approach,
                                           arrow_radius=0.005,
                                           arrow_tip_radius=0.01,
                                           arrow_length=0.05,
                                           idx=idx) for idx, (filtered_point, filtered_approach) in enumerate(zip(filtered_points, filtered_approaches))]
    filtered_grasps_pub.publish(filtered_grasps)

    sorted_scores = filtered_scores.copy()
    sorted_scores.sort()
    RANK = 0
    try:
        while True:
            RANK += 1
            abb_go_home()
            #STEP 2: Select the grasp point with the highest score (highest area of range of wrench space)
            #STEP 2-1: Select
            selected_idx = np.where(filtered_scores == sorted_scores[-RANK])[0]
            decided_grasp_point = filtered_points[selected_idx][0]
            decided_approach = filtered_approaches[selected_idx][0]

            #STEP 2-2: Visualize the selected grasp
            selected_grasp_pub.publish(arrowmarker(color=[1, 0, 0, 1],
                                                framename="base_link",
                                                position=decided_grasp_point,
                                                direction=decided_approach,
                                                arrow_radius=0.01,
                                                arrow_tip_radius=0.02,
                                                arrow_length=0.1,
                                                idx=0))

            #STEP 3: Use Moveit to make the robot move (or virtually simulate the movement)
            #STEP 3-1: The robot arm position itself a little bit(APPROACH_DIST) apart from surface
            ret = move_abb(point=decided_grasp_point + APPROACH_DIST*decided_approach,
                        approach=decided_approach)
            if not ret:
                print("No plan is found for this grasp, continuing for other grasp.")
                continue
            
            #STEP 3-2: The robot arm actually approaches the surface, and pull the object using the end effector.
            ret = move_abb(point=decided_grasp_point,
                        approach=decided_approach)
            if not ret:
                print("No plan is found for this grasp, continuing for other grasp.")
                continue
            
            #STEP 3-3: The robot goes a little backward, resulting it being at the first state.
            ret = move_abb(point=decided_grasp_point + APPROACH_DIST*decided_approach,
                        approach=decided_approach)
            if not ret:
                print("No plan is found for this grasp, continuing for other grasp.")
                continue

            #STEP 3-4: Make the robot goes to home pose.
            abb_go_home()
            break
    except IndexError:
        print("Couldn't find the appropriate plan!")

    return MainRobotResponse()

if __name__ == "__main__":
    init()
    rospy.init_node('robot')
    service_as_server = rospy.Service("main_robot", MainRobot, callback)
    rospy.spin()