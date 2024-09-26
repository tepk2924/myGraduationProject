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
    global selected_grasp_pub
    selected_grasp_pub = rospy.Publisher('selected_grasp',
                                         Marker,
                                         queue_size=10)
    #The robot's home position, the figures is the angles of robot joints.
    global HOME_JOINT
    HOME_JOINT = [0.0, 0.0, 0.0, 0.0, pi/2, 0]
    abb_go_home()
    global K_VEC
    K_VEC = np.array([[0, 0, 1]])

def abb_go_home():
    '''
    Make the robot(abb) go to its home position defined at init stage.
    '''
    move_group.go(HOME_JOINT, wait=True)
    move_group.stop()

def move_abb(point:np.ndarray | list,
             approach:np.ndarray | list):
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
    minXYZ = np.array([0.2, -0.35, 0])
    maxXYZ = np.array([0.9, 0.35, 0.5])
    
    bounding_box_center = (minXYZ + maxXYZ)/2
    bounding_box_scale = maxXYZ - minXYZ
    
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

    #STEP 2: Select the grasp point with the highest score (highest area of range of wrench space)
    #STEP 2-1: Select
    highest_idx = np.argmax(filtered_scores, axis=0)
    decided_grasp_point = filtered_points[highest_idx]
    decided_approach = filtered_approaches[highest_idx]

    #STEP 2-2: Visualize the selected grasp
    selected_grasp = Marker()
    #Selected grasp is red.
    selected_grasp.color = ColorRGBA(1, 0, 0, 1)

    selected_grasp.header.frame_id = "base_link"
    selected_grasp.type = selected_grasp.ARROW
    selected_grasp.pose.position = Point(*decided_grasp_point)

    selected_grasp.id = 0
    selected_grasp.pose.orientation = Quaternion(0, 0, 0, 1)
    selected_grasp.scale = Vector3(0.01, 0.02, 0)
    selected_grasp.points = [Point(0, 0, 0), Point(*(0.1*decided_approach))]

    selected_grasp_pub.publish(selected_grasp)

    #STEP 3: Use Moveit to make the robot move (or virtually simulate the movement)
    #STEP 3-1: The robot arm position itself a little bit(APPROACH_DIST) apart from surface
    move_abb(point=decided_grasp_point + APPROACH_DIST*decided_approach,
             approach=decided_approach)
    
    #STEP 3-2: The robot arm actually approaches the surface, and pull the object using the end effector.
    move_abb(point=decided_grasp_point,
             approach=decided_approach)
    
    #STEP 3-3: The robot goes a little backward, resulting it being at the first state.
    move_abb(point=decided_grasp_point + APPROACH_DIST*decided_approach,
             approach=decided_approach)
    
    #STEP 3-4: Make the robot goes to home pose.
    abb_go_home()

    return MainRobotResponse()

if __name__ == "__main__":
    init()
    rospy.init_node('robot')
    service_as_server = rospy.Service("main_robot", MainRobot, callback)
    rospy.spin()