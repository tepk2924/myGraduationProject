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
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from scipy.spatial.transform import Rotation

grasp_point: np.ndarray = np.array([[1, 0, 0, 0.4],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0.4],
                                    [0, 0, 0, 1]])
P = np.expand_dims(grasp_point[:3, 3], axis=-1)
print(P)
R = grasp_point[:3, :3]
print(R)
approach = 0.05 #Metre

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('robot', anonymous=True)
robot = RobotCommander()
scene = PlanningSceneInterface()
group_name = "irb120_arm"
move_group:MoveGroupCommander = MoveGroupCommander(group_name)

display_trajectory_pub = rospy.Publisher('/move_group/display_planned_path',
                                         moveit_msgs.msg.DisplayTrajectory,
                                         queue_size=20)

home_joint = [0.0, 0.0, 0.0, 0.0, pi/2, 0]
current_joint = move_group.get_current_joint_values()
move_group.go(home_joint, wait=True)
move_group.stop()

pose_goal = geometry_msgs.msg.Pose()
rot = Rotation.from_matrix(-R).as_quat().tolist()
pos = (P + R@np.array([[0], [0], [approach]])).tolist()
pose_goal.orientation.x = rot[0]
pose_goal.orientation.y = rot[1]
pose_goal.orientation.z = rot[2]
pose_goal.orientation.w = rot[3]
pose_goal.position.x = pos[0][0]
pose_goal.position.y = pos[1][0]
pose_goal.position.z = pos[2][0]
move_group.set_pose_target(pose_goal)
plan = move_group.go(wait=True)
move_group.stop()
move_group.clear_pose_targets()

pose_goal = geometry_msgs.msg.Pose()
rot = Rotation.from_matrix(-R).as_quat().tolist()
pos = P.tolist()
pose_goal.orientation.x = rot[0]
pose_goal.orientation.y = rot[1]
pose_goal.orientation.z = rot[2]
pose_goal.orientation.w = rot[3]
pose_goal.position.x = pos[0][0]
pose_goal.position.y = pos[1][0]
pose_goal.position.z = pos[2][0]
move_group.set_pose_target(pose_goal)
plan = move_group.go(wait=True)
move_group.stop()
move_group.clear_pose_targets()

pose_goal = geometry_msgs.msg.Pose()
rot = Rotation.from_matrix(-R).as_quat().tolist()
pos = (P + R@np.array([[0], [0], [approach]])).tolist()
pose_goal.orientation.x = rot[0]
pose_goal.orientation.y = rot[1]
pose_goal.orientation.z = rot[2]
pose_goal.orientation.w = rot[3]
pose_goal.position.x = pos[0][0]
pose_goal.position.y = pos[1][0]
pose_goal.position.z = pos[2][0]
move_group.set_pose_target(pose_goal)
plan = move_group.go(wait=True)
move_group.stop()
move_group.clear_pose_targets()

move_group.go(home_joint, wait=True)
move_group.stop()