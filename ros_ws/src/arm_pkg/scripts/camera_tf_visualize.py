#This is not a ROS node, this is just normal Python script
import numpy as np
import trimesh
import trimesh.creation

with open("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/arm_pkg/scripts/camera_tf.txt", "r") as f:
    lines = f.readlines()

array = []
for line in lines:
    array.append(list(map(float, line.split())))

transform = np.array(array)
print(transform)

scene = trimesh.Scene()
axis_world = trimesh.creation.axis(origin_color=(0, 0, 0), origin_size=0.002, axis_radius=0.001, axis_length=0.1)
axis_cam:trimesh.Trimesh = trimesh.creation.axis(transform=transform, origin_size=0.001, axis_radius=0.001, axis_length=0.05)

scene.add_geometry(axis_world)
scene.add_geometry(axis_cam)

scene.show()