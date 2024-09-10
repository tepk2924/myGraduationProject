import trimesh
import trimesh.creation
import numpy as np

scene = trimesh.Scene()

joint = np.array([0, 0, 0], dtype=np.float32)

base_link = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/base_link.stl")
base_link.visual.face_colors = [255, 0, 0, 255]

joint += np.array([0, 0, 0], dtype=np.float32)

link_1:trimesh.Trimesh = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/link_1.stl")
link_1.apply_translation(joint)
link_1.visual.face_colors = [0, 255, 0, 255]

joint += np.array([0, 0, 0.29])

link_2 = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/link_2.stl")
link_2.apply_translation(joint)
link_2.visual.face_colors = [0, 0, 255, 255]

joint += np.array([0, 0, 0.27])

link_3 = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/link_3.stl")
link_3.apply_translation(joint)
link_3.visual.face_colors = [255, 255, 0, 255]

joint += np.array([0, 0, 0.07])

link_4 = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/link_4.stl")
link_4.apply_translation(joint)
link_4.visual.face_colors = [0, 255, 255, 255]

joint += np.array([0.302, 0, 0])

link_5 = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/link_5.stl")
link_5.apply_translation(joint)
link_5.visual.face_colors = [255, 0, 255, 255]

joint += np.array([0.072, 0, 0])

link_6 = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/link_6.stl")
link_6.apply_translation(joint)
link_6.visual.face_colors = [255, 255, 255, 255]

rotate = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])

base:trimesh.Trimesh = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/epick_base.stl")
base.apply_scale(0.001)
base.apply_transform(rotate)
base.apply_translation(joint)
base.visual.face_colors = [127, 0, 0, 255]

body:trimesh.Trimesh = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/epick_body.stl")
body.apply_scale(0.001)
body.apply_transform(rotate)
body.apply_translation(joint)
body.visual.face_colors = [0, 127, 0, 255]

extend:trimesh.Trimesh = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/epick_extend.stl")
extend.apply_scale(0.001)
extend.apply_transform(rotate)
extend.apply_translation(joint)
extend.visual.face_colors = [0, 0, 127, 255]

suction_cup:trimesh.Trimesh = trimesh.load_mesh("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/drivers/irb120_description/meshes/collision/suction_cup.stl")
suction_cup.apply_scale(0.001)
suction_cup.apply_transform(rotate)
suction_cup.apply_translation(joint)
suction_cup.visual.face_colors = [127, 127, 0, 255]

scene.add_geometry(base_link)
scene.add_geometry(link_1)
scene.add_geometry(link_2)
scene.add_geometry(link_3)
scene.add_geometry(link_4)
scene.add_geometry(link_5)
scene.add_geometry(link_6)
scene.add_geometry(base)
scene.add_geometry(body)
scene.add_geometry(extend)
scene.add_geometry(suction_cup)
scene.add_geometry(trimesh.creation.axis())
scene.show()