#!/usr/bin/env python
import os
import sys

# import trimesh.geometry
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import RobotMain, RobotMainResponse
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from arm_pkg.srv import MainUnet, MainUnetRequest, MainUnetResponse
from arm_pkg.srv import MainSgnet, MainSgnetRequest, MainSgnetResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as pilimage
import numpy as np
import trimesh
from trimesh import creation
import rospy

pub = rospy.Publisher('chatter', Image, queue_size=2)
bridge = CvBridge()

def callback(req):
    rospy.wait_for_service("main_camera")
    service_req_image_depth = rospy.ServiceProxy("main_camera", MainCamera)
    resp1:MainCameraResponse = service_req_image_depth()
    image = resp1.img
    depth = resp1.depth
    pc = resp1.pointcloud
    pc_np = np.array(pc.data).reshape((-1, 3)) #(720*1280, 3)
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")
    pilimage.fromarray(img_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Image.png"))
    service_req_unet_segmap = rospy.ServiceProxy("main_unet", MainUnet)
    resp2:MainUnetResponse = service_req_unet_segmap(image, depth)
    segmap_np: np.ndarray = np.array(resp2.segmap.data) #(720*1280)
    # print(f"{segmap_np = }")
    pilimage.fromarray(np.array([[255, 0, 0],
                                 [0, 255, 0],
                                 [0, 0, 255]], dtype=np.uint8)[segmap_np.reshape((720, 1280))], "RGB").save(os.path.join(os.path.dirname(__file__), "Segmap.png"))
    service_req_sgnet = rospy.ServiceProxy("main_sgnet", MainSgnet)
    resp3:MainSgnetResponse = service_req_sgnet(pc)
    # print(f"{resp3 = }")
    pc_result_np = np.array(resp3.pointcloudresult.data).reshape((-1, 3))
    scores_np = np.array(resp3.scores.data).reshape((-1))
    approaches_np = np.array(resp3.approaches.data).reshape((-1, 3))

    point_segmap_dict = dict()
    for point, segmentation in zip(pc_np, segmap_np):
        point_segmap_dict[tuple(point)] = segmentation

    # print(f"{pc_result_np = }")
    # print(f"{scores_np = }")
    # print(f"{approaches_np = }")

    # print(f"{pc_result_np.shape = }")
    # print(f"{scores_np.shape = }")
    # print(f"{approaches_np.shape = }")

    # print(f"{np.any(np.isnan(pc_result_np)) = }")
    # print(f"{np.any(np.isnan(scores_np)) = }")
    # print(f"{np.any(np.isnan(approaches_np)) = }")

    THRESHOLD = np.quantile(scores_np, 0.5)
    selected_idx = np.where(scores_np >= THRESHOLD)[0]
    pc_thresholded = pc_result_np[selected_idx]
    scores_thresholded = scores_np[selected_idx]
    approaches_thresholded = approaches_np[selected_idx]

    # print(f"{pc_thresholded = }")
    # print(f"{scores_thresholded = }")
    # print(f"{approaches_thresholded = }")

    pc_filtered = np.zeros((0, 3), dtype=np.float32)
    scores_filtered = np.zeros((0), dtype=np.float32)
    approaches_filtered = np.zeros((0, 3), dtype=np.float32)

    for point, score, approach in zip(pc_thresholded, scores_thresholded, approaches_thresholded):
        if point_segmap_dict[tuple(point)] == 2:
            pc_filtered = np.vstack((pc_filtered, point))
            scores_filtered = np.hstack((scores_filtered, score))
            approaches_filtered = np.vstack((approaches_filtered, approach))
    
    # print(f"{pc_filtered = }")
    # print(f"{scores_filtered = }")
    # print(f"{approaches_filtered = }")

    pc_filtered_msg = Float32MultiArray()
    scores_filtered_msg = Float32MultiArray()
    approaches_filtered_msg = Float32MultiArray()

    pc_filtered_msg.data = pc_filtered.reshape((-1)).tolist()
    scores_filtered_msg.data = scores_filtered.reshape((-1)).tolist()
    approaches_filtered_msg.data = approaches_filtered.reshape((-1)).tolist()

    pc_np = np.nan_to_num(pc_np)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(pc_np, np.pad(img_np.reshape((-1, 3)), ((0, 0), (0, 1)), mode='constant', constant_values=255)))
    for point, approach in zip(pc_filtered, approaches_filtered):
        rot_trans = trimesh.geometry.align_vectors([0, 0, 1], approach)
        cyl = creation.cylinder(0.001, 0.05, 3)
        cyl.apply_translation([0, 0, 0.025])
        cyl.apply_transform(rot_trans)
        cyl.apply_translation(point)
        scene.add_geometry(cyl)
        scene.add_geometry(creation.axis())
    scene.show(line_settings={'point_size':0.05})    
    
    return RobotMainResponse(pc_filtered_msg,
                             scores_filtered_msg,
                             approaches_filtered_msg)

rospy.init_node("main")
service_as_server = rospy.Service("robot_main_service", RobotMain, callback)
rospy.spin()