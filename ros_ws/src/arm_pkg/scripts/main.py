#!/usr/bin/env python
import os
import sys

import trimesh.geometry
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import RobotMain, RobotMainResponse
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from arm_pkg.srv import MainUnet, MainUnetRequest, MainUnetResponse
from arm_pkg.srv import MainSgnet, MainSgnetRequest, MainSgnetResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
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
    pc_np = np.array(pc.data).reshape((-1, 3))
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")
    pilimage.fromarray(img_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Image.png"))
    service_req_unet_segmap = rospy.ServiceProxy("main_unet", MainUnet)
    resp2:MainUnetResponse = service_req_unet_segmap(image, depth)
    segmap_np: np.ndarray = bridge.imgmsg_to_cv2(resp2.segmap, "rgb8")
    pilimage.fromarray(segmap_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Segmap.png"))
    service_req_sgnet = rospy.ServiceProxy("main_sgnet", MainSgnet)
    resp3:MainSgnetResponse = service_req_sgnet(pc)
    pc_result_np = np.array(resp3.pointcloudresult.data).reshape((-1, 3))
    scores_np = np.array(resp3.scores.data).reshape((-1))
    approaches_np = np.array(resp3.approaches.data).reshape((-1, 3))

    print(f"{pc_result_np = }")
    print(f"{scores_np = }")
    print(f"{approaches_np = }")

    print(f"{pc_result_np.shape = }")
    print(f"{scores_np.shape = }")
    print(f"{approaches_np.shape = }")

    print(f"{np.any(np.isnan(pc_result_np)) = }")
    print(f"{np.any(np.isnan(scores_np)) = }")
    print(f"{np.any(np.isnan(approaches_np)) = }")

    THRESHOLD = np.quantile(scores_np, 0.5)
    selected_idx = np.where(scores_np >= THRESHOLD)[0]
    pc_thresholded = pc_result_np[selected_idx]
    scores_thresholded = scores_np[selected_idx]
    approaches_thresholded = approaches_np[selected_idx]

    print(f"{pc_thresholded = }")
    print(f"{scores_thresholded = }")
    print(f"{approaches_thresholded = }")

    segmap_np = segmap_np.reshape((-1, 3))

    # pc_np = np.nan_to_num(pc_np)
    # scene = trimesh.Scene()
    # scene.add_geometry(trimesh.PointCloud(pc_np, np.pad(segmap_np, ((0, 0), (0, 1)), mode='constant', constant_values=255)))
    # scene.show()

    pc_filtered = np.zeros((0, 3), dtype=np.float32)
    scores_filtered = np.zeros((0), dtype=np.float32)
    approaches_filtered = np.zeros((0, 3), dtype=np.float32)

    for point, score, approach in zip(pc_thresholded, scores_thresholded, approaches_thresholded):
        idx = np.where(np.all(pc_np == point, axis=1))[0][0]
        if segmap_np[idx, 2] == 255:
            pc_filtered = np.vstack((pc_filtered, point))
            scores_filtered = np.hstack((scores_filtered, score))
            approaches_filtered = np.vstack((approaches_filtered, approach))
    
    print(f"{pc_filtered = }")
    print(f"{scores_filtered = }")
    print(f"{approaches_filtered = }")

    # pc_np = np.nan_to_num(pc_np)
    # scene = trimesh.Scene()
    # scene.add_geometry(trimesh.PointCloud(pc_np, np.pad(img_np.reshape((-1, 3)), ((0, 0), (0, 1)), mode='constant', constant_values=255)))
    # for point, approach in zip(pc_filtered, approaches_filtered):
    #     rot_trans = trimesh.geometry.align_vectors([0, 0, 1], approach)
    #     cyl = creation.cylinder(0.001, 0.05, 3)
    #     cyl.apply_translation([0, 0, 0.025])
    #     cyl.apply_transform(rot_trans)
    #     cyl.apply_translation(point)
    #     scene.add_geometry(cyl)
    #     scene.add_geometry(creation.axis())
    # scene.show()    
    
    return RobotMainResponse()

rospy.init_node("main")
service_as_server = rospy.Service("robot_main_service", RobotMain, callback)
rospy.spin()