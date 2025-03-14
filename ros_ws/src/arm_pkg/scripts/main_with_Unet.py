#!/usr/bin/env python
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from arm_pkg.srv import MainRobot, MainRobotRequest, MainRobotResponse
from arm_pkg.srv import MainCamera, MainCameraRequest, MainCameraResponse
from arm_pkg.srv import MainUnet, MainUnetRequest, MainUnetResponse
from arm_pkg.srv import MainSgnet, MainSgnetRequest, MainSgnetResponse
from arm_pkg.srv import Execution, ExecutionRequest, ExecutionResponse
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as pilimage
import numpy as np
import rospy
import ros_numpy

def init():
    #Publisher initialization
    global pointcloud_publisher
    global grasps_pub
    pointcloud_publisher = rospy.Publisher('/pointcloud', PointCloud2, queue_size=10)

    #To convert ROS msg from np.ndarray used by cv2, this code is needed.
    global bridge
    bridge = CvBridge()

    #Reads the last 4 lines of camera_tf.txt and take them as transform matrix of camera relative to the world(==base_link) frame
    with open(os.path.join(os.path.dirname(__file__), "camera_tf.txt"), "r") as f:
        lines = f.readlines()[-4:]
    global camera_tf
    camera_tf = np.array([list(map(float, line.split())) for line in lines])
    
    global IMAGE_HEIGHT
    global IMAGE_WIDTH
    IMAGE_HEIGHT = rospy.get_param("image_height")
    IMAGE_WIDTH = rospy.get_param("image_width")

def callback(req: ExecutionRequest):
    #Getting color image & depth & pointcloud from camera node
    rospy.wait_for_service("main_camera")
    service_req_image_depth = rospy.ServiceProxy("main_camera", MainCamera)
    resp1:MainCameraResponse = service_req_image_depth()
    image = resp1.img
    depth = resp1.depth
    np.save(os.path.join(os.path.dirname(__file__), "Depth.npy"), np.array(depth.data).reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
    pc = resp1.pointcloud

    #From msg to np.ndarray
    pc_np = np.array(pc.data).reshape((-1, 3)) #(720*1280, 3)
    pc_world = np.ones((4, pc_np.shape[0]), np.float32)
    pc_world[:3, :] = pc_np.T
    pc_world = camera_tf @ pc_world
    img_np = bridge.imgmsg_to_cv2(image, "rgb8")

    #Convert np.ndarray to PointCloud2 message, and publish
    r = img_np[:, :, 0].reshape((-1))
    g = img_np[:, :, 1].reshape((-1))
    b = img_np[:, :, 2].reshape((-1))
    xyz = np.array(pc_world.T[:, :3]).reshape((-1, 3))
    x = xyz[:, 0].reshape((-1))
    y = xyz[:, 1].reshape((-1))
    z = xyz[:, 2].reshape((-1))

    data = np.zeros(len(r), dtype=[('x', np.float32),
                                   ('y', np.float32),
                                   ('z', np.float32),
                                   ('r', np.uint8),
                                   ('g', np.uint8),
                                   ('b', np.uint8)])

    data['x'] = x
    data['y'] = y
    data['z'] = z
    data['r'] = r
    data['g'] = g
    data['b'] = b

    pointcloud_xyzrgb_msg = ros_numpy.point_cloud2.array_to_pointcloud2(data,
                                                                        None,
                                                                        "base_link")
    pointcloud_publisher.publish(pointcloud_xyzrgb_msg)

    #Save the color RGB image as PNG file
    pilimage.fromarray(img_np, "RGB").save(os.path.join(os.path.dirname(__file__), "Image.png"))

    #Getting unet segmep from unet node
    service_req_unet_segmap = rospy.ServiceProxy("main_unet", MainUnet)
    resp2:MainUnetResponse = service_req_unet_segmap(image, depth)
    segmap_np:np.ndarray = np.array(resp2.segmap.data) #(720*1280)

    #Save segmap as PNG file
    pilimage.fromarray(np.array([[255, 0, 0],
                                 [0, 255, 0],
                                 [0, 0, 255]], dtype=np.uint8)[segmap_np.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))], "RGB").save(os.path.join(os.path.dirname(__file__), "Segmap.png"))
    
    #Getting result from sgnet node
    service_req_sgnet = rospy.ServiceProxy("main_sgnet", MainSgnet)
    resp3:MainSgnetResponse = service_req_sgnet(pc)

    #Converting sgnet result msg to np.ndarray
    pc_result_np = np.array(resp3.pointcloudresult.data).reshape((-1, 3))
    scores_np = np.array(resp3.scores.data).reshape((-1))
    approaches_np = np.array(resp3.approaches.data).reshape((-1, 3))

    #Some processing, filtering grasps with higher scores
    point_segmap_dict = dict()
    for point, segmentation in zip(pc_np, segmap_np):
        point_segmap_dict[tuple(point)] = segmentation

    theshold = np.quantile(scores_np, 0.5)
    selected_idx = np.where(scores_np >= theshold)[0]
    pc_thresholded = pc_result_np[selected_idx]
    scores_thresholded = scores_np[selected_idx]
    approaches_thresholded = approaches_np[selected_idx]

    pc_filtered = np.zeros((0, 3), dtype=np.float32)
    scores_filtered = np.zeros((0), dtype=np.float32)
    approaches_filtered = np.zeros((0, 3), dtype=np.float32)

    #Filtering grasps using segmap created by my Unet.
    for point, score, approach in zip(pc_thresholded, scores_thresholded, approaches_thresholded):
        #If the point is considered as valid (index 2)
        if point_segmap_dict[tuple(point)] == 2:
            pc_filtered = np.vstack((pc_filtered, point))
            scores_filtered = np.hstack((scores_filtered, score))
            approaches_filtered = np.vstack((approaches_filtered, approach))

    #Converting np.ndarray sgnet node result from camera frame to world frame
    pc_filtered_world = ((camera_tf[:3, :3] @ pc_filtered.T) + camera_tf[:3, 3:4]).T
    approaches_filtered_world = (camera_tf[:3, :3] @ approaches_filtered.T).T

    #Creating & Returning service for robot node
    pc_filtered_msg = Float32MultiArray()
    scores_filtered_msg = Float32MultiArray()
    approaches_filtered_msg = Float32MultiArray()

    pc_filtered_msg.data = pc_filtered_world.reshape((-1)).tolist()
    scores_filtered_msg.data = scores_filtered.reshape((-1)).tolist()
    approaches_filtered_msg.data = approaches_filtered_world.reshape((-1)).tolist()
    
    service_req_robot_move = rospy.ServiceProxy("main_robot", MainRobot)
    resp4:MainRobotResponse = service_req_robot_move(pc_filtered_msg,
                                                     scores_filtered_msg,
                                                     approaches_filtered_msg)

    return ExecutionResponse()

if __name__ == "__main__":
    init()
    rospy.init_node("main_with_Unet")
    service_as_server = rospy.Service("execution", Execution, callback)
    rospy.spin()