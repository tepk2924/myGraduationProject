#!/usr/bin/env python
import rospy
import cv2
import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.typeDict = np.sctypeDict
import tensorflow as tf
import importlib
import sys
import os
import yaml
import glob
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from scipy import signal
from arm_pkg.srv import main_unet, main_unetResponse

bridge = CvBridge()
model_name = rospy.get_param("unet_model_name")
target_epoch = rospy.get_param("target_epoch")
image_height = rospy.get_param("image_height")
image_width = rospy.get_param("image_width")
RGBDE_normalized = np.ndarray((image_height, image_width, 5), dtype=float)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(project_dir)
arch = importlib.import_module(f"unet_checkpoints.{model_name}.unet_architecture")
saved_model_dir = os.path.join(os.path.join(project_dir, "unet_checkpoints"), model_name)
with open(os.path.join(saved_model_dir, "hyperparameters.yml"), 'r') as f:
    hyperparameters = yaml.safe_load(f)
inputs, outputs = arch.build_unet_graph(hyperparameters)
model = arch.Unet(inputs, outputs)
weight_filelist = glob.glob(os.path.join(saved_model_dir, "weights/*.h5"))
weight_filelist.sort()
epoch_list = [int(os.path.basename(weight_file).split('-')[0]) for weight_file in weight_filelist]
epoch_list = np.array(epoch_list)
if target_epoch == -1:
    weight_file = weight_filelist[-1]
    target_epoch = epoch_list[-1]
else:
    idx = np.where(epoch_list == target_epoch)[0][0]
    weight_file = weight_filelist[idx]
    target_epoch = epoch_list[idx]
model.load_weights(weight_file)

def callback(req):
    cv_image:np.ndarray = bridge.imgmsg_to_cv2(req.img, "rgb8")
    laplacian = np.array([[1, 4, 1],
                            [4,-20, 4],
                            [1, 4, 1]])
    RGBDE_normalized[:, :, :3] = cv_image/255.
    kernaled = (signal.convolve2d(cv_image[:, :, 0], laplacian, mode="same", boundary="symm")**2 + 
                signal.convolve2d(cv_image[:, :, 1], laplacian, mode="same", boundary="symm")**2 + 
                signal.convolve2d(cv_image[:, :, 2], laplacian, mode="same", boundary="symm")**2)
    RGBDE_normalized[:, :, 4] = (kernaled - np.mean(kernaled))/np.std(kernaled)
    depth_np = np.array(req.depth, dtype=float).reshape((image_height, image_width))
    RGBDE_normalized[:, :, 3] = (depth_np - np.mean(depth_np))/np.std(depth_np)
    logit = model(tf.expand_dims(tf.convert_to_tensor(RGBDE_normalized, dtype=tf.float32), axis=0), training=False)
    pred_segmap_prob = tf.nn.softmax(logit)
    pred = tf.argmax(pred_segmap_prob, axis=-1)
    pred_segmap=tf.squeeze(pred, axis=0).numpy()

    convert_RGB = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255]], dtype=np.uint8)

    pred_segmap_RGB = convert_RGB[pred_segmap]
    segmap_msg = bridge.cv2_to_imgmsg(pred_segmap_RGB, "rgb8")
    return main_unetResponse(segmap_msg)

rospy.init_node("unet")
service = rospy.Service("/main_unet", main_unet, callback)
rospy.spin()