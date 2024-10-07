#!/usr/bin/env python
import rospy
import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.typeDict = np.sctypeDict
import importlib
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import yaml
import glob
from std_msgs.msg import Float32MultiArray
from scipy import signal
from arm_pkg.srv import MainSgnet, MainSgnetResponse, MainSgnetRequest

def init():

    #Solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')

    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        
    global model
    global hyperparameters
    #Parameters
    model_name = rospy.get_param("sgnet_model_name")
    target_epoch = rospy.get_param("sgnet_target_epoch")

    #Initializations
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    sys.path.append(project_dir)
    arch = importlib.import_module(f"sgnet_checkpoints.{model_name}.sgnet_architecture")
    saved_model_dir = os.path.join(os.path.join(project_dir, "sgnet_checkpoints"), model_name)
    with open(os.path.join(saved_model_dir, "hyperparameters.yml"), 'r') as f:
        hyperparameters = yaml.safe_load(f)
    inputs, outputs = arch.build_suction_pointnet_graph(hyperparameters)
    model = arch.SuctionGraspNet(inputs, outputs)
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

def callback(req:MainSgnetRequest):
    point_cloud_np = np.array(req.pointcloud.data).reshape((-1, 3))
    nan_mask = np.where(np.any(np.isnan(point_cloud_np), axis=-1) == False)[0]
    point_cloud_np = point_cloud_np[nan_mask]
    selected_point_idxs = np.random.choice(range(point_cloud_np.shape[0]),
                                           hyperparameters["RAW_NUM_POINTS"],
                                           False)
    
    selected_point_cloud = point_cloud_np[selected_point_idxs]
    selected_point_cloud = np.expand_dims(selected_point_cloud, axis=0)
    pc_tensor = tf.convert_to_tensor(selected_point_cloud, dtype=tf.float32)
    output_pc: tf.Tensor
    score_output: tf.Tensor
    approach_output: tf.Tensor
    with tf.device('/device:GPU:1'):
        output_pc, score_output, approach_output = model(pc_tensor, training=False)

    pc_np:np.ndarray = output_pc[0].numpy()
    score_np:np.ndarray = score_output[0].numpy()
    approach_np:np.ndarray = approach_output[0].numpy()

    pc_msg = Float32MultiArray()
    score_msg = Float32MultiArray()
    approach_msg = Float32MultiArray()

    pc_msg.data = pc_np.reshape((-1)).tolist()
    score_msg.data = score_np.reshape((-1)).tolist()
    approach_msg.data = approach_np.reshape((-1)).tolist()

    return MainSgnetResponse(pc_msg, score_msg, approach_msg)

if __name__ == "__main__":
    init()
    rospy.init_node("sgnet")
    service = rospy.Service("main_sgnet", MainSgnet, callback)
    rospy.spin()