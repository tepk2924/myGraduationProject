#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import yaml
import importlib

import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.typeDict = np.sctypeDict
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import trimesh
from trimesh import creation

if __name__ == '__main__':
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True) # allow memory growth

    saved_model_dir = input("Directory of the folder containing saved model (must be in the folder 'sgnet_checkpoints'): ")
    saved_model_name = os.path.basename(saved_model_dir)
    target_epoch = int(input("Target epoch : "))

    if not os.path.isdir(saved_model_dir):
        raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))
    log_dir = os.path.join(saved_model_dir, "logs")

    target_hdf5folder = input("Directory of the folder containiing test hdf5scene files : ")
    arch = importlib.import_module(f"sgnet_checkpoints.{saved_model_name}.sgnet_architecture")

    with open(os.path.join(saved_model_dir, "hyperparameters.yml"), 'r') as f:
        hyperparameters = yaml.safe_load(f)

    test_dataset = arch.DataGenerator(target_hdf5folder, hyperparameters)

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
    print("Loaded weights from {}".format(weight_file))

    ###########
    # predict #
    ###########
    # evaluate the model


    scene_number = int(input("The # of hdf5scene : "))

    with h5py.File(os.path.join(target_hdf5folder, f"{scene_number}.hdf5"), "r") as f:
        point_cloud = np.array(f["pc"])
        colors = np.array(f["colors"]) #(image_height, image_width, 3) np.uint8
        extrinsic = np.array(f["extrinsic"])
    
    camera_inverse = np.linalg.inv(extrinsic)
    point_cloud_1padded = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float32)), axis=-1).T
    point_cloud_reformatted = ((camera_inverse@point_cloud_1padded).T)[:, :3]
    color_r = np.concatenate((colors.reshape((-1, 3)), 255*np.ones((colors.shape[0]*colors.shape[1], 1), dtype=np.uint8)), axis=-1)
    camera_inverse = np.linalg.inv(extrinsic)
    point_cloud_1padded = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float32)), axis=-1).T
    point_cloud_camera_frame = ((camera_inverse@point_cloud_1padded).T)[:, :3] #Full point clound at camera frame, (image_height*image_width, 3)

    selected_point_idxs = np.random.choice(range(point_cloud_camera_frame.shape[0]),
                                            hyperparameters["RAW_NUM_POINTS"],
                                            False)
    
    selected_point_cloud = point_cloud_camera_frame[selected_point_idxs]
    selected_point_cloud = np.expand_dims(selected_point_cloud, axis=0)
    pc_tensor = tf.convert_to_tensor(selected_point_cloud, dtype=tf.float32)
    output_pc: tf.Tensor
    score_output: tf.Tensor
    approach_output: tf.Tensor
    output_pc, score_output, approach_output = model(pc_tensor, training=False)

    scene = trimesh.Scene()
    pc_np = output_pc[0].numpy()
    score_np = score_output[0].numpy()
    approach_np = approach_output[0].numpy()

    scene.add_geometry(trimesh.PointCloud(point_cloud_reformatted, color_r))

    #Print min, max, each quartile value scores
    print(f"{np.min(score_np) = }")
    print(f"{np.quantile(score_np, 0.25) = }")
    print(f"{np.quantile(score_np, 0.5) = }")
    print(f"{np.quantile(score_np, 0.75) = }")
    print(f"{np.max(score_np) = }")

    #match the value of 10% quantile to 0, 90% quantile to 1, trim the values onto [0, 1]
    per10 = np.quantile(score_np, .1)
    per90 = np.quantile(score_np, .9)

    grasp_score_nor = (score_np - per10) / (per90 - per10)
    grasp_score_nor = np.where(grasp_score_nor > 1, 1, np.where(grasp_score_nor < 0, 0, grasp_score_nor))

    for score, approach, point in zip(grasp_score_nor, approach_np, pc_np):
        rot_trans = trimesh.geometry.align_vectors([0, 0, 1], approach)
        cyl = creation.cylinder(0.001, 0.05, 3)
        cyl.apply_translation([0, 0, 0.025])
        cyl.apply_transform(rot_trans)
        cyl.apply_translation(point)
        cyl.visual.face_colors = [int(255*score), 0, 255 - int(255*score), 255]
        scene.add_geometry(cyl)
    scene.show()