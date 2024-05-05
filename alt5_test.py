#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

from alt4_1_data_generator_Unet import DataGenerator
from alt4_2_unet import Unet, build_unet_graph

#HyperParameters
BC = 32

if __name__ == '__main__':
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    saved_model_dir = input("저장된 모델의 디렉토리 입력 : ")
    target_epoch = int(input("대상 Epoch 입력 : "))

    if not os.path.isdir(saved_model_dir):
        raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))
    log_dir = os.path.join(saved_model_dir, "logs")


    #################
    # Load Datatset #
    #################
    target_hdf5folder = input("Test hdf5 파일이 들어있는 폴더의 디렉토리 입력 : ")
    test_dataset = DataGenerator(target_hdf5folder)

    ##############
    # Load Model #
    ##############
    # build model

    inputs, outputs = build_unet_graph(BC)
    model = Unet(inputs, outputs)

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
    scene_number = int(input("hdf5scene 번호 입력 : "))
    RGBD_normalized, gt_segmap_onehot = test_dataset[scene_number]

    with h5py.File(os.path.join(target_hdf5folder, f"{scene_number}.hdf5"), "r") as f:
        original_image = np.array(f["colors"])

    output_tensor = model(RGBD_normalized)
    logit = tf.nn.softmax(output_tensor)
    gt_segmap=tf.squeeze(tf.argmax(gt_segmap_onehot, axis=-1), axis=0).numpy()
    pred_segmap=tf.squeeze(tf.argmax(logit, axis=-1), axis=0).numpy()
    
    convert_RGB = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255]], dtype=np.uint8)
    gt_segmap_RGB = convert_RGB[gt_segmap]
    pred_segmap_RGB = convert_RGB[pred_segmap]

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("original_image")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_segmap_RGB)
    plt.axis("off")
    plt.title("gt_segmap")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_segmap_RGB)
    plt.axis("off")
    plt.title("pred_segmap")

    plt.show()