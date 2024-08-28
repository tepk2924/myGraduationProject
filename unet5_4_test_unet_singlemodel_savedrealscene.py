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
from PIL import Image
from scipy import signal

if __name__ == '__main__':

    IMAGE_HEIGHT = 720
    IMAGE_WIDTH = 1280

    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')

    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True) # allow memory growth

    saved_model_dir = input("Directory of the folder containing saved model (must be in the folder 'unet_checkpoints'): ")
    saved_model_name = os.path.basename(saved_model_dir)
    target_epoch = int(input("Target epoch : "))

    if not os.path.isdir(saved_model_dir):
        raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))
    log_dir = os.path.join(saved_model_dir, "logs")

    arch = importlib.import_module(f"unet_checkpoints.{saved_model_name}.unet_architecture")

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
    print("Loaded weights from {}".format(weight_file))

    ###########
    # predict #
    ###########
    # evaluate the model

    image_folder = "/home/riseabb/johan_ws/build_dataset_chohan/Images_Annotations_All"
    depth_folder = "/home/riseabb/johan_ws/build_dataset_chohan/Depths"
    segmap_save_folder = os.path.join("/home/riseabb/johan_ws/build_dataset_chohan/Pred_segmaps", saved_model_name)
    if not os.path.isdir(segmap_save_folder):
        os.makedirs(segmap_save_folder)

    for sceneidx in range(1, 301):
        RGBDE_normalized = np.ndarray((IMAGE_HEIGHT, IMAGE_WIDTH, 5), dtype=float)

        original_image = np.array(Image.open(os.path.join(image_folder, f"Image_{sceneidx:03d}.png")))
        original_depth = np.load(os.path.join(depth_folder, f"Depth_{sceneidx:03d}.npy"))

        laplacian = np.array([[1, 4, 1],
                              [4, -20, 4],
                              [1, 4, 1]])
        
        RGBDE_normalized[:, :, :3] = original_image/255.
        kernaled = (signal.convolve2d(original_image[:, :, 0], laplacian, mode="same", boundary="symm")**2 + 
                    signal.convolve2d(original_image[:, :, 1], laplacian, mode="same", boundary="symm")**2 + 
                    signal.convolve2d(original_image[:, :, 2], laplacian, mode="same", boundary="symm")**2)
        RGBDE_normalized[:, :, 4] = (kernaled - np.mean(kernaled))/np.std(kernaled)
        RGBDE_normalized[:, :, 3] = (original_depth - np.mean(original_depth))/np.std(original_depth)
        
        logit = model(tf.expand_dims(tf.convert_to_tensor(RGBDE_normalized, dtype=tf.float32), axis=0), training=False)
        pred_segmap_prob = tf.nn.softmax(logit)
        pred = tf.argmax(pred_segmap_prob, axis=-1)
        pred_segmap:np.ndarray = tf.squeeze(pred, axis=0).numpy().astype(np.uint8)

        palette = [255, 0, 0, #Red: Background
                0, 255, 0, #Green: Invalid
                0, 0, 255] #Blue: Valid
        
        imag = Image.fromarray(pred_segmap)
        imag.putpalette(palette)
        imag.save(os.path.join(segmap_save_folder, f"Segmap_{sceneidx:03d}.png"))
        print(f"scene {sceneidx:03d} done")