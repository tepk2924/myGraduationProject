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

if __name__ == '__main__':
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

    target_hdf5folder = input("Directory of the folder containiing test hdf5scene files : ")
    arch = importlib.import_module(f"unet_checkpoints.{saved_model_name}.unet_architecture")

    test_dataset = arch.DataGenerator(target_hdf5folder)

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


    scene_number = int(input("The # of hdf5scene : "))

    RGBDE_normalized, gt_segmap_onehot = test_dataset[scene_number]

    with h5py.File(os.path.join(target_hdf5folder, f"{scene_number}.hdf5"), "r") as f:
        original_image = np.array(f["colors"])
        original_depth = np.array(f["depth"])

    logit = model(RGBDE_normalized, training=False)
    pred_segmap_prob = tf.nn.softmax(logit)

    total_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(gt_segmap_onehot, pred_segmap_prob))

    gt = tf.argmax(gt_segmap_onehot, axis=-1)
    pred = tf.argmax(pred_segmap_prob, axis=-1)

    accuracy_tracker = tf.keras.metrics.Accuracy(name='accuracy')
    effective_accuracy_tracker = tf.keras.metrics.Accuracy(name='effective_accuracy')

    accuracy = accuracy_tracker(y_true=gt, y_pred=pred)

    mask = tf.where(gt >= 1, True, False)
    effective_accuracy = effective_accuracy_tracker(y_true=tf.boolean_mask(gt, mask), y_pred=tf.boolean_mask(pred, mask))

    gt_isvalid = tf.where(gt == 2, 1, 0)
    pred_isvalid = tf.where(pred == 2, 1, 0)

    recall_tracker = tf.keras.metrics.Recall(name='recall')
    precision_tracker = tf.keras.metrics.Precision(name='precision')

    recall = recall_tracker(gt_isvalid, pred_isvalid)
    precision = precision_tracker(gt_isvalid, pred_isvalid)

    gt_segmap=tf.squeeze(gt, axis=0).numpy()
    pred_segmap=tf.squeeze(pred, axis=0).numpy()
    
    convert_RGB = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255]], dtype=np.uint8)
    gt_segmap_RGB = convert_RGB[gt_segmap]
    pred_segmap_RGB = convert_RGB[pred_segmap]

    depth_image = np.tile(np.expand_dims(np.uint8(255*(original_depth - np.min(original_depth))/(np.max(original_depth) - np.min(original_depth))), axis=-1), (1, 1, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("original_image")

    plt.subplot(1, 4, 2)
    plt.imshow(depth_image)
    plt.axis("off")
    plt.title("original_depth")

    plt.subplot(1, 4, 3)
    plt.imshow(gt_segmap_RGB)
    plt.axis("off")
    plt.title("gt_segmap")

    plt.subplot(1, 4, 4)
    plt.imshow(pred_segmap_RGB)
    plt.axis("off")
    plt.title("pred_segmap")

    plt.figtext(0.5, 0.01, f"{os.path.basename(saved_model_dir)}\n{target_epoch = }\n{scene_number = }\n{total_loss = :.6f}\n{accuracy = :.4f}\n{effective_accuracy = :.4f}\n{recall = :.4f}\n{precision = :.4f}", ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(saved_model_dir, f"{os.path.basename(saved_model_dir)}_{target_epoch}Ep_datanum{scene_number}.png"), dpi=300)