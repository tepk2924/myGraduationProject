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
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    plt.figure(figsize=(10, 6))
    number_of_models = int(input("The number of models : "))
    list_of_models_dir = [input(f"Directory of {i + 1}th model (must be in 'unet_checkpoints' folder): ") for i in range(number_of_models)]
    target_hdf5folder = input("Directory of the folder containiing test hdf5scene files : ")
    scene_number = int(input("The # of hdf5scene : "))
    plt.subplot(1, 3 + number_of_models, 1)
    with h5py.File(os.path.join(target_hdf5folder, f"{scene_number}.hdf5"), "r") as f:
        original_image = np.array(f["colors"])
        original_depth = np.array(f["depth"])
        original_gt_segmap = np.array(f["category_id_segmaps"])
    convert_RGB = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255]], dtype=np.uint8)
    gt_segmap_RGB = convert_RGB[original_gt_segmap]
    depth_image = np.tile(np.expand_dims(np.uint8(255*(original_depth - np.min(original_depth))/(np.max(original_depth) - np.min(original_depth))), axis=-1), (1, 1, 3))
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("original_image")

    plt.subplot(1, 3 + number_of_models, 2)
    plt.imshow(depth_image)
    plt.axis("off")
    plt.title("original_depth")

    plt.subplot(1, 3 + number_of_models, 3)
    plt.imshow(gt_segmap_RGB)
    plt.axis("off")
    plt.title("gt_segmap")
    saved_model_names = []
    target_epochs = []
    total_losses = []
    accuracies = []
    effective_accuracies = []
    recalls = []
    precisions = []
    for i, saved_model_dir in enumerate(list_of_models_dir, start=1):
        saved_model_name = os.path.basename(saved_model_dir)
        saved_model_names.append(saved_model_name)
        target_epoch = int(input(f"Target epoch of {i}th model : "))
        if not os.path.isdir(saved_model_dir):
            raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))
        log_dir = os.path.join(saved_model_dir, "logs")

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
        target_epochs.append(target_epoch)
        print("Loaded weights from {}".format(weight_file))

        ###########
        # predict #
        ###########
        # evaluate the model

        RGBDE_normalized, gt_segmap_onehot = test_dataset[scene_number]


        logit = model(RGBDE_normalized, training=False)
        pred_segmap_prob = tf.nn.softmax(logit)

        total_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(gt_segmap_onehot, pred_segmap_prob))
        total_losses.append(round(float(total_loss), 6))

        gt = tf.argmax(gt_segmap_onehot, axis=-1)
        pred = tf.argmax(pred_segmap_prob, axis=-1)

        accuracy_tracker = tf.keras.metrics.Accuracy(name='accuracy')
        effective_accuracy_tracker = tf.keras.metrics.Accuracy(name='effective_accuracy')

        accuracy = accuracy_tracker(y_true=gt, y_pred=pred)
        accuracies.append(round(float(accuracy), 4))

        mask = tf.where(gt >= 1, True, False)
        effective_accuracy = effective_accuracy_tracker(y_true=tf.boolean_mask(gt, mask), y_pred=tf.boolean_mask(pred, mask))
        effective_accuracies.append(round(float(effective_accuracy), 4))
        pred_segmap=tf.squeeze(pred, axis=0).numpy()
    
        pred_segmap_RGB = convert_RGB[pred_segmap]

        recall_tracker = tf.keras.metrics.Recall(name='recall')
        precision_tracker = tf.keras.metrics.Precision(name='precision')

        gt_isvalid = tf.where(gt == 2, 1, 0)
        pred_isvalid = tf.where(pred == 2, 1, 0)

        recall = recall_tracker(y_true=gt_isvalid, y_pred=pred_isvalid)
        precision = precision_tracker(y_true=gt_isvalid, y_pred=pred_isvalid)
        recalls.append(round(float(recall), 4))
        precisions.append(round(float(precision), 4))

        plt.subplot(1, 3 + number_of_models, 3 + i)
        plt.imshow(pred_segmap_RGB)
        plt.axis("off")
        plt.title(saved_model_name)
        
    plt.figtext(0.5, 0.01, f"{saved_model_names = }\n{target_epochs = }\n{scene_number = }\n{total_losses = }\n{accuracies = }\n{effective_accuracies = }\n{recalls = }\n{precisions = }", ha='center', va='bottom', fontsize=12)
    plt.savefig(os.path.join(os.path.dirname(list_of_models_dir[0]), f"multi_test_scene{scene_number:02d}.png"), dpi=300)