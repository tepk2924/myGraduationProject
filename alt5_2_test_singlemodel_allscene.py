#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import yaml
import importlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

if __name__ == '__main__':
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    saved_model_dir = input("저장된 모델의 디렉토리 입력 (unet_checkpoints 폴더 안에 있어야 함.): ")
    saved_model_name = os.path.basename(saved_model_dir)
    target_epoch = int(input("대상 Epoch 입력 : "))

    if not os.path.isdir(saved_model_dir):
        raise ValueError('Model directory does not exist: {}'.format(saved_model_dir))
    log_dir = os.path.join(saved_model_dir, "logs")

    target_hdf5folder = input("Test hdf5 파일이 들어있는 폴더의 디렉토리 입력 : ")
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

    scene_numbers = [*range(len(target_hdf5folder))]
    results = []
    for scene_number in scene_numbers:

        RGBDE_normalized, gt_segmap_onehot = test_dataset[scene_number]

        with h5py.File(os.path.join(target_hdf5folder, f"{scene_number}.hdf5"), "r") as f:
            original_image = np.array(f["colors"])

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

        results.append((scene_number, float(total_loss), float(accuracy), float(effective_accuracy), float(recall), float(precision)))
    results.sort(key = lambda x:x[1], reverse=True)
    with open(os.path.join(saved_model_dir, f"test_result_{saved_model_name}.txt"), "w") as f:
        f.write(f"result of {saved_model_name}\n")
        f.write("#  loss   acc    ef_acc recall precis\n")
        for r in results:
            f.write(f"{r[0]:02d} {r[1]:.04f} {r[2]:.04f} {r[3]:.04f} {r[4]:.04f} {r[5]:.04f}\n")