#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
import tensorflow as tf
import yaml
import importlib
import shutil

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    OPTION = input("처음부터 학습 시작시 s 입력, 기존 모델 불러오기 시 l 입력 : ")
    if OPTION == "s":
        print("--- No model directory provided. Training from scratch ---")

        # set model dirs
        model_name = input("새 모델 이름 입력 (unet_checkpoints 폴더 안에 저장됨. 해당 폴더가 없으면 만들어짐.): ")
        model_dir = os.path.join(ROOT_DIR, os.path.join("unet_checkpoints", model_name))

        # load the model architecture
        architecture_source_path = input("모델 구조 소스코드 경로 입력 (unet_architecture 폴더 안에 있어야 함.): ")
        architecture_name = os.path.basename(architecture_source_path).replace(".py","")
        arch = importlib.import_module(f"unet_architecture.{architecture_name}")

        # Load HyperParameter saved in yml file
        with open(input("불러올 하이퍼파라미터 : "), 'r') as f:
            hyperparameters = yaml.safe_load(f)
        BC = hyperparameters["BC"]
        LR = hyperparameters["LR"]
        EPOCHS = hyperparameters["EPOCHS"]
        os.makedirs(os.path.join(model_dir, 'weights'))
        with open(os.path.join(model_dir, "hyperparameters.yml"), 'w') as f:
            yaml.safe_dump(hyperparameters, f)
        log_dir = os.path.join(model_dir, "logs")

        shutil.copyfile(architecture_source_path, os.path.join(model_dir, "unet_architecture.py"))
        print("Directory: ", model_dir, ". Created")

        init_epoch = 0
    #TODO: 나중에 시간 있을 때.
    # elif OPTION == "l":
    #     saved_model_dir = input("학습된 모델의 디렉토리 입력 : ")
    #     print(f"--- Continuing training from: {saved_model_dir} ---")
    #     log_dir = os.path.join(saved_model_dir, "logs")
    #     model_dir = saved_model_dir

    #     #Load HyperParameter saved in yml file
    #     with open(os.path.join(saved_model_dir, "hyperparameters.yml"), 'r') as f:
    #         hyperparameters = yaml.safe_load(f)
    #     BC = hyperparameters["BC"]
    #     LR = hyperparameters["LR"]
    #     EPOCHS = hyperparameters["EPOCHS"]
    #     init_epoch = int(input("시작 시의 Epoch 횟수 입력, -1 입력시 기존 Epoch 횟수에서 시작 : "))
    else:
        print("잘못된 값 입력, 프로그램 종료.")
        exit()

    
    #################
    # Load Datatset #
    #################
    
    train_dataset = arch.DataGenerator(input("Train hdf5 파일이 들어있는 폴더의 디렉토리 입력 : "))
    validation_dataset = arch.DataGenerator(input("Validation hdf5 파일이 들어있는 폴더의 디렉토리 입력 : "))

    ##############
    # Load Model #
    ##############
    # build model

    inputs, outputs = arch.build_unet_graph(hyperparameters)
    model:tf.keras.models.Model = arch.Unet(inputs, outputs)

    # LR_SCHEDULE 설정
    if "LR_SCHEDULE" in hyperparameters:
        LR_SCHEDULE = hyperparameters["LR_SCHEDULE"]
    else:
        LR_SCHEDULE = "InverseTimeDecay"

    if LR_SCHEDULE == "InverseTimeDecay":
        lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=LR,
            decay_steps=1,
            decay_rate=hyperparameters["DECAY"])
    elif LR_SCHEDULE == "CosineDecay":
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LR,
            decay_steps=1,
            alpha=hyperparameters["ALPHA"])
    elif LR_SCHEDULE == "ExponentialDecay":
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LR,
            decay_steps=1,
            decay_rate=hyperparameters["DECAY"])
    elif LR_SCHEDULE == "Constant":
        lr_scheduler = LR
    else:
        print("잘못된 하이퍼파라미터(LR_SCHEDULE)")
        exit()
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler))

    #TODO: 나중에 시간 있을 때.
    # if OPTION == "l":
    #     # load weights
    #     weight_filelist = glob.glob(os.path.join(saved_model_dir, "weights/*.h5"))
    #     weight_filelist.sort()
    #     epoch_list = [int(os.path.basename(weight_file).split('-')[0]) for weight_file in weight_filelist]
    #     epoch_list = np.array(epoch_list)
    #     if init_epoch == -1:
    #         weight_file = weight_filelist[-1]
    #         init_epoch = epoch_list[-1]
    #     else:
    #         idx = np.where(epoch_list == init_epoch)[0][0]
    #         weight_file = weight_filelist[idx]
    #         init_epoch = epoch_list[idx]
    #     model.load_weights(weight_file)
    #     print("Loaded weights from {}".format(weight_file))

    ####################
    # Prepare Training #
    ####################
    # callbacks
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights", "E{epoch:03d}-L{val_total_loss:.6f}-A{val_accuracy:.4f}-EA{val_effective_accuracy:.4f}.h5"),
        save_weights_only=False,
        monitor="val_total_loss",
        mode="min",
        save_best_only=hyperparameters["SAVE_BEST_ONLY"] if "SAVE_BEST_ONLY" in hyperparameters else False)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=0,
                                                          profile_batch=0,
                                                          write_graph=True,
                                                          write_images=False)
    
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[save_callback, tensorboard_callback],
        validation_data=validation_dataset,
        validation_freq=1,
        max_queue_size=100,
        initial_epoch=init_epoch)