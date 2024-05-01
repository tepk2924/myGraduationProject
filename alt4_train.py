#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import datetime

import numpy as np
import tensorflow as tf

from alt4_1_data_generator_Unet import DataGenerator
from alt4_2_unet import Unet, build_unet_graph

#HyperParameters
BC = 32
LR = 0.001
DECAY = 5.0e-05
EPOCHS = 20

if __name__ == '__main__':
    #tepk2924 조한 수정 : 모니터가 없는 컴퓨터에서 돌리기 위해 필요.
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    OPTION = input("처음부터 학습 시작시 s 입력, 기존 모델 불러오기 시 l 입력 : ")
    if OPTION == "s":
        print("--- No model directory provided. Training from scratch ---")
        # set model dirs
        cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(ROOT_DIR, os.path.join("checkpoints_Unet", cur_date))
        log_dir = os.path.join(model_dir, "logs")

        os.makedirs(os.path.join(model_dir, 'weights'))
        print("Directory: ", model_dir, ". Created")

        init_epoch = 0
    elif OPTION == "l":
        saved_model_dir = input("학습된 모델의 디렉토리 입력 : ")
        print(f"--- Continuing training from: {saved_model_dir} ---")
        log_dir = os.path.join(saved_model_dir, "logs")
        model_dir = saved_model_dir

        init_epoch = int(input("시작 시의 Epoch 횟수 입력, -1 입력시 기존 Epoch 횟수에서 시작 : "))
    else:
        print("잘못된 값 입력, 프로그램 종료.")
        exit()

    
    #################
    # Load Datatset #
    #################
    
    #tepk2924 조한 수정 : batch size와 raw_num_point를 자동으로 컨피그 파일에 맞추도록
    data_dir = os.path.join(ROOT_DIR, "data")
    train_dataset = DataGenerator(input("Train hdf5 파일이 들어있는 폴더의 디렉토리 입력 : "))
    validation_dataset = DataGenerator(input("Validation hdf5 파일이 들어있는 폴더의 디렉토리 입력 : "))

    ##############
    # Load Model #
    ##############
    # build model

    inputs, outputs = build_unet_graph(BC)
    model = Unet(inputs, outputs)

    # compile model
    lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=LR,
        decay_steps=1,
        decay_rate=DECAY)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler))

    if OPTION == "l":
        # load weights
        weight_filelist = glob.glob(os.path.join(saved_model_dir, "weights/*.h5"))
        weight_filelist.sort()
        epoch_list = [int(os.path.basename(weight_file).split('-')[0]) for weight_file in weight_filelist]
        epoch_list = np.array(epoch_list)
        if init_epoch == -1:
            weight_file = weight_filelist[-1]
            init_epoch = epoch_list[-1]
        else:
            idx = np.where(epoch_list == init_epoch)[0][0]
            weight_file = weight_filelist[idx]
            init_epoch = epoch_list[idx]
        model.load_weights(weight_file)
        print("Loaded weights from {}".format(weight_file))

    ####################
    # Prepare Training #
    ####################
    # callbacks
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights", "{epoch:03d}-{val_total_loss:.3f}.h5"),
        save_weights_only=False,
        monitor="val_total_loss",
        mode="min",
        save_best_only=False)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=0,
                                                          profile_batch=0,
                                                          write_graph=True,
                                                          write_images=False)


    # train model
    #TODO: 데이터셋 생성 + 트레이닝 개시
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[save_callback, tensorboard_callback],
        validation_data=validation_dataset,
        validation_freq=1,
        max_queue_size=100,
        initial_epoch=init_epoch)