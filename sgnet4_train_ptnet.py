#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob

import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.typeDict = np.sctypeDict
import tensorflow as tf
import yaml
import importlib
import shutil

if __name__ == '__main__':
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # allow memory growth

    OPTION = input("input 's' if you want start from scratch, 'l' if you want to continue (not implemented): ")
    if OPTION == "s":
        print("--- No model directory provided. Training from scratch ---")
        # set model dirs
        model_name = input("Input the name of the new model (will be stored in the folder 'sgnet_checkpoints'): ")
        model_dir = os.path.join(ROOT_DIR, os.path.join("sgnet_checkpoints", model_name))

        # load the model architecture
        architecture_source_path = input("Input the directory of the source file of model architecture (must be in 'sgnet_architecture' folder) : ")
        architecture_name = os.path.basename(architecture_source_path).replace(".py","")
        arch = importlib.import_module(f"sgnet_architecture.{architecture_name}")

        # Load HyperParameter saved in yml file
        with open(input("Input the directory of hyperparameter config file : "), 'r') as f:
            hyperparameters = yaml.safe_load(f)
        os.makedirs(os.path.join(model_dir, 'weights'))
        with open(os.path.join(model_dir, "hyperparameters.yml"), "w") as f:
            yaml.safe_dump(hyperparameters, f)
        log_dir = os.path.join(model_dir, "logs")

        shutil.copyfile(architecture_source_path, os.path.join(model_dir, "sgnet_architecture.py"))
        print("Directory: ", model_dir, ". Created")
        
        init_epoch = 0
    #TODO: Whenever I have a free time
    # elif OPTION == "l":
    #     saved_model_dir = input("학습된 모델의 디렉토리 입력 : ")
    #     print(f"--- Continuing training from: {saved_model_dir} ---")
    #     log_dir = os.path.join(saved_model_dir, "logs")
    #     model_dir = saved_model_dir

    #     # load train config from the saved model directory
    #     config = Config(os.path.join(saved_model_dir, 'config.yml'))
    #     train_config = config.load()
    #     init_epoch = int(input("시작 시의 Epoch 횟수 입력, -1 입력시 기존 Epoch 횟수에서 시작 : "))
    else:
        print("Invalid input. exiting program.")
        exit(1)

    
    #################
    # Load Datatset #
    #################
    
    #tepk2924 조한 수정 : batch size와 raw_num_point를 자동으로 컨피그 파일에 맞추도록
    data_dir = os.path.join(ROOT_DIR, "data")
    train_dataset = arch.DataGenerator(input("Train Scene이 들어있는 폴더의 디렉토리 입력 : "), hyperparameters)
    validation_dataset = arch.DataGenerator(input("Validation Scene이 들어있는 폴더의 디렉토리 입력 : "), hyperparameters)

    ##############
    # Load Model #
    ##############
    # build model
    inputs, outputs = arch.build_suction_pointnet_graph(hyperparameters)
    model:tf.keras.models.Model = arch.SuctionGraspNet(inputs, outputs)

    # compile model
    lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=hyperparameters["LR"],
        decay_steps=1,
        decay_rate=hyperparameters["DECAY"])
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler))

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
        filepath=os.path.join(model_dir, "weights", "{epoch:03d}-{val_total_loss:.3f}.h5"),
        save_weights_only=False,
        monitor="val_total_loss",
        mode="min",
        save_best_only=hyperparameters["SAVE_BEST_ONLY"])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=0,
                                                          profile_batch=0,
                                                          write_graph=True,
                                                          write_images=False)


    # train model

    #tepk2924 조한 : 여기 코드를 돌리는 중에 메모리 이슈가 발생하는 것으로 추정. => 최대한 작은 모델을 만들 것.
    #기존 config 파일에서 거의 모든 값들을 1/4 수준으로 낮추어서 학습 진행 성공.
    
    model.fit(
        train_dataset,
        epochs=hyperparameters["EPOCH"],
        callbacks=[save_callback, tensorboard_callback],
        validation_data=validation_dataset,
        validation_freq=1,
        max_queue_size=100,
        initial_epoch=init_epoch)