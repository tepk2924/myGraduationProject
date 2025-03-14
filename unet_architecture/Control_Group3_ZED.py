#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import numpy as np
import tensorflow as tf

from scipy import signal

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_unet_graph(hyperparameters:dict):
    """
    Build computational graph for Suction-GraspNet

    --------------
    Args:
        hyperparameters : a dictionary containing some hyperparameters
    --------------
    Returns:
        input_tensor tf.Tensor : input data with 5 channels (batch_num, image_height: 720, image_width: 1280, 5).
        output_tensor: tf.Tensor : output_tensor to be compared to gt (batch_num, image_height: 720, image_width: 1280, 3)
    """
    BC = hyperparameters["BC"]
    DROP_RATE = hyperparameters["DROP_RATE"] if "DROP_RATE" in hyperparameters else 0
    input_tensor = tf.keras.Input((720, 1280, 5), batch_size=1) #(B, 720, 1280, 5)
    relu = tf.keras.layers.ReLU()
    drop = tf.keras.layers.Dropout(rate=DROP_RATE)
    layer00 = tf.pad(input_tensor, ((0, 0), (94, 94), (94, 94), (0, 0)), mode="SYMMETRIC") #(B, 908, 1468, 5)
    layer01 = drop(relu(tf.keras.layers.Conv2D(filters=1*BC, kernel_size=(3, 3))(layer00))) #(B, 906, 1466, BC)
    layer02 = drop(relu(tf.keras.layers.Conv2D(filters=1*BC, kernel_size=(3, 3))(layer01))) #(B, 904, 1464, BC)
    layer10 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer02) #(B, 452, 732, BC)
    layer11 = drop(relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer10))) #(B, 450, 730, 2BC)
    layer12 = drop(relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer11))) #(B, 448, 728, 2BC)
    layer20 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer12) #(B, 224, 364, 2BC)
    layer21 = drop(relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer20))) #(B, 222, 362, 4BC)
    layer22 = drop(relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer21))) #(B, 220, 360, 4BC)
    layer30 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer22) #(B, 110, 180, 4BC)
    layer31 = drop(relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer30))) #(B, 108, 178, 8BC)
    layer32 = drop(relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer31))) #(B, 106, 176, 8BC)
    layer40 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer32) #(B, 53, 88, 8BC)
    layer41 = drop(relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer40))) #(B, 51, 86, 16BC)
    layer42 = drop(relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer41))) #(B, 49, 84, 16BC)
    layer32_cropped = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(layer32) #(B, 98, 168, 8BC)
    layer42_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer42) #(B, 98, 168, 16BC)
    layer42_conved22 = tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(2, 2), padding="same")(layer42_upsample) #(B, 98, 168, 8BC)
    layer50 = tf.keras.layers.Concatenate(axis=-1)((layer32_cropped, layer42_conved22)) #(B, 98, 168, 16BC)
    layer51 = drop(relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer50))) #(B, 96, 166, 8BC)
    layer52 = drop(relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer51))) #(B, 94, 164, 8BC)
    layer22_cropped = tf.keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(layer22) #(B, 188, 328, 4BC)
    layer52_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer52) #(B, 188, 328, 8BC)
    layer52_conved22 = tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(2, 2), padding="same")(layer52_upsample) #(B, 188, 328, 4BC)
    layer60 = tf.keras.layers.Concatenate(axis=-1)((layer22_cropped, layer52_conved22)) #(B, 188, 328, 8BC)
    layer61 = drop(relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer60))) #(B, 186, 326, 4BC)
    layer62 = drop(relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer61))) #(B, 184, 324, 4BC)
    layer12_cropped = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(layer12) #(B, 368, 648, 2BC)
    layer62_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer62) #(B, 368, 648, 4BC)
    layer62_conved22 = tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(2, 2), padding="same")(layer62_upsample) #(B, 368, 648, 2BC)
    layer70 = tf.keras.layers.Concatenate(axis=-1)((layer12_cropped, layer62_conved22)) #(B, 368, 648, 4BC)
    layer71 = drop(relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer70))) #(B, 366, 646, 2BC)
    layer72 = drop(relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer71))) #(B, 364, 644, 2BC)
    layer02_cropped = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(layer02) #(B, 728, 1288, BC)
    layer72_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer72) #(B, 728, 1288, 2BC)
    layer72_conved22 = tf.keras.layers.Conv2D(filters=1*BC, kernel_size=(2, 2), padding="same")(layer72_upsample) #(B, 728, 1288, BC)
    layer80 = tf.keras.layers.Concatenate(axis=-1)((layer02_cropped, layer72_conved22)) #(B, 728, 1288, 2BC)
    layer81 = drop(relu(tf.keras.layers.Conv2D(filters=1*BC, kernel_size=(5, 5))(layer80))) #(B, 724, 1284, BC)
    layer82 = drop(relu(tf.keras.layers.Conv2D(filters=1*BC, kernel_size=(5, 5))(layer81))) #(B, 720, 1280, BC)
    output_tensor = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1))(layer82) #(B, 720, 1280, 3)

    return input_tensor, output_tensor #(B, 720, 1280, 5), (B, 720, 1280, 3)

class Unet(tf.keras.models.Model):
    def __init__(self, inputs, outputs):
        super(Unet, self).__init__(inputs, outputs)

    def compile(self, optimizer='adam', run_eagerly=None):
        super(Unet, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)

        # define trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.accuracy_tracker = tf.keras.metrics.Accuracy(name='accuracy')
        self.effective_accuracy_tracker = tf.keras.metrics.Accuracy(name='effective_accuracy')
        self.recall_tracker = tf.keras.metrics.Recall(name='recall')
        self.precision_tracker = tf.keras.metrics.Precision(name='precision')

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [
            self.total_loss_tracker,
            self.accuracy_tracker,
            self.effective_accuracy_tracker,
            self.recall_tracker,
            self.precision_tracker]
        return metrics

    def train_step(self, data):
        # unpack data
        RGBDE_normalized, gt_segmap_onehot = data #(B=1, H, W, 5), (B=1, H, W, 3)

        # get gradient
        with tf.GradientTape() as tape:
            # get network forward output
            logit = self(RGBDE_normalized, training=True) #(B=1, H, W, 3)
            pred_segmap_prob = tf.nn.softmax(logit) #(B=1, H, W, 3)

            CEvals = tf.keras.losses.categorical_crossentropy(gt_segmap_onehot, pred_segmap_prob) #(B=1, H, W)
            total_loss = tf.reduce_mean(CEvals) #scalar

        # udate gradient
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        gt = tf.argmax(gt_segmap_onehot, axis=-1)
        pred = tf.argmax(pred_segmap_prob, axis=-1)

        # update loss and metric trackers
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y_true=gt, y_pred=pred)
        
        mask = tf.where(gt >= 1, True, False)
        self.effective_accuracy_tracker.update_state(y_true=tf.boolean_mask(gt, mask),
                                                     y_pred=tf.boolean_mask(pred, mask))
        
        gt_isvalid = tf.where(gt == 2, 1, 0)
        pred_isvalid = tf.where(pred == 2, 1, 0)

        self.recall_tracker.update_state(y_true=gt_isvalid, y_pred=pred_isvalid)
        self.precision_tracker.update_state(y_true=gt_isvalid, y_pred=pred_isvalid)

        # pack return
        ret = {
            'total_loss': self.total_loss_tracker.result(),
            'accuracy': self.accuracy_tracker.result(),
            'effective_accuracy': self.effective_accuracy_tracker.result(),
            'recall': self.recall_tracker.result(),
            'precision': self.precision_tracker.result()}
        return ret

    def test_step(self, data):
        # unpack data
        RGBDE_normalized, gt_segmap_onehot = data #(H, W, 5), (H, W, 3)

        # get netwokr output
        logit = self(RGBDE_normalized, training=False) #(H, W, 3)
        pred_segmap_prob = tf.nn.softmax(logit) #(H, W, 3)

        CEvals = tf.keras.losses.categorical_crossentropy(gt_segmap_onehot, pred_segmap_prob) #(H, W)
        total_loss = tf.reduce_mean(CEvals) #scalar

        gt = tf.argmax(gt_segmap_onehot, axis=-1)
        pred = tf.argmax(pred_segmap_prob, axis=-1)

        # update loss and metric trackers
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y_true=gt, y_pred=pred)

        mask = tf.where(gt >= 1, True, False)
        self.effective_accuracy_tracker.update_state(y_true=tf.boolean_mask(gt, mask),
                                                     y_pred=tf.boolean_mask(pred, mask))
        
        gt_isvalid = tf.where(gt == 2, 1, 0)
        pred_isvalid = tf.where(pred == 2, 1, 0)

        self.recall_tracker.update_state(y_true=gt_isvalid, y_pred=pred_isvalid)
        self.precision_tracker.update_state(y_true=gt_isvalid, y_pred=pred_isvalid)

        # pack return
        ret = {
            'total_loss': self.total_loss_tracker.result(),
            'accuracy': self.accuracy_tracker.result(),
            'effective_accuracy': self.effective_accuracy_tracker.result(),
            'recall': self.recall_tracker.result(),
            'precision': self.precision_tracker.result()}
        return ret

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 data_dir:str):
        """
        A data object that can be used to load scenes and generate point cloud batches for training .
        ----------
        Args:
            data_dir {str}: The path to the data directory containing hdf5 file
        ----------
        """
        self.data_dir = data_dir

        # Get the amount of available scenes

        # self.scenes_3d_dir = os.path.join(os.path.join(os.path.join(self.data_dir, 'scenes_3d'), splits))
        self.num_of_scenes_3d = len(os.listdir(self.data_dir))
        self.scene_name_list = os.listdir(self.data_dir)

        # Scene order
        self.scene_order = np.arange(self.num_of_scenes_3d)

     
    def __getitem__(self, idx):
        """
        Returns a batch of data from the scene given the index of the scene.
        ----------
        Args:
            index : Int, Slice or tuple of desired scenes
        ----------
        Return:
            RGBD: (image_height, image_width, 4), 4 channels(Red, Green, Blue, Depth) all normalized to (0, 1).
            gt_segmap_onehot: (image_height, image_width, 3)
        """

        # Generate batch data
        scene_name = f"{self.scene_order[idx]}.hdf5"
        with h5py.File(os.path.join(self.data_dir, scene_name), "r") as f:
            segmap = np.array(f["category_id_segmaps"]) #(image_height, image_width), np.int64
            colors = np.array(f["colors"]) #(image_height, image_width, 3) np.uint8
            # point_cloud = np.array(f["pc"]) #(image_height*image_width, 3) np.float32
            depth = np.array(f["depth"]) #(image_height, image_width) np.float32
            # grasps_tf = np.array(f["grasps_tf"])
            # grasps_scores = np.array(f["grasps_scores"])

        image_height = colors.shape[0]
        image_width = colors.shape[1]
        laplacian = np.array([[1, 4, 1],
                              [4,-20, 4],
                              [1, 4, 1]])
        RGBDE_normalized = np.empty((image_height, image_width, 5), dtype=np.float)
        RGBDE_normalized[:, :, :3] = colors/255.
        RGBDE_normalized[:, :, 3] = (depth - np.mean(depth))/np.std(depth)

        kernaled = (signal.convolve2d(colors[:, :, 0], laplacian, mode="same", boundary="symm")**2 + 
                    signal.convolve2d(colors[:, :, 1], laplacian, mode="same", boundary="symm")**2 + 
                    signal.convolve2d(colors[:, :, 2], laplacian, mode="same", boundary="symm")**2)
        RGBDE_normalized[:, :, 4] = (kernaled - np.mean(kernaled))/np.std(kernaled)

        onehot = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]], dtype=np.float)

        gt_segmap_onehot = onehot[segmap]


        #일단 현재는 Batch Size를 1로 고정시켜 놓는다. 나중에 Batch Size를 조정할 수 있도록 만들면 좋으련만.....
        gt_segmap_onehot = tf.expand_dims(tf.convert_to_tensor(gt_segmap_onehot, dtype=tf.float32), axis=0)
        RGBDE_normalized = tf.expand_dims(tf.convert_to_tensor(RGBDE_normalized, dtype=tf.float32), axis=0)

        #B = 1
        return RGBDE_normalized, gt_segmap_onehot #(B, 720, 1280, 5), (B, 720, 1280, 3)

    def __len__(self):
        return self.num_of_scenes_3d

    def shuffle(self):
        self.scene_order = np.random.shuffle(self.scene_order)