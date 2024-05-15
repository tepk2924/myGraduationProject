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
        input_tensor tf.Tensor : input data with 4 channels (batch_num, image_height: 480, image_width: 640, 4).
        output_tensor: tf.Tensor : output_tensor to be compared to gt (batch_num, image_height: 480, image_width: 640, 3)
    """
    BC = hyperparameters["BC"]
    input_tensor = tf.keras.Input((480, 640, 4), batch_size=1) #(B, 480, 640, 4)
    relu = tf.keras.layers.ReLU()
    layer00 = tf.pad(input_tensor, ((0, 0), (94, 94), (94, 94), (0, 0)), mode="SYMMETRIC") #(B, 668, 828, 4)
    layer01 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(3, 3))(layer00)) #(B, 666, 826, BC)
    layer02 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(3, 3))(layer01)) #(B, 664, 824, BC)
    layer10 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer02) #(B, 332, 412, BC)
    layer11 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer10)) #(B, 330, 410, 2BC)
    layer12 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer11)) #(B, 328, 408, 2BC)
    layer20 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer12) #(B, 164, 204, 2BC)
    layer21 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer20)) #(B, 162, 202, 4BC)
    layer22 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer21)) #(B, 160, 200, 4BC)
    layer30 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer22) #(B, 80, 100, 4BC)
    layer31 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer30)) #(B, 78, 98, 8BC)
    layer32 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer31)) #(B, 76, 96, 8BC)
    layer40 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer32) #(B, 38, 48, 8BC)
    layer41 = relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer40)) #(B, 36, 46, 16BC)
    layer42 = relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer41)) #(B, 34, 44, 16BC)
    layer32_cropped = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(layer32) #(B, 68, 88, 8BC)
    layer42_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer42) #(B, 68, 88, 16BC)
    layer42_conved22 = tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(2, 2), padding="same")(layer42_upsample) #(B, 68, 88, 8BC)
    layer50 = tf.keras.layers.Concatenate(axis=-1)((layer32_cropped, layer42_conved22)) #(B, 68, 88, 16BC)
    layer51 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer50)) #(B, 66, 86, 8BC)
    layer52 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer51)) #(B, 64, 84, 8BC)
    layer22_cropped = tf.keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(layer22) #(B, 128, 168, 4BC)
    layer52_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer52) #(B, 128, 168, 8BC)
    layer52_conved22 = tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(2, 2), padding="same")(layer52_upsample) #(B, 128, 168, 4BC)
    layer60 = tf.keras.layers.Concatenate(axis=-1)((layer22_cropped, layer52_conved22)) #(B, 128, 168, 8BC)
    layer61 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer60)) #(B, 126, 166, 4BC)
    layer62 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer61)) #(B, 124, 164, 4BC)
    layer12_cropped = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(layer12) #(B, 248, 328, 2BC)
    layer62_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer62) #(B, 248, 328, 4BC)
    layer62_conved22 = tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(2, 2), padding="same")(layer62_upsample) #(B, 248, 328, 2BC)
    layer70 = tf.keras.layers.Concatenate(axis=-1)((layer12_cropped, layer62_conved22)) #(B, 248, 328, 4BC)
    layer71 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer70)) #(B, 246, 326, 2BC)
    layer72 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer71)) #(B, 244, 324, 2BC)
    layer02_cropped = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(layer02) #(B, 488, 648, BC)
    layer72_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer72) #(B, 488, 648, 2BC)
    layer72_conved22 = tf.keras.layers.Conv2D(filters=BC, kernel_size=(2, 2), padding="same")(layer72_upsample) #(B, 488, 648, BC)
    layer80 = tf.keras.layers.Concatenate(axis=-1)((layer02_cropped, layer72_conved22)) #(B, 488, 648, 2BC)
    layer81 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(5, 5))(layer80)) #(B, 484, 644, BC)
    layer82 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(5, 5))(layer81)) #(B, 480, 640, BC)
    output_tensor = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1))(layer82) #(B, 480, 640, 3)

    return input_tensor, output_tensor #(B, H, W, 4), (B, H, W, 3)

class Unet(tf.keras.models.Model):
    def __init__(self, inputs, outputs):
        super(Unet, self).__init__(inputs, outputs)

    def compile(self, optimizer='adam', run_eagerly=None):
        super(Unet, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)

        # define trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.accuracy_tracker = tf.keras.metrics.Accuracy(name='accuracy')
        self.effective_accuracy_tracker = tf.keras.metrics.Accuracy(name='effective_accuracy')

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
            self.effective_accuracy_tracker]
        return metrics

    def train_step(self, data):
        # unpack data
        RGBDE_normalized, gt_segmap_onehot = data #(B=1, H, W, 4), (B=1, H, W, 3)

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

        # pack return
        ret = {
            'total_loss': self.total_loss_tracker.result(),
            'accuracy': self.accuracy_tracker.result(),
            'effective_accuracy': self.effective_accuracy_tracker.result()}
        return ret

    def test_step(self, data):
        # unpack data
        RGBDE_normalized, gt_segmap_onehot = data #(H, W, 5), (H, W, 3)

        # get netwokr output
        logit = self(RGBDE_normalized, training=True) #(H, W, 3)
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
        
        # pack return
        ret = {
            'total_loss': self.total_loss_tracker.result(),
            'accuracy': self.accuracy_tracker.result(),
            'effective_accuracy': self.effective_accuracy_tracker.result()}
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
            point_cloud = np.array(f["pc"]) #(image_height*image_width, 3) np.float32
            depth = np.array(f["depth"]) #(image_height, image_width) np.float32
            grasps_tf = np.array(f["grasps_tf"])
            grasps_scores = np.array(f["grasps_scores"])

        image_height = colors.shape[0]
        image_width = colors.shape[1]
        RGBD_normalized = np.empty((image_height, image_width, 4), dtype=np.float)
        RGBD_normalized[:, :, :3] = colors/255.
        depth_max = np.max(depth)
        depth_min = np.min(depth)
        RGBD_normalized[:, :, 3] = (depth - depth_min)/(depth_max - depth_min)

        onehot = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]], dtype=np.float)

        gt_segmap_onehot = onehot[segmap]


        #일단 현재는 Batch Size를 1로 고정시켜 놓는다. 나중에 Batch Size를 조정할 수 있도록 만들면 좋으련만.....
        gt_segmap_onehot = tf.expand_dims(tf.convert_to_tensor(gt_segmap_onehot, dtype=tf.float32), axis=0)
        RGBD_normalized = tf.expand_dims(tf.convert_to_tensor(RGBD_normalized, dtype=tf.float32), axis=0)

        #B = 1
        return RGBD_normalized, gt_segmap_onehot #(B, H, W, 5), (B, H, W, 3)

    def __len__(self):
        return self.num_of_scenes_3d

    def shuffle(self):
        self.scene_order = np.random.shuffle(self.scene_order)