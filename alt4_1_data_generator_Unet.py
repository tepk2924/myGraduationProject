import os
import sys
import random
import tensorflow as tf
import numpy as np
import h5py

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 data_dir:str):
        """
        A data object that can be used to load scenes and generate point cloud batches for training .
        ----------
        Args:
            data_dir {str}: The path to the data directory containing scenes meshes and grasps
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
        scene_name = self.scene_name_list[self.scene_order[idx]]
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

        gt_segmap_onehot = tf.convert_to_tensor(gt_segmap_onehot, dtype=tf.float32)
        RGBD_normalized = tf.convert_to_tensor(RGBD_normalized, dtype=tf.float32)

        return RGBD_normalized, gt_segmap_onehot

    def __len__(self):
        return self.num_of_scenes_3d

    def shuffle(self):
        self.scene_order = np.random.shuffle(self.scene_order)