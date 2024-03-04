import os
import sys
import random
import tensorflow as tf
import numpy as np

from trimeshVisualize import Scene
from network.config import Config

import point_cloud_reader as pcr
# import scene_render.create_table_top_scene as create_scene


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 data_dir:str,
                 grasp_root_folder:str,
                 batch_size:int,
                 raw_num_points:int,
                 threshold=0.05, 
                 search_radius = 0.008):
        """
        A data object that can be used to load scenes and generate point cloud batches for training .
        ----------
        Args:
            data_dir {str}: The path to the data directory containing scenes meshes and grasps
            batch_size {int}: Batch size (batch samples one scene from multiple random angles)
        Keyword Args:
            threshold {float}: The threshold for the ground truth scores. Valid grasps will be the ones that have the score higher than the threshold.
            search_radius {float}: How for from the each point on point cloud we look for a valid grasp.
        ----------
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.threshold = threshold
        self.search_radius = search_radius

        # Get the amount of available scenes

        # self.scenes_3d_dir = os.path.join(os.path.join(os.path.join(self.data_dir, 'scenes_3d'), splits))
        self.num_of_scenes_3d = len(os.listdir(self.data_dir))
        self.scene_name_list = os.listdir(self.data_dir)

        # Create a pcr
        self.pcreader = pcr.PointCloudReader(
            grasp_root_folder=grasp_root_folder,
            batch_size=batch_size,
            raw_num_points=raw_num_points,
            intrinsics="zivid2",
            use_uniform_quaternions=False,
            elevation=(10, 40),  # The pitch of the camera(테이블의 z축과 카메라의 시선과의 각도) in degrees
            distance_range=(0.6, 1),  # How far away the camera is in m
            estimate_normals=False,
            depth_augm_config={"sigma": 0.001,
                               "clip": 0.005, "gaussian_kernel": 0},
            pc_augm_config={"occlusion_nclusters": 0,
                            "occlusion_dropout_rate": 0.0, "sigma": 0.000, "clip": 0.005}  
        )

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
            pc_tensor: (batch_size, num_points, 3)
            gt_scene_tensor: (batch_size, num_points)
            gt_approach_tensor: (batch_size, num_points, 3)
        """

        self.pcreader._renderer.create()

        

        # ------ Prepare the output arrays -------
        pc_numpy = np.empty(
            [self.batch_size, self.pcreader._raw_num_points, 3])
        # Binary mask for each point
        gt_scores = np.empty(
            [self.batch_size, self.pcreader._raw_num_points])
        # Approach vectors for positive points
        gt_approach = np.empty(
            [self.batch_size, self.pcreader._raw_num_points, 3])


        # Generate batch data
        scene_name = self.scene_name_list[self.scene_order[idx]]
        batch_data, cam_poses, scene_idx, batch_segmap, obj_pcs_batch = self.pcreader.get_scene_batch(
            os.path.join(self.data_dir , scene_name))
        self.pcreader._renderer.destroy()

        # Get camera tf to world frame
        world_to_cam = self.pcreader.pc_convert_cam(cam_poses)

        # Compbine object PC's to one PC
        pc_segmap = []
        for obj_pcs in obj_pcs_batch:
            pc_objects = None
            for pc in obj_pcs:
                if pc_objects is None:
                    pc_objects = pc[:,0:3]
                else:
                    pc_objects = np.append(pc_objects, pc[:, 0:3], axis=0)
            pc_segmap.append(pc_objects)
        
        #pc_segmap[0 ~ N - 1] : N개의 batch 중 0 ~ N - 1번째의 pc, 테이블 제외함

        # Convert all point clouds to world frame (to find GT)
        pc_segmap = self.pcreader.pc_to_world(
            pc_segmap, cam_poses)


        batch_data = self.pcreader.pc_to_world(
            batch_data, cam_poses)
        
        #=====================================================

        my_scene = Scene()
        my_scene.plot_point_cloud(pc_segmap[0])
        print(f"pc_segmap[0] 시각화 중 (지점 {len(pc_segmap[0])} 개)")
        my_scene.display()
        del my_scene

        my_scene = Scene()
        my_scene.plot_point_cloud(batch_data[0])
        print(f"batch_data[0] 시각화 중 (지점 {len(batch_data[0])} 개)")
        my_scene.display()
        del my_scene

        #======================================================

        # Get ground truth
        gt_scores, gt_approach = self.pcreader.get_ground_truth(
            batch_data, os.path.join(self.data_dir, scene_name), pc_segmap=pc_segmap, threshold=self.threshold, search_radius=self.search_radius)
        pc_numpy = batch_data

        #======================================================

        # print(gt_scores.shape)
        # print(batch_data.shape)

        maskM1 = np.where(gt_scores[0] == -1)
        mask0 = np.where(gt_scores[0] == 0)
        mask1 = np.where(gt_scores[0] == 1)

        my_scene = Scene()
        my_scene.plot_point_cloud(batch_data[0][maskM1])
        my_scene.display()
        del my_scene

        my_scene = Scene()
        my_scene.plot_point_cloud(batch_data[0][mask0])
        my_scene.display()
        del my_scene

        my_scene = Scene()
        my_scene.plot_point_cloud(batch_data[0][mask1])
        my_scene.display()
        del my_scene

        del maskM1, mask0, mask1

        #======================================================
        
        # Convert back to OpenCV camera frame
        for batch_idx in range(self.batch_size):
            # Make homogenous PC
            batch_pc_hom = np.ones((len(gt_scores[batch_idx]), 4))
            batch_pc_hom[:, :3] = pc_numpy[batch_idx]

            pc_numpy[batch_idx] = np.dot(
                world_to_cam[batch_idx], batch_pc_hom.T).T[:, 0: 3]
            gt_approach[batch_idx] = np.dot(
                world_to_cam[batch_idx, 0:3, 0:3], gt_approach[batch_idx].T).T[:, 0: 3]

        self.lb_cam_inverse = world_to_cam
        
        pc_tensor = tf.convert_to_tensor(pc_numpy, dtype=tf.float32)

        # Normalize the input PC
        pc_mean = tf.reduce_mean(pc_tensor, axis=1, keepdims=True)
        self.lb_mean = pc_mean
        pc_tensor = pc_tensor - pc_mean

        #gt_score의 한 행에 1이 하나도 없는 상황이 나오면 학습시 NaN이 나와 방해하므로, 행마다 1의 갯수를 조사해서 없으면 그 행의 0인 값 중 하나를 1로 바꾼다
        for i in range(len(gt_scores)):
            mask1 = np.where(gt_scores[i] == 1)
            if len(mask1[0]) == 0:
                mask0 = np.where(gt_scores[i] == 0)
                print(mask0[0])
                gt_scores[i][mask0[0][0]] = 1.

        gt_scores_tensor = tf.convert_to_tensor(gt_scores, dtype=tf.int32)
        gt_approach_tensor = tf.convert_to_tensor(gt_approach, dtype=tf.float32)

        #=====================================================

        my_scene = Scene()
        my_scene.plot_point_cloud(pc_tensor[0].numpy(), color=[255, 0, 0, 255])
        my_scene.display()
        del my_scene

        #======================================================

        #tepk2924 수정 : 쓸모없는 tf.squeeze와 tf.expand_dims 코드 삭제
        return pc_tensor, (gt_scores_tensor, gt_approach_tensor)     

    def __len__(self):
        return self.num_of_scenes_3d

    def shuffle(self):
        self.scene_order = np.random.shuffle(self.scene_order)

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    config = Config(os.path.join(ROOT_DIR, 'network/config.yml'))
    train_config = config.load()
    # data_dir = input("Scene이 들어있는 폴더의 디렉토리 입력 : ")
    # grasp_root_folder = input(".pkl 파일이 들어있는 폴더 입력")
    data_dir = "/home/tepk2924/tepk2924Works/scenes_generated/train"
    grasp_root_folder = "/home/tepk2924/tepk2924Works/successful_grasp"
    dg = DataGenerator(data_dir,
                       grasp_root_folder,
                       train_config["BATCH_SIZE"],
                       train_config["RAW_NUM_POINTS"],
                       threshold=0.05,
                       search_radius=0.003)
    dg[300]