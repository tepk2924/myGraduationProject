#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import h5py
import numpy as np
import tensorflow as tf

from scipy.spatial.ckdtree import cKDTree
from pnet2_layers.layers import Pointnet_SA, Pointnet_SA_MSG, Pointnet_FP
from pnet2_layers.cpp_modules import select_top_k

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# add config as arg later
def build_suction_pointnet_graph(hyperparameters):
    """
    Build computational graph for Suction-GraspNet

    --------------
    Args:
        config (`Config`) : `Config` instance.
    --------------
    Returns:
        input (`tf.Tensor`) : Input point cloud (B, N, 3).
        outputs (`Tuple`) : (pc_contacts, grasp_socre, grasp_approach).
            pc_contacts (`tf.Tensor`) : Contact point cloud (B, M, 3)
            grasp_score (`tf.Tensor`) : Confidence of the grasps (B, M).
            grasp_approach (`tf.Tensor`) : Approach vector of the grasps (B, M, 3).
    """
    # Input layer
    # (20000, 3)
    input_pc = tf.keras.Input(
        shape=(hyperparameters["RAW_NUM_POINTS"], 3),
        name='input_point_cloud')


    # Set Abstraction layers
    # (2048, 3), (2048, 320)
    sa_xyz_0, sa_points_0 = Pointnet_SA_MSG(
        npoint=hyperparameters["SA_NPOINT_0"],
        radius_list=hyperparameters["SA_RADIUS_LIST_0"],
        nsample_list=hyperparameters["SA_NSAMPLE_LIST_0"],
        mlp=hyperparameters["SA_MLP_LIST_0"],
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(input_pc, None)
    # (512, 3), (512, 640)
    sa_xyz_1, sa_points_1 = Pointnet_SA_MSG(
        npoint=hyperparameters["SA_NPOINT_1"],
        radius_list=hyperparameters["SA_RADIUS_LIST_1"],
        nsample_list=hyperparameters["SA_NSAMPLE_LIST_1"],
        mlp=hyperparameters["SA_MLP_LIST_1"],
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(sa_xyz_0, sa_points_0)
    # (128, 3), (128, 640)
    sa_xyz_2, sa_points_2 = Pointnet_SA_MSG(
        npoint=hyperparameters["SA_NPOINT_2"],
        radius_list=hyperparameters["SA_RADIUS_LIST_2"],
        nsample_list=hyperparameters["SA_NSAMPLE_LIST_2"],
        mlp=hyperparameters["SA_MLP_LIST_2"],
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(sa_xyz_1, sa_points_1)

    # Global feature layer
    # (1, 3), (1024...?)
    sa_xyz_3, sa_points_3 = Pointnet_SA(
        npoint=None,
        radius=None,
        nsample=None,
        mlp=hyperparameters["SA_MLP_GROUP_ALL"],
        group_all=True,
        knn=False,
        use_xyz=True,
        activation=tf.nn.relu,
        bn=False)(sa_xyz_2, sa_points_2)

    # Feature propagation layers.
    # (128, 256)
    fp_points_2 = Pointnet_FP(
        mlp=hyperparameters["FP_MLP_0"],
        activation=tf.nn.relu,
        bn=False)(sa_xyz_2, sa_xyz_3, sa_points_2, sa_points_3)
    # (512, 128)
    fp_points_1 = Pointnet_FP(
        mlp=hyperparameters["FP_MLP_1"],
        activation=tf.nn.relu,
        bn=False)(sa_xyz_1, sa_xyz_2, sa_points_1, fp_points_2)
    # (2048, 128)
    fp_points_0 = Pointnet_FP(
        mlp=hyperparameters["FP_MLP_2"],
        activation=tf.nn.relu,
        bn=False)(sa_xyz_0, sa_xyz_1, sa_points_0, fp_points_1)

    # Output from the pointnet++
    # (2048, 3)
    # (2048, 1024)
    output_pc = sa_xyz_0
    output_feature = fp_points_0

    # grasp_score
    # (2048, 1)
    grasp_score = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid')(output_feature)
    grasp_score = tf.keras.layers.LeakyReLU()(grasp_score)
    grasp_score = tf.keras.layers.Dropout(rate=0.5)(grasp_score)
    grasp_score = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='valid')(grasp_score)
    grasp_score = tf.keras.activations.sigmoid(grasp_score)
    grasp_score = tf.squeeze(grasp_score, axis=-1)

    # grasp_approach
    # (2048, 3)
    grasp_approach = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid')(output_feature)
    grasp_approach = tf.keras.layers.LeakyReLU()(grasp_approach)
    grasp_approach = tf.keras.layers.Dropout(rate=0.5)(grasp_approach)
    grasp_approach = tf.keras.layers.Conv1D(filters=3, kernel_size=1, strides=1, padding='valid')(grasp_approach)
    grasp_approach = tf.math.l2_normalize(grasp_approach, axis=-1)


    return input_pc, (output_pc, grasp_score, grasp_approach)


@tf.function
def knn_point(k, xyz1, xyz2):
    #idx = tf.constanct(tf.int32, shape=(None,2048))

    # This did not work for me so changed
    b = tf.shape(xyz1)[0]
    n = tf.shape(xyz1)[1]
    c = tf.shape(xyz1)[2]
    m = tf.shape(xyz2)[1]


    xyz1 = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)

    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0, 0, 0], [-1, -1, k])
    val = tf.slice(out, [0, 0, 0], [-1, -1, k])

    return val, idx


@tf.function
def score_loss_fn(gt_scores, pred_scores, max_k=512):
    """
    Calculate score loss given ground truth and predicted scores. 
    The inputs must be of dimension [batch_size, num_points].
    --------------
    Args:
        gt_scores (tf.Tensor) : Ground truth scores. (B, N)
        pred_scores (tf.Tensor) : Predicted scores. (B, N)
    --------------
    Returns:
        loss (tf.Tensor) : Binary crossentropy
    """
    # Expand dimensions
    gt_scores = tf.expand_dims(gt_scores, axis=-1)
    pred_scores = tf.expand_dims(pred_scores, axis=-1)
    mask = tf.where(gt_scores != -1, 1, 0)

    # Calculate elementvise binary cross entropy loss
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss = bce(gt_scores, pred_scores, sample_weight=mask)
    # Get the indeces for the sorted losses
    sort_i = tf.argsort(loss, axis=1, direction='DESCENDING')

    #tepk2924 수정 : batch 1개일 때와 batch가 2개 이상일 때와 같은 동작을 하도록 코드 수정.
    # Use the indeces to sort input data

    #gt_scores = tf.squeeze(gt_scores)
    gt_scores = tf.squeeze(gt_scores, axis=-1)
    gt_scores = tf.gather(gt_scores, sort_i, batch_dims=-1)

    #pred_scores = tf.squeeze(pred_scores)
    pred_scores = tf.squeeze(pred_scores, axis=-1)
    pred_scores = tf.gather(pred_scores, sort_i, batch_dims=-1)

    # Calculate the loss for top k points
    loss = tf.gather(loss, sort_i, batch_dims=-1)
    loss = tf.reduce_mean(loss[:, :max_k])
    return loss


@tf.function
def approach_loss_fn(gt_approach, pred_approach):
    """
    Calculate the loss for the approach vectors.
    The inputs must be of dimension (B, M, 3).
    Where m are only the predicted points where the ground truth for those points are True !!!!
    --------------
    Args:
        gt_approach (tf.Tensor) : Ground truth approach vectors. (B, M, 3)
        pred_approach (tf.Tensor) : Predicted approach vectors. (B, M, 3)
    --------------
    Returns:
        loss (tf.Tensor): CosineSimilarity
    """
    #TODO : 가끔씩 NaN을 내보내서 전체 학습 과정을 방해하는 경우가 있음.
    loss = tf.reduce_mean(
        tf.keras.losses.cosine_similarity(gt_approach, pred_approach)+1)
    
    if tf.math.is_nan(loss):
        loss = tf.convert_to_tensor(0.0, dtype=float)
    return loss


@tf.function
def loss_fn(gt_scores, pred_scores, gt_approach, pred_approach, max_k=256):
    """
    Given formatted ground truth boxes and network output, calculate score and approach loss.
    --------------
    Args:
        gt_scores (tf.Tensor) : Ground truth scores. (B, N)
        pred_scores (tf.Tensor) : Predicted scores. (B, N)
        gt_approach (tf.Tensor) : Ground truth approach vectors. (B, N, 3)
        pred_approach (tf.Tensor) : Predicted approach vectors. (B, N, 3)
    Keyword Args:
        max_k (int) : Amount of points to use for the score loss.
    --------------
    Returns:
        l_score (tf.Tensor) : Score loss value
        l_approach (tf.Tensor) : Approach loss value
    """

    # Calculate score loss
    l_score = score_loss_fn(gt_scores, pred_scores, 512)
    # Filter only grasps that should be positive
    mask = tf.where(gt_scores == 1, True, False)
    gt_approach = tf.boolean_mask(gt_approach, mask)
    pred_approach = tf.boolean_mask(pred_approach, mask)

    # Calculate approach loss
    l_approach = approach_loss_fn(gt_approach, pred_approach)

    return l_score, l_approach


class SuctionGraspNet(tf.keras.models.Model):
    def __init__(self, inputs, outputs):
        super(SuctionGraspNet, self).__init__(inputs, outputs)

    def compile(self, optimizer='adam', run_eagerly=None):
        super(SuctionGraspNet, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)

        # define trackers
        self.grasp_score_acc_tracker = tf.keras.metrics.BinaryAccuracy(name='grasp_sc_acc')
        self.grasp_score_precision_tracker = tf.keras.metrics.Precision(thresholds=0.5, name='grasp_sc_pcs')
        self.grasp_score_loss_tracker = tf.keras.metrics.Mean(name='grasp_sc_loss')
        self.grasp_app_mae_tracker = tf.keras.metrics.MeanAbsoluteError(name='grasp_app_mae')
        self.grasp_app_loss_trakcer = tf.keras.metrics.Mean(name='grasp_app_loss')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        metrics = [
            self.grasp_score_acc_tracker,
            self.grasp_score_precision_tracker,
            self.grasp_score_loss_tracker,
            self.grasp_app_mae_tracker,
            self.grasp_app_loss_trakcer,
            self.total_loss_tracker]
        return metrics

    def train_step(self, data):
        # unpack data
        input_pc, (score_target, approach_target) = data

        # get gradient
        with tf.GradientTape() as tape:
            # get network forward output
            output_pc, score_output, approach_output = self(input_pc, training=True)

            # fromat ground truth boxes
            indeces_all = tf.squeeze(knn_point(1, input_pc, output_pc)[1])

            #tepk2924 조한 수정 : 차원 맞추는 코드 삽입.
            indeces_all = tf.expand_dims(indeces_all, 0)

            indeces_all = tf.ensure_shape(
                indeces_all, (None, 2048), name=None
            )
            # Match output points and original PC points
            score_target = tf.gather(
                score_target, indeces_all, axis=1, batch_dims=1)
            approach_target = tf.gather(
                approach_target, indeces_all, axis=1, batch_dims=1)

            # get loss
            score_loss, approach_loss = loss_fn(score_target, score_output,
                                                approach_target, approach_output)
            
            #tepk2924 조한 : 굳이 loss의 비율을 1대 1로? 여러번의 trial and error로 알맞은 비율을 찾는 게 중요할 지도 모르겠군.
            total_loss = score_loss + approach_loss

        # udate gradient
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        mask_scores = tf.where(score_target != -1, 1, 0)
        mask_scores_exp = tf.expand_dims(mask_scores, -1)
        
        # update loss and metric trackers
        self.grasp_score_acc_tracker.update_state(
            score_target, score_output, sample_weight=mask_scores_exp)
        self.grasp_score_precision_tracker.update_state(
            score_target, score_output, sample_weight=mask_scores_exp)
        self.grasp_score_loss_tracker.update_state(score_loss)
        self.grasp_app_mae_tracker.update_state(approach_target, approach_output)
        self.grasp_app_loss_trakcer.update_state(approach_loss)
        self.total_loss_tracker.update_state(total_loss)

        # pack return
        ret = {
            'score_acc': self.grasp_score_acc_tracker.result(),
            'score_prec': self.grasp_score_precision_tracker.result(),
            'score_loss': self.grasp_score_loss_tracker.result(),
            'app_mae': self.grasp_app_mae_tracker.result(),
            'app_loss': self.grasp_app_loss_trakcer.result(),
            'total_loss': self.total_loss_tracker.result()}
        return ret

    def test_step(self, data):
        # unpack data
        input_pc, (score_target, approach_target) = data

        # get netwokr output
        output_pc, score_output, approach_output = self(input_pc, training=False)

        # fromat ground truth boxes
        indeces_all = tf.squeeze(knn_point(1, input_pc, output_pc)[1])

        #tepk2924 조한 수정 : 차원 맞추는 코드 삽입.
        indeces_all = tf.expand_dims(indeces_all, 0)

        indeces_all = tf.ensure_shape(
            indeces_all, (None, 2048), name=None
        )
        # Match output points and original PC points
        score_target = tf.gather(
            score_target, indeces_all, axis=1, batch_dims=1)
        approach_target = tf.gather(
            approach_target, indeces_all, axis=1, batch_dims=1)

        # get loss
        score_loss, approach_loss = loss_fn(score_target, score_output,
                                            approach_target, approach_output)
        total_loss = score_loss + approach_loss

        mask_scores = tf.where(score_target != -1, 1, 0)
        mask_scores = tf.expand_dims(mask_scores, -1)
        # update loss and metric trackers
        self.grasp_score_acc_tracker.update_state(score_target, score_output, sample_weight=mask_scores)
        self.grasp_score_precision_tracker.update_state(
            score_target, score_output, sample_weight=mask_scores)
        self.grasp_score_loss_tracker.update_state(score_loss)
        self.grasp_app_mae_tracker.update_state(approach_target, approach_output)
        self.grasp_app_loss_trakcer.update_state(approach_loss)
        self.total_loss_tracker.update_state(total_loss)

        # pack return
        ret = {
            'score_acc': self.grasp_score_acc_tracker.result(),
            'score_prec': self.grasp_score_precision_tracker.result(),
            'score_loss': self.grasp_score_loss_tracker.result(),
            'app_mae': self.grasp_app_mae_tracker.result(),
            'app_loss': self.grasp_app_loss_trakcer.result(),
            'total_loss': self.total_loss_tracker.result()}
        return ret

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 data_dir:str,
                 hyperparameters: dict):
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
        self.threshold = hyperparameters["THRESHOLD"]
        self.raw_num_points = hyperparameters["RAW_NUM_POINTS"]
        self.search_radius = hyperparameters["SEARCH_RADIUS"]

        # Get the amount of available scenes

        # self.scenes_3d_dir = os.path.join(os.path.join(os.path.join(self.data_dir, 'scenes_3d'), splits))
        self.num_of_scenes_3d = len(os.listdir(self.data_dir))
        self.scene_name_list = os.listdir(self.data_dir)

        # # Create a pcr
        # self.pcreader = pcr.PointCloudReader(
        #     grasp_root_folder=grasp_root_folder,
        #     batch_size=batch_size,
        #     raw_num_points=raw_num_points,
        #     intrinsics="zivid2",
        #     use_uniform_quaternions=False,
        #     elevation=(10, 40),  # The pitch of the camera(테이블의 z축과 카메라의 시선과의 각도) in degrees
        #     distance_range=(0.6, 1),  # How far away the camera is in m
        #     estimate_normals=False,
        #     depth_augm_config={"sigma": 0.001,
        #                        "clip": 0.005, "gaussian_kernel": 0},
        #     pc_augm_config={"occlusion_nclusters": 0,
        #                     "occlusion_dropout_rate": 0.0, "sigma": 0.000, "clip": 0.005}  
        # )

        # Scene order
        self.scene_order = np.arange(self.num_of_scenes_3d)

     
    def __getitem__(self, idx):
        """
        Returns a data from the scene given the index of the scene.
        ----------
        Args:
            index : Int, Slice or tuple of desired scenes
        ----------
        Return:
            pc_tensor: (raw_num_points, 3)
            gt_scene_tensor: (raw_num_points)
            gt_approach_tensor: (raw_num_points, 3)
        """   

        # Generate batch data
        scene_name = self.scene_name_list[self.scene_order[idx]]

        with h5py.File(os.path.join(self.data_dir, scene_name), "r") as f:
            segmap = np.array(f["category_id_segmaps"]) #(image_height, image_width), np.int64, 0 if table, 1 if non-grippable object, 2 if grippable object 
            # colors = np.array(f["colors"]) #(image_height, image_width, 3) np.uint8
            point_cloud = np.array(f["pc"]) #(image_height*image_width, 3) np.float32 in world frame
            # depth = np.array(f["depth"]) #(image_height, image_width) np.float32
            extrinsic = np.array(f["extrinsic"]) #camera extrinsic, 4x4 np.ndarray
            grasps_tf = np.array(f["grasps_tf"])
            grasps_scores = np.array(f["grasps_scores"])
            # obj_file_path_np = np.array(f["original_obj_file"])

        camera_inverse = np.linalg.inv(extrinsic)
        point_cloud_1padded = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float32)), axis=-1).T
        point_cloud_camera_frame = ((camera_inverse@point_cloud_1padded).T)[:, :3] #Full point clound at camera frame, (image_height*image_width, 3)
        grasps_tf_camera_frame = camera_inverse@grasps_tf #Grasp tfs at camera frame
        #Each grasp in 4x4 form:
        #[A, B, C, D]
        #[E, F, G, H]
        #[I, J, K, L]
        #[0, 0, 0, 1]

        #vec [A, E, I] : i-vector
        #vec [B, F, J] : j-vector
        #vec [C, G, K] : k-vector, approach vector, almost normal outward vector of obj surface
        #vec [D, H, L] : translation of grasp, center of grasp, at surface of obj
        segmap_longlist = segmap.reshape(-1)  #Segmap idxs in long vector form, (image_height*image_width)
        selected_point_idxs = np.random.choice(range(point_cloud_camera_frame.shape[0]),
                                               self.raw_num_points,
                                               False) #Randomly chooses (self.raw_num_points) numbers from 0 to image_height*image_width - 1
        selected_point_cloud = point_cloud_camera_frame[selected_point_idxs] #Select partial point cloud from original point cloud, (raw_num_points, 3)
        selected_segmap = segmap_longlist[selected_point_idxs] #Select parial segmap, (raw_num_points)
        selected_segmap_object_pts_idx = np.where(selected_segmap >= 1)

        positive_grasps_idx_mask = np.where(grasps_scores > self.threshold) #(num_of_positive_grasp)
        positive_grasps_tf = grasps_tf_camera_frame[positive_grasps_idx_mask, :, :] #(num_of_positive_grasp, 4, 4)
        
        #vec [D, H, L] : translation of grasp, center of grasp, at surface of obj
        positive_grasps_translation = positive_grasps_tf[:, :3, 3] #(num_of_positive_grasp, 3)
        
        #vec [C, G, K] : k-vector, approach vector, almost normal outward vector of obj surface
        positive_grasps_approach_vector = positive_grasps_tf[:, :3, 2] #(num_of_positive_grasp, 3)

        #initialize all value to -1
        gt_score = np.full([selected_point_cloud.shape[0]], -1, dtype=np.int32) #(raw_num_points)

        #initialize all appr_vec to [0, 0, 1]
        gt_approach = np.zeros([selected_point_cloud.shape[0], 3]) #(raw_num_points, 3)
        gt_approach[..., 2] = 1

        tree = cKDTree(selected_point_cloud)
        #Below returns a list of lists of indexs of points in selected point cloud sufficiently close to each grasp points
        good_point_lists_idx = tree.query_ball_point(positive_grasps_translation, self.search_radius)

        #All points at surface of objs are assigned score 0
        gt_score[selected_segmap_object_pts_idx] = 0

        for appr_vec, point_idx_list in zip(positive_grasps_approach_vector, good_point_lists_idx):
            for point_idx in point_idx_list:
                #All points near at least one grasp translation pts gets score 1
                gt_score[point_idx] = 1

                #All points near a grasp translation pts gets appr_vec same as k-vector of corresponding grasp
                gt_approach[point_idx, :] = appr_vec
        
        pc_tensor = tf.convert_to_tensor(selected_point_cloud, dtype=tf.float32)
        gt_scores_tensor = tf.convert_to_tensor(gt_score, dtype=tf.int32)
        gt_approach_tensor = tf.convert_to_tensor(gt_approach, dtype=tf.float32)
        return pc_tensor, (gt_scores_tensor, gt_approach_tensor) 
        # return selected_point_cloud, (gt_score, gt_approach)

        # batch_data, cam_poses, scene_idx, batch_segmap, obj_pcs_batch = self.pcreader.get_scene_batch(
        #     os.path.join(self.data_dir , scene_name))
        # self.pcreader._renderer.destroy()

        # # Get camera tf to world frame
        # world_to_cam = self.pcreader.pc_convert_cam(cam_poses)

        # # Compbine object PC's to one PC
        # pc_segmap = []
        # for obj_pcs in obj_pcs_batch:
        #     pc_objects = None
        #     for pc in obj_pcs:
        #         if pc_objects is None:
        #             pc_objects = pc[:,0:3]
        #         else:
        #             pc_objects = np.append(pc_objects, pc[:, 0:3], axis=0)
        #     pc_segmap.append(pc_objects)

        # # Convert all point clouds to world frame (to find GT)
        # pc_segmap = self.pcreader.pc_to_world(
        #     pc_segmap, cam_poses)

        # batch_data = self.pcreader.pc_to_world(
        #     batch_data, cam_poses)

        # # Get ground truth
        # gt_scores, gt_approach = self.pcreader.get_ground_truth(
        #     batch_data, os.path.join(self.data_dir , scene_name), pc_segmap=pc_segmap, threshold=self.threshold, search_radius=self.search_radius)
        # pc_numpy = batch_data

        # # Convert back to OpenCV camera frame
        # for batch_idx in range(self.batch_size):
        #     # Make homogenous PC
        #     batch_pc_hom = np.ones((len(gt_scores[batch_idx]), 4))
        #     batch_pc_hom[:, :3] = pc_numpy[batch_idx]

        #     pc_numpy[batch_idx] = np.dot(
        #         world_to_cam[batch_idx], batch_pc_hom.T).T[:, 0: 3]
        #     gt_approach[batch_idx] = np.dot(
        #         world_to_cam[batch_idx, 0:3, 0:3], gt_approach[batch_idx].T).T[:, 0: 3]

        # self.lb_cam_inverse = world_to_cam
        
        # pc_tensor = tf.convert_to_tensor(pc_numpy, dtype=tf.float32)
  
        # # Normalize the input PC
        # pc_mean = tf.reduce_mean(pc_tensor, axis=1, keepdims=True)
        # self.lb_mean = pc_mean
        # pc_tensor = pc_tensor - pc_mean

        # #gt_score의 한 행에 1이 하나도 없는 상황이 나오면 학습시 NaN이 나와 방해하므로, 행마다 1의 갯수를 조사해서 없으면 그 행의 0인 값 중 하나를 1로 바꾼다
        # for i in range(len(gt_scores)):
        #     mask1 = np.where(gt_scores[i] == 1)
        #     if len(mask1[0]) == 0:
        #         mask0 = np.where(gt_scores[i] == 0)
        #         print(mask0[0])
        #         gt_scores[i][mask0[0][0]] = 1.

        # gt_scores_tensor = tf.convert_to_tensor(gt_scores, dtype=tf.int32)
        # gt_approach_tensor = tf.convert_to_tensor(gt_approach, dtype=tf.float32)

        # #tepk2924 수정 : 쓸모없는 tf.squeeze와 tf.expand_dims 코드 삭제
        # return pc_tensor, (gt_scores_tensor, gt_approach_tensor)     

    def __len__(self):
        return self.num_of_scenes_3d

    def shuffle(self):
        self.scene_order = np.random.shuffle(self.scene_order)