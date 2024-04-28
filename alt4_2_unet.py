#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# add config as arg later
def build_unet_graph(BC: int):
    """
    Build computational graph for Suction-GraspNet

    --------------
    Args:
        BC : hyperparameter, Basic Channels
    --------------
    Returns:
        input_tensor tf.Tensor : Input point cloud (image_height: 480, image_width: 640, 4).
        output_tensor: tf.Tensor : output_tensor to be compared to gt (image_height: 480, image_width: 640, 3)
    """

    input_tensor = tf.keras.Input((480, 640, 4)) #(480, 640, 4)
    relu = tf.keras.layers.ReLU()
    layer00 = tf.expand_dims(tf.pad(input_tensor, ((94, 94), (94, 94), (0, 0)), mode="SYMMETRIC"), axis=0) #(1, 668, 828, 4)
    layer01 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(3, 3))(layer00)) #(1, 666, 826, BC)
    layer02 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(3, 3))(layer01)) #(1, 664, 824, BC)
    layer10 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer02) #(1, 332, 412, BC)
    layer11 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer10)) #(1, 330, 410, 2BC)
    layer12 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer11)) #(1, 328, 408, 2BC)
    layer20 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer12) #(1, 164, 204, 2BC)
    layer21 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer20)) #(1, 162, 202, 4BC)
    layer22 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer21)) #(1, 160, 200, 4BC)
    layer30 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer22) #(1, 80, 100, 4BC)
    layer31 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer30)) #(1, 78, 98, 8BC)
    layer32 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer31)) #(1, 76, 96, 8BC)
    layer40 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(layer32) #(1, 38, 48, 8BC)
    layer41 = relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer40)) #(1, 36, 46, 16BC)
    layer42 = relu(tf.keras.layers.Conv2D(filters=16*BC, kernel_size=(3, 3))(layer41)) #(1, 34, 44, 16BC)
    layer32_cropped = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(layer32) #(1, 68, 88, 8BC)
    layer42_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer42) #(1, 68, 88, 16BC)
    layer42_conved22 = tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(2, 2), padding="same")(layer42_upsample) #(1, 68, 88, 8BC)
    layer50 = tf.keras.layers.Concatenate(axis=1)((layer32_cropped, layer42_conved22)) #(1, 68, 88, 16BC)
    layer51 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer50)) #(1, 66, 86, 8BC)
    layer52 = relu(tf.keras.layers.Conv2D(filters=8*BC, kernel_size=(3, 3))(layer51)) #(1, 64, 84, 8BC)
    layer22_cropped = tf.keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(layer22) #(1, 128, 168, 4BC)
    layer52_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer52) #(1, 128, 168, 8BC)
    layer52_conved22 = tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(2, 2), padding="same")(layer52_upsample) #(1, 128, 168, 4BC)
    layer60 = tf.keras.layers.Concatenate(axis=1)((layer22_cropped, layer52_conved22)) #(1, 128, 168, 8BC)
    layer61 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer60)) #(1, 126, 166, 4BC)
    layer62 = relu(tf.keras.layers.Conv2D(filters=4*BC, kernel_size=(3, 3))(layer61)) #(1, 124, 164, 4BC)
    layer12_cropped = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(layer12) #(1, 248, 328, 2BC)
    layer62_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer62) #(1, 248, 328, 4BC)
    layer62_conved22 = tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(2, 2), padding="same")(layer62_upsample) #(1, 248, 328, 2BC)
    layer70 = tf.keras.layers.Concatenate(axis=1)((layer12_cropped, layer62_conved22)) #(1, 248, 328, 4BC)
    layer71 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer70)) #(1, 246, 326, 2BC)
    layer72 = relu(tf.keras.layers.Conv2D(filters=2*BC, kernel_size=(3, 3))(layer71)) #(1, 244, 324, 2BC)
    layer02_cropped = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(layer02) #(1, 488, 648, BC)
    layer72_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))(layer72) #(1, 488, 648, 2BC)
    layer72_conved22 = tf.keras.layers.Conv2D(filters=BC, kernel_size=(2, 2), padding="same")(layer72_upsample) #(1, 488, 648, BC)
    layer80 = tf.keras.layers.Concatenate(axis=1)((layer02_cropped, layer72_conved22)) #(1, 488, 648, 2BC)
    layer81 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(5, 5))(layer80)) #(1, 484, 644, BC)
    layer82 = relu(tf.keras.layers.Conv2D(filters=BC, kernel_size=(5, 5))(layer81)) #(1, 480, 640, BC)
    output_tensor = tf.squeeze(tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1))(layer82)) #(480, 640, 3)

    return input_tensor, output_tensor

    # # Input layer
    # # (B, 20000, 3)
    # input_pc = tf.keras.Input(
    #     shape=(config["RAW_NUM_POINTS"], 3),
    #     name='input_point_cloud')


    # # Set Abstraction layers
    # # (B, 2048, 3), (B, 2048, 320)
    # sa_xyz_0, sa_points_0 = Pointnet_SA_MSG(
    #     npoint=config["SA_NPOINT_0"],
    #     radius_list=config["SA_RADIUS_LIST_0"],
    #     nsample_list=config["SA_NSAMPLE_LIST_0"],
    #     mlp=config["SA_MLP_LIST_0"],
    #     use_xyz=True,
    #     activation=tf.nn.relu,
    #     bn=False)(input_pc, None)
    # # (B, 512, 3), (B, 512, 640)
    # sa_xyz_1, sa_points_1 = Pointnet_SA_MSG(
    #     npoint=config["SA_NPOINT_1"],
    #     radius_list=config["SA_RADIUS_LIST_1"],
    #     nsample_list=config["SA_NSAMPLE_LIST_1"],
    #     mlp=config["SA_MLP_LIST_1"],
    #     use_xyz=True,
    #     activation=tf.nn.relu,
    #     bn=False)(sa_xyz_0, sa_points_0)
    # # (B, 128, 3), (B, 128, 640)
    # sa_xyz_2, sa_points_2 = Pointnet_SA_MSG(
    #     npoint=config["SA_NPOINT_2"],
    #     radius_list=config["SA_RADIUS_LIST_2"],
    #     nsample_list=config["SA_NSAMPLE_LIST_2"],
    #     mlp=config["SA_MLP_LIST_2"],
    #     use_xyz=True,
    #     activation=tf.nn.relu,
    #     bn=False)(sa_xyz_1, sa_points_1)

    # # Global feature layer
    # # (B, 1, 3), (1024...?)
    # sa_xyz_3, sa_points_3 = Pointnet_SA(
    #     npoint=None,
    #     radius=None,
    #     nsample=None,
    #     mlp=config["SA_MLP_GROUP_ALL"],
    #     group_all=True,
    #     knn=False,
    #     use_xyz=True,
    #     activation=tf.nn.relu,
    #     bn=False)(sa_xyz_2, sa_points_2)

    # # Feature propagation layers.
    # # (B, 128, 256)
    # fp_points_2 = Pointnet_FP(
    #     mlp=config["FP_MLP_0"],
    #     activation=tf.nn.relu,
    #     bn=False)(sa_xyz_2, sa_xyz_3, sa_points_2, sa_points_3)
    # # (B, 512, 128)
    # fp_points_1 = Pointnet_FP(
    #     mlp=config["FP_MLP_1"],
    #     activation=tf.nn.relu,
    #     bn=False)(sa_xyz_1, sa_xyz_2, sa_points_1, fp_points_2)
    # # (B, 2048, 128)
    # fp_points_0 = Pointnet_FP(
    #     mlp=config["FP_MLP_2"],
    #     activation=tf.nn.relu,
    #     bn=False)(sa_xyz_0, sa_xyz_1, sa_points_0, fp_points_1)

    # # Output from the pointnet++
    # # (B, 2048, 3)
    # # (B, 2048, 1024)
    # output_pc = sa_xyz_0
    # output_feature = fp_points_0

    # # grasp_score
    # # (B, 2048, 1)
    # grasp_score = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid')(output_feature)
    # grasp_score = tf.keras.layers.LeakyReLU()(grasp_score)
    # grasp_score = tf.keras.layers.Dropout(rate=0.5)(grasp_score)
    # grasp_score = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1, padding='valid')(grasp_score)
    # grasp_score = tf.keras.activations.sigmoid(grasp_score)
    # grasp_score = tf.squeeze(grasp_score, axis=-1)

    # # grasp_approach
    # # (B, 2048, 3)
    # grasp_approach = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid')(output_feature)
    # grasp_approach = tf.keras.layers.LeakyReLU()(grasp_approach)
    # grasp_approach = tf.keras.layers.Dropout(rate=0.5)(grasp_approach)
    # grasp_approach = tf.keras.layers.Conv1D(filters=3, kernel_size=1, strides=1, padding='valid')(grasp_approach)
    # grasp_approach = tf.math.l2_normalize(grasp_approach, axis=-1)


    # return input_pc, (output_pc, grasp_score, grasp_approach)


#TODO: 나머지 loss 함수 작성하고 train 코드도 재작성하고 가상 데이터 생성하고 테스트 해보기. 또한 loss 함수를 변경한다거나 padding 방식도 변경해보기.

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


class Unet(tf.keras.models.Model):
    def __init__(self, inputs, outputs):
        super(Unet, self).__init__(inputs, outputs)

    def compile(self, optimizer='adam', run_eagerly=None):
        super(Unet, self).compile(optimizer=optimizer, run_eagerly=run_eagerly)

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