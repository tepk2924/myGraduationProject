import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import trimesh

from trimeshVisualize import Scene

def visualize_network_input(pc, gt, scene = None, return_scene = False):
    """ Given a point cloud and a ground truth grasp, visualize the point cloud and the grasp.
    If a batch of data is provided only the first batch element will get shown

    Arguments:
    ------------
    pc = ({B}, N, 3) np.array
    gt = (({B}, N) np,array, ({B}, N, 3) np.array)
    """
    
    gt_scores, gt_app = gt

    if isinstance(pc, tf.Tensor):
        pc = pc.numpy()
    if isinstance(gt_app, tf.Tensor):
        gt_app = gt_app.numpy()
    if isinstance(gt_scores, tf.Tensor):
        gt_scores = gt_scores.numpy()

    if pc.ndim == 3:
        # print("Showing only the first batch element")
        pc, gt_scores, gt_app = pc[0], gt_scores[0], gt_app[0]

    grasp_tf = grasps_to_tf(pc, gt_app)

    maskM1 = np.where(gt_scores == -1)
    mask0 = np.where(gt_scores == 0)
    mask1 = np.where(gt_scores == 1)

    if scene is None:
        my_scene = Scene()
    print("빨강색 점 : ground_truth == -1, 테이블")
    print("초록색 점 : ground_truth ==  0, 물체")
    print("파랑색 점 : ground_truth == +1, 물체 위 형성된 grasp에 가까운 지점들")
    my_scene.plot_point_cloud(pc[maskM1], color=[255, 0, 0, 255])
    my_scene.plot_point_cloud(pc[mask0], color=[0, 255, 0, 255])    
    my_scene.plot_point_cloud(pc[mask1], color=[0, 0, 255, 255])

    print("파랑색 선 : groun_truth == +1 인 grasp")
    my_scene.plot_grasp(grasp_tf[mask1], gt_scores[mask1], color=[0, 0, 255, 255])
    if return_scene:
        return my_scene
    else:
        my_scene.display()
        return None

def visualize_network_output(pc, score, app, scene = None, return_scene = False):
    """ Given a point cloud and a ground truth grasp, visualize the point cloud and the grasp.
    If a batch of data is provided only the first batch element will get shown

    Arguments:
    ------------
    pc = ({B}, N, 3) np.array
    app = ({B}, N, 3) np.array
    score = ({B}, N) np.array
    """

    if isinstance(pc, tf.Tensor):
        pc = pc.numpy()
    if isinstance(app, tf.Tensor):
        app = app.numpy()
    if isinstance(score, tf.Tensor):
        score = score.numpy()

    if pc.ndim == 3:
        # print("Showing only the first batch element")
        pc, app, score = pc[0], app[0], score[0]

    grasp_tf = grasps_to_tf(pc, app)

    print(f"최소 스코어 : {np.min(score)}")
    print(f"제 1사분위 값 : {np.quantile(score, 0.25)}")
    print(f"중앙값 : {np.quantile(score, 0.5)}")
    print(f"제 3사분위 값 : {np.quantile(score, 0.75)}")
    print(f"최대 스코어 : {np.max(score)}")

    per10 = np.quantile(score, .1)
    per90 = np.quantile(score, .9)

    #grasp_score의 값을 10% 부분을 0, 90% 부분을 1로 잡고, 절사한 값으로 바꿈
    grasp_score_nor = (score - per10) / (per90 - per10)
    grasp_score_nor = np.where(grasp_score_nor > 1, 1, np.where(grasp_score_nor < 0, 0, grasp_score_nor))
   
    if scene is None:
        my_scene = Scene()
    else:
        my_scene = scene
    print("검정색 점 : model이 score를 평가한 지점들")
    my_scene.plot_point_cloud(pc, color=[0, 0, 0, 255])
    #my_scene.plot_grasp(grasp_tf, score, color=[0, 0, 0, 255])

    print("연보라색 ~ 청록색 작은 선 : model이 판단하는 grasp, 청록색으로 갈 수록 점수가 높음")
    for i in range(len(grasp_tf)):
        if round(score[i], 4) == 0:
            continue
        grasp_point = np.array([0, 0, 0])
        #일단 10mm의 화살표로 처리.
        grasp_dir = np.array([0, 0, 0.01])
        points_transformed = trimesh.transform_points(
            [grasp_point, grasp_dir], grasp_tf[i])
        grasp_point = np.array(points_transformed[0])
        grasp_dir = np.array(points_transformed[1])
        id = my_scene.plot_vector(grasp_point, grasp_dir,
                                  color=[255 - int(255*grasp_score_nor[i]), int(255*grasp_score_nor[i]), 255, 255],
                                  radius_cyl=0.001, arrow=True)

    if return_scene:
        return my_scene
    else:
        my_scene.display()
        return None

def grasps_to_tf(pc, approach):
    """Transforms a combination of a point and approach vector to a 4x4 tf matrix
    Arguments:
    -----------
    pc = (N, 3) np.array
    approach = (N, 3) np.array

    Returns:
    -----------
    (N, 4, 4) np.array
    """

    z = approach
    x = np.cross(np.array([0, 1, 0]), z)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    grasp_tf = np.zeros((pc.shape[0], 4, 4))
    rot = np.stack([x, y, z], axis=-1)
    grasp_tf[:, :3, :3] = rot
    grasp_tf[:, :3, 3] = pc
    grasp_tf[:, 3, 3] = 1
    
    return grasp_tf
    
def best_grasp(pred_pc, pred_scores, pred_approach):
    """
    Given the network outputs, return the best grasps
    """
    # Get the max score
    max_grasp_i = tf.argmax(pred_scores, axis = 1)
    location = tf.gather(pred_pc, max_grasp_i, axis=1, batch_dims=1) # (B, 1, 3)
    approach = tf.gather(pred_approach, max_grasp_i, axis=1, batch_dims=1) # (B, 1, 3)
    score = tf.gather(pred_scores, max_grasp_i, axis=1, batch_dims=1) # (B, 1)
    return location, approach, score