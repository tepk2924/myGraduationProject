
import os
import sys
import random
import copy
import numpy as np
import numpy.random as npr
from scipy.spatial.ckdtree import cKDTree


import trimesh
import trimesh.transformations as tra
import cv2

from render_utils import inverse_transform
from scene_renderer import SceneRenderer

def load_scene_data(scene_path):
    with np.load(scene_path, allow_pickle=True) as scene_data:

        # Extrude data
        scene_grasps_tf = scene_data["scene_grasps_tf"]
        scene_grasps_scores = scene_data["scene_grasps_scores"]
        object_names = scene_data["object_names"]
        object_transforms = scene_data["obj_transforms"]
        obj_grasp_idcs = scene_data["obj_grasp_idcs"]

    return scene_grasps_tf, scene_grasps_scores, object_names, object_transforms, obj_grasp_idcs

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))

def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than n points, it oversamples.
      Otherwise, it downsample the input pc to have n point points.
      use_farthest_point: indicates 
      
      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """

    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def vectorized_normal_computation(pc, neighbors):
    """
    Vectorized normal computation with numpy
    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        neighbors {np.ndarray} -- Nxkx3 neigbours
    Returns:
        [np.ndarray] -- Nx3 normal directions
    """
    diffs = neighbors - np.expand_dims(pc, 1)  # num_point x k x 3
    covs = np.matmul(np.transpose(diffs, (0, 2, 1)),
                     diffs)  # num_point x 3 x 3
    covs /= diffs.shape[1]**2
    # takes most time: 6-7ms
    eigen_values, eigen_vectors = np.linalg.eig(
        covs)  # num_point x 3, num_point x 3 x 3
    orders = np.argsort(-eigen_values, axis=1)  # num_point x 3
    orders_third = orders[:, 2]  # num_point
    directions = eigen_vectors[np.arange(
        pc.shape[0]), :, orders_third]  # num_point x 3
    dots = np.sum(directions * pc, axis=1)  # num_point
    directions[dots >= 0] = -directions[dots >= 0]
    return directions


class PointCloudReader:
    def __init__(self,
                 grasp_root_folder,
                 batch_size=3,
                 raw_num_points=20000,
                 intrinsics="zivid2",
                 use_uniform_quaternions = False,
                 elevation=(0, 70),  # The pitch of the camera in degrees
                 distance_range=(3, 4.5), # How far away the camera is in m
                 pc_augm_config=None,
                 depth_augm_config=None,
                 estimate_normals=False,
                 use_farthest_point = False
                 ) -> None:
        """
        PointCloudReader 클래스. .npz로 된 데이터를 받아서 raw_num_point의 갯수만큼 점들을 각 pointcloud에서 고른 다음 batch 갯수대로 묶어서 데이터셋을 생성하는 클래스.
        --------------
        Arguments:
            grasp_root_folder {str} -- 물체의 grasp을 나타내는 pkl파일이 들어있는 폴더의 경로
        Keyword arguments:
            batch_size {int} -- The size og the batches we want (default: 1)
            raw_num_points {int} -- how many points should point clouds have (default: 20000)
            intrinsics {dict} -- camera intrinsics (default: zivid2)
            use_uniform_quaternions {bool} -- Not yet implimented (default: False)
            elevation {tuple} -- How much the camera is tilted from vertical in degrees  (default: (-50,50))
            distance_range {tuple} -- The range of distance for the pose of camera from the table in m. (default: (3, 4.5))
            pc_augm_config {dict} -- (default: None)
                - "sigma"-> variance of std.dist. for the applied jitter 
                - "clip" -> max values for jitter
                - "occlusion_nclusters" -> number of clusters we split the pc into
                - "occlusion_nclusters_rate" -> probability of removal of a cluster
            depth_augm_config {dict} -- (default: None)
                - "sigma"-> variance of std.dist. for the depth augmentation
                - "clip" -> max values for depth augmentation
                - "gaussian_kernel" -> smoothing for depth augmentation
            estimate_normals {bool} -- not yet implemented (default: False)
            use_farthest_point {bool} -- if use furthest point sampling to filter points (default: False)
        --------------
        """

        self._grasp_root_folder = grasp_root_folder
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point

        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        # Pyrender uses OpenGL camera coordinates:
        # z_axis - away from camera scene
        # x_axis - right in image space
        # y_axis - up in image space

        if use_uniform_quaternions:
            raise NotImplementedError
        else:
            #카메라의 가능한 각도 900(카메라의 테이블 기준 z축회전 각도 30개 * 카메라의 높이각도 30개)개에 대해 모든 변환행렬을 미리 저장
            self._cam_orientations = []
            self._elevation = np.pi*np.array(elevation)/180.
            for az in np.linspace(-np.pi/2, np.pi/2, 30):
                for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                    self._cam_orientations.append(
                        tra.euler_matrix(az, el, npr.normal(0, np.pi/2), axes="rzxz"))  # az, el, npr.normal(0, np.pi/2)
            

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center 
        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix
        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """
        
        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])
        dist_vec = np.array([0,0,distance,1])
        dist_vec_augm = cam_orientation.dot(dist_vec)

        #이제 카메라의 위치와 카메라의 방향과 카메라가 비추는 위치를 모두 고려해서 카메라의 pose(extrinsic을 생성.)
        extrinsics = np.eye(4)
        # pos
        extrinsics[0:3, 3] = dist_vec_augm[0:3]
        # table height
        extrinsics[2, 3] += self._renderer._table_dims[2]/2
        # pos + orientation
        extrinsics[0:3,0:3] = cam_orientation[0:3,0:3]

        return extrinsics

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config
        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
        Returns:
            np.ndarray -- augmented point cloud
        """

        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'],
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            #provider.jitter_point_cloud
            pc = jitter_point_cloud(pc[np.newaxis, :, :],
                                             sigma=self._pc_augm_config['sigma'],
                                             clip=self._pc_augm_config['clip'])[0]

        return pc[:, :3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config
        Arguments:
            depth {np.ndarray} -- depth map
        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth, (kernel, kernel), 0)
            depth[depth_copy == 0] = depth_copy[depth_copy == 0]

        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability
        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- no of cluster to remove
            occlusion_dropout_rate {float} -- prob of removal
        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc


        labels = farthest_points(
            pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(
            removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def get_scene_batch(self, scene_path, return_segmap=True, save=False):
        """
        Render a batch of scene point clouds
        --------------
        Keyword Arguments:
            scene_3d_idx {int} -- index of the scene to load (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})
        --------------
        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        # Prepare the arrays for where to store the scene data
        dims = 6 if self._estimate_normals else 3
        batch_data = np.empty(
            (self._batch_size, self._raw_num_points, dims), dtype=np.float32)
        cam_poses = np.empty((self._batch_size, 4, 4), dtype=np.float32)

        # Path to objects
        # objects_path = os.path.join(
        #     self._data_root_folder, "meshes", self._splits)

        # Get the saved data for that 3d scene
        #tepk2924 조한수정 : object_names는 object mesh가 들어있는 GraspData 클래스의 pickle 파일의 이름(확장자 .pkl 제외)이 된다.
        scene_grasp_tf, scene_grasp_scores, object_names, obj_tf, obj_grasp_idcs = load_scene_data(scene_path)

        # Get paths to all pkl objects in the scene
        grasp_paths = []
        for name in object_names:
            grasp_paths.append(os.path.join(self._grasp_root_folder, name)+".pkl")

        # Set the scene so we can render it
        # 주로 mm 단위를 m 단위로 변환한 다음에 렌더링을 하는 과정인 것 같다.
        # 이 메소드는 세 번째 인자로 스케일 값을 넣을 수 있는데, 기본 값이 mm에서 m 변환인 0.001이다.
        self.change_scene(grasp_paths, obj_tf)

        # Render the scene with camera in various positions
        batch_segmap, batch_obj_pcs = [], []
        for i in range(self._batch_size):
            # 0.005s
            pc_cam, pc_normals, camera_pose, depth = self.render_random_scene(
                estimate_normals=self._estimate_normals)

            # Here we can also render a segmentation map
            if return_segmap:
                segmap, _, obj_pcs = self._renderer.render_labels(
                    depth, object_names, render_pc=True)
                batch_obj_pcs.append(obj_pcs)
                batch_segmap.append(segmap)

            # Set the pc and cam_pose data
            batch_data[i, :, 0:3] = pc_cam[:, :3]
            if self._estimate_normals:
                batch_data[i, :, 3:6] = pc_normals[:, :3]
            cam_poses[i, :, :] = camera_pose

        scene_3d_idx = os.path.splitext(os.path.basename(scene_path))[0]

        # Save the generated batch
        if save:
            K = np.array([[616.36529541, 0, 310.25881958], [
                         0, 616.20294189, 236.59980774], [0, 0, 1]])
            data = {'depth': depth, 'K': K,
                    'camera_pose': camera_pose, 'scene_idx': scene_3d_idx}
            if return_segmap:
                data.update(segmap=segmap)
            np.savez('results/{}_acronym.npz'.format(scene_3d_idx), data)

        if return_segmap:
            return batch_data, cam_poses, scene_3d_idx, batch_segmap, batch_obj_pcs
        else:
            return batch_data, cam_poses, scene_3d_idx

    def render_random_scene(self, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations
        --------------
        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {np.array(4,4)} -- camera pose to render the scene from. (default: {None})
        --------------
        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(
                0, high=len(self._cam_orientations))
            camera_orientation = self._cam_orientations[viewing_index]
            camera_pose = self.get_cam_pose(camera_orientation)
        
        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        _, depth, _, camera_pose = self._renderer.render(
            in_camera_pose, render_pc = True, display = False)
        depth = self._augment_depth(depth)

        pc = self._renderer._to_pointcloud(depth)
        pc = regularize_pc_point_count(
            pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
        pc = self._augment_pc(pc)

        pc_normals = []

        return pc, pc_normals, camera_pose, depth

    def change_scene(self, obj_paths, obj_transforms, obj_scales = None):
        """
        Change pyrender scene
        --------------
        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models
        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        --------------
        """
        if obj_scales == None:
            obj_scales = [0.001 for obj in obj_paths]
        self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)

    def pc_to_world(self, pc, camera_pose):
        """
        Converts point cloud to world coordinates. The input PC is assumed to be in lefthanded OpenGL CS.
        --------------
        Arguments:
            pc {np.ndarray} -- point cloud (batch_size, npoints, 3)
            camera_pose {np.ndarray} -- camera pose (batch_size, 4, 4)
        --------------
        Returns:
            np.ndarray -- point cloud in world coordinates
        """
        
        for i in range(len(pc)):
            pc[i][:, 2] = -pc[i][:, 2]
            pc[i][:, 1] = -pc[i][:, 1]
            tf = np.eye(4)
            tf[0:3, 0:3] = camera_pose[i, 0:3, 0:3]
            tf[0:3, 3] = camera_pose[i, 0:3, 3]
            pc[i] = np.array(
                trimesh.transformations.transform_points(pc[i], tf))
       
        return pc

    def pc_convert_cam(self, poses):
        """
        Converts from OpenGL to OpenCV coordinates. Returns the transformation from world frame to OpenCV frame.
        
        :param cam_poses: (bx4x4) Camera poses in OpenGL format
        :param batch_data: (bxNx3) point clouds 
        :returns: (cam_poses, batch_data) converted
        """
        cam_poses = np.copy(poses)
        # OpenCV OpenGL conversion
        for j in range(len(cam_poses)):
            cam_poses[j, :3, 1] = -cam_poses[j, :3, 1]
            cam_poses[j, :3, 2] = -cam_poses[j, :3, 2]
            cam_poses[j] = inverse_transform(cam_poses[j])


        return cam_poses

    def get_ground_truth(self, pc, scene_path, pc_segmap = None, search_radius=0.006, threshold=0.2):
        """
        각 pc의 지점들마다 그 지점을 중심으로 search_radius 반경 내에 있는 grasp을 찾아서 그 grasp의 접근 벡터와 grasp score를 가져오는 메소드.
        --------------
        Arguments:
            pc {np.ndarray} -- point cloud
            scene_3d_idx {int} -- index of scene
        Keyword Arguments:
            search_radius {float} -- search radius for ground truth (default: {0.003})
            threshold {float} -- threshold for ground truth (default: {0.05})
        --------------
        Returns:
            gt_score -- binary score for each point
            gt_approach -- 3d approach vector for each point (Negative points have approach vector of (0,0,0))
        """

        # Get the saved data for that 3d scene
        scene_grasp_tf, scene_grasp_scores, object_names, obj_tf, obj_grasp_idcs = load_scene_data(scene_path)

        #tepk2924 조한 수정 : mm에서 m단위로 변환하는 코드 삽입
        scene_grasp_tf[:, 0:3, 3] *= 0.001

        # Positives mask
        mask = np.where(scene_grasp_scores > threshold)[0]
        # Positive grasps points
        positive_grasps_tf = np.empty([len(mask), 3])
        positive_grasps_tf = scene_grasp_tf[mask,:, :]
        positive_grasps_rot = scene_grasp_tf[mask, 0:3, 0:3]

        vectors = np.array([0, 0, 1])

        positive_grasps_vec = positive_grasps_rot*vectors
        positive_grasps_vec = np.sum(positive_grasps_vec, axis=2)


        # Make the output array for scores
        gt_score = np.full([pc.shape[0], pc.shape[1]], -1, dtype=np.int32)
        # Make the output array for approach
        #gt_approach의 값을 (0, 0, 1)로 초기화한다.
        gt_approach = np.zeros([pc.shape[0], pc.shape[1], 3])
        gt_approach[..., 2] = 1

        for i in range(len(pc)):
            my_tree = cKDTree(pc[i])
            if positive_grasps_tf[:, 0:3, 3].shape[0] == 0:
                raise Exception("No positive grasps found in the scene")
                
            #good_point는 pc(지터링 처리가 된 20000개의 지점) 내의 지점들 중 형성된 grasp으로부터 가까이에 있는 지점들이다
            good_point = my_tree.query_ball_point(
                positive_grasps_tf[:, 0:3, 3], search_radius)
            
            #pc_segmap(테이블이 아닌 물체 부분에 있는 점들)로부터 가까이에 있는 pc 내의 지점들은 gt_score가 0이다
            if pc_segmap is not None:
                object_points = my_tree.query_ball_point(pc_segmap[i], search_radius+0.002)
                for grasp_idx, point_list in enumerate(object_points):
                    for point in point_list:
                        gt_score[i, point] = 0

            #good_point의 지점들은 gt_score가 1이고, (0, 0, 1)이 아닌 gt_approach 값(==가까웠던 grasp의 접근벡터)을 부여받는다.
            for grasp_idx, point_list in enumerate(good_point):
                for point in point_list:
                    gt_approach[i, point,:] = positive_grasps_vec[grasp_idx, :]
                    gt_score[i, point] = 1

        return gt_score, gt_approach