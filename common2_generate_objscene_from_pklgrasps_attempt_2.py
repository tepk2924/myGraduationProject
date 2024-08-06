import os
import trimesh
import trimesh.path
import trimesh.transformations as tra
import numpy as np
import random
import time
import signal
import pickle
import os

from trimesh import creation
from typing import Dict, List
from os import walk
# from trimesh.permutate import transform
from multiprocessing import freeze_support

from common1_4_graspdata import GraspData
from common2_1_scenedata import SceneData
import multiprocessing_utils

def transform_grasp(grasp_dict, tf, add_transform_score = True):
    """
    Given a new transformation matrix for the object, transform all the grasps for that object.

    Parameters
    ----------
    grasp_dict : (dict) : the dictionary containing grasp data for the given object
            - "tf"  : np.array() [num_grasps, 4, 4] 
                4x4 transformation matrices of the object grasps
            - "score" : np.array() [num_grasps]
                The score of each grasp
    tf : np.array() [4, 4]
        New transformation matrix for the object
    add_transform_score : bool, optional
        Scales the score of the grasps based on their new orientation.
        (grasp with z-axis pointing down get score of 0, grasp with z-axis pointing up get score of 1), by default True

    Returns
    -------
    grasp_dict : (dict)
        Updated grasp dictionary.
    """

    new_grasp_tf = []
    new_grasp_scores = []
    
    grasp_tf = grasp_dict["tf"] # [num_grasps, 4, 4]
    if np.shape(grasp_tf) == (0,):
        return grasp_dict
    # Calculate new tf for the grasp np.dot(tf, grasp_tf) 
    new_grasp_tf = np.matmul(tf, grasp_tf)
    # Calculate new grasp score
    if add_transform_score:
        temp_points = np.zeros((new_grasp_tf.shape[0], 4))
        temp_points[:, -2] = 1
        temp_points = np.einsum("ijk,ik->ij", new_grasp_tf, temp_points)
        approach_vector = temp_points[:, -2]
        new_grasp_scores = grasp_dict["scores"] * (0.5 + 0.5 * approach_vector)

    grasp_dict["tf"] = new_grasp_tf
    if add_transform_score:
        grasp_dict["scores"] = new_grasp_scores

    return grasp_dict

class Scene(object):
    def __init__(self) -> None:
        """
        Scene은 Mesh와 변환행렬의 집합체로 정의됨
        ---------------
        """
        self._objects = {}
        self._poses = {}
        self._support_objects = []

        #octomap, python-fcl 설치 필요
        self.collision_manager = trimesh.collision.CollisionManager()

    def add_object(self, obj_id, obj_mesh:trimesh.base.Trimesh, pose, support=False):
        """
        Scene에 Mesh를 추가하는 메소드
        --------------
        Args:
            obj_id (str): Mesh의 이름
            obj_mesh (trimesh.Trimesh): Mesh
            pose (np.ndarray): 변환행렬
            support (bool, optional): Indicates whether this object has support surfaces for other objects. Defaults to False.
        --------------
        """
        self._objects[obj_id] = obj_mesh
        self._poses[obj_id] = pose
        if support:
            self._support_objects.append(obj_mesh)

        self.collision_manager.add_object(
            name=obj_id, mesh=obj_mesh, transform=pose)

    def _get_random_stable_pose(self, stable_poses, stable_poses_probs):
        """
        Mesh가 취할 수 있는 안정적인 자세를 하나 골라서 리턴
        ----------------
        Args:
            stable_poses (list[np.ndarray]): List of stable poses as 4x4 matrices.
            stable_poses_probs (list[float]): List of probabilities.
        Returns:
            np.ndarray: homogeneous 4x4 matrix
        """
        index = np.random.choice(len(stable_poses), p=stable_poses_probs)
        inplane_rot = tra.rotation_matrix(
            angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1]
        )
        return inplane_rot.dot(stable_poses[index])

    def _get_support_polygons(self, min_area=0.01, gravity=np.array([0, 0, -1.0]), erosion_distance=0.02):
        """
        Extract support facets by comparing normals with gravity vector and checking area.
        ----------------
        Args:
            min_area (float, optional): Minimum area of support facets [m^2]. Defaults to 0.01.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.
        Returns:
            list[trimesh.path.polygons.Polygon]: list of support polygons.
            list[np.ndarray]: list of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
        """
        # Check if gravity is a unit vector
        assert np.isclose(np.linalg.norm(gravity), 1.0)

        support_polygons = []
        support_polygons_T = []

        # Add support plane if it is set (although not infinite)
        support_meshes:list[trimesh.base.Trimesh] = self._support_objects

        for obj_mesh in support_meshes:
            # get all facets that are aligned with -gravity and bigger than min_area
            support_facet_indices = np.argsort(obj_mesh.facets_area)
            support_facet_indices = [
                idx
                for idx in support_facet_indices
                if np.isclose(obj_mesh.facets_normal[idx].dot(-gravity), 1.0, atol=0.5)
                and obj_mesh.facets_area[idx] > min_area
            ]

            for inds in support_facet_indices:
                index = inds
                normal = obj_mesh.facets_normal[index]
                origin = obj_mesh.facets_origin[index]

                T = trimesh.geometry.plane_transform(origin, normal)
                vertices = trimesh.transform_points(
                    obj_mesh.vertices, T)[:, :2]

                # find boundary edges for the facet
                edges = obj_mesh.edges_sorted.reshape((-1, 6))[
                    obj_mesh.facets[index]
                ].reshape((-1, 2))
                group = trimesh.grouping.group_rows(edges, require_count=1)

                # run the polygon conversion
                polygon = trimesh.path.polygons.edges_to_polygons(
                    edges=edges[group], vertices=vertices
                )

                assert len(polygon) == 1

                # erode to avoid object on edges
                polygon[0] = polygon[0].buffer(-erosion_distance)

                if not polygon[0].is_empty and polygon[0].area > min_area:
                    support_polygons.append(polygon[0])
                    support_polygons_T.append(
                        trimesh.transformations.inverse_matrix(T))

        return support_polygons, support_polygons_T

class TableScene(Scene):
    
    def __init__(self, data_dir, lower_table=0.00002):
        """
        Scene의 자식 글래스. Mesh를 올려놓을 테이블을 하나 가정함.
        --------------
        Keyword Arguments:
            lower_table {float} -- lower table to permit slight grasp collisions between table and object/gripper (default: {0.02})
        --------------
        """

        super().__init__()

        tf = trimesh.transformations.translation_matrix(np.array([0, 0, 0.017]))
        #If this gripper mesh is too small (such as being radius ~ 0.001 and height ~ 0.001), segmentation fault (core dumped) happens.
        self.gripper_mesh = creation.cylinder(radius=0.001, height=0.03, transform=tf)
        print("Gripper mesh loaded")

        # Table
        self._table_dims = [0.75, 0.75, 0.6]
        self._table_support = [0.75, 0.75, 0.6]
        self._table_pose = np.eye(4)
        self.table_mesh = trimesh.creation.box(self._table_dims)
        self.table_support = trimesh.creation.box(self._table_support)

        # Obj meshes
        self.data_dir = data_dir

        self._lower_table = lower_table

        self._scene_count = 0

    def _filter_colliding_grasps(self, transformed_grasps):
        """
        다른 오브젝트와 충돌하는 grasp을 거르는 메소드
        ----------------
        Arguments:
            transformed_grasps {np.ndarray} -- Nx4x4 grasps
            transformed_contacts {np.ndarray} -- 2Nx3 contact points
        Returns:
            {"tf" : np.ndarray, "scores" : np.ndarray} -- Mx4x4 filtered grasps, Mx2x3 filtered contact points
        """
        filtered_grasps = []
        filtered_scores = []
        for grasp_tf, grasp_score in zip(transformed_grasps["tf"], transformed_grasps["scores"]):
            if not self.is_colliding(self.gripper_mesh, grasp_tf):
                filtered_grasps.append(grasp_tf)
                filtered_scores.append(grasp_score)
        return {"tf": np.array(filtered_grasps).reshape(-1, 4, 4), "scores": np.array(filtered_scores)}

    def get_random_object(self):
        """
        successful_grasp 폴더에 있는 pkl파일을 로드
        --------------
        Returns:
            obj_name, obj_mesh, obj_grasp : 각각 pkl파일의 이름, mesh, graspdata
        """
        # Get an object
    
        pklgrasp_file_names = os.listdir(self.data_dir)
        choosen_pklgrasp_file_name = random.choice(pklgrasp_file_names)
        with open(os.path.join(data_dir, choosen_pklgrasp_file_name), "rb") as f:
            graspdata:GraspData = pickle.load(f)

        # load mesh
        obj_mesh = trimesh.load(graspdata.obj_path)
        obj_path = graspdata.obj_path
        # load coresponding grasp
        obj_grasp = graspdata.grasp_info

        # mesh_mean = np.mean(obj_mesh.vertices, 0, keepdims=True)
        # obj_mesh.vertices -= mesh_mean
        return obj_path, obj_mesh, obj_grasp

    def find_object_placement(self, obj_mesh:trimesh.base.Trimesh, max_iter):
        """
        Mesh를 support_surface상의 support_polygon에 올려놓는 메소드
        ------------------
        Args:
            obj_mesh (trimesh.Trimesh): Object mesh to be placed.
            max_iter (int): Maximum number of attempts to place to object randomly.
        Raises:
            RuntimeError: In case the support object(s) do not provide any support surfaces.
        Returns:
            bool: Whether a placement pose was found.
            np.ndarray: Homogenous 4x4 matrix describing the object placement pose. Or None if none was found.
        """
        support_polys, support_T = self._get_support_polygons()
        if len(support_polys) == 0:
            raise RuntimeError("No support polygons found!")

        # get stable poses for object
        stable_obj = obj_mesh.copy()
        stable_obj.vertices -= stable_obj.center_mass
        stable_poses, stable_poses_probs = stable_obj.compute_stable_poses(
            threshold=0, sigma=0, n_samples=3
        )
        #stable_poses, stable_poses_probs = obj_mesh.compute_stable_poses(threshold=0, sigma=0, n_samples=5)
        # Sample support index
        support_index = max(enumerate(support_polys),
                            key=lambda x: x[1].area)[0]

        iter = 0
        colliding = True
        while iter < max_iter and colliding:

            # Sample position in plane
            pts = trimesh.path.polygons.sample(
                support_polys[support_index], count=1
            )

            # To avoid collisions with the support surface
            pts3d = np.append(pts, 0)

            # Transform plane coordinates into scene coordinates
            placement_T = np.dot(
                support_T[support_index],
                trimesh.transformations.translation_matrix(pts3d),
            )

            pose = self._get_random_stable_pose(
                stable_poses, stable_poses_probs)

            placement_T = np.dot(
                np.dot(placement_T,
                       pose), tra.translation_matrix(-obj_mesh.center_mass)
            )

            # Check collisions
            colliding = self.is_colliding(obj_mesh, placement_T)

            iter += 1


        return not colliding, placement_T if not colliding else None

    def is_colliding(self, mesh, transform, eps=1e-6):
        """
        Whether given mesh collides with scene
        --------------
        Arguments:
            mesh {trimesh.Trimesh} -- mesh 
            transform {np.ndarray} -- mesh transform
        Keyword Arguments:
            eps {float} -- minimum distance detected as collision (default: {1e-3})
        --------------
        Returns:
            [bool] -- colliding or not
        """
        dist = self.collision_manager.min_distance_single(
            mesh, transform=transform)
        return dist < eps

    def arrange(self, num_obj, max_iter=100, time_out=8):
        """
        Arrange random table top scene with contact grasp annotations
        --------------
        Arguments:
            num_obj {int} -- number of objects
        Keyword Arguments:
            max_iter {int} -- maximum iterations to try placing an object (default: {100})
            time_out {int} -- maximum time to try placing an object (default: {8})
        --------------
        Returns:
            scene_filtered_grasps {list} -- list of valid grasps for the scene. That is grasps which the gripper can reach.
            scene_filtered_scores {list} -- A corresponding list of grasp scores
            object_names {list} -- names of all objects in the scene
            obj_transforms {list} --  transformation matrices for all the scene objects
            obj_grasp_idcs {list} : List of ints indicating to which object some grasps belong to.
        """

        # Add table
        self._table_pose[2, 3] -= self._lower_table
        self.add_object('table', self.table_mesh, self._table_pose)

        self._support_objects.append(self.table_support)

        object_transforms = []
        object_grasps = []
        total_success = 0
        total_attempt = 0

        while total_success < num_obj and total_attempt < 2*num_obj:
            total_attempt += 1
            obj_filepath, obj_mesh, obj_grasp = self.get_random_object()
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(10)
            try:
                success, placement_T = self.find_object_placement(
                    obj_mesh, max_iter)
                
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            
            except Exception as exc:
                print(exc, " after {} seconds!".format(time_out))
                continue
            signal.alarm(0)
            if success:
                anonymous = f"{random.randint(0, (1 << 32) - 1):08x}"
                obj_name = obj_filepath + anonymous
                self.add_object(obj_name, obj_mesh, placement_T)
                # obj_scales.append(
                #     float(random_grasp_path.split('_')[-1].split('.h5')[0]))
                object_transforms.append(placement_T)
                object_grasps.append(obj_grasp)
                total_success += 1
            else:
                print("Couldn't place object"," after {} iterations!".format(max_iter))

        scene_filtered_grasps = []
        scene_filtered_scores = []
        obj_grasp_idcs = []
        grasp_count = 0

        for obj_transform, object_grasp in zip(object_transforms, object_grasps):
            transformed_obj_grasp = transform_grasp(object_grasp, obj_transform)
            # Expand first dimension to list of np arrays
            transformed_obj_grasp["tf"] = list(transformed_obj_grasp["tf"])
            transformed_obj_grasp["scores"] = list(transformed_obj_grasp["scores"])

            filtered_grasps = self._filter_colliding_grasps(transformed_obj_grasp)

            scene_filtered_grasps.append(filtered_grasps["tf"])
            scene_filtered_scores.append(filtered_grasps["scores"])
            grasp_count += len(filtered_grasps["tf"])
            obj_grasp_idcs.append(grasp_count)

        

        scene_filtered_grasps = np.concatenate(scene_filtered_grasps, 0)
        scene_filtered_scores = np.concatenate(scene_filtered_scores, 0)


        self._table_pose[2, 3] += self._lower_table
        self.set_mesh_transform('table', self._table_pose)

        # return scene_filtered_grasps, scene_filtered_scores, object_names, object_transforms, obj_grasp_idcs
        return scene_filtered_grasps, scene_filtered_scores, self._objects, self._poses

    def set_mesh_transform(self, name, transform):
        """
        Set mesh transform for collision manager
        --------------
        Arguments:
            name {str} -- mesh name
            transform {np.ndarray} -- 4x4 homog mesh pose
        --------------
        """
        self.collision_manager.set_transform(name, transform)
        self._poses[name] = transform

    def handler(self, signum, frame):
        raise Exception("Could not place object ")

    def reset(self):
        """
        --------------
        Reset, i.e. remove scene objects
        --------------
        """
        for name in self._objects:
            self.collision_manager.remove_object(name)
        self._objects = {}
        self._poses = {}
        self._support_objects = []

class SceneDatasetGenerator():
    def __init__(self,
                 data_dir,
                 save_dir,
                 gripper_path,
                 min_num_objects = 8,
                 max_num_objects = 13,
                 max_iterations = 100,
                 number_of_scenes_generating = 1
                 ) -> None:
        self._data_dir = data_dir
        self._save_dir = save_dir
        self._gripper_path = gripper_path
        self._min_num_objects = min_num_objects
        self._max_num_objects = max_num_objects
        self._max_iterations = max_iterations
        self._number_of_scenes_generating = number_of_scenes_generating
        self._initial_file_num = len(os.listdir(self._save_dir))
        self.fails = []
    
    def generate_save_scene(self, scene_id: str):
        """
        scene_id의 이름을 가진 scene을 지정된 폴더에 저장하는 메소드
        ----------
        Args:
        scene_id : int
            Scene의 이름

        Returns
        Success : bool
            Whether the scene was successfully generated.
        """
        start_time = time.time()
        print(f"Evaluating scene {scene_id}")
        
        # Create scene
        table_scene = TableScene(self._data_dir)
        table_scene.reset()
        num_objects = np.random.randint(self._min_num_objects, self._max_num_objects+1)

        
        try:
            scene_grasps_tf, scene_grasps_scores, obj_meshes, obj_poses= table_scene.arrange(num_objects, self._max_iterations)
            print(f"Arrange Finished for scene {scene_id}")
            self.save_scene(scene_id, scene_grasps_tf, scene_grasps_scores, obj_meshes, obj_poses)
            print(f"Created {scene_id} with {len(obj_meshes) - 1} objects, time taken {time.time()-start_time:.2f}secs ({(len(os.listdir(self._save_dir)) - self._initial_file_num)}/{self._number_of_scenes_generating})")
            return True
        
        except KeyboardInterrupt:
            raise KeyboardInterrupt

        except:
            self.fails.append(scene_id)
            return False
        
    def generate_save_N_scenes(self, n_processors = 1):
        """
        입력받은 갯수의 Scene을 생성하고 저장하는 메소드
        ----------
        Args:
        number_of_scenes_generating : int
        n_processors : int, optional
            Number of processors to use, by default 1
        overwrite : bool, optional
            Whether to overwrite existing scenes, by default False
        ----------
        """

        start_time = time.time()
        scene_id_list = [f"scene_{random.randint(0, (1<<32) - 1):08x}" for _ in range(self._number_of_scenes_generating)]
        
        if n_processors == 1:
            for scene_id in scene_id_list:
                self.generate_save_scene(scene_id)
        else:
            out = multiprocessing_utils.run_multiprocessing(self.generate_save_scene,
                                            scene_id_list, n_processors)
        print(f"Evaluated {len(scene_id_list)} scenes, time taken {time.time()-start_time}")
        print(f"Failed to evaluate {self.fails} scenes")
        return True
    
    def save_scene(self, 
                scene_id,
                scene_grasps_tf,
                scene_grasps_scores,
                obj_meshes: Dict[str, trimesh.base.Trimesh],
                obj_poses_dict: Dict[str, np.ndarray]):
        """
        지정된(self._save_dir) 폴더에 scene 저장하는 메소드
        --------------
        Arguments:
        scene_id {int} : Scene index
        scene_grasps_tf {list} : A list of grasps "tf".
        scene_grasps_scores {list} : A corresponding list of scores for individual grasps.
        """
        obj_names = obj_poses_dict.keys()
        obj_poses_list = []
        obj_filepaths_list = []
        for obj_name in obj_names:
            if obj_name != 'table':
                obj_poses_list.append(obj_poses_dict[obj_name])
                obj_filepaths_list.append(obj_name[:-8])

        scenedata = SceneData(obj_filepaths_list, obj_poses_list, scene_grasps_tf, scene_grasps_scores)
        with open(os.path.join(self._save_dir, f"pklscene_{scene_id}.pkl"), "wb") as f:
            pickle.dump(scenedata, f)

if __name__ == "__main__":
    freeze_support()
    number_of_scenes_generating = int(input("생성할 pklscene의 갯수 입력 : "))
    data_dir = input("pklgrasp 파일이 들어있는 폴더의 디렉토리 입력 : ")
    save_dir = input("pklscene을 저장할 폴더의 디렉토리 입력 : ")
    # Generate a dataset of 3D Scenes
    dg = SceneDatasetGenerator(data_dir=data_dir,
                               save_dir=save_dir,
                               gripper_path=None,
                               min_num_objects=8,
                               max_num_objects=13,
                               max_iterations=100,
                               number_of_scenes_generating=number_of_scenes_generating
    )
    
    # Generate a dataset of 3D Scenes
    dg.generate_save_N_scenes(n_processors=8)