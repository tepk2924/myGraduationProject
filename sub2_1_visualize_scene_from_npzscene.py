import trimesh
import numpy as np
import pickle
import random
import os

import trimeshVisualize
from common1_4_graspdata import GraspData

if __name__ == "__main__":
    scene = trimeshVisualize.Scene()
    #계산은 밀리미터 단위였지만, 일단 시각화는 미터단위, 즉 1/1000 스케일로 줄인다.
    table_mesh = trimesh.primitives.Box([1.0, 1.2, 0.6])
    scene.plot_mesh(table_mesh, color=[127, 127, 127, 255], id="table")
    scene.plot_vector(np.array([0, 0, 0]), np.array([1, 0, 0]), color = [255, 0, 0, 255], radius_cyl = .01)
    scene.plot_vector(np.array([0, 0, 0]), np.array([0, 1, 0]), color = [0, 255, 0, 255], radius_cyl = .01)
    scene.plot_vector(np.array([0, 0, 0]), np.array([0, 0, 1]), color = [0, 0, 255, 255], radius_cyl = .01)

    grasp_path = input(".pkl 파일이 들어있는 폴더의 디렉토리 입력 : ")
    scene_path = input("불러올 npz 파일의 디렉토리 입력 : ")
    with np.load(scene_path, allow_pickle=True) as scene_data:

        # Extrude data
        scene_grasps_tf = scene_data["scene_grasps_tf"]
        scene_grasps_scores = scene_data["scene_grasps_scores"]
        object_names = scene_data["object_names"]
        obj_transforms = scene_data["obj_transforms"]
        obj_grasp_idcs = scene_data["obj_grasp_idcs"]

    obj_idx = 0

    for obj_name, obj_trans in zip(object_names, obj_transforms):
        with open(os.path.join(grasp_path, f"{obj_name}.pkl"), "rb") as f:
            graspdata:GraspData = pickle.load(f)
        current_mesh = graspdata.mesh
        current_mesh.apply_transform(obj_trans)
        #여기도 1/1000 스케일
        current_mesh = current_mesh.apply_scale(0.001)
        #여기서 알아낸 사실 : plot_mesh에서 아이디가 같으면 한 개만 렌더링이 됨 ㅋㅋ
        scene.plot_mesh(current_mesh, color=[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255], id=f"obj_{obj_idx:03d}")
        obj_idx += 1

    for scene_grasp_tf in scene_grasps_tf:
        grasp_point = np.array([0, 0, 0])
        #일단 10mm == 0.01m 의 화살표로 처리.
        grasp_dir = np.array([0, 0, 10])
        points_transformed = trimesh.transform_points(
            [grasp_point, grasp_dir], scene_grasp_tf)
        grasp_point = np.array(points_transformed[0])
        grasp_dir = np.array(points_transformed[1])
        grasp_point = np.array([[.001, 0, 0],
                                [0, .001, 0],
                                [0, 0, .001]]) @ grasp_point
        grasp_dir = np.array([[.001, 0, 0],
                              [0, .001, 0],
                              [0, 0, .001]]) @ grasp_dir
        
        scene.plot_vector(grasp_point, grasp_dir, color=[0, 0, 0, 255],
                          radius_cyl=.001, arrow=True)
    
    print(f"{len(object_names)}개의 오브젝트와 테이블 시각화 중")
    print(f"오브젝트 이름 : {object_names}")
    print(obj_grasp_idcs)
    scene.display()