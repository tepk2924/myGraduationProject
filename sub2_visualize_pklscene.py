import pickle
from trimeshVisualize import Scene
from common2_1_scenedata import SceneData
import numpy as np
import trimesh

if __name__ == "__main__":

    file_loc = input("input the path of pklscene file : ")
    my_scene = Scene()

    table = trimesh.creation.box([7.5, 7.5, 0.6])
    my_scene.plot_mesh(table, color=[127, 127, 127, 100])
    table_support = trimesh.creation.box([1.0, 1.0, 0.6])
    my_scene.plot_mesh(table_support, color=[255, 0, 0, 255])


    with open(file_loc, "rb") as f:
        scenedata:SceneData = pickle.load(f)
    
    obj_file_list = scenedata.obj_file_list
    obj_poses = scenedata.obj_poses
    grasps_tf = scenedata.grasps_tf
    grasps_score = scenedata.grasps_score

    for obj_path, pose in zip(obj_file_list, obj_poses):
        print(obj_path)
        mesh = trimesh.load(obj_path)
        mesh.apply_transform(pose)
        my_scene.plot_mesh(mesh)

    print("Displaying scene")

    print(f"minimum score : {np.min(grasps_score):.5f}")
    print(f"Maximum score : {np.max(grasps_score):.5f}")

    per10 = np.quantile(grasps_score, .1)
    per90 = np.quantile(grasps_score, .9)
    grasp_score_nor = (grasps_score - per10) / (per90 - per10)
    grasp_score_nor = np.where(grasp_score_nor > 1, 1,
                               np.where(grasp_score_nor < 0, 0, grasp_score_nor))
    rendered_grasps = 0
    for i in range(len(grasps_tf)):
        if round(grasps_score[i], 4) == 0:
            continue
        grasp_point = np.array([0, 0, 0])
        #Single grasp as arrow with length of 0.01m.
        grasp_dir = np.array([0, 0, 0.01])
        points_transformed = trimesh.transform_points(
            [grasp_point, grasp_dir], grasps_tf[i])
        grasp_point = np.array(points_transformed[0])
        grasp_dir = np.array(points_transformed[1])
        id = my_scene.plot_vector(grasp_point, grasp_dir,
                                  color=[255 - int(255*grasp_score_nor[i]), int(255*grasp_score_nor[i]), 0, 255],
                                  radius_cyl=0.001, arrow=True)
        rendered_grasps += 1
    print(f"Trying to render {rendered_grasps} grasps")
    #coordinate arrows with length of 0.1m.
    my_scene.plot_vector(np.array([0, 0, 0]), np.array([0.1, 0, 0]),
        color=[255, 0, 0, 255], radius_cyl=0.001)
    my_scene.plot_vector(np.array([0, 0, 0]), np.array([0, 0.1, 0]),
        color=[0, 255, 0, 255], radius_cyl=0.001)
    my_scene.plot_vector(np.array([0, 0, 0]), np.array([0, 0, 0.1]),
        color=[0, 0, 255, 255], radius_cyl=0.001)
    my_scene.display()