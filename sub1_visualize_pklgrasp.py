import pickle
from trimeshVisualize import Scene
from common1_4_graspdata import GraspData
import numpy as np
import trimesh

if __name__ == "__main__":

    file_loc = input("input the path of pklgrasp file : ")

    with open(file_loc, "rb") as f:
        graspdata:GraspData = pickle.load(f)
    
    mesh = trimesh.load(graspdata.obj_path)
    grasp_info = graspdata.grasp_info
    meta_data = graspdata.meta_data

    print("Displaying scene")
    my_scene = Scene()
    #my_scene.plot_point_multiple(samples, radius=1)
    my_scene.plot_mesh(mesh)
    grasp_score = grasp_info["scores"]

    print(f"minimum score : {np.min(grasp_score):.5f}")
    print(f"Maximum score : {np.max(grasp_score):.5f}")

    per10 = np.quantile(grasp_score, .1)
    per90 = np.quantile(grasp_score, .9)
    grasp_score_nor = (grasp_score - per10) / (per90 - per10)
    grasp_score_nor = np.where(grasp_score_nor > 1, 1,
                               np.where(grasp_score_nor < 0, 0, grasp_score_nor))
    rendered_grasps = 0
    for i in range(len(grasp_info["tf"])):
        if round(grasp_info["scores"][i], 4) == 0:
            continue
        grasp_point = np.array([0, 0, 0])
        #Single grasp as arrow with length of 0.01m.
        grasp_dir = np.array([0, 0, 0.01])
        points_transformed = trimesh.transform_points(
            [grasp_point, grasp_dir], grasp_info["tf"][i])
        grasp_point = np.array(points_transformed[0])
        grasp_dir = np.array(points_transformed[1])
        id = my_scene.plot_vector(grasp_point, grasp_dir,
                                  color=[255 - int(255*grasp_score_nor[i]), int(255*grasp_score_nor[i]), 0, 255],
                                  radius_cyl=0.001, arrow=True)
        rendered_grasps += 1
    print(f"Meta Data : {meta_data}")
    print(f"Trying to render {rendered_grasps} grasps")
    #coordinate arrows with length of 0.1m.
    my_scene.plot_vector(np.array([0, 0, 0]), np.array([0.1, 0, 0]),
        color=[255, 0, 0, 255], radius_cyl=0.001)
    my_scene.plot_vector(np.array([0, 0, 0]), np.array([0, 0.1, 0]),
        color=[0, 255, 0, 255], radius_cyl=0.001)
    my_scene.plot_vector(np.array([0, 0, 0]), np.array([0, 0, 0.1]),
        color=[0, 0, 255, 255], radius_cyl=0.001)
    my_scene.display()