import os
import sys
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import multiprocessing
import trimesh

from multiprocessing import freeze_support
from os import close, path, walk

import suction_cup_logic as scl
import suction_cup_lib as sclib
from trimeshVisualize import Scene

def evaluate_object_one(obj_name,
                        scalar: float,
                        config_path = None,
                        number_of_points = 3000,
                        display=True,
                        splits="test"):

    """
    Evalute N points on a single object
    ---------
    Args:
        - obj_name {str} : The name of the object to evaluate.
        - root_dir {str} : The path to root_dir of data.
    Kwargs:
        - config_path {str} : The path to the model config file.
        - number_of_points {int} : The number of points to evaluate on the object.
        - display {bool} : Whether to display the results.
        - splits {str} : The name of subdirectory in meshes/ to evaluate. (Used for train/test split)
        - save_path {str} : The path to save the results.
            if None, results are not saved.
    ----------
    Returns:
        - results {dict} : The results of the evaluation.
    """

    start_time = time.time()

    print(f"-----Evaluating {obj_name}----")
    mesh = trimesh.load(f"sample_objects/{obj_name}.stl")

    if not mesh.is_watertight:
        print("Mesh가 watertight하지 않음!")
        raise ValueError

    scalemat = trimesh.transformations.scale_matrix(scalar)
    mesh.apply_transform(scalemat)

    obj_model = sclib.ModelData(mesh, None, units=("millimeters", "millimeters"), subdivide=True)

    evaluation_object = scl.EvaluateMC(
        obj_model, n_processors=8, number_of_points=number_of_points, multiprocessing=True)

    results = evaluation_object.evaluate_model(display=display)

    print("Elapsed Time:" + str(time.time() - start_time))
    print("-------------------------------")

    return results

if __name__ == "__main__":
    obj_name = input("오브젝트 이름 입력: ")
    #Thingi10K의 오브젝트들이 너무 작기 때문에 제대로 grasp을 형성 하지 않아서 오브젝트 크기를 키워야 함.
    #스케일 배율에 1을 넣어서 오브젝트 크기를 유지한다면 grasp이 형성이 되지 않음.
    obj_scale = float(input("오브젝트 스케일 배율 입력: ")) 
    result = evaluate_object_one(obj_name, scalar = obj_scale, number_of_points=600)
    print(result)
    print(f"{len(result['tf'])} grasps 찾음")