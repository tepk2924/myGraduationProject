import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
import random
import trimesh
import pickle

from multiprocessing import freeze_support
from os import close, path, walk

import common1_1_suction_cup_logic as scl
import common1_2_suction_cup_lib as sclib
from trimeshVisualize import Scene
from common1_4_graspdata import GraspData

def main(num_of_files, number_of_points = 600):

    """
    IPA_3D1K 폴더 안에 있는 아무 .obj 형태의 mesh+texture를 골라서 평가하여 지정한 폴더에 .pkl로 저장
    ----------
    """

    start_time = time.time()

    success = 0
    obj_folder = input("Please enter the directory of the ipa-3d1k_models3d folder containing subfolder containing .obj and .jpeg: ")
    target_folder = input("Please enter the directory of the folder to save the results in the form of .pkl files : ")

    while success < num_of_files:
        try:
            obj_subfolderpath = os.path.join(obj_folder, random.choice(os.listdir(obj_folder)))
            obj_path = os.path.join(obj_subfolderpath, "model3d_textured.obj")
            print(f"-----Evaluating {os.path.basename(obj_subfolderpath)}----")
            mesh = trimesh.load(obj_path)
            # if not mesh.is_watertight:
            #     print("Mesh is not watertight!")
            #     raise ValueError
            
            # print(mesh.body_count)
            # if mesh.body_count != 1:
            #     print("The mesh does not consists of only one objects!")
            #     raise ValueError

            #Meter to millimeter
            scalar = 1000
            scalemat = trimesh.transformations.scale_matrix(scalar)
            mesh.apply_transform(scalemat)

            obj_model = sclib.ModelData(mesh, None, units=("millimeters", "millimeters"), subdivide=True)
            evaluation_object = scl.EvaluateMC(
        obj_model, n_processors=8, number_of_points=number_of_points, multiprocessing=True)
            results = evaluation_object.evaluate_model(display=False)
            if len(results["tf"]) <= 10:
                print("형성된 grasp의 갯수가 충분하지 않음!")
                raise ValueError
            #Since the 4x4 tf matrix is in millimeter, we then convert the result back into meter unit.
            results["tf"][:, :3, 3] /= 1000
            meta_data = {"number_of_points" : number_of_points,
                         "number_of_grasps" : len(results["tf"])}
            scalar = 0.001
            scalemat = trimesh.transformations.scale_matrix(scalar)
            mesh.apply_transform(scalemat)
            #GraspData 클래스를 생성해서 mesh와 mesh의 grasp 정보, 약간의 메타데이터를 담는다.
            graspdata = GraspData(mesh, obj_path, results, meta_data)
            idx = random.randint(0, (1<<32) - 1)
            with open(os.path.join(target_folder, f"pklgrasp_{idx:08x}.pkl"), "wb") as f:
                pickle.dump(graspdata, f)
            success += 1
            print(f"총 {success}개의 파일 생성함")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except ValueError as e:
            pass
        except Exception as e:
            print(e)
            pass

    print("Elapsed Time:" + str(time.time() - start_time))
    print("-------------------------------")

if __name__ == "__main__":
    num_of_files = int(input("생성할 pkl 파일의 갯수 : "))
    number_of_points = int(input("테스트할 포인트의 갯수 : "))
    main(num_of_files, number_of_points=number_of_points)