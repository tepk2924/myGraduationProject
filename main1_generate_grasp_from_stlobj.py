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

import suction_cup_logic as scl
import suction_cup_lib as sclib
from trimeshVisualize import Scene
from graspdata import GraspData

def main(num_of_files, number_of_points = 600):

    """
    지정한 폴더 안에 있는 아무 .stl 형태의 mesh를 골라서 크기를 적당히 조절 후 평가하여 지정한 폴더에 .pkl로 저장
    ----------
    """

    start_time = time.time()

    success = 0
    obj_folder = input(".stl 파일이 들어있는 폴더의 디렉토리 입력 : ")
    obj_filenames = os.listdir(obj_folder)
    pkl_folder = input(".pkl 파일을 저장할 폴더의 디렉토리 입력 : ")

    while success < num_of_files:
        try:
            obj_filename = random.choice(obj_filenames)
            print(f"-----Evaluating {obj_filename}----")
            mesh = trimesh.load(os.path.join(obj_folder, obj_filename))
            if not mesh.is_watertight:
                print("Mesh가 watertight하지 않음!")
                raise ValueError
            
            if mesh.body_count != 1:
                print("Mesh가 한 개의 오브젝트가 아님!")
                raise ValueError

            mesh_size = mesh.extents

            #가장 긴 축이 300mm 보다 작게, 가장 짧은 축이 20mm 보다 크게
            minimum_scalar = max([20/mesh_size[0], 20/mesh_size[1], 20/mesh_size[2]])
            maxinum_scalar = min([300/mesh_size[0], 300/mesh_size[1], 300/mesh_size[2]])
            if maxinum_scalar < minimum_scalar:
                print("크기가 적당하지 않음!")
                raise ValueError
            scalar = minimum_scalar + (maxinum_scalar - minimum_scalar)*random.random()

            #Mesh의 전체적인 크기를 조절하기 위해 변환 적용. 스칼라값은 위와 같이 제한됨
            scalemat = trimesh.transformations.scale_matrix(scalar)
            mesh.apply_transform(scalemat)
            obj_model = sclib.ModelData(mesh, None, units=("millimeters", "millimeters"), subdivide=True)
            evaluation_object = scl.EvaluateMC(
        obj_model, n_processors=8, number_of_points=number_of_points, multiprocessing=True)
            results = evaluation_object.evaluate_model(display=False)
            if len(results["tf"]) <= 10:
                print("형성된 grasp의 갯수가 충분하지 않음!")
                raise ValueError
            meta_data = {"scalar" : scalar,
                         "mesh_size" : mesh.extents,
                         "number_of_points" : number_of_points,
                         "number_of_grasps" : len(results["tf"])}
            #GraspData 클래스를 생성해서 mesh와 mesh의 grasp 정보, 약간의 메타데이터를 담는다.
            graspdata = GraspData(mesh, results, meta_data)
            idx = random.randint(0, (1<<32) - 1)
            with open(os.path.join(pkl_folder, f"{os.path.splitext(obj_filename)[0]}_{idx:08x}.pkl"), "wb") as f:
                pickle.dump(graspdata, f)
            success += 1
            print(f"총 {success}개의 파일 생성함")
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            pass

    print("Elapsed Time:" + str(time.time() - start_time))
    print("-------------------------------")

if __name__ == "__main__":
    num_of_files = int(input("생성할 mesh와 파지를 담을 pkl 파일의 갯수 : "))
    number_of_points = int(input("테스트할 포인트의 갯수 : "))
    main(num_of_files, number_of_points=number_of_points)