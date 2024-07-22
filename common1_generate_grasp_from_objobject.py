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

def main(number_of_points = 600):

    """
    inspect ipa-3d1k_models3d folder, convert mesh & texture in the form of .obj + .png, evaluate them and save them as pickle file
    ----------
    """

    start_time = time.time()

    success = 0
    obj_folder = input("Please enter the directory of the dataset folder containing subfolder containing .obj and .jpeg: ")
    subfoldername_list = os.listdir(obj_folder)
    target_folder = input("Please enter the directory of the folder to save the results in the form of .pkl files : ")

    for subfoldername in subfoldername_list:
        try:
            obj_subfolderpath = os.path.join(obj_folder, subfoldername)
            filenames = os.listdir(obj_subfolderpath)
            for name in filenames:
                if name[-4:] == ".obj":
                    filename = name
                    break
            obj_path = os.path.join(obj_subfolderpath, filename)
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
                print("not enough grasps generated!")
                raise ValueError
            #Since the 4x4 tf matrix is in millimeter, we then convert the result back into meter unit.
            results["tf"][:, :3, 3] /= 1000
            meta_data = {"number_of_points" : number_of_points,
                         "number_of_grasps" : len(results["tf"])}
            scalar = 0.001
            scalemat = trimesh.transformations.scale_matrix(scalar)
            mesh.apply_transform(scalemat)

            graspdata = GraspData(obj_path, results, meta_data)
            idx = random.randint(0, (1<<32) - 1)
            with open(os.path.join(target_folder, f"pklgrasp_{idx:08x}.pkl"), "wb") as f:
                pickle.dump(graspdata, f)
            success += 1
            print(f"{success} files generated")
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
    number_of_points = int(input("point number : "))
    main(number_of_points=number_of_points)