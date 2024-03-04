import numpy as np
import os

if __name__ == "__main__":

    scenes_root_path = input("불러올 scene들이 들어있는 폴더 경로 입력 : ")
    npz_file_namelist = os.listdir(scenes_root_path)

    num_of_objects = []
    num_of_grasps = []

    for filename in npz_file_namelist:
        with np.load(os.path.join(scenes_root_path, filename), allow_pickle=True) as scene_data:
            num_of_objects.append(len(scene_data["object_names"]))
            num_of_grasps.append(len(scene_data["scene_grasps_tf"]))

    num_of_scene = len(npz_file_namelist)
    print(f"npz 파일의 갯수 : {num_of_scene}")
    print(f"가장 많은 오브젝트를 가진 scene의 오브젝트 갯수 : {max(num_of_objects)}")
    print(f"가장 적은 오브젝트를 가진 scene의 오브젝트 갯수 : {min(num_of_objects)}")
    print(f"한 scene 당 평균 오브젝트 갯수 : {sum(num_of_objects)/num_of_scene}")
    print(f"가장 많은 파지의 개수를 가진 scene의 파지 갯수 : {max(num_of_grasps)}")
    print(f"가장 적은 파지의 개수를 가진 scene의 파지 갯수 : {min(num_of_grasps)}")
    print(f"한 scene 당 평균 파지 갯수 : {sum(num_of_grasps)/num_of_scene}")