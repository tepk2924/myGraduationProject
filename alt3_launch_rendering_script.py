import os
import time

if __name__ == "__main__":
    launching_folder = input("objscene과 npzgrasp이 들어있는 폴더 선택 : ")
    target_folder = input("저장할 폴더 : ")
    texture_folder = input("텍스쳐 3종류가 들어있는 폴더 : ")
    iterations = int(input("같은 objscene 파일에서 렌더링할 횟수 : "))
    filenames = [filename for filename in os.listdir(launching_folder) if filename[-4:] == ".obj"]

    scenes_per_iteration = len(filenames)
    total_scenes = scenes_per_iteration*iterations

    for idx in range(iterations):
        for jdx, filename in enumerate(filenames):

            print(f"Rendering {idx*scenes_per_iteration + jdx + 1}/{total_scenes}")
            filepath = os.path.join(launching_folder, filename)
            os.system(f"blenderproc run {os.path.join(os.path.dirname(__file__), 'alt3_1_render_objfile_blenderproc.py')} --filepath {filepath} --target_folder {target_folder} --texture_folder {texture_folder}")
            time.sleep(0.3)