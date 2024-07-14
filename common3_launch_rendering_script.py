import os
import time

if __name__ == "__main__":
    # launching_folder = input("directory of the folder containing pklscenes : ")
    launching_folder = "/home/tepk2924/tepk2924Works/myGraduationProject/pklscenes_generated/test"
    # target_folder = input("target folder directory : ")
    target_folder = "/home/tepk2924/tepk2924Works/myGraduationProject/hdf5scenes_generated/ZED/test"
    # texture_folder = input("texture folder containing textures (invalid, background) : ")
    texture_folder = "/home/tepk2924/tepk2924Works/myGraduationProject/texture_dataset/test"
    # iterations = int(input("multiple rendering from single pklscene : "))
    iterations = 1
    filenames = [filename for filename in os.listdir(launching_folder) if filename[-4:] == ".pkl"]

    scenes_per_iteration = len(filenames)
    total_scenes = scenes_per_iteration*iterations

    for idx in range(iterations):
        for jdx, filename in enumerate(filenames):

            print(f"Rendering {idx*scenes_per_iteration + jdx + 1}/{total_scenes}")
            filepath = os.path.join(launching_folder, filename)
            os.system(f"blenderproc run {os.path.join(os.path.dirname(__file__), 'common3_1_render_objfile_blenderproc.py')} --filepath {filepath} --target_folder {target_folder} --texture_folder {texture_folder}")
            time.sleep(0.3)