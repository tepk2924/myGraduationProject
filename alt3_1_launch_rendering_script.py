import os

if __name__ == "__main__":
    filepath = input("입력 : ")
    os.system(f"blenderproc run /home/tepk2924/tepk2924Works/myGraduationProject/alt3_2_render_objfile_blenderproc.py --obj_file {filepath}")
    os.system("blenderproc vis hdf5 /home/tepk2924/tepk2924Works/myGraduationProject/hdf5output/0.hdf5")