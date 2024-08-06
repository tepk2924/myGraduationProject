import os

path = "/home/tepk2924/tepk2924Works/myGraduationProject/DataSet/test/37786_Thingi10K/37786.obj"
os.system(f"blenderproc run /home/tepk2924/tepk2924Works/myGraduationProject/blenderproc_sandbox_1.py --filepath {path}")
os.system("blenderproc vis hdf5 /home/tepk2924/tepk2924Works/myGraduationProject/blenderproc_test/result/0.hdf5")