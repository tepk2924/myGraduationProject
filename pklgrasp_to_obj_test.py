import trimesh
from common1_4_graspdata import GraspData
import pickle
import numpy as np
import os

def main(filepath:str, target_folder:str):
    with open(filepath, "rb") as f:
        graspdata:GraspData = pickle.load(f)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    extension = ".obj"
    print(filename)
    mesh:trimesh.base.Trimesh = graspdata.mesh
    # print(len(mesh.vertices))
    # print(mesh.vertices)
    # print(len(mesh.vertex_normals))
    # print(mesh.vertex_normals)
    # print(len(mesh.faces))
    print(mesh.faces)
    with open(os.path.join(target_folder, filename + extension), "w") as f:
        f.write(f"o {filename}\n")
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]/1000:.6f} {vertex[1]/1000:.6f} {vertex[2]/1000:.6f}\n")
        for normal in mesh.vertex_normals:
            f.write(f"vn {normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}\n")

if __name__ == "__main__":
    # filepath = input("입력 : ")
    # target_folder = "/home/tepk2924/tepk2924Works/myGraduationProject/objscenes_generated"
    # main(filepath, target_folder)

    target_input_folder = "/home/tepk2924/tepk2924Works/myGraduationProject/successful_grasp"
    target_output_folder = "/home/tepk2924/tepk2924Works/myGraduationProject/objscenes_generated"
    for filename in os.listdir(target_input_folder):
        main(os.path.join(target_input_folder, filename), target_output_folder)