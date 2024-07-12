import trimesh
from trimesh import creation
import numpy as np
import h5py


def plot_grasp(scene: trimesh.Scene,
               grasp_tf: np.ndarray):
    cyl = creation.cylinder(0.001, 0.01, 3)
    cyl.apply_translation([0, 0, 0.005])
    cyl.apply_transform(grasp_tf)
    scene.add_geometry(cyl)

# def plot_grasp(scene: trimesh.Scene,
#                grasp_tf: np.ndarray):
#     pt = creation.uv_sphere(0.002)
#     pt.apply_translation(grasp_tf[:3, 3])
#     scene.add_geometry(pt)

if __name__ == "__main__":
    with h5py.File(input("Directory : "), "r") as f:
        segmap = np.array(f["category_id_segmaps"]) #(image_height, image_width), np.int64
        colors = np.array(f["colors"]) #(image_height, image_width, 3) np.uint8
        point_cloud = np.array(f["pc"]) #(image_height*image_width, 3) np.float32
        depth = np.array(f["depth"]) #(image_height, image_width) np.float32
        extrinsic = np.array(f["extrinsic"])
        grasps_tf = np.array(f["grasps_tf"])
        grasps_scores = np.array(f["grasps_scores"])
        obj_file_path_np = np.array(f["original_obj_file"])

    OPTION = input("real or segmap : ")
    color_r = (np.concatenate((colors.reshape((-1, 3)), 255*np.ones((colors.shape[0]*colors.shape[1], 1), dtype=np.uint8)), axis=-1) if OPTION == "real" else
               np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], np.uint8)[segmap.reshape(-1)] if OPTION == "segmap" else
               np.tile(np.array([[0, 0, 0, 255]], dtype=np.uint8), (colors.shape[0]*colors.shape[1], 1)))
    scene = trimesh.Scene()
    camera_inverse = np.linalg.inv(extrinsic)
    obj_file_path = "".join([chr(ords) for ords in obj_file_path_np.tolist()])
    print(obj_file_path)
    original_obj = trimesh.load(obj_file_path, "obj")
    original_obj.apply_transform(np.array([[0, 0, 1, 0],
                                           [1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]], dtype=np.float32))
    original_obj.apply_transform(np.array([[0, 1, 0, 0],
                                           [-1, 0, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], dtype=np.float32))
    original_obj.apply_transform(camera_inverse)
    scene.add_geometry(original_obj)
    point_cloud_1padded = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float32)), axis=-1).T
    point_cloud_reformatted = ((camera_inverse@point_cloud_1padded).T)[:, :3]
    pc_scene = trimesh.PointCloud(point_cloud_reformatted, color_r)
    [plot_grasp(scene, camera_inverse@one_grasp_tf) for one_grasp_tf in grasps_tf]
    scene.add_geometry(pc_scene)
    scene.add_geometry(creation.axis(0.04))

    print(point_cloud)
    scene.show()