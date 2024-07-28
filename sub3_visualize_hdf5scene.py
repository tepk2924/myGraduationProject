import trimesh
from trimesh import creation
import numpy as np
import h5py


def plot_grasp(scene: trimesh.Scene,
               grasp_tf: np.ndarray):
    cyl = creation.cylinder(0.001, 0.01, 3)
    cyl.apply_translation([0, 0, 0.005])
    cyl.apply_transform(grasp_tf)
    cyl.visual.face_colors = [255, 255, 255, 255]
    scene.add_geometry(cyl)

if __name__ == "__main__":
    with h5py.File(input("Directory : "), "r") as f:
        segmap = np.array(f["category_id_segmaps"]) #(image_height, image_width), np.int64
        colors = np.array(f["colors"]) #(image_height, image_width, 3) np.uint8
        point_cloud = np.array(f["pc"]) #(image_height*image_width, 3) np.float32
        depth = np.array(f["depth"]) #(image_height, image_width) np.float32
        extrinsic = np.array(f["extrinsic"])
        grasps_tf = np.array(f["grasps_tf"])
        grasps_scores = np.array(f["grasps_scores"])
        original_obj_paths_np = np.array(f["original_obj_paths"])
        obj_poses = np.array(f["obj_poses"])

    grasps_tf = np.array([[0, 1, 0, 0],
                          [-1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]) @ grasps_tf

    OPTION = input("real or segmap : ")
    color_r = (np.concatenate((colors.reshape((-1, 3)), 255*np.ones((colors.shape[0]*colors.shape[1], 1), dtype=np.uint8)), axis=-1) if OPTION == "real" else
               np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], np.uint8)[segmap.reshape(-1)] if OPTION == "segmap" else
               np.tile(np.array([[0, 0, 0, 255]], dtype=np.uint8), (colors.shape[0]*colors.shape[1], 1)))
    scene = trimesh.Scene()
    camera_inverse = np.linalg.inv(extrinsic)

    original_obj_paths = list(map(lambda x: str(x)[2:-1], original_obj_paths_np.tolist()))

    point_cloud_1padded = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float32)), axis=-1).T
    point_cloud_reformatted = ((camera_inverse@point_cloud_1padded).T)[:, :3]
    pc_scene = trimesh.PointCloud(point_cloud_reformatted, color_r)
    [plot_grasp(scene, camera_inverse@one_grasp_tf) for one_grasp_tf in grasps_tf]
    scene.add_geometry(pc_scene)
    scene.add_geometry(creation.axis(0.04))

    scene.show()