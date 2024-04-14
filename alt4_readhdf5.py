import h5py
import numpy as np

def main():
    f = h5py.File(input("입력 : "), "r")
    segmap = np.array(f["category_id_segmaps"])
    colors = np.array(f["colors"])
    point_cloud = np.array(f["pc"])
    depth = np.array(f["depth"])
    grasps_tf = np.array(f["grasps_tf"])
    grasps_scores = np.array(f["grasps_scores"])
    f.close()
    print(segmap.shape)
    print(segmap)
    print(len(np.where(segmap == 0)[0]))
    print(len(np.where(segmap == 1)[0]))
    print(len(np.where(segmap == 2)[0]))
    print(depth.shape)
    print(depth)
    colors = np.transpose(colors, (2, 0, 1))
    print(colors.shape)
    print(colors)
    print(point_cloud.shape)
    print(point_cloud)
    print(grasps_tf)
    print(grasps_scores)

if __name__ == "__main__":
    main()