import h5py
import numpy as np

def main():
    f = h5py.File(input("입력 : "), "r")
    gt = np.array(f["category_id_segmaps"])
    colors = np.array(f["colors"])
    point_cloud = np.array(f["pc"])
    f.close()

    print(gt.shape)
    print(colors.shape)
    print(point_cloud.shape)

if __name__ == "__main__":
    main()