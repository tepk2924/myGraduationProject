import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

target_folder = os.path.dirname(__file__)

OPTION = 2
if OPTION == 0:
    depth:np.ndarray = np.load(os.path.join(target_folder, "depth_binary.npy"))
    SIZE = 200
    startheight = 200
    startwidth = 200
    part = depth[startheight:startheight+SIZE, startwidth:startwidth+SIZE]

    x = np.array(list(range(SIZE)))
    y = np.array(list(range(SIZE)))
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, part, cmap='viridis')

    plt.show()
elif OPTION == 1:
    depth:np.ndarray = np.load(os.path.join(target_folder, "depth_binary.npy"))
    depth = np.nan_to_num(depth)
    x = np.array(list(range(depth.shape[1])))
    y = np.array(list(range(depth.shape[0])))
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, depth, cmap='viridis')

    plt.show()
else:
    import trimesh
    scene = trimesh.Scene()
    pc:np.ndarray = np.load(os.path.join(target_folder, "point_cloud.npy"))[:, :, :3]
    pc = pc.reshape((-1, 3))
    nan_mask = np.where(np.any(np.isnan(pc), axis=1) == False)
    rgb:np.ndarray = np.load(os.path.join(target_folder, "rgb.npy"))
    rgb = rgb.reshape((-1, 3))
    pc_trimesh = trimesh.PointCloud(pc[nan_mask], np.concatenate((rgb[nan_mask], np.full((nan_mask[0].shape[0], 1), 255)), axis=-1))
    scene.add_geometry(pc_trimesh)

    scene.show()