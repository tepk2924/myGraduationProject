import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

target_folder = os.path.dirname(__file__)

OPTION = 3
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
elif OPTION == 2:
    import trimesh
    from trimesh import creation
    scene = trimesh.Scene()
    pc:np.ndarray = np.load(os.path.join(target_folder, "point_cloud.npy"))[:, :, :3]
    pc = pc.reshape((-1, 3))
    nan_mask = np.where(np.any(np.isnan(pc), axis=1) == False)
    rgb:np.ndarray = np.load(os.path.join(target_folder, "rgb.npy"))
    rgb = rgb.reshape((-1, 3))
    pc_trimesh = trimesh.PointCloud(pc[nan_mask], np.concatenate((rgb[nan_mask], np.full((nan_mask[0].shape[0], 1), 255)), axis=-1))
    scene.add_geometry(pc_trimesh)
    scene.add_geometry(creation.axis(origin_size=0.01,
                                     axis_radius=0.005))
    
    scene.show(line_settings={'point_size':0.05})
elif OPTION == 3:
    import trimesh
    from trimesh import creation
    with open("/home/riseabb/johan_ws/myGraduationProject/ros_ws/src/arm_pkg/scripts/camera_tf.txt", "r") as f:
        lines = f.readlines()[-4:]

    array = []
    for line in lines:
        array.append(list(map(float, line.split())))

    transform = np.array(array)

    scene = trimesh.Scene()
    pc:np.ndarray = np.load(os.path.join(target_folder, "point_cloud.npy"))[:, :, :3]
    pc = pc.reshape((-1, 3))
    pc[:, 2] -= 0.06 #ZED uses the REAR side of left camera as origin, which can be different from the origin of camera I calculated from PnP algorithm.
    pc[:, 1] += 0.03
    nan_mask = np.where(np.any(np.isnan(pc), axis=1) == False)
    rgb:np.ndarray = np.load(os.path.join(target_folder, "rgb.npy"))
    rgb = rgb.reshape((-1, 3))
    pc_trimesh:trimesh.PointCloud = trimesh.PointCloud(pc[nan_mask], np.concatenate((rgb[nan_mask], np.full((nan_mask[0].shape[0], 1), 255)), axis=-1))
    pc_trimesh.apply_transform(transform)
    scene.add_geometry(pc_trimesh)
    scene.add_geometry(creation.axis(origin_size=0.01,
                                     axis_radius=0.005,
                                     transform=transform))
    scene.add_geometry(creation.axis(origin_size=0.01,
                                    axis_radius=0.005))

    scene.show(line_settings={'point_size':0.05})