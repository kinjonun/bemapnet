import pdb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
import random

color = {0: 'orange', 1: 'blue', 2: 'red'}
def av2_npz():
    folder_path = '/home/sun/Bev/BeMapNet/data/argoverse2/customer'
    file_names = os.listdir(folder_path)
    # random_file = random.choice(file_names)
    random_file = '315970547759868000.npz'
    # print("av2_npz:", random_file)

    file_path = os.path.join(folder_path, random_file)
    data = np.load(file_path, allow_pickle=True)
    # ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']
    data_dict = {key: data[key].tolist() for key in data.files}

    input_dict = data_dict["input_dict"]
    # ['timestamp', 'pts_filename', 'lidar_path', 'ego2global_translation', 'ego2global_rotation', 'log_id', 'scene_token',
    # 'camego2global', 'img_filename', 'lidar2img', 'camera_intrinsics', 'ego2cam', 'camera2ego', 'cam_type', 'lidar2ego', 'ann_info']
    # print(input_dict["img_filename"])

    timestamp = "av2 " + input_dict["timestamp"]
    intrinsic = np.stack([np.eye(3) for _ in range(len(input_dict["camera_intrinsics"]))], axis=0)
    camera_intrinsics = np.array(input_dict['camera_intrinsics'])
    intrinsic[:, :, :] = camera_intrinsics[:, :3, :3]

    instance_mask8 = data_dict["instance_mask8"]
    # print("instance_mask8.shape: ", len(instance_mask8))
    ctr_points = data_dict["ctr_points"]
    ego_points = data_dict["ego_points"]
    # print("ctr_points:", ctr_points)
    return ctr_points, ego_points, instance_mask8, timestamp

def nuscenes_npz():
    folder_path = '/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet_lidar'
    file_names = os.listdir(folder_path)
    # random_file = random.choice(file_names)
    # print("nuscenes_npz:", random_file)
    random_file = "1c05e9f2e6364d2c8ceb0ff2f0f72fdb.npz"

    file_path = os.path.join(folder_path, random_file)
    data = np.load(file_path, allow_pickle=True)
    # pdb.set_trace()
    image_paths = data['image_paths']
    trans = data['trans']
    rots = data['rots']
    intrins = data['intrins']
    semantic_mask = data['semantic_mask']  # (3, 400, 200)
    instance_mask = data['instance_mask']  # (3, 400, 200)
    instance_mask8 = data['instance_mask8']
    ego_vectors = data['ego_vectors']
    ctr_points = data['ctr_points']
    map_vectors = data['map_vectors']
    return ctr_points, ego_vectors, instance_mask8, random_file

def plot_ctr_points(ctr_points, ego_vectors, timestamp):
    plt.figure(figsize=(3, 6))
    plt.title(timestamp)
    for item in ctr_points:
        pts = item['pts']
        y = [pt[0] - 15 for pt in pts]
        x = [-pt[1] + 30 for pt in pts]
        plt.scatter(y, x, c=color[item['type']])

    for item in ego_vectors:
        pts = item['pts']
        for i in range(len(pts) - 1):
            plt.plot([-pts[i][1], -pts[i + 1][1]], [pts[i][0], pts[i + 1][0]], c=color[item['type']])
def plot_instance_mask(instance_mask8, timestamp):
    plt.figure(figsize=(8, 6))
    plt.title(timestamp)
    plt.subplot(1, 3, 1)
    # plt.imshow(instance_mask8[0], cmap='gray')
    plt.subplot(1, 3, 2)
    # print("instance_mask8:", len(instance_mask8[0]))
    # print("instance_mask8[1]", instance_mask8[1])
    plt.imshow(instance_mask8[1], cmap='gray')
    plt.subplot(1, 3, 3)
    # plt.imshow(instance_mask8[2], cmap='gray')


av2_ctr_points, av2_ego_points, av2_instance_mask8, av2_timestamp = av2_npz()
cross = av2_instance_mask8[2]
# for i in range(len(cross)):
#     for j in range(len(cross[i])):
#         aa = np.int16(cross[i][j])
#         if aa > 0:
#             print(aa)
        # pdb.set_trace()

# plot_ctr_points(av2_ctr_points, av2_ego_points, av2_timestamp)
# plot_instance_mask(av2_instance_mask8, av2_timestamp)
#
nus_ctr_points, nus_ego_points, nus_instance_mask8, nus_timestamp = nuscenes_npz()
# plot_ctr_points(nus_ctr_points, nus_ego_points, nus_timestamp)
# plot_instance_mask(nus_instance_mask8, nus_timestamp)

# plt.show()


