import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

color = {0: 'orange', 1: 'blue', 2: 'red'}
data = np.load('/home/sun/Bev/BeMapNet/data/argoverse2/customer_train/315965385959975000.npz', allow_pickle=True)
# ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']
data_dict = {key: data[key].tolist() for key in data.files}

input_dict = data_dict["input_dict"]
# ['timestamp', 'pts_filename', 'lidar_path', 'ego2global_translation', 'ego2global_rotation', 'log_id', 'scene_token',
# 'camego2global', 'img_filename', 'lidar2img', 'camera_intrinsics', 'ego2cam', 'camera2ego', 'cam_type', 'lidar2ego', 'ann_info']
# print(input_dict["img_filename"])

intrinsic = np.stack([np.eye(3) for _ in range(len(input_dict["camera_intrinsics"]))], axis=0)
camera_intrinsics = np.array(input_dict['camera_intrinsics'])
intrinsic[:, :, :] = camera_intrinsics[:, :3, :3]
# print("intrinsic", intrinsic.shape)

instance_mask8 = data_dict["instance_mask8"]
# print("instance_mask8.shape: ", len(instance_mask8[0]))
plt.subplot(2, 3, 4)
plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(instance_mask8[0], cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(instance_mask8[1], cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(instance_mask8[2], cmap='gray')
# plt.show()
plt.figure(figsize=(3, 6))
ctr_points = data_dict["ctr_points"]
# print("ctr_points", ctr_points)
for item in ctr_points:
    pts = item['pts']
    y = [pt[0]-15 for pt in pts]
    x = [-pt[1]+30 for pt in pts]
    plt.scatter(y, x, c = color[item['type']])

plt.show()