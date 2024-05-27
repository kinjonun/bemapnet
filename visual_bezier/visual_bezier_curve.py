import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os.path as osp

sample_dir = "/home/sun/Bev/BeMapNet/visual_bezier"
# data = np.load('1b9a789e08bb4b7b89eacb2176c70840.npz', allow_pickle=True)
data = np.load('/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet/0a0a8655794d462c8a27941b9d5ad1a3.npz', allow_pickle=True)


image_paths = data['image_paths']
trans = data['trans']
rots = data['rots']
intrins = data['intrins']
semantic_mask = data['semantic_mask']         # (3, 400, 200)
instance_mask = data['instance_mask']         # (3, 400, 200)
instance_mask8 = data['instance_mask8']
ego_vectors = data['ego_vectors']
ctr_points = data['ctr_points']
map_vectors = data['map_vectors']
# print("image_paths", image_paths)
# print("trans", trans)
# print("rots", rots)
# print("intrins", intrins)
# print("ctr_points", ctr_points)
# print("ego_vectors", ego_vectors)
color = {0: 'orange', 1: 'blue', 2: 'red'}
pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')

# image = plt.imread('images/n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533153857912404.jpg')
# fig, ax = plt.subplots()
# ax.imshow(image)

tensor_tran = torch.tensor(trans).cuda()
# print(image_paths.shape)

plt.figure(figsize=(3, 6))
# plt.subplot(2, 3, 1)
# plt.figure(figsize=(2, 4))
plt.ylim(pc_range[1], pc_range[4])  # -30, 30
plt.xlim(pc_range[0], pc_range[3])  # -15, 15
# plt.axis('off')

for item in data['ctr_points']:
    pts = item['pts']
    y = [pt[0]-15 for pt in pts]
    x = [-pt[1]+30 for pt in pts]
    plt.scatter(y, x, c = color[item['type']])
#
for item in data['ego_vectors']:
    pts = item['pts']
    for i in range(len(pts)-1):
        plt.plot([-pts[i][1], -pts[i + 1][1]], [pts[i][0], pts[i + 1][0]], c=color[item['type']])

# plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
# plt.text(-15, 31, 'GT', color='red', fontsize=12)

plt.figure(figsize=(6, 6))
plt.subplot(2, 3, 4)
plt.title('divider semantic')
plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(instance_mask[0], cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(semantic_mask[1], cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(instance_mask[2], cmap='gray')


plt.tight_layout()

# map_path = osp.join(sample_dir, 'GT.png')
# plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
plt.show()

# sample_path = "/home/sun/Bev/BeMapNet/visual_bezier"
# img_name_path = osp.join(sample_path, 'instance_mask.txt')
# # 设置 NumPy 打印选项，以避免省略号
# np.set_printoptions(threshold=np.inf)
# # 将数组转换为字符串
# array_str = np.array2string(instance_mask[0], separator=' ')
#
# with open(img_name_path, "w") as f:
#         f.write(array_str)
#
# # 恢复 NumPy 默认打印选项
# np.set_printoptions(threshold=1000)
#
#
# for i in range(instance_mask8[2].shape[0]):
#         for j in range(instance_mask8[2].shape[1]):
#                 if instance_mask8[2][i][j] > 0:
#                         print(instance_mask8[2][i][j])
