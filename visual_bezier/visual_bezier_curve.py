import numpy as np
import matplotlib.pyplot as plt
import torch

# data = np.load('1b9a789e08bb4b7b89eacb2176c70840.npz', allow_pickle=True)
data = np.load('/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet/02651ca22b08484294d473d9c26cd9af.npz', allow_pickle=True)


map_vectors = data['map_vectors']
image_paths = data['image_paths']
trans = data['trans']
rots = data['rots']
intrins = data['intrins']
semantic_mask = data['semantic_mask']
instance_mask = data['instance_mask']
instance_mask8 = data['instance_mask8']
ego_vectors = data['ego_vectors']
ctr_points = data['ctr_points']

color = {0: 'orange', 1: 'blue', 2: 'r'}

# image = plt.imread('images/n008-2018-08-01-16-03-27-0400__CAM_FRONT__1533153857912404.jpg')
fig, ax = plt.subplots()
# ax.imshow(image)

tensor_tran = torch.tensor(trans).cuda()
# print(image_paths.shape)

plt.subplot(2, 1, 1)

for item in data['ctr_points']:
    pts = item['pts']
    y = [-pt[0]+15 for pt in pts]
    x = [-pt[1]+30 for pt in pts]
    plt.scatter(x, y, c = color[item['type']])

for item in data['ego_vectors']:
    pts = item['pts']
    for i in range(len(pts)-1):
        plt.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], c=color[item['type']])

plt.subplot(2, 3, 4)
plt.title('divider semantic')
plt.xlabel('Y')
plt.ylabel('X')
plt.imshow(semantic_mask[0], cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(semantic_mask[1], cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(semantic_mask[2], cmap='gray')


plt.tight_layout()

plt.show()


# fig, ax = plt.subplots()
# ax.imshow(image)