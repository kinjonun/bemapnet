import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os


# data = np.load('/home/sun/Bev/BeMapNet/assets/evaluation/results/0a0d6b8c2e884134a3b48df43d54c36a.npz', allow_pickle=True)
data = np.load('0afa886817744d61be66f0bcea2be47c.npz', allow_pickle=True)

dt_res = data['dt_res'].tolist()
dt_mask =data['dt_mask']
image = data["image"]

print(image.shape)
# print(dt_mask[0])

aa =np.array({'map': [None, ([[[ 60.302734 , 161.13281  ]]]),],
              'confidence_level': [-1, 0.9999086856842041, 0.9919501543045044]},)


# plt.imread("/media/sun/z/nuscenes/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_LEFT__1533201485404844.jpg")
image = Image.open("/media/sun/z/nuscenes/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_LEFT__1533201485404844.jpg")

plt.xlabel('Y')
plt.ylabel('X')
# plt.subplot(2, 1, 1)
# plt.imshow(dt_mask[2], cmap='gray')
# image.show()

folder_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35/evaluation/results"

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)
print(len(file_list))