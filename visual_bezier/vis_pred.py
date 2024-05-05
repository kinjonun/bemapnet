import numpy as np
import array
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os.path as osp

sample_dir = "/home/sun/Bev/BeMapNet/visual_bezier"
car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
color = {0:'r', 1:'orange', 2:'b', 3:'g', 4:"c", 5:"m", 6:"k", 7:"y", 8:"deeppink"}
data = np.load("/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35/evaluation/results/0f77ffe576ac436a87787eb343dc3f27.npz", allow_pickle=True)
dt_res = data['dt_res'].tolist()
res = dict(dt_res)
points = res["map"]
label = res["pred_label"]

# plt.subplot(1, 2, 1)
plt.figure(figsize=(3, 6))
plt.ylim(-30, 30)
plt.xlim(-15, 15)
# print(len(res["map"][10]))
print("map长度", len(res["map"]))
print(res["pred_label"])                 # [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]

# print(res["map"][11])
# for i in range(len(res["map"])):
skla = 15/100

for i in range(1, len(res["map"])):
    ins = res["map"][i]
    for j in range(len(ins)-1):
        x = ins[j][0]
        y = 400 - ins[j][1]
        # plt.scatter(x, y, c = color[label[i]])
        plt.plot([(ins[j][0]-100)*skla, (ins[j+1][0]-100)*skla], [(200 - ins[j][1])*skla, (200 - ins[j+1][1])*skla], c=color[label[i]])

plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

plt.text(-15, 31, 'PRED', color='red', fontsize=12)
map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
# plt.subplot(1, 2, 2)
# mask = data['dt_mask']
#
# plt.title('Prediction Boundary')
# plt.imshow(mask[0], cmap='gray')
plt.show()

# cv2.imshow('Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()