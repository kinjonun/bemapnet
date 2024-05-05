import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import os.path as osp
from nuscenes import NuScenes
import cv2

anno_path = "/home/sun/Bev/BeMapNet/data/nuscenes/customer/bemapnet"
eval_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35"
folder_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35/evaluation/results"
car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
color = {0:'r', 1:'orange', 2:'b', 3:'g', 4:"c", 5:"m", 6:"k", 7:"y", 8:"deeppink"}
pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
img_key_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
dataroot = "/media/sun/z/nuscenes/nuscenes"

vis_path = os.path.join(eval_path, 'visual')
if not os.path.exists(vis_path):
    os.makedirs(vis_path, exist_ok=True)

file_list = os.listdir(folder_path)
print(len(file_list))

def save_pred_visual(file):
    npz_path = os.path.join(folder_path, file)
    data = np.load(npz_path, allow_pickle=True)
    dt_res = data['dt_res'].tolist()
    res = dict(dt_res)
    points = res["map"]
    label = res["pred_label"]

    plt.figure(figsize=(3, 6))
    plt.ylim(-30, 30)
    plt.xlim(-15, 15)

    skla = 15 / 100
    for i in range(1, len(res["map"])):
        ins = res["map"][i]
        for j in range(len(ins) - 1):
            x = ins[j][0]
            y = 400 - ins[j][1]

            plt.plot([(ins[j][0] - 100) * skla, (ins[j + 1][0] - 100) * skla],
                     [(200 - ins[j][1]) * skla, (200 - ins[j + 1][1]) * skla], c=color[label[i]])

    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    plt.text(-15, 31, 'PRED', color='red', fontsize=12)
    map_path = osp.join(sample_path, 'PRED_MAP_plot.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
    plt.close()

def save_GT_visual(file):
    gt_path = os.path.join(anno_path, file)
    data = np.load(gt_path, allow_pickle=True)

    plt.figure(figsize=(3, 6))
    plt.ylim(pc_range[1], pc_range[4])           # -30, 30
    plt.xlim(pc_range[0], pc_range[3])           # -15, 15
    # plt.axis('off')

    # for item in data['ctr_points']:
    #     pts = item['pts']
    #     y = [pt[0] - 15 for pt in pts]
    #     x = [-pt[1] + 30 for pt in pts]
    #     plt.scatter(y, x, c=color[item['type']+1])

    for item in data['ego_vectors']:
        pts = item['pts']
        for i in range(len(pts) - 1):
            plt.plot([-pts[i][1], -pts[i + 1][1]], [pts[i][0], pts[i + 1][0]], c=color[item['type']+1])

    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    plt.text(-15, 31, 'GT', color='red', fontsize=12)
    plt.tight_layout()

    map_path = osp.join(sample_path, 'GT.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)

def save_surroud(token):
    sample = nusc.get('sample', token)
    row_1_list = []
    row_2_list = []
    image_name_list = []
    for cam in img_key_list[:3]:
        img = nusc.get('sample_data', sample['data'][cam])
        filename = img['filename']
        # print(filename)
        img_path = os.path.join(dataroot, filename)
        img = cv2.imread(img_path)
        row_1_list.append(img)
        image_name_list.append(filename)

    for cam in img_key_list[3:]:
        img = nusc.get('sample_data', sample['data'][cam])
        filename = img['filename']
        img_path = os.path.join(dataroot, filename)
        img = cv2.imread(img_path)
        row_2_list.append(img)
        image_name_list.append(filename)

    row_1_img = cv2.hconcat(row_1_list)       # 水平拼接成一张图像
    row_2_img = cv2.hconcat(row_2_list)
    cams_img = cv2.vconcat([row_1_img, row_2_img])
    cams_img_path = osp.join(sample_path, 'surroud_view.jpg')
    cv2.imwrite(cams_img_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    img_name_path = osp.join(sample_path, 'images_name_list.txt')
    with open(img_name_path, "w") as f:
        for item in image_name_list:
            f.write("%s\n" % item)

def concat():
    gt_path = osp.join(sample_path, 'GT.png')
    pred_path = osp.join(sample_path, 'PRED_MAP_plot.png')
    surroud_path = osp.join(sample_path, 'surroud_view.jpg')
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)
    surroud = cv2.imread(surroud_path)

    surroud_h, surroud_w, _ = surroud.shape
    pred_h, pred_w, _ = pred.shape
    resize_ratio = surroud_h / pred_h

    resized_w = pred_w * resize_ratio
    resized_pred = cv2.resize(pred, (int(resized_w), int(surroud_h)))
    resized_gt_map_img = cv2.resize(gt, (int(resized_w), int(surroud_h)))

    img = cv2.hconcat([surroud, resized_pred, resized_gt_map_img])

    cams_img_path = osp.join(sample_path, 'con.jpg')
    cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
num_visual = 0

for file in file_list:
    print(file)
    token = os.path.splitext(file)[0]                 # a4354e58aaaa454493ec48f11176530c
    sample_path = os.path.join(vis_path, token)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path, exist_ok=True)

    save_pred_visual(file)
    save_GT_visual(file)
    save_surroud(token)
    concat()


    num_visual = num_visual + 1
    if num_visual >= 5:
        break