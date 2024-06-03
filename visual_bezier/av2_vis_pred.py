import pdb
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import os.path as osp
from nuscenes import NuScenes
import cv2
from pathlib import Path
import random

caption_by_cam = {
    'ring_front_center': 'CAM_FRONT_CENTER',
    'ring_front_right': 'CAM_FRONT_RIGHT',
    'ring_front_left': 'CAM_FRONT_LEFT',
    'ring_rear_right': 'CAM_REAR_RIGHT',
    'ring_rear_left': 'CAM_REAT_LEFT',
    'ring_side_right': 'CAM_SIDE_RIGHT',
    'ring_side_left': 'CAM_SIDE_LEFT',
}


def save_gt_visual(data_dict, car_img, sample_path):
    plt.figure(figsize=(3, 6))
    plt.ylim(-30, 30)
    plt.xlim(-15, 15)
    color = {0: 'g', 1: 'orange', 2: 'b', 3: 'r', 4: "c", }

    ego_points = data_dict["ego_points"]
    ctr_points = data_dict["ctr_points"]
    for item in ctr_points:
        pts = item['pts']
        y = [-pt[0] + 15 for pt in pts]
        x = [-pt[1] + 30 for pt in pts]
        plt.scatter(y, x, c=color[item['type'] + 1])

    for item in ego_points:
        pts = item['pts']
        for i in range(len(pts) - 1):
            plt.plot([pts[i][1], pts[i + 1][1]], [pts[i][0], pts[i + 1][0]], c=color[item['type'] + 1])

    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    plt.text(-15, 31, 'GT', color='red', fontsize=12)
    plt.tight_layout()

    map_path = osp.join(sample_path, 'GT.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
    plt.close()


def save_pred_visual(pred_path, sample_path):
    _, _, epoch_num = pred_path.split('/')[-4].rpartition('_')

    data = np.load(pred_path, allow_pickle=True)
    dt_res = data['dt_res'].tolist()
    res = dict(dt_res)
    points = res["map"]
    label = res["pred_label"]

    plt.figure(figsize=(3, 6))
    plt.ylim(-30, 30)
    plt.xlim(-15, 15)
    color = {0: 'r', 1: 'orange', 2: 'b', 3: 'g', 4: "c", }
    skla = 15 / 100
    for i in range(1, len(res["map"])):
        ins = np.int16(res["map"][i])
        for j in range(len(ins) - 1):
            x = ins[j][0]
            y = 400 - ins[j][1]

            plt.plot([(ins[j][0] - 100) * skla, (ins[j + 1][0] - 100) * skla],
                     [(200 - ins[j][1]) * skla, (200 - ins[j + 1][1]) * skla], c=color[label[i]])

    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    plt.text(-15, 31, 'PRED_epoch_'+epoch_num, color='red', fontsize=12)
    map_path = osp.join(sample_path, 'PRED_epoch_'+epoch_num+'.png')
    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
    plt.close()


def save_surround(cams_dict, sample_dir, timestamp):
    rendered_cams_dict = {}
    for key, cam_dict in cams_dict.items():
        cam_img = cv2.imread(cam_dict)
        # render_anno_on_pv(cam_img, pred_anno, cam_dict['lidar2img'])
        if 'front' not in key:
            #         cam_img = cam_img[:,::-1,:]
            cam_img = cv2.flip(cam_img, 1)
        lw = 8
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(caption_by_cam[key], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (0, 0)
        p2 = (w, h + 3)
        color = (0, 0, 0)
        txt_color = (255, 255, 255)
        cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(cam_img,
                    caption_by_cam[key], (p1[0], p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        rendered_cams_dict[key] = cam_img

    new_image_height = 2048
    new_image_width = 1550 + 2048 * 2
    color = (255, 255, 255)
    first_row_canvas = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)
    first_row_canvas[(2048 - 1550):, :2048, :] = rendered_cams_dict['ring_front_left']
    first_row_canvas[:, 2048:(2048 + 1550), :] = rendered_cams_dict['ring_front_center']
    first_row_canvas[(2048 - 1550):, 3598:, :] = rendered_cams_dict['ring_front_right']

    new_image_height = 1550
    new_image_width = 2048 * 4
    color = (255, 255, 255)
    second_row_canvas = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)
    second_row_canvas[:, :2048, :] = rendered_cams_dict['ring_side_left']
    second_row_canvas[:, 2048:4096, :] = rendered_cams_dict['ring_rear_left']
    second_row_canvas[:, 4096:6144, :] = rendered_cams_dict['ring_rear_right']
    second_row_canvas[:, 6144:, :] = rendered_cams_dict['ring_side_right']

    resized_first_row_canvas = cv2.resize(first_row_canvas, (8192, 2972))
    full_canvas = np.full((2972 + 1550, 8192, 3), color, dtype=np.uint8)
    full_canvas[:2972, :, :] = resized_first_row_canvas
    full_canvas[2972:, :, :] = second_row_canvas
    cams_img_path = osp.join(sample_dir, 'surround_view.jpg')
    cv2.imwrite(cams_img_path, full_canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])

def concat(sample_path):
    gt_path = osp.join(sample_path, 'GT.png')
    surround_path = osp.join(sample_path, 'surround_view.jpg')
    pred_paths = glob.glob(osp.join(sample_path, 'PRED*'))

    pred_images = []
    for pred_path in pred_paths:
        img = cv2.imread(pred_path)
        pred_images.append(img)

    gt = cv2.imread(gt_path)
    # pred = cv2.imread(pred_path)
    surround = cv2.imread(surround_path)

    surround_h, surround_w, _ = surround.shape
    pred_h, pred_w, _ = gt.shape
    resize_ratio = surround_h / pred_h

    resized_w = pred_w * resize_ratio
    resized_pred = [cv2.resize(pred, (int(resized_w), int(surround_h))) for pred in pred_images]
    resized_gt_map_img = cv2.resize(gt, (int(resized_w), int(surround_h)))

    img = cv2.hconcat([surround, resized_gt_map_img])
    # img = cv2.hconcat([surround, resized_gt_map_img, resized_pred])

    cams_img_path = osp.join(sample_path, 'Sample_vis.jpg')
    cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def main():
    project_path = "/home/sun/Bev/BeMapNet"
    anno_path = "/home/sun/Bev/BeMapNet/data/argoverse2/customer"
    output_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_effb0/2024-05-23T12:40:01"
    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')

    num_visual = 0
    file_list = os.listdir(anno_path)
    # print("num of files: {}".format(len(file_list)))
    file_names = os.listdir(anno_path)
    random_files = random.sample(file_names, 20)
    # print(random_files)


    vis_path = os.path.join(output_path, 'visual')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path, exist_ok=True)

    file_list = ["315965566660183000.npz"]
    for file_name in file_list:
        # print("file_name: ", file_name)
        anno_data_path = osp.join(anno_path, file_name)
        anno_data = np.load(anno_data_path, allow_pickle=True)
        # ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']
        anno_data_dict = {key: anno_data[key].tolist() for key in anno_data.files}
        input_dict = anno_data_dict["input_dict"]
        # ['timestamp', 'pts_filename', 'lidar_path', 'ego2global_translation', 'ego2global_rotation', 'log_id',
        # 'scene_token', 'camego2global', 'img_filename', 'lidar2img', 'camera_intrinsics', 'ego2cam', 'camera2ego',
        # 'cam_type', 'lidar2ego', 'ann_info']
        timestamp = input_dict["timestamp"]

        sample_path = os.path.join(vis_path, timestamp)
        if not os.path.exists(sample_path):
            os.makedirs(sample_path, exist_ok=True)

        # save_images
        img_filename = input_dict["img_filename"]
        # ('data/argoverse2/sensor/val/201fe83b-768638a9/sensors/cameras/ring_front_center/315969617449927219', '.jpg')
        cams_dict = {}
        for img in img_filename:
            path = Path(img)
            img_name = path.parts[-2]
            img_path = osp.join(project_path, img)
            cams_dict[img_name] = img_path
        save_surround(cams_dict, sample_path, timestamp)

        all_eval = glob.glob(osp.join(output_path, 'evaluation*'))
        for path in all_eval:
            pred_path = os.path.join(path, 'evaluation', 'results', timestamp)
            save_pred_visual(pred_path, sample_path)

        save_gt_visual(anno_data_dict, car_img, sample_path)
        # concat(sample_path)
        # if cam_img is not None:
        #     cv2.imshow("cam_img", cam_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # else:
        #     print("Error: Processed image for key  is None.")

        num_visual = num_visual + 1
        if num_visual >= 1:
            break


if __name__ == "__main__":
    main()
