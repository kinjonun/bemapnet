import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import os.path as osp
from nuscenes import NuScenes
import cv2
from pathlib import Path

caption_by_cam = {
    'ring_front_center': 'CAM_FRONT_CENTER',
    'ring_front_right': 'CAM_FRONT_RIGHT',
    'ring_front_left': 'CAM_FRONT_LEFT',
    'ring_rear_right': 'CAM_REAR_RIGHT',
    'ring_rear_left': 'CAM_REAT_LEFT',
    'ring_side_right': 'CAM_SIDE_RIGHT',
    'ring_side_left': 'CAM_SIDE_LEFT',
}

def save_surroud(cams_dict, sample_dir, timestamp):
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
    cams_img_path = osp.join(sample_dir, timestamp+'surroud_view.jpg')
    cv2.imwrite(cams_img_path, full_canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])

def main():
    anno_path = "/home/sun/Bev/BeMapNet/data/argoverse2/customer_train"
    # eval_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35"
    project_path = "/home/sun/Bev/BeMapNet"
    sample_dir = "/home/sun/Bev/BeMapNet/visual_bezier"
    # folder_path = "/home/sun/Bev/BeMapNet/outputs/bemapnet_nuscenes_res50/2024-05-03T10:33:35/evaluation/results"
    car_img = Image.open('/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png')
    color = {0: 'r', 1: 'orange', 2: 'b', 3: 'g', 4: "c", 5: "m", 6: "k", 7: "y", 8: "deeppink"}
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

    num_visual = 0
    file_list = os.listdir(anno_path)
    # print("num of files: {}".format(len(file_list)))

    file_list = ["315965566660183000.npz"]
    for file_name in file_list:
        # print("file_name: ", file_name)
        data_path = osp.join(anno_path, file_name)
        data = np.load(data_path, allow_pickle=True)  # ['input_dict', 'instance_mask', 'instance_mask8', 'semantic_mask', 'ctr_points', 'ego_points', 'map_vectors']
        data_dict = {key: data[key].tolist() for key in data.files}
        input_dict = data_dict["input_dict"]  # ['timestamp', 'pts_filename', 'lidar_path', 'ego2global_translation', 'ego2global_rotation', 'log_id', 'scene_token', 'camego2global', 'img_filename', 'lidar2img', 'camera_intrinsics', 'ego2cam', 'camera2ego', 'cam_type', 'lidar2ego', 'ann_info']
        timestamp = input_dict["timestamp"]

        img_filename =input_dict["img_filename"]  # ('data/argoverse2/sensor/val/201fe83b-7dd7-38f4-9d26-7b4a668638a9/sensors/cameras/ring_front_center/315969617449927219', '.jpg')
        cams_dict = {}
        for img in img_filename:
            path = Path(img)
            img_name = path.parts[-2]
            img_path = osp.join(project_path, img)
            cams_dict[img_name] = img_path
        save_surroud(cams_dict, sample_dir, timestamp)

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

