import pdb
from functools import partial
from multiprocessing import Pool, cpu_count
import multiprocessing
import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt
# from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
# from mmdet.datasets.pipelines import to_tensor
import json
from PIL import Image
import cv2
import os
from tqdm import tqdm
from scipy.special import comb as n_over_k


class PiecewiseBezierCurve(object):
    def __init__(self, num_points=100, num_degree=2, margin=0.05, threshold=0.1):
        super().__init__()
        self.num_points = num_points
        self.num_degree = num_degree
        self.margin = margin
        self.bezier_coefficient = self._get_bezier_coefficients(np.linspace(0, 1, self.num_points))
        self.threshold = threshold

    def _get_bezier_coefficients(self, t_list):
        bernstein_fn = lambda n, t, k: (t ** k) * ((1 - t) ** (n - k)) * n_over_k(n, k)
        bezier_coefficient_fn = \
            lambda ts: [[bernstein_fn(self.num_degree, t, k) for k in range(self.num_degree + 1)] for t in t_list]
        return np.array(bezier_coefficient_fn(t_list))

    def _get_interpolated_points(self, points):
        line = LineString(points)
        distances = np.linspace(0, line.length, self.num_points)
        sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        return sampled_points

    def _get_chamfer_distance(self, points_before, points_after):
        points_before = torch.from_numpy(points_before).float()
        points_after = torch.from_numpy(points_after).float()
        dist = torch.cdist(points_before, points_after)
        dist1, _ = torch.min(dist, 2)
        dist1 = (dist1 * (dist1 > self.margin).float())
        dist2, _ = torch.min(dist, 1)
        dist2 = (dist2 * (dist2 > self.margin).float())
        return (dist1.mean(-1) + dist2.mean(-1)) / 2

    def bezier_fitting(self, curve_pts):
        curve_pts_intered = self._get_interpolated_points(curve_pts)
        curve_pts_intered = curve_pts_intered[:, :2]                        # 3D --> 2D
        # pdb.set_trace()
        bezier_ctrl_pts = np.linalg.pinv(self.bezier_coefficient).dot(curve_pts_intered)
        bezier_ctrl_pts = np.concatenate([curve_pts[0:1, :2], bezier_ctrl_pts[1:-1], curve_pts[-1:, :2]], axis=0)
        curve_pts_recovery = self.bezier_coefficient.dot(bezier_ctrl_pts)
        criterion = self._get_chamfer_distance(curve_pts_intered[None, :, :], curve_pts_recovery[None, :, :]).item()
        return bezier_ctrl_pts, criterion

    @staticmethod
    def sequence_reverse(ctr_points):
        ctr_points = np.array(ctr_points)
        (xs, ys), (xe, ye) = ctr_points[0], ctr_points[-1]
        if ys > ye:
            ctr_points = ctr_points[::-1]
        return ctr_points

    def __call__(self, curve_pts):
        ctr_points_piecewise = []
        num_points = curve_pts.shape[0]
        start, end = 0, num_points - 1
        while start < end:
            ctr_points, loss = self.bezier_fitting(curve_pts[start: end + 1])
            if loss < self.threshold:
                start, end = end, num_points - 1
                if start >= end:
                    ctr_points_piecewise += ctr_points.tolist()
                else:
                    ctr_points_piecewise += ctr_points.tolist()[:-1]
            else:
                end = end - 1
        # pdb.set_trace()
        ctr_points_piecewise = self.sequence_reverse(ctr_points_piecewise)
        return ctr_points_piecewise


class BezierConverter(object):
    def __init__(self,
                 data_info,
                 ):
        super().__init__()
        self.color = {0:"orange", 1:"blue", 2:"red", 3:"yellow", 4:"black"}
        self.car_img = Image.open("/home/sun/Bev/BeMapNet/assets/figures/car.png")
        self.lidar_car_img = Image.open("/home/sun/Bev/BeMapNet/assets/figures/lidar_car.png")
        self.scale_width = 20 / 3
        self.scale_height = 20 / 3
        self.thickness = [1, 8]
        self.patch_size = [30, 60]
        self.canvas_size = [200, 400]
        self.max_channel = 3
        self.num_degrees = [2, 1, 3]
        self.vec_class = ['divider', 'boundary']
        # self.vec_class = ['divider', 'ped_crossing', 'boundary', 'centerline']
        self.CLASS2LABEL = {'divider': 0, 'ped_crossing': 1, 'boundary': 2, 'centerline': 3, 'others': -1}
        self.pbc_funcs = {d: PiecewiseBezierCurve(num_points=100, num_degree=d, margin=0.05, threshold=0.1) for d in self.num_degrees}
        self.trans_x, self.trans_y = 30, 15
        self.data_info = data_info

    def get_data_info(self, data_info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
                :param data_info:
        """
        info = data_info
        # standard protocal modified from SECOND.Pytorch

        input_dict = dict(
            timestamp=info['timestamp'],
            pts_filename=info['lidar_path'],
            lidar_path=info['lidar_path'],
            ego2global_translation=info['e2g_translation'],
            ego2global_rotation=info['e2g_rotation'],
            log_id=info['log_id'],
            scene_token=info['log_id'],
        )
        if True:
            image_paths = []
            cam_intrinsics = []
            ego2img_rts = []
            ego2cam_rts = []
            cam_types = []
            cam2ego_rts = []
            input_dict["camego2global"] = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['img_fpath'])
                camera_intrinsics = np.eye(4).astype(np.float32)  # camera intrinsics
                camera_intrinsics[:3, :3] = cam_info["intrinsics"]
                # input_dict["camera_intrinsics"].append(camera_intrinsics)

                # ego2img,    ego = lidar
                ego2cam_rt = cam_info['extrinsics']
                cam2ego_rts.append(np.matrix(ego2cam_rt).I)  # 使用 .I 方法对其求逆

                intrinsic = cam_info['intrinsics']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2img_rt = (viewpad @ ego2cam_rt)
                ego2img_rts.append(ego2img_rt)
                ego2cam_rts.append(ego2cam_rt)
                cam_intrinsics.append(viewpad)
                cam_types.append(cam_type)

                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = cam_info['e2g_rotation']
                camego2global[:3, 3] = cam_info['e2g_translation']
                # camego2global = torch.from_numpy(camego2global)
                input_dict["camego2global"].append(camego2global)

            lidar2ego = np.eye(4).astype(np.float32)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=ego2img_rts,                # 认为lidar和ego是同一个坐标系
                    camera_intrinsics=cam_intrinsics,
                    ego2cam=ego2cam_rts,
                    camera2ego=cam2ego_rts,
                    cam_type=cam_types,
                    lidar2ego=lidar2ego,
                ))

        input_dict['ann_info'] = info['annotation']

        return input_dict

    def plot_line(self, points, ctr_point_type=0, ):
        for i in range(len(points) - 1):
            plt.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], color=self.color[ctr_point_type])
    def plot_ctr_points(self, ctr_points):
        # plt.figure(figsize=(16, 8))
        for i in range(len(ctr_points)):
            ctr_point, ctr_point_type = ctr_points[i]
            for j in range(len(ctr_point)):
                x = -ctr_point[j][1] + np.array(self.patch_size[1])/2
                y = -ctr_point[j][0] + np.array(self.patch_size[0])/2
                plt.scatter(x, y, c=self.color[ctr_point_type])
    def plot_ego_points(self, ego_points):
        plt.figure(figsize=(16, 8))
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        for i in range(len(ego_points)):
            ego_point, ego_point_type = ego_points[i]
            self.plot_line(ego_point, ego_point_type)
        plt.imshow(self.car_img, extent=[-1.5, 1.5, -1.5, 1.5])
        plt.text(-25, 16, 'lidar timestamp:  ' + self.data_info["timestamp"], color='red', fontsize=26)
    def plot_map_points(self, map_vectors):
        plt.figure(figsize=(8, 16))
        plt.xlim(0, 200)
        plt.ylim(0, 400)
        plt.imshow(self.lidar_car_img, extent=[90, 110, 185, 215])
        for i in range(len(map_vectors)):
            map_vector, map_vector_type = map_vectors[i]
            self.plot_line(map_vector, map_vector_type)
    def plot_semantic_map(self, semantic_map):
        plt.figure(figsize=(16, 10))
        plt.subplot(1, 3, 1)
        plt.title('divider')
        plt.imshow(semantic_map[0][0], cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('ped_crossing')
        plt.imshow(semantic_map[0][1], cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('boundary')
        plt.imshow(semantic_map[0][2], cmap='gray')
    def plot_curve_recovery_by_ctr_points(self, ctr_points):
        plt.figure(figsize=(16, 8))
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        # pdb.set_trace()
        for i in range(len(ctr_points)):
            ctr_point, ctr_point_type = ctr_points[i]
            # pdb.set_trace()
            degree = self.num_degrees[ctr_point_type]
            pbc_func = self.pbc_funcs[degree]
            # print(ctr_point)
            if ctr_point_type == 1:
                points = []
                for point in ctr_point:
                    new_point = -point[::-1] + np.array(self.patch_size[::-1]) / 2
                    points.append(new_point)
                self.plot_line(points, ctr_point_type)
            else:
                ctr_point = -ctr_point[:, ::-1] + np.array(self.patch_size[::-1]) / 2
                num_piece = len(ctr_point) // degree
                for j in range(num_piece):
                    curve_recovery = pbc_func.bezier_coefficient.dot(ctr_point[degree * j:degree * (j + 1) + 1])
                    self.plot_line(curve_recovery, ctr_point_type)

        plt.text(-25, 16, 'curve recovery by ctr_points ', color='red', fontsize=26)
        plt.imshow(self.car_img, extent=[-1.5, 1.5, -1.5, 1.5])

    def mask_for_lines(self, lines, mask, thickness, idx, trans_type='index'):
        coords = np.asarray(list(lines.coords), np.int32)
        coords = coords[:, :2].reshape((-1, 2))  # 3D --> 2D
        if len(coords) < 2:
            return mask, idx
        for i, t in enumerate(thickness):
            # pdb.set_trace()
            if trans_type == 'index':
                cv2.polylines(mask[i], np.array([coords]), False, color=idx, thickness=t)
                idx += 1
        return mask, idx

    def ped_crossing_bezier_convert(self, idx, ctr_points, ego_points, instance_masks, map_vectors):
        ped_crossing = self.data_info["annotation"]["ped_crossing"]
        num_ped_crossing = len(ped_crossing)
        # print("num_ped_crossing", num_ped_crossing)

        if num_ped_crossing > 0:
            for i in range(num_ped_crossing):
                ped_instance = ped_crossing[i]
                num_points = ped_instance.shape[0]
                if num_points <= 1:
                    continue
                segment_lengths = []
                for j in range(num_points - 1):
                    p1 = np.array(ped_instance[j][:2])
                    p2 = np.array(ped_instance[j + 1][:2])
                    dist = np.linalg.norm(p2 - p1)
                    segment_lengths.append(dist)

                if not segment_lengths:
                    print("timestamp", self.data_info["timestamp"])    # 315965800260014000
                    raise ValueError("LineString does not contain enough points to form a segment.")
                max_length = max(segment_lengths)
                max_index = segment_lengths.index(max_length)
                pt1 = np.array(self.patch_size) / 2 - ped_instance[max_index][:2][::-1]
                pt2 = np.array(self.patch_size) / 2 - ped_instance[max_index + 1][:2][::-1]
                # pdb.set_trace()
                ctr_points.append(((pt1, pt2), 1))
                idx = self.ego_points_and_map_vectors(ped_instance[max_index][:2], ped_instance[max_index + 1][:2], idx,
                                                ego_points, instance_masks, map_vectors)
                if num_points > 3:
                    idx2 = np.argsort(segment_lengths)[-2]        # 第二长
                    pt3 = np.array(self.patch_size) / 2 - ped_instance[idx2][:2][::-1]
                    pt4 = np.array(self.patch_size) / 2 - ped_instance[idx2 + 1][:2][::-1]
                    ctr_points.append(((pt3, pt4), 1))
                    idx = self.ego_points_and_map_vectors(ped_instance[idx2][:2], ped_instance[idx2 + 1][:2], idx,
                                                          ego_points, instance_masks, map_vectors)

    # def ped_crossing_bezier_converter(self, idx, ctr_points, ego_points, instance_masks, map_vectors):
    #     ped_crossing = self.data_info["annotation"]["ped_crossing"]
    #     num_ped_crossing = len(ped_crossing)
    #     # print("num_ped_crossing", num_ped_crossing)
    #
    #     if num_ped_crossing > 0:
    #         for i in range(num_ped_crossing):
    #             ped_instance = ped_crossing[i]
    #             num_points = ped_instance.shape[0]
    #             segment_lengths = []
    #             for j in range(num_points - 1):
    #                 p1 = np.array(ped_instance[j][:2])
    #                 p2 = np.array(ped_instance[j + 1][:2])
    #                 dist = np.linalg.norm(p2 - p1)
    #                 segment_lengths.append(dist)
    #
    #             # print(segment_lengths)
    #             max_length = max(segment_lengths)
    #             max_index = segment_lengths.index(max_length)
    #             # print("max_index", max_index)
    #             if num_points < 4:
    #                 ctr_points.append(((ped_instance[max_index][:2], ped_instance[max_index + 1][:2]), 1))
    #                 self.ego_points_and_map_vectors(ped_instance[max_index][:2], ped_instance[max_index + 1][:2], idx, ego_points, instance_masks, map_vectors)
    #             else:
    #                 ctr_points.append(((ped_instance[max_index][:2], ped_instance[max_index + 1][:2]), 1))
    #                 self.ego_points_and_map_vectors(ped_instance[max_index][:2], ped_instance[max_index + 1][:2], idx, ego_points, instance_masks, map_vectors)
    #                 if max_index < 2:
    #                     ctr_points.append(((ped_instance[max_index + 2][:2], ped_instance[max_index + 3][:2]), 1))
    #                     self.ego_points_and_map_vectors(ped_instance[max_index + 2][:2], ped_instance[max_index + 3][:2],
    #                                                idx, ego_points, instance_masks, map_vectors)
    #                 else:
    #                     ctr_points.append(((ped_instance[max_index - 2][:2], ped_instance[max_index - 1][:2]), 1))
    #                     self.ego_points_and_map_vectors(ped_instance[max_index - 2][:2], ped_instance[max_index - 1][:2],
    #                                                idx, ego_points, instance_masks, map_vectors)

        # for i in range(len(ctr_points)):
        #     plot_line(ctr_points[i],"1")

    def ego_points_and_map_vectors(self, p1, p2, idx, ego_points, instance_masks, map_vectors):
        map_masks_ret = []
        map_masks = np.zeros((len(self.thickness), *self.canvas_size), np.uint8)  # [2, 200, 400]

        ped_line = LineString((p1, p2))
        distances = np.arange(0, ped_line.length, 1)
        sampled_point = np.array([list(ped_line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        sampled_point = np.concatenate((sampled_point, p2.reshape(1, -1)), axis=0)
        ego_points.append((sampled_point, 1))
        # print("number of sampled points", len(sampled_point))

        vectorize_line = LineString(sampled_point)
        # print("koordinates: ", np.array(list(instance_line.coords)))
        ped_line = affinity.affine_transform(vectorize_line, [1.0, 0.0, 0.0, 1.0, self.trans_x, self.trans_y])
        ped_line = affinity.scale(ped_line, xfact=self.scale_width, yfact=self.scale_height, origin=(0, 0))
        # pdb.set_trace()
        map_masks, idx = self.mask_for_lines(ped_line, map_masks, self.thickness, idx, )
        for i in range(len(self.thickness)):
            map_masks_ret.append(np.flip(np.rot90(map_masks[i][None], k=1, axes=(1, 2)), axis=2)[0])

        map_masks_ret = np.array(map_masks_ret)
        instance_masks[:, 1, :, :] += map_masks_ret

        map_points = np.array(self.canvas_size) - np.array(ped_line.coords[:])[:, :2][:, ::-1]
        map_vectors.append((map_points, 1))
        return idx

    def divider_and_boundary_bezier_converter(self, vectors, ctr_points, ego_points, instance_masks, map_vectors, idx):
        for instance, instance_type in vectors:
            map_masks_ret = []
            map_masks = np.zeros((len(self.thickness), *self.canvas_size), np.uint8)  # [2, 200, 400]
            if instance_type != -1:
                if instance.geom_type == 'LineString':
                    distances = np.arange(0, instance.length, 1)
                    sampled_point = np.array(
                        [list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 3)
                else:
                    print(instance.geom_type)


            last_point = np.array(list(instance.coords)[-1]).reshape(1, -1)
            sampled_point = np.concatenate((sampled_point, last_point), axis=0)  # 线段长度小于1时，sampled_point就只剩一个点
            ego_points.append((sampled_point, instance_type))
            # print("number of sampled points", len(sampled_point))


            instance_line = LineString(sampled_point)
            # print("koordinates: ", np.array(list(instance_line.coords)))
            new_line = affinity.affine_transform(instance_line, [1.0, 0.0, 0.0, 1.0, self.trans_x, self.trans_y])
            new_line = affinity.scale(new_line, xfact=self.scale_width, yfact=self.scale_height, origin=(0, 0))

            map_masks, idx = self.mask_for_lines(new_line, map_masks, self.thickness, idx, )
            for i in range(len(self.thickness)):
                map_masks_ret.append(np.flip(np.rot90(map_masks[i][None], k=1, axes=(1, 2)), axis=2)[0])
            # pdb.set_trace()
            map_masks_ret = np.array(map_masks_ret)
            instance_masks[:, instance_type, :, :] += map_masks_ret


            pts = np.array(self.canvas_size) - np.array(new_line.coords[:])[:, :2][:, ::-1]
            map_vectors.append((pts, instance_type))


            pts2 = np.array(self.patch_size)/2 - sampled_point[:, :2][:, ::-1]
            pbc_func = self.pbc_funcs[self.num_degrees[instance_type]]
            ctr_points.append((pbc_func(pts2), instance_type))
            # pdb.set_trace()
        return ctr_points, ego_points, map_vectors, instance_masks, idx

    def convert(self):
        vectors = []
        ego_points = []
        ctr_points = []
        idx = 1
        map_vectors = []
        instance_masks = np.zeros((len(self.thickness), self.max_channel, self.canvas_size[1], self.canvas_size[0]), np.uint8)            # 2,3,400,200

        for i in range(len(self.vec_class)):
            instance_list = self.data_info["annotation"][self.vec_class[i]]
            for instance in instance_list:
                if instance.shape[0] < 2:
                    continue
                vectors.append((LineString(np.array(instance)), self.CLASS2LABEL.get(self.vec_class[i], -1)))

        ctr_points, ego_points, map_vectors, instance_masks, idx = self.divider_and_boundary_bezier_converter(vectors,
                                                                                                              ctr_points,
                                                                                                              ego_points,
                                                                                                              instance_masks,
                                                                                                              map_vectors,
                                                                                                              idx)
        self.ped_crossing_bezier_convert(idx, ctr_points, ego_points, instance_masks, map_vectors)
        return ctr_points, ego_points, map_vectors, instance_masks

def process_data(args):
    data_info, save_dir = args
    # 处理数据逻辑
    bezier_convert = BezierConverter(data_info)
    input_dict = bezier_convert.get_data_info(data_info)
    ctr_points, ego_points, map_vectors, instance_masks = bezier_convert.convert()
    semantic_masks = (instance_masks != 0).astype(np.uint8)  # 将不同的实例都变成1，保持语义一致

    instance_map_points, instance_ego_points, instance_ctr_points = [], [], []
    for i in range(len(ctr_points)):
        pts, pts_type = ctr_points[i]
        instance_ctr_points.append({'pts': pts, 'pts_num': len(pts), 'type': pts_type})
    for i in range(len(map_vectors)):
        pts, pts_type = map_vectors[i]
        instance_map_points.append({'pts': pts, 'pts_num': len(pts), 'type': pts_type})
    for i in range(len(ego_points)):
        pts, pts_type = ego_points[i]
        instance_ego_points.append({'pts': pts, 'pts_num': len(pts), 'type': pts_type})

    file_path = os.path.join(save_dir, input_dict["timestamp"] + '.npz')
    np.savez_compressed(file_path, input_dict=input_dict, instance_mask=instance_masks[0], instance_mask8=instance_masks[1],
                        semantic_mask=semantic_masks[0], ctr_points=instance_ctr_points, ego_points=instance_ego_points,
                        map_vectors=instance_map_points)
    return file_path

def process_data_infos(data_infos, save_dir):
    # 创建一个包含数据和保存目录的参数列表
    args = [(data_info, save_dir) for data_info in data_infos]

    # 使用多进程池并行处理数据
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.map_async(process_data, args).get(), total=len(data_infos)))

    return results

def main():
    save_dir = "/home/sun/Bev/BeMapNet/data/argoverse2/customer_v"
    ann_file = "/home/sun/MapTR/data/argoverse2/sensor/av2_map_infos_val.pkl"
    load_interval = 1

    data = mmcv.load(ann_file, file_format='pkl')
    data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))
    data_infos = data_infos[::load_interval]

    results = process_data_infos(data_infos, save_dir)
    print(f"Processing completed. Files saved to: {save_dir}")




if __name__ == '__main__':
    main()