import os
import pdb
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from skimage import io as skimage_io
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes import NuScenes
import matplotlib.pyplot as plt

def depth_transform(cam_depth, img_shape, resize_dims, crop, flip, rotate):   # (900, 1600)
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """
    # pdb.set_trace()
    resize_dims = np.array(resize_dims)[::-1]
    H, W = resize_dims            # (512, 896)
    resize_h = H /img_shape[0]
    resize_w = W /img_shape[1]
    cam_depth[:, 0] = cam_depth[:, 0] * resize_w       # (3379, 3)
    cam_depth[:, 1] = cam_depth[:, 1] * resize_h       # (3379, 3)
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]                         # (3379, 3)
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0           # 移动原点到中心便于做旋转
    cam_depth[:, 1] -= H / 2.0
    h = rotate / 180 * np.pi
    rot_matrix = [[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)],]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T
    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros((crop[3]-crop[1], crop[2]-crop[0]))    # [384, 896]
    valid_mask = ((depth_coords[:, 1] < resize_dims[0]) & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0) & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)


def map_pointcloud_to_image(lidar_points, img, lidar_calibrated_sensor, lidar_ego_pose, cam_calibrated_sensor,
                            cam_ego_pose,  min_dist: float = 0.0,):

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud. Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths
    # pdb.set_trace()
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(lidar_points.points[:3, :], np.array(cam_calibrated_sensor['camera_intrinsic']), normalize=True)  # (3, 34752)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make sure points are at least 1m in front of the camera to
    # avoid seeing the lidar points on the camera casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    # mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)      # 1600
    mask = np.logical_and(mask, points[0, :] < img[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    # mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)      # 900
    mask = np.logical_and(mask, points[1, :] < img[0] - 1)
    points = points[:, mask]         # (3, 3379)
    coloring = coloring[mask]        # (3379,)
    return points, coloring


class NuScenesMapDatasetDepth(Dataset):
    def __init__(self, img_key_list, map_conf, ida_conf, bezier_conf, transforms, data_split="training"):
        super().__init__()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        self.ida_conf = ida_conf
        self.bez_conf = bezier_conf
        self.ego_size = map_conf["ego_size"]
        self.mask_key = map_conf["mask_key"]
        self.nusc_root = map_conf["nusc_root"]
        self.anno_root = map_conf["anno_root"]
        self.split_dir = map_conf["split_dir"]
        self.num_degree = bezier_conf["num_degree"]
        self.max_pieces = bezier_conf["max_pieces"]
        self.max_instances = bezier_conf["max_instances"]
        self.split_mode = 'train' if data_split == "training" else 'val'
        split_path = os.path.join(self.split_dir, f'{self.split_mode}.txt')
        # split_path = os.path.join('/home/sun/Bev/BeMapNet/assets/splits/nuscenes/ein.txt')
        self.tokens = [token.strip() for token in open(split_path).readlines()]
        self.transforms = transforms
        self.return_depth = True

    def __getitem__(self, idx: int):
        token = self.tokens[idx]
        sample = np.load(os.path.join(self.anno_root, f'{token}.npz'), allow_pickle=True)
        resize_dims, crop, flip, rotate = self.sample_ida_augmentation()   # (896, 512), (0, 128, 896, 512),
        # pdb.set_trace()
        images, ida_mats, lidar_depth = [], [], []
        lidar_filename = np.array(sample['lidar_filename']).tolist()
        lidar_calibrated_sensor = dict(
            rotation=sample['lidar_rot'],
            translation=sample['lidar_tran'])
        lidar_ego_pose = dict(
            rotation=sample['lidar_ego_pose_rot'],
            translation=sample['lidar_ego_pose_tran'])

        lidar_points = np.fromfile(os.path.join(self.nusc_root, lidar_filename), dtype=np.float32,
                                   count=-1).reshape(-1, 5)[..., :4]

        cam_ego = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        # pdb.set_trace()
        for i in range(len(sample['image_paths'])):
            im_path = sample['image_paths'][i]
            im_path = os.path.join(self.nusc_root, im_path)
            img = skimage_io.imread(im_path)
            cam_calibrated_sensor = dict(
                rotation=sample['rots'][i],
                translation=sample['trans'][i],
                camera_intrinsic=sample['intrins'][i])
            cam_ego_pose = dict(
                rotation=sample['cam_ego_pose_rots'][i],
                translation=sample['cam_ego_pose_trans'][i])
            cam_ego[i, :3, :3] = Quaternion(sample['cam_ego_pose_rots'][i]).rotation_matrix
            cam_ego[i, :3, 3] = sample['cam_ego_pose_trans'][i]
            # pdb.set_trace()
            if self.return_depth:
                pts_img, depth = map_pointcloud_to_image(lidar_points.copy(), img.shape[:2], lidar_calibrated_sensor.copy(),
                                                     lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
                point_depth = np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1).astype(np.float32)
                point_depth_augmented = depth_transform(point_depth, img.shape[:2], resize_dims, crop, flip, rotate)
                lidar_depth.append(point_depth_augmented)

            img, ida_mat = self.img_transform(img, resize_dims, crop, flip, rotate)
            images.append(img)
            ida_mats.append(ida_mat)

        # plt.imshow(lidar_depth[0], cmap='hot')
        # plt.colorbar()  # 添加颜色条
        # plt.show()  # 显示图片
        # pdb.set_trace()

        rots = []
        for rot in sample['rots']:
            extrin = Quaternion(rot).rotation_matrix
            rots.append(extrin)

        extrinsic = np.stack([np.eye(4) for _ in range(sample["trans"].shape[0])], axis=0)
        lidar2ego = np.eye(4)
        extrinsic[:, :3, :3] = np.array(rots)
        extrinsic[:, :3, 3] = sample["trans"]
        lidar2ego[:3, :3] = np.array(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar2ego[:3, 3] = lidar_calibrated_sensor['translation']
        intrinsic = sample['intrins']
        # image_paths = torch.tensor(sample['image_paths'].tolist())
        ctr_points = np.zeros((self.max_instances, max(self.max_pieces) * max(self.num_degree) + 1, 2), dtype=np.float)
        ins_labels = np.zeros((self.max_instances, 3), dtype=np.int16) - 1

        for ins_id, ctr_info in enumerate(sample['ctr_points']):
            cls_id = int(ctr_info['type'])
            ctr_pts_raw = np.array(ctr_info['pts'])
            max_points = self.max_pieces[cls_id] * self.num_degree[cls_id] + 1
            num_points = max_points if max_points <= ctr_pts_raw.shape[0] else ctr_pts_raw.shape[0]
            assert num_points >= self.num_degree[cls_id] + 1
            ctr_points[ins_id][:num_points] = np.array(ctr_pts_raw[:num_points])
            ins_labels[ins_id] = [cls_id, (num_points - 1) // self.num_degree[cls_id] - 1, num_points]

        masks = sample[self.mask_key]

        if flip:
            new_order = [2, 1, 0, 5, 4, 3]
            img_key_list = [self.img_key_list[i] for i in new_order]
            images = [images[i] for i in new_order]
            ida_mats = [ida_mats[i] for i in new_order]
            extrinsic = [extrinsic[i] for i in new_order]
            intrinsic = [intrinsic[i] for i in new_order]
            masks = [np.flip(mask, axis=1) for mask in masks]
            ctr_points = self.point_flip(ctr_points, ins_labels, self.ego_size)

        item = dict(
            images=images, targets=dict(masks=masks, points=ctr_points, labels=ins_labels),
            extrinsic=np.stack(extrinsic), intrinsic=np.stack(intrinsic), ida_mats=np.stack(ida_mats),
            extra_infos=dict(token=token, img_key_list=self.img_key_list, map_size=self.ego_size, do_flip=flip),
            cam_ego_pose=cam_ego, lidar_ego_pose=lidar_ego_pose, lidar_calibrated_sensor=lidar2ego,
        )
        if self.transforms is not None:
            item = self.transforms(item)

        if self.return_depth:
            item['lidar_depth'] = lidar_depth
        return item

    def __len__(self):
        return len(self.tokens)

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        resize_dims = w, h = self.ida_conf["resize_dims"]
        crop = (0, 0, w, h)
        if self.ida_conf["up_crop_ratio"] > 0:
            crop = (0, int(self.ida_conf["up_crop_ratio"] * h), w, h)
        flip, color, rotate_ida = False, False, 0
        if self.split_mode == "train":
            if self.ida_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            if self.ida_conf["rot_lim"]:
                assert isinstance(self.ida_conf["rot_lim"], (tuple, list))
                rotate_ida = np.random.uniform(*self.ida_conf["rot_lim"])
        return resize_dims, crop, flip, rotate_ida

    def img_transform(self, img, resize_dims, crop, flip, rotate):      # (896, 512), (0, 128, 896, 512)
        img = Image.fromarray(img)
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        w, h = img.size
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        scales = torch.tensor([resize_dims[0] / w, resize_dims[1] / h])
        ida_rot *= torch.Tensor(scales)
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = ida_rot.new_zeros(3, 3)
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return np.asarray(img), ida_mat

    @staticmethod
    def point_flip(points, labels, map_shape):

        def _flip(pts):
            pts[:, 0] = map_shape[1] - pts[:, 0]
            return pts.copy()

        points_ret = deepcopy(points)
        for ins_id in range(points.shape[0]):
            end = labels[ins_id, 2]
            points_ret[ins_id][:end] = _flip(points[ins_id][:end])

        return points_ret

    @staticmethod
    def get_rot(h):
        return torch.Tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
