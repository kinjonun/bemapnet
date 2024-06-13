from bemapnet.dataset.nuscenes_depth import NuScenesMapDatasetDepth
from torchvision.transforms import Compose
from bemapnet.dataset.transform import Normalize, ToTensor

map_conf = dict(
    dataset_name="nuscenes",
    nusc_root="data/nuscenes",
    anno_root="data/nuscenes/customer/bemapnet_lidar",
    split_dir="assets/splits/nuscenes",
    num_classes=3,
    ego_size=(60, 30),
    map_region=(30, 30, 15, 15),
    map_resolution=0.15,
    map_size=(400, 200),
    mask_key="instance_mask8",
    line_width=8,
    save_thickness=1,
)

bezier_conf = dict(
    num_degree=(2, 1, 3),
    max_pieces=(3, 1, 7),
    num_points=(7, 2, 22),
    piece_length=100,
    max_instances=40,
)

dataset_setup = dict(
    img_key_list=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    img_norm_cfg=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True),
)
ida_conf = dict(resize_dims=(896, 512), up_crop_ratio=0.25, rand_flip=True, rot_lim=False)

transform = Compose(
            [
                Normalize(**dataset_setup["img_norm_cfg"]),
                ToTensor(),
            ]
        )

nusc = NuScenesMapDatasetDepth(img_key_list=dataset_setup["img_key_list"],
            map_conf=map_conf,
            ida_conf=ida_conf,
            bezier_conf=bezier_conf,
            transforms=transform,
            data_split="training",)

print(nusc[0]["lidar_depth"])
print("extrinsic", nusc[0]["extrinsic"])