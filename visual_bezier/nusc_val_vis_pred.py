import os
import os.path as osp
import pdb
import argparse
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from torch.optim import AdamW
from torchvision.transforms import Compose
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from bemapnet.engine.callbacks import CheckPointLoader, CheckPointSaver, ClearMLCallback, ProgressBar
from bemapnet.engine.callbacks import TensorBoardMonitor, TextMonitor, ClipGrad
from bemapnet.models.network import BeMapNet
from bemapnet.engine.core import BeMapNetCli
from bemapnet.engine.experiment import BaseExp
from bemapnet.dataset.nusc_dataset import NuScenesMapDataset
from bemapnet.dataset.transform import Normalize, ToTensor
from bemapnet.utils.misc import get_param_groups, is_distributed
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2

class EXPConfig:

    CLASS_NAMES = ["lane_divider", "ped_crossing", "drivable_area"]
    IMAGE_SHAPE = (900, 1600)
    ida_conf = dict(resize_dims=(896, 512), up_crop_ratio=0.25, rand_flip=True, rot_lim=False)
    INPUT_SHAPE = [int(ida_conf["resize_dims"][1] * (1 - ida_conf["up_crop_ratio"])), int(ida_conf["resize_dims"][0])]

    map_conf = dict(
        dataset_name="nuscenes",
        nusc_root="data/nuscenes",
        anno_root="data/nuscenes/customer/bemapnet",
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

    model_setup = dict(
        im_backbone=dict(
            arch_name="resnet",
            bkb_kwargs=dict(
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='assets/weights/resnet50-0676ba61.pth'),  # from pytorch
                with_cp=True,
            ),
            ret_layers=3,
            fpn_kwargs=dict(
                conv_channels=(512, 1024, 2048),
                fpn_cell_repeat=3,
                fpn_num_filters=120,  # 128 -> 120  avoid OOM only
                norm_layer=nn.SyncBatchNorm,
                use_checkpoint=True,
                tgt_shape=(21, 49)
            ),
        ),
        bev_decoder=dict(
            arch_name="transformer",
            net_kwargs=dict(
                key='im_nek_features',
                in_channels=600,
                src_shape=(21, 49*6),
                query_shape=(64, 32),
                d_model=512,
                nhead=8,
                num_encoder_layers=2,
                num_decoder_layers=4,
                dim_feedforward=1024,
                src_pos_embed='ipm_learned',
                tgt_pos_embed='ipm_learned',
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=True,
                use_checkpoint=True,
                ipm_proj_conf=dict(
                    map_size=map_conf["map_size"],
                    map_resolution=map_conf["map_resolution"],
                    input_shape=(512, 896)
                )
            ),
        ),
        ins_decoder=dict(
            arch_name="mask2former",
            net_kwargs=dict(
                decoder_ids=[0, 1, 2, 3, 4, 5],
                in_channels=512,
                tgt_shape=(200, 100),
                num_feature_levels=1,
                mask_classification=True,
                num_classes=1,
                hidden_dim=512,
                num_queries=60,
                nheads=8,
                dim_feedforward=2048,
                dec_layers=6,
                pre_norm=False,
                mask_dim=512,
                enforce_input_project=False
            ),
        ),
        output_head=dict(
            arch_name="bezier_output_head",
            net_kwargs=dict(
                in_channel=512,
                num_queries=[20, 25, 15],
                tgt_shape=map_conf['map_size'],
                num_degree=bezier_conf["num_degree"],
                max_pieces=bezier_conf["max_pieces"],
                bev_channels=512,
                ins_channel=64,
            )
        ),
        post_processor=dict(
            arch_name="bezier_post_processor",
            net_kwargs=dict(
                map_conf=map_conf,
                bezier_conf=bezier_conf,
                criterion_conf=dict(
                    bev_decoder=dict(
                        weight=[0.5, 0.8, 1.2, 1.8],
                        sem_mask_loss=dict(
                            ce_weight=1, dice_weight=1, use_point_render=True,
                            num_points=20000, oversample=3.0, importance=0.9)
                    ),
                    ins_decoder=dict(
                        weight=[0.4, 0.4, 0.4, 0.8, 1.2, 1.6],
                    ),
                    loss_weight=dict(
                        sem_loss=0.5,
                        obj_loss=2, ctr_loss=5, end_loss=2, msk_loss=5, curve_loss=10, recovery_loss=1)
                ),
                matcher_conf=dict(
                    cost_obj=2, cost_ctr=5, cost_end=2, cost_mask=5, cost_curve=10, cost_recovery=1,
                    ins_mask_loss_conf=dict(ce_weight=1, dice_weight=1,
                                            use_point_render=True, num_points=20000, oversample=3.0, importance=0.9),
                    point_loss_conf=dict(ce_weight=0, dice_weight=1, curve_width=5, tgt_shape=map_conf["map_size"])
                ),
                no_object_coe=0.5,
            )
        )
    )

    optimizer_setup = dict(
        base_lr=2e-4, wd=1e-4, backb_names=["backbone"], backb_lr=5e-5, extra_names=[], extra_lr=5e-5, freeze_names=[]
    )

    scheduler_setup = dict(milestones=[0.7, 0.9, 1.0], gamma=1 / 3)

    metric_setup = dict(
        class_names=CLASS_NAMES,
        map_resolution=map_conf["map_resolution"],
        iou_thicknesses=(1,),
        cd_thresholds=(0.2, 0.5, 1.0, 1.5, 5.0)
    )

    VAL_TXT = [
        "assets/splits/nuscenes/val.txt",
        "assets/splits/nuscenes/day.txt", "assets/splits/nuscenes/night.txt",
        "assets/splits/nuscenes/sunny.txt", "assets/splits/nuscenes/cloudy.txt", "assets/splits/nuscenes/rainy.txt",
    ]
def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)
    config = EXPConfig()

    if args.show_dir is None:
        args.show_dir = osp.join('./outputs',
                                'bemapnet_nuscenes_res50',
                                'vis_pred')

    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    # cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    dataset_setup = config.dataset_setup

    transform = Compose(
        [
            Normalize(**dataset_setup["img_norm_cfg"]),
            ToTensor(),
        ]
    )

    val_set = NuScenesMapDataset(
        img_key_list=dataset_setup["img_key_list"],
        map_conf=config.map_conf,
        ida_conf=config.ida_conf,
        bezier_conf=config.bezier_conf,
        transforms=transform,
        data_split="validation",
    )

    sampler = None
    if is_distributed():
        sampler = DistributedSampler(val_set, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        sampler=sampler,
    )
    # print(len(val_set))

    # get color map: divider->or, ped->b, boundary->r
    colors_plt = ['orange', 'b', 'r']

    val_iter = iter(val_loader)
    model = BeMapNet(config.model_setup)
    checkpoint = torch.load(open(args.checkpoint, "rb"), map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"], strict=False)

    model.cuda()
    model.eval()

    for step in tqdm(range(len(val_loader))):
        batch = next(val_iter)
        batch['extrinsic'] = torch.tensor(batch['extrinsic']).cuda()
        batch['intrinsic'] = torch.tensor(batch['intrinsic']).cuda()
        batch['ida_mats'] = torch.tensor(batch['ida_mats']).cuda()
        # pdb.set_trace()
        with torch.no_grad():
            batch["images"] = batch["images"].float().cuda()
            outputs = model(batch)
            results, dt_masks, _ = model.post_processor(outputs["outputs"])



if __name__ == '__main__' :
    main()