import pdb
import torch
import numpy as np
import torch.nn as nn
from bemapnet.models.bev_decoder.transformer import Transformer
from bemapnet.models.bev_decoder.LSS import TransformerDepth, LSSTransform


class TransformerBEVDecoder(nn.Module):
    def __init__(self, key='im_bkb_features', **kwargs):
        super(TransformerBEVDecoder, self).__init__()
        self.bev_encoder = Transformer(**kwargs)
        self.key = key

    def forward(self, inputs):
        assert self.key in inputs
        feats = inputs[self.key]
        fuse_feats = feats[-1]                                                                # [6, 600, 21, 49]
        # pdb.set_trace()
        fuse_feats = fuse_feats.reshape(*inputs['images'].shape[:2], *fuse_feats.shape[-3:])  # [1, 6, 600, 21, 49]
        fuse_feats = torch.cat(torch.unbind(fuse_feats, dim=1), dim=-1)                       # [1, 600, 21, 294]

        cameras_info = {
            'extrinsic': inputs.get('extrinsic', None),
            'intrinsic': inputs.get('intrinsic', None),
            'ida_mats': inputs.get('ida_mats', None),
            'do_flip': inputs['extra_infos'].get('do_flip', None)
        }

        _, _, bev_feats = self.bev_encoder(fuse_feats, cameras_info=cameras_info)             # [4, 1, 512, 64, 32]

        return {"bev_enc_features": list(bev_feats)}



class TransformerBEVDecoderDepth(nn.Module):
    def __init__(self, key='im_bkb_features', **kwargs):
        super(TransformerBEVDecoderDepth, self).__init__()
        self.bev_encoder = TransformerDepth(**kwargs)
        self.key = key

    def forward(self, inputs):
        assert self.key in inputs
        feats = inputs[self.key]
        fuse_feats = feats[-1]                                                                # [6, 600, 21, 49]
        # pdb.set_trace()
        fuse_feats = fuse_feats.reshape(*inputs['images'].shape[:2], *fuse_feats.shape[-3:])  # [1, 6, 600, 21, 49]
        fuse_feats = torch.cat(torch.unbind(fuse_feats, dim=1), dim=-1)                       # [1, 600, 21, 294]

        cameras_info = {
            'extrinsic': inputs.get('extrinsic', None),
            'intrinsic': inputs.get('intrinsic', None),
            'ida_mats': inputs.get('ida_mats', None),
            'do_flip': inputs['extra_infos'].get('do_flip', None)
        }

        _, _, bev_feats = self.bev_encoder(fuse_feats, cameras_info=cameras_info)             # [4, 1, 512, 64, 32]

        return {"bev_enc_features": list(bev_feats)}


class TransformerBEVDecoderLSS(nn.Module):
    def __init__(self, key='im_bkb_features', **kwargs):
        super(TransformerBEVDecoderLSS, self).__init__()
        self.bev_encoder = LSSTransform(**kwargs)
        self.key = key
        self.ouput_proj = nn.Conv2d(256, 512, kernel_size=1)


    def forward(self, inputs):
        assert self.key in inputs
        feats = inputs[self.key]
        fuse_feats = feats[-1]                                                                # [6, 600, 21, 49]
        # pdb.set_trace()
        fuse_feats = fuse_feats.reshape(*inputs['images'].shape[:2], *fuse_feats.shape[-3:])  # [1, 6, 600, 21, 49]
        # fuse_feats = torch.cat(torch.unbind(fuse_feats, dim=1), dim=-1)                       # [1, 600, 21, 294]

        ida_mats = inputs.get('ida_mats', None)

        cameras_info = {
            'camera2ego': inputs.get('extrinsic', None),                         # [1, 6, 4, 4]
            'camera_intrinsics': inputs.get('intrinsic', None),                  # [1, 6, 3, 3]
            'img_aug_matrix': inputs.get('ida_mats', None),                      # [1, 6, 3, 3]
            'do_flip': inputs['extra_infos'].get('do_flip', None),
            'lidar2ego': inputs.get('lidar_calibrated_sensor', None),
            'img_shape': inputs['images'].shape,
        }

        ret_dict = self.bev_encoder(fuse_feats, img_metas=cameras_info)   # origin [4, 1, 512, 64, 32]. bev [1, 256, 200, 100] depth [1, 6, 68, 21, 49]

        bev_enc_features = self.ouput_proj(ret_dict['bev'])
        bev_enc_features = bev_enc_features.view(-1, *bev_enc_features.shape)
        # pdb.set_trace()
        return {"bev_enc_features": list(bev_enc_features), 'pred_depth': ret_dict['depth']}