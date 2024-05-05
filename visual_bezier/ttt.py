import numpy as np
import torch
from nuscenes import NuScenes
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import cv2

a =np.array(['samples/CAM_FRONT_LEFT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_LEFT__1533201481004844.jpg',
 'samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201481012460.jpg',
 'samples/CAM_FRONT_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_RIGHT__1533201481020339.jpg',
 'samples/CAM_BACK_LEFT/n015-2018-08-02-17-16-37+0800__CAM_BACK_LEFT__1533201481047423.jpg',
 'samples/CAM_BACK/n015-2018-08-02-17-16-37+0800__CAM_BACK__1533201481037525.jpg',
 'samples/CAM_BACK_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_BACK_RIGHT__1533201481027893.jpg'])

# {'token': '3be45164a35e465cbfbc3c20ef060718',
# 'timestamp': 1533201485449407,
# 'prev': '57a79294b87e4b55a7bcf58f7f8c6326',
# 'next': '7d1d71e3ce5745f7b49b9c2e5991a45b',
# 'scene_token': 'e7ef871f77f44331aefdebc24ec034b7',
# 'data': {'RADAR_FRONT': '29647f00b6ba466b82ec95e4b4c49de7',
# 'RADAR_FRONT_LEFT': '2ce16e0fadd64c3d94f44b21118ba812',
# 'RADAR_FRONT_RIGHT': '820c0794119d431d90f9091245628244',
# 'RADAR_BACK_LEFT': '43bbd98a37944339bafa5a076f7c5fe2',
# 'RADAR_BACK_RIGHT': '3587faf842b845b091dc4053a1008a46',
# 'LIDAR_TOP': '5941570fa1fa47dd8ca9f4996589e1db',
# 'CAM_FRONT': '0931140a78f04c77916259e85a57869a',
# 'CAM_FRONT_RIGHT': 'ba8e6b774da1401f862e5fc50ec45a14',
# 'CAM_BACK_RIGHT': '19d6ca1593c84aba8e064d2dc8b69eca',
# 'CAM_BACK': 'a77b3f9c712844d6b6e3527f44f768a9',
# 'CAM_BACK_LEFT': '42ae83f9a4d542ee95853535e01f139c',
# 'CAM_FRONT_LEFT': 'ab12c7892eba4353a6b337bbb8570d2c'},
# 'anns': ['e0ad1c894bf94cc6af12b41f733d61d7', '8bf7995cc5054f0fb1d716cc28edcc36', 'bf5b6942b7094e168243527ba208190b',
# '9755d191b829405c9ab1ae8720ceaa76', '2a47b35285b04043aea8dfcc5251eb26', '9945768c9c1d47cbaf326fd9b8255877',
# 'f73f989b278447f1966ad4e89846e391', 'ea5f8d23edd54e89bbd4976ea42a7b61', 'f123a2cdaf494761bc3a200890394892',
# 'eef7b6d114d74561b6a3be965b54ac33', '5b8e78fcb5c6425ea54ee9f31874d361', '06c3e212303440c89312eb3bf3ac4e31',
# '45518ac5e5274b639681ba980d92689b', 'b5e772f36f2e42c9a20922f07ccf208a', 'ca268934ab014a018cbb8cb666346b51',
# '98d48e55325044108b0a64a0b871c331', '353d3532ae0a464398db87b9ad02f1ea', '7ef7914a3f074c32b53cc752e5851109',
# '5fcf26b7189f47d8b1d918fdfbf4ffc4', '25e3ed4b846c4119996c2b43ae7a7b16', '87b5cc7d76154a2e8be6a1de2a533f77',
# '13148f662cfb4554ababa85e483cd717', 'a869244810764808aa6312d5e9fbd444', 'd976892aeabd4f7e968d5ea85882ea10',
# '817f2812deb64bbda8f6addca827fcb5', 'f5a6fb5a947042c7b01d840a2527ac07', '584254fd3dcc4dc28c1f9680b1eabeb4',
# 'd73b5ae3ad8d415b86ce413bad0c368e', 'f8b2af3543504c3687cb740b682e6325', '85718467e88149a8910621d8a041d503',
# '43fffaa5a0fd43b3b671a601f46f9389', 'e5a56af9cbef4066a56875b32ce71721', 'f42408602207427ca06f57e0c22d0e40',
# 'bd484a7c4f034a2b8390cf07336ab382', '942aeacd53e84b7286378631f84b2b43', '48085b470a3146df8f157b50d204f06f',
# '24cc60bda01e40fda9cf9b9a623a75b5', 'c9a68470207641a99eb1e2ea33f37938', 'cceff505bf174c9786ec4357f5aefab5',
# 'dc6a5d1c08c647698fcace76df8f5457', '6396de7d39c449e38f5564fe951a8140', 'a39b278ec56d4535b27da8c1123ef5d6',
# 'f64f13fe483646638f40fee7242cc353', 'abb3f29f47084d4bbffcad061ea08aed', '8c9069b1fdfe417098a21c5eaf720ba4',
# 'ae105ac571b344b488840a53d338acc0', '9f8962b498f641ef8e6792d5bea541a2']}

aa_list =a.tolist()

img_key_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
dataroot = "/media/sun/z/nuscenes/nuscenes"
sample_dir = "/home/sun/Bev/BeMapNet/visual_bezier"

# token = "3be45164a35e465cbfbc3c20ef060718"
token = "0f77ffe576ac436a87787eb343dc3f27"
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
sample = nusc.get('sample', token)

row_1_list = []
row_2_list = []
for cam in img_key_list[:3]:
    img = nusc.get('sample_data', sample['data'][cam])
    filename = img['filename']
    print(filename)
    img_path = os.path.join(dataroot, filename)
    img = cv2.imread(img_path)
    row_1_list.append(img)

for cam in img_key_list[3:]:
    img = nusc.get('sample_data', sample['data'][cam])
    filename = img['filename']
    img_path = os.path.join(dataroot, filename)
    img = cv2.imread(img_path)
    row_2_list.append(img)

row_1_img = cv2.hconcat(row_1_list)  # 水平拼接成一张图像
row_2_img = cv2.hconcat(row_2_list)
cams_img = cv2.vconcat([row_1_img, row_2_img])
cams_img_path = osp.join(sample_dir, 'surroud_view.jpg')
cv2.imwrite(cams_img_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

# for cam in img_key_list:
#  img =
# print(img_path)
#     image = Image.open(img_path)

# image.show()