from PIL import Image
import cv2
import os.path as osp

sample_dir = "/home/sun/Bev/BeMapNet/visual_bezier"
gt_path = "/home/sun/Bev/BeMapNet/visual_bezier/GT.png"
pred_path ="/home/sun/Bev/BeMapNet/visual_bezier/PRED_MAP_plot.png"
surroud_path ="/home/sun/Bev/BeMapNet/visual_bezier/surroud_view.jpg"

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

cams_img_path = osp.join(sample_dir, 'con.jpg')
cv2.imwrite(cams_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])