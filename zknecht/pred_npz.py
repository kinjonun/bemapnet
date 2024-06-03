import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

color = {0: 'orange', 1: 'blue', 2: 'red'}
data = np.load('/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50/2024-05-31/evaluation_epoch_1/results/315965785159642000.npz', allow_pickle=True)
# ['dt_res', 'dt_mask']
dt_mask = data['dt_mask']       # [3, 400, 200]
dt_res = data['dt_res'].tolist()            # ['map', 'confidence_level', 'pred_label']
res = dict(dt_res)
print(res.keys())
points = res["map"]
label = res["pred_label"]
# print(label)