import os
from os import path

img_lst = []
for r, d, f in os.walk('/data/raw/robotics/kitti/raw_sequences-20200224133836/data'):
    for file in f:
        if ('image_02' in r or 'image_03' in r) and ('groundtruth' not in r) and 'png' in file:
            img_lst.append(os.path.join(r, file))

depth_lst = []
for img_path in img_lst:
    parts = img_path.split('/')
    depth_lst.append(os.path.join('/'.join(parts[:9]), 'proj_depth/groundtruth/', parts[9], parts[11]))

img_path = []
depth_path = []
for i in range(len(img_lst)):
    if path.exists(depth_lst[i]):
        img_path.append(img_lst[i])
        depth_path.append(depth_lst[i])

with open('image_sequence.txt', 'w') as file:
    for img in img_path:
        file.write(img+'\n')

with open('depth_sequence.txt', 'w') as file:
    for depth in depth_path:
        file.write(depth+'\n')

