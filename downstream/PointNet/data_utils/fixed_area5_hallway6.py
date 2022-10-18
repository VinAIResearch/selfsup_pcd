import indoor3d_util
import numpy as np


# fp = open('/home/ubuntu/Stanford3dDataset/Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt')
# i=0
# for line in fp.readlines():
#     i+=1
#     if len(line.strip().split(' '))!=6:
#         print(i)
anno_path = "/home/ubuntu/Stanford3dDataset/Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations"
elements = anno_path.split("/")
out_filename = elements[-3] + "_" + elements[-2] + ".npy"  # Area_1_hallway_1.npy
indoor3d_util.collect_point_label(anno_path, "/home/ubuntu/stanford_indoor3d/Area_5_hallway_6.npy", "numpy")
