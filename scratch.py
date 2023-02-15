import json
import random

import cv2
import os
import shutil
import numpy as np
#
# from preprocess import Rescale
#
str1 = 'train_data/gt_image/'
str2 = 'train_data/gt_binary_image/'
str3 = 'train_data/gt_instance_image/'
str11 = 'test_data/gt_image/'
str22 = 'test_data/gt_binary_image/'
str33 = 'test_data/gt_instance_image/'

liss = os.listdir(str1)
indices=list(range(len(liss)))
random.shuffle(indices)
tto = len(liss)
print(tto)
tto = int(tto * 0.2)
cnt = 0
# for i in indices:
#     img = cv2.imread(os.path.join(str1, liss[i]))
#     # print(np.any(np.isnan(img)))
#     # img = cv2.resize(img, (512, 256))
#     # print(img.shape)
#     j = liss[i].split('.')[-2]
#     if img.shape[0] != 540:
#         src1 = str1 + j + '.jpg'
#         src2 = str2 + j + '.png'
#         src3 = str3 + j + '.png'
#
#         dst1 = str11 + j + '.jpg'
#         dst2 = str22 + j + '.png'
#         dst3 = str33 + j + '.png'
#         shutil.move(src1, dst1)
#         shutil.move(src2, dst2)
#         shutil.move(src3, dst3)
#     cnt += 1
#     if cnt == tto:
#         break
    # cv2.imwrite('train_data/gt_binary_image/'+i, img)
#
# # map = {}
# # json_gt = [json.loads(line) for line in open("./train_data/label(5).json").readlines()]
# # name1 = [l['raw_file'].split('/')[-1] for l in json_gt]
# # print(len(name1))
# # aa = np.unique(name1)
# # print(len(aa))
# # str1 = 'train_data/gt_image/'
# # liss = os.listdir(str1)
# # print(len(liss))
# # cnt = 0
# # for i in name1:
# #     if cnt == 0:
# #         cnt += 1
# #         continue
# #     if name1[cnt-1] == name1[cnt]:
# #         print(i)
# #         break
# #     cnt += 1
# # for i in name1:
# #     i = str(i)
# #     if map[i] == 1:
# #         map[i] = 2
# #         continue
# #     map[i] = 1
# # for i in name1:
# #     if map[i] == 2:
# #         print(i)
# # for i in name1:
# #     if i not in liss:
# #         print(i)
# #         break
# #     # cnt += 1
# #     # print(cnt)
#
# # a = [1,2,3,[444,555,666],3]
# # for b in a:
# #     if type(b).__name__ == 'list':
# #         print(b)
#
# import cv2
# image_size = tuple([512,256])
# image = cv2.imread('test_images/Snipaste_2022-04-21_15-57-58.png')
# print(image.shape[0], image.shape[1])
# image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)
# print(image.shape[0], image.shape[1])
# # cv2.imshow("demo", image)
# # cv2.waitKey(0)
# # print(image[:,:,0])
# print(image[:,:,2])
import math
# import numpy as np
# import torch
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# def hierarchy_cluster(train_data, method='average', threshold=5.0):
#
#     train_data = np.array(train_data)
#     Z = linkage(train_data, method=method)
#
#     # dendrogram(Z)
#     # plt.xticks(fontsize=5, rotation=30)
#     # plt.show()
#
#     cluster_assignments = fcluster(Z, threshold, criterion='distance')
#
#     type(cluster_assignments)
#     num_clusters = cluster_assignments.max()
#     indices = get_cluster_indices(cluster_assignments)
#
#     return num_clusters, indices
#
# def get_cluster_indices(cluster_assignments):
#
#     n = cluster_assignments.max()
#     indices = []
#     for cluster_number in range(1, n + 1):
#         indices.append(np.where(cluster_assignments == cluster_number)[0])
#
#     return indices
#
# arr = [[653], [654], [655], [656], [657], [658], [659], [660], [661], [662], [663], [664], [665], [666], [667], [668], [669], [828], [829], [830], [831], [832], [833], [834], [835], [836], [837], [838], [839], [840], [841], [842], [843], [844]]
# num_clusters, indices = hierarchy_cluster(arr)
# print(num_clusters)
# print(indices)


