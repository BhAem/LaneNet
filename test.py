import json
import os
import numpy as np 
import cv2
from lane import LaneEval
from model import LaneNet 
from torch.nn import DataParallel
from clustering import lane_cluster
import torch
import ffmpeg
import argparse
import matplotlib.pyplot as plt


def _load_model(mode, model_path):
    model = LaneNet()
    if mode == 'parallel':
        model = DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    return model

def _frame_process(image_frame, model, image_size, threshold):
    image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)
    img = image.copy()

    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, :, :, :]
    image = image / 255
    image = torch.tensor(image, dtype=torch.float)
    segmentation, embeddings = model(image.cuda())

    binary_mask = segmentation.data.cpu().numpy()
    binary_mask = binary_mask.squeeze()

    exp_mask = np.exp(binary_mask - np.max(binary_mask, axis=0))
    binary_mask = exp_mask / exp_mask.sum(axis=0)
    threshold_mask = binary_mask[1, :, :] > threshold
    threshold_mask = threshold_mask.astype(np.uint8)
    threshold_mask = threshold_mask * 255
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(4, 4))
    threshold_mask = cv2.dilate(threshold_mask, kernel, iterations=1)
    mask = cv2.connectedComponentsWithStats(threshold_mask, connectivity=8, ltype=cv2.CV_32S)
    output_mask = np.zeros(threshold_mask.shape, dtype=np.uint8)
    for label in np.unique(mask[1]):
        if label == 0:
            continue
        labelMask = np.zeros(threshold_mask.shape, dtype="uint8")
        labelMask[mask[1] == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 400:
            output_mask = cv2.add(output_mask, labelMask)
    output_mask = output_mask.astype(np.float64) / 255
    return embeddings, output_mask, img

if __name__=='__main__':

    args=argparse.ArgumentParser()

    args.add_argument('-i','--input',default="./test_images")
    args.add_argument('-o','--output',default='./test_result')
    args.add_argument('-mp','--model',default='./logs/models/model_1_1648481010_326_2.7336108684539795.pkl')
    args.add_argument('-m','--mode',default='gpu')
    args.add_argument('-s','--size',default=[512,256],type=int,nargs='+')
    args.add_argument('-t','--threshold',default=.5,type=float)
    args.add_argument('-b','--bandwidth',default=3)
    
    args=args.parse_args()

    input_ad = args.input
    output_ad = args.output
    model_path = args.model
    bandwidth = args.bandwidth
    mode = args.mode
    image_size = tuple(args.size)
    threshold = args.threshold

    model = _load_model(mode, model_path)
    model.eval()
    img_files = os.listdir(input_ad)

    dump_to_json = []
    str_input = 'gt_image/'
    # print(str_input)

    json_gt = [json.loads(line) for line in open("train_data/label(5).json").readlines()]
    gts = {l['raw_file']: l for l in json_gt}
    total = 0
    for i in img_files:
        img_frame = cv2.imread(os.path.join(input_ad, i), cv2.IMREAD_UNCHANGED)
        original_h = img_frame.shape[0]
        original_w = img_frame.shape[1]
        # scale_width = original_shape[1]/image_size[0]
        # scale_height = original_shape[0] / image_size[1]
        embeddings, threshold_mask, img = _frame_process(img_frame, model, image_size, threshold)
        cluster = lane_cluster(bandwidth, img, embeddings.squeeze().data.cpu().numpy(), threshold_mask,
                               method='Meanshift')
        fitted_image, instance_mask, segmentation_mask, lane_idx, labels, unique_label = cluster()
        segmentation_mask = cv2.resize(segmentation_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        if not os.path.exists(os.path.join(output_ad, 'instance/')):
            os.mkdir(os.path.join(output_ad, 'instance/'))
        if not os.path.exists(os.path.join(output_ad, "fitted/")):
            os.mkdir(os.path.join(output_ad, "fitted/"))
        instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_RGB2BGR)
        fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("demo", instance_mask)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(output_ad, 'instance/', '.'.join([i.split('.')[0], 'png'])), instance_mask)
        cv2.imwrite(os.path.join(output_ad, 'fitted/', '.'.join([i.split('.')[0], 'png'])), fitted_image)

    #     raw_file = os.path.join(str_input, i)
    #     gt = gts[raw_file]
    #     y_samples = gt['h_samples']
    #     x_samples = gt['lanes']
    #     x_len = len(x_samples)
    #     val_idx = []
    #     for j in range(x_len):
    #         for jj in range(len(y_samples)):
    #             if x_samples[j][jj] == -2:
    #                 continue
    #             k = [x_samples[j][jj], y_samples[jj]]
    #             val_idx.append(k)
    #
    #     lane_idx2 = []
    #     for ii in range(original_w):
    #         for jj in range(original_h):
    #             if segmentation_mask[jj, ii, 2] == 255:
    #                 lane_idx2.append([ii, jj])
    #
    #
    #     # print(lane_idx2)
    #     # print("--"*5)
    #     # print(val_idx)
    #
    #     sum = len(val_idx)
    #     num = 0
    #     for ii in range(len(lane_idx2)):
    #         for jj in range(len(val_idx)):
    #             if lane_idx2[ii] == val_idx[jj]:
    #                 num += 1
    #
    #     print(num / sum)
    #     total += num/sum
    #
    # print(total/len(img_files))

















    #     json_dict = {}
    #     json_dict['lanes'] = []
    #     json_dict['raw_file'] = os.path.join(str_input, i)
    #
    #     json_gt = [json.loads(line) for line in open("./testpy/label(5).json").readlines()]
    #     gts = {l['raw_file']: l for l in json_gt}
    #
    #     raw_file = json_dict['raw_file']
    #     gt = gts[raw_file]
    #     y_samples = gt['h_samples']
    #     json_dict['h_sample'] = y_samples
    #
    #     lane_idx2 = []
    #     for t in lane_idx:
    #         w = t[0]
    #         h = t[1]
    #         lane_idx2.append([w,h])
    #
    #     lane_coords = []
    #     for lb1 in unique_label:
    #         temp = []
    #         for idx, lb2 in enumerate(labels):
    #             if lb2 == lb1:
    #                 temp.append(lane_idx2[idx])
    #         lane_coords.append(temp)
    #
    #     # for l in lane_coords: # 整张图片
    #     #     if len(l) == 0:
    #     #         continue
    #     #     json_dict['lanes'].append([])
    #     #     for (x, y) in l:
    #     #         json_dict['lanes'][-1].append(int(x))
    #
    #
    #     for l in lane_coords: # 整张图片
    #         if len(l) == 0:
    #             continue
    #         json_dict['lanes'].append([])
    #         for h in y_samples:
    #             flag = False
    #             for (x, y) in l:
    #                 if y == h:
    #                     json_dict['lanes'][-1].append(int(x))
    #                     flag = True
    #                     break
    #             if not flag:
    #                 json_dict['lanes'][-1].append(int(-2))
    #
    #     dump_to_json.append(json.dumps(json_dict))
    #
    #
    # with open("predict_test.json", "w") as f:
    #     for line in dump_to_json:
    #         print(line, end="\n", file=f)
    #
    # eval_result = LaneEval.bench_one_submit("predict_test.json", "./testpy/label(5).json")
    # print(eval_result)


    





        



    