#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import re
import os
import numpy as np
import cv2
import mmcv
import math
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload_midline import TextDataset, TextInstance
from util.io import read_lines
from util.misc import norm2


class TD500HUSTText_mid(TextDataset):

    def __init__(self, data_root, is_training=True, ignore_list=None, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_list = []
        self.anno_list = []
        img_check = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        gt_check = re.compile('.gt')

        if is_training:
            data_root = os.path.join(self.data_root, 'HUST-TR400/')
            fnames = os.listdir(data_root)
            self.image_list = self.image_list + sorted([os.path.join(data_root, fname) for fname in fnames if img_check.findall(fname)])
            self.anno_list = self.anno_list + sorted([os.path.join(data_root, fname) for fname in fnames if gt_check.findall(fname)])
            data_root = os.path.join(self.data_root, 'MSRA-TD500/train/')
            fnames = os.listdir(data_root)
            self.image_list = self.image_list + sorted([os.path.join(data_root, fname) for fname in fnames if img_check.findall(fname)])
            self.anno_list = self.anno_list + sorted([os.path.join(data_root, fname) for fname in fnames if gt_check.findall(fname)])
        else:
            data_root = os.path.join(data_root, 'MSRA-TD500/test/')
            fnames = os.listdir(data_root)
            self.image_list = self.image_list + sorted([os.path.join(data_root, fname) for fname in fnames if img_check.findall(fname)])
            self.anno_list = self.anno_list + sorted([os.path.join(data_root, fname) for fname in fnames if gt_check.findall(fname)])
    
    # @staticmethod
    def parse_txt(self,gt_path):
        lines = mmcv.list_from_file(gt_path)
        bboxes = []
        for line in lines:
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')

            gt = line.split(' ')

            w_ = np.float(gt[4])
            h_ = np.float(gt[5])
            x1 = np.float(gt[2]) + w_ / 2.0
            y1 = np.float(gt[3]) + h_ / 2.0
            theta = np.float(gt[6]) / math.pi * 180

            bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
            bbox = bbox.reshape(-1,2).astype(int)
            bboxes.append(TextInstance(bbox, 'c', "word"))

        return bboxes

    def load_img_gt(self, item):
        image_path = self.image_list[item]
        image_id = image_path.split("/")[-1]
        # Read image data
        image = pil_load_img(image_path)
        annotation_path = self.anno_list[item]
        polygons = self.parse_txt(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):
        data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(data["image"], data["polygons"],
                                          image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(data["image"], data["polygons"],
                                      image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import os
    import cv2
    from util.augmentation import Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = TD500HUSTText(
        data_root='./data/',
        is_training=True,
        transform=transform
    )

    for idx in range(0, len(trainset)):
        idx  = 15
        t0 = time.time()
        image_info = trainset[idx]
        print(trainset.image_list[idx])
        img, train_mask, tr_mask = image_info[0], image_info[1], image_info[2]
        img, train_mask, tr_mask = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask))
        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape)
        # tr_mask = (tr_mask > 0)
        ignore_tag = image_info[8]
        gt = image_info[6][ignore_tag == 1].numpy().astype(np.int)
        poly = image_info[7][ignore_tag == 1].numpy().astype(np.int)
        print(gt.shape)
        print(poly.shape)

        for i in range(len(poly)):
            for j in range(20):
                cv2.circle(img, tuple(gt[i,j]), 3, (255,0,0), -1)
                cv2.circle(img, tuple(poly[i,j]), 3, (0,255,0), -1)
        cv2.imwrite("tr_mask.jpg", np.array((tr_mask > 0) * 255, dtype=np.uint8))
        cv2.imwrite('imgsshow.jpg', img)

        distance_field = image_info[3].numpy()
        distance_map = cav.heatmap(np.array((distance_field) * 255 / np.max(distance_field), dtype=np.uint8))
        cv2.imwrite("distance_map.jpg", distance_map*255)

        break