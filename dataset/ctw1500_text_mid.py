#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload_midline import TextDataset, TextInstance
from util.io import read_lines
import cv2


class Ctw1500Text_mid(TextDataset):
    def __init__(self, data_root, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'train' if is_training else 'test', "text_image")
        self.annotation_root = os.path.join(data_root, 'train' if is_training else 'test', "text_label_circum")
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            # line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = list(map(int, line.split(',')))
            pts = np.stack([gt[4::2], gt[5::2]]).T.astype(np.int32)

            pts[:, 0] = pts[:, 0] + gt[0]
            pts[:, 1] = pts[:, 1] + gt[1]
            polygons.append(TextInstance(pts, 'c', "**"))

        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert (c == 3)
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_carve_txt(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        if self.load_memory:
            data = self.datas[item]
        else:
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
    import time
    from util.augmentation import Augmentation
    from util import canvas as cav

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    # transform = BaseTransformNresize(
    #     mean=means, std=stds
    # )

    trainset = Ctw1500Text_mid(
        data_root='./data/ctw1500',
        ignore_list=None,
        is_training=True,
        transform=transform
    )
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags, midline_pts, gt_mid_pts = trainset[idx]
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags ,midline_pts, gt_mid_pts \
            = map(lambda x: x.cpu().numpy(),
                  (img, train_mask, tr_mask, distance_field,
                   direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags, midline_pts, gt_mid_pts))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        boundary_point = gt_mid_pts[np.where(ignore_tags!=0)[0]]
        for i, bpts in enumerate(boundary_point):
            cv2.drawContours(img, [bpts.astype(np.int32)], -1, (0, 255, 0), 1)
            for j,  pp in enumerate(bpts):
                if j==0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j==1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)

            ppts = midline_pts[i]
            cv2.drawContours(img, [ppts.astype(np.int32)], -1, (0, 0, 255), 1)
            for j,  pp in enumerate(ppts):
                if j==0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j==1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)
            cv2.imwrite('imgs.jpg', img)