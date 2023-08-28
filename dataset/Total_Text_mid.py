# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import warnings
warnings.filterwarnings("ignore")
import os
import re
import numpy as np
import scipy.io as io
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload_midline import TextDataset, TextInstance
import cv2
from util import io as libio


class TotalText_mid(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

        # self.balance_weight = True

    @staticmethod
    def parse_mat(mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path + ".mat")
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = libio.read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line, '\xef\xbb\xbf')
            gt = line.split(',')
            xx = gt[0].replace("x: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            yy = gt[1].replace("y: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            try:
                xx = [int(x) for x in re.split(r" *", xx)]
                yy = [int(y) for y in re.split(r" *", yy)]
            except:
                xx = [int(x) for x in re.split(r" +", xx)]
                yy = [int(y) for y in re.split(r" +", yy)]
            if len(xx) < 4 or len(yy) < 4:  # too few points
                continue
            text = gt[-1].split('\'')[1]
            try:
                ori = gt[-2].split('\'')[1]
            except:
                ori = 'c'
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))
        # print(polygon)
        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
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

    trainset = TotalText_mid(
        data_root='./data/total-text-mat',
        ignore_list=None,
        is_training=True,
        transform=transform
    )
    for idx in range(0, len(trainset)):
        print(trainset.image_list[idx])
        t0 = time.time()
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags, gt_mid_pts = trainset[idx]
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags , gt_mid_pts \
            = map(lambda x: x.cpu().numpy(),
                  (img, train_mask, tr_mask, distance_field,
                   direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags, gt_mid_pts))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        boundary_point = ctrl_points[np.where(ignore_tags!=0)[0]]
        mid_point = gt_mid_pts[np.where(ignore_tags!=0)[0]]
        for i, bpts in enumerate(boundary_point):
            cv2.drawContours(img, [bpts.astype(np.int32)], -1, (0, 255, 125), 1)
            for j,  pp in enumerate(bpts):
                if j==0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 1, (255, 0, 255), -1)
                elif j==1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 1, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 1, (0, 0, 255), -1)

            ppts = mid_point[i]
            cv2.polylines(img, [ppts.astype(np.int32)], False, (0, 125, 255), 1)
            for j,  pp in enumerate(ppts):
                if j==0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 1, (255, 0, 255), -1)
                elif j==1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 1, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 1, (0, 0, 255), -1)
            cv2.imwrite('midline_demo/'+trainset.image_list[idx], img)