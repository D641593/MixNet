# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

import scipy.io as scio


class SynthText(TextDataset):

    def __init__(self, data_root, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.image_root = data_root
        self.load_memory = load_memory

        # self.annotation_root = os.path.join(data_root, 'gt')
        self.annotation_root = os.path.join(data_root, 'gt.mat')

        data = scio.loadmat(self.annotation_root)
        self.img_paths = data['imnames'][0]
        self.gts = data['wordBB'][0]
        self.texts = data['txt'][0]
        # with open(os.path.join(data_root, 'image_list.txt')) as f:
        #     self.annotation_list = [line.strip() for line in f.readlines()]

    @staticmethod
    def parse_txt(annotation_path):

        with open(annotation_path) as f:
            lines = [line.strip() for line in f.readlines()]
            image_id = lines[0]
            polygons = []
            for line in lines[1:]:
                points = [float(coordinate) for coordinate in line.split(',')]
                points = np.array(points, dtype=int).reshape(4, 2)
                polygon = TextInstance(points, 'c', 'abc')
                polygons.append(polygon)
        return image_id, polygons

    def txt_to_polygon(self,bboxes):
        bboxes = np.array(bboxes)
        bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
        bboxes = bboxes.transpose(2, 1, 0)
        # print(bboxes.shape)
        polygons = []
        for bbox in bboxes:
            points = np.rint(bbox)
            # print(points)
            polygon = TextInstance(points, 'c', 'abc')
            polygons.append(polygon)
        return polygons

    def load_img_gt(self, item):
        # Read annotation
        # annotation_id = self.annotation_list[item]
        # annotation_path = os.path.join(self.annotation_root, annotation_id)
        # image_id, polygons = self.parse_txt(annotation_path)

        # # Read image data
        # image_path = os.path.join(self.image_root, image_id)
        # image = pil_load_img(image_path)

        # Read annotation
        annotation = self.gts[item]
        polygons = self.txt_to_polygon(annotation)

        # Read image data
        image_id = self.img_paths[item][0]
        # print(image_id)
        image_path = os.path.join(self.data_root, image_id)
        image = pil_load_img(image_path)

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
        return len(self.img_paths)


if __name__ == '__main__':
    from util.augmentation import BaseTransform, Augmentation
    import time
    import cv2
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = SynthText(
        data_root='../FAST/data/SynthText',
        is_training=True,
        transform=transform
    )

    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask = trainset[idx][:3]
        img, train_mask, tr_mask = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape, tr_mask.shape)
        tr_mask = np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)
        print(tr_mask.shape)
        cv2.imwrite("tr_mask.jpg",tr_mask)

        cv2.imwrite('imgs.jpg', img)
        # cv2.waitKey(0)
        break
