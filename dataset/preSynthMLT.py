# -*- coding: utf-8 -*-
__author__ = "S.X.Zhang"
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import cv2
from pycocotools.coco import COCO

class MLTTextJson(TextDataset):
    def __init__(self, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)

        self.is_training = is_training
        self.load_memory = load_memory
        image_root = [
            "data/MLT/train_images/",
            "data/SynthCurve/img_part1/emcs_imgs/",
            "data/SynthCurve/img_part2/syntext_word_eng/",
        ]
        gt_root = [
            "data/MLT/gts/",
            "data/SynthCurve/img_part1/train_poly_pos.json",
            "data/SynthCurve/img_part2/train_poly_pos.json",
        ]

        image_list = []
        anno_list = []
        for path, gtpath in zip(image_root, gt_root):
            imgfnames = sorted(os.listdir(path))
            image_list.extend([os.path.join(path, fname) for fname in imgfnames])
            if ".json" in gtpath:
                coco_api = COCO(gtpath)
                img_ids = sorted(coco_api.imgs.keys())
                anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
                for anno in anns:
                    polygons = []
                    for label in anno:
                        poly = label["polys"]
                        poly = np.array(list(map(int, poly)))
                        poly = poly.reshape(-1,2)
                        polygons.append(TextInstance(poly, 'c', "word"))
                    anno_list.append(polygons)
            else:
                gtfnames = sorted(os.listdir(gtpath))
                anno_list.extend([self.read_txt(os.path.join(gtpath, fname)) for fname in gtfnames])

        self.image_list = []
        self.anno_list = []
        for imgpath, gtpath in zip(image_list, anno_list):
            if ".jpg" in imgpath or ".png" in imgpath:
                self.image_list.append(imgpath)
                self.anno_list.append(gtpath)

    def read_txt(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        polygons = []
        for line in lines:
            line = line.strip('\ufeff')
            line = line.replace('\xef\xbb\xbf','')
            gt = line.split(',')
            poly = np.array(list(map(int, gt[:8]))).reshape(-1,2)
            if gt[-1].strip() == "###":
                label = gt[-1].strip().replace("###", "#")
            else:
                label = "word"
            polygons.append(TextInstance(poly, 'c', label))

        return polygons


    def load_img_gt(self, item):
        image_path = self.image_list[item]
        image_id = image_path.split("/")[-1]

        image = pil_load_img(image_path)
        # print(image.shape)
        try:
            assert image.shape[-1] == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

            # Read annotation
        polygons = self.anno_list[item]

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):
        data = self.load_img_gt(item)

        return self.get_training_data(
                    data["image"], data["polygons"],
                    image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    # execute in base dir `PYTHONPATH=.:$PYTHONPATH python dataset/Icdar19ArT_Text.py`
    from util.augmentation import Augmentation
    from util.misc import regularize_sin_cos
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds)

    trainset = MLTTextJson(
        is_training=True,
        transform=transform,
    )
    print(len(trainset.image_list))
    print(len(trainset.anno_list))
    # t0 = time.time()
    # for i in range(len(trainset)):
        # print(trainset.image_list[i])
        # img, train_mask, tr_mask, distance_field, \
        # direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags = trainset[i]
    # img, train_mask, tr_mask, distance_field, \
    # direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags \
    #     = map(lambda x: x.cpu().numpy(),
    #           (img, train_mask, tr_mask, distance_field,
    #            direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags))

    # img = img.transpose(1, 2, 0)
    # img = ((img * stds + means) * 255).astype(np.uint8)
    # cv2.imwrite("show_img.jpg", img)
    # distance_map = np.array(distance_field * 255 / np.max(distance_field), dtype=np.uint8)
    # cv2.imwrite("distance_map.jpg", distance_map)

    # direction_map = np.array(direction_field[0] * 255 / np.max(direction_field[0]), dtype=np.uint8)
    # cv2.imwrite("direction_field.jpg", direction_map)
