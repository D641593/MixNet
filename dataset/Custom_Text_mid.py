# -*- coding: utf-8 -*-
"""
Custom_Text_mid.py
-------------------
Dataset loader for MixNet's "_mid" (midline) training pipeline that reads
PaddleOCR-style detection label files directly (the same
`train_det.txt` / `val_det.txt` format PPOCRLabel / PaddleOCR export) --
no conversion step needed.

Place this file at MixNet/dataset/Custom_Text_mid.py.

Expected folder layout
-----------------------
    data/<name>/
        imgs/...              (whatever relative paths appear in the txt)
        train_det.txt
        val_det.txt            (optional, only needed for eval)

Label file format (tab-separated, one line per image)
--------------------------------------------------------
    <path relative to data_root>\t<json array>

e.g.:
    imgs/10122_5.jpg\t[{"transcription": "", "points": [[20.0,13.8],[17.4,24.1],[43.8,26.7],[45.7,16.0]]}, ...]

Rules for "points":
  * At least 4 vertices.
  * An EVEN number of vertices, ordered so the first half traces one long
    side of the text region and the second half traces the other long
    side -- needed to build the midline supervision this "_mid" variant
    uses. A 4-point quad [top-left, top-right, bottom-right, bottom-left]
    (PaddleOCR's usual convention) satisfies this automatically. If a box
    has an odd point count (rare labeling artifact), the last point is
    dropped to make it even; if that leaves fewer than 4 points the box
    is skipped.

"transcription":
  * Empty string "" (common for detection-only labels, i.e. no OCR text
    was annotated) is treated as a NORMAL, valid text box -- NOT as
    "don't care". Only "###" (PaddleOCR/ICDAR's actual "difficult /
    don't care" marker) is treated as ignore.
"""
import json
import os

import numpy as np

from dataset.dataload_midline import TextDataset, TextInstance


def pil_load_img(path):
    from PIL import Image
    image = Image.open(path)
    image = np.array(image)
    return image


class CustomText_mid(TextDataset):

    def __init__(self, data_root, is_training=True, load_memory=False, transform=None,
                 train_file="train_det.txt", val_file="val_det.txt"):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        label_file = train_file if is_training else val_file
        label_path = os.path.join(data_root, label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        self.samples = []  # list of (image_rel_path, boxes)
        n_skipped = 0
        with open(label_path, encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                if "\t" not in line:
                    print(f"[skip] {label_file}:{line_no} -> no tab separator found")
                    n_skipped += 1
                    continue
                img_rel_path, js = line.split("\t", 1)
                try:
                    boxes = json.loads(js)
                except json.JSONDecodeError as e:
                    print(f"[skip] {label_file}:{line_no} -> invalid JSON ({e})")
                    n_skipped += 1
                    continue
                if not os.path.exists(os.path.join(data_root, img_rel_path)):
                    print(f"[skip] {label_file}:{line_no} -> image not found: {img_rel_path}")
                    n_skipped += 1
                    continue
                self.samples.append((img_rel_path, boxes))

        if len(self.samples) == 0:
            raise RuntimeError(f"No usable samples parsed from {label_path}")
        print(f"[CustomText_mid] {label_file}: {len(self.samples)} samples loaded, {n_skipped} lines skipped")

        if self.load_memory:
            self.datas = [self.load_img_gt(i) for i in range(len(self.samples))]

    def parse_polygons(self, boxes):
        polygons = []
        for b in boxes:
            text = str(b.get("transcription", "")).strip()
            # Only an explicit "don't care" marker means ignore -- an empty
            # transcription just means "no OCR text was labeled" and the
            # box is still a normal, valid detection target.
            if text in ("###", "##"):
                text = "#"
            pts = np.array(b["points"], dtype=np.float32)
            if pts.shape[0] % 2 != 0:
                pts = pts[:-1]  # drop the odd extra point (rare annotation artifact)
            if pts.shape[0] < 4:
                continue
            polygons.append(TextInstance(pts, "c", text))
        return polygons

    def load_img_gt(self, item):
        image_rel_path, boxes = self.samples[item]
        image_path = os.path.join(self.data_root, image_rel_path)
        image_id = os.path.basename(image_rel_path)

        image = pil_load_img(image_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        polygons = self.parse_polygons(boxes)

        return {
            "image": image,
            "polygons": polygons,
            "image_id": image_id,
            "image_path": image_path,
        }

    def __getitem__(self, item):
        data = self.datas[item] if self.load_memory else self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(
                data["image"], data["polygons"],
                image_id=data["image_id"], image_path=data["image_path"],
            )
        return self.get_test_data(
            data["image"], data["polygons"],
            image_id=data["image_id"], image_path=data["image_path"],
        )

    def __len__(self):
        return len(self.samples)