# -*- coding: utf-8 -*-
"""
eval_custom.py
----------------
Standalone precision/recall/F1 evaluator for a MixNet checkpoint on a
Custom_Text_mid dataset (val_det.txt). Unlike eval_mixNet.py -- which, in
this repo, only dumps detections to output/*.txt and never actually scores
them (the scoring code for each official dataset is commented out and
depends on separate external benchmark tools that don't exist for custom
data) -- this script computes IoU-based precision/recall/F1 directly and
prints the numbers.

Same simplified metric as train_mixNet.py's validate(): greedy polygon
IoU@iou_thresh matching between predictions and ground truth. This is a
practical proxy for monitoring quality, not a reimplementation of official
per-dataset benchmark protocols (DetEval / TIoU / ...).

Usage
-----
    python3 eval_custom.py \
        --checkpoint model/Custom_mid/MixNet_FSNet_M_50.pth \
        --data_root data/receipt \
        --net FSNet_M --mid True \
        --test_size 640 1024 \
        --dis_threshold 0.3 --cls_threshold 0.85 \
        --iou_thresh 0.5
"""
import argparse

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import DataLoader

from cfglib.config import config as cfg
from dataset.Custom_Text_mid import CustomText_mid
from network.textnet import TextNet
from util.augmentation import BaseTransform
from util.misc import rescale_result


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def polygon_iou(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    if p1.is_empty or p2.is_empty:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def build_config(args):
    cfg.exp_name = "Custom_mid"
    cfg.net = args.net
    cfg.mid = args.mid
    cfg.onlybackbone = False
    cfg.know = False
    cfg.pos = False
    cfg.resume = None
    cfg.scale = args.scale
    cfg.test_size = args.test_size
    cfg.dis_threshold = args.dis_threshold
    cfg.cls_threshold = args.cls_threshold
    cfg.cuda = args.cuda and torch.cuda.is_available()
    cfg.device = torch.device("cuda") if cfg.cuda else torch.device("cpu")
    if cfg.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
    return cfg


def load_model(checkpoint_path):
    """Same embed_head auto-detect/patch as infer_single_image.py -- see that
    file's load_model() docstring for why this is needed: TextNet.__init__
    crashes outright on current PyTorch if cfg.embed=True is set naively."""
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    ckpt_keys = state_dict["model"].keys()
    needs_embed = any(k.startswith("embed_head.") for k in ckpt_keys)

    cfg.embed = False
    model = TextNet(backbone=cfg.net, is_training=False)

    if needs_embed:
        model.embed_head = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            torch.nn.PReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            torch.nn.PReLU(),
            torch.nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )
        cfg.embed = True

    print(f"Loading from {checkpoint_path} (embed_head: {cfg.embed})")
    model.load_state_dict(state_dict["model"], strict=True)
    model.to(cfg.device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, val_loader, iou_thresh):
    total_gt, total_pred, total_matched = 0, 0, 0

    for i, (image, meta) in enumerate(val_loader):
        image = image.to(cfg.device, non_blocking=True)
        H, W = meta['Height'][0].item(), meta['Width'][0].item()

        output_dict = model({'img': image})
        contours = output_dict["py_preds"][-1].int().cpu().numpy()

        img_show = image[0].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
        # dataset.get_test_data() (dataset/dataload_midline.py) captures H/W
        # BEFORE resizing, but stores polygon points AFTER resizing -- so
        # meta['annotation'] is in resized-network-input coordinates while
        # meta['Height']/['Width'] are original-image dimensions. Compute
        # the same scale factors rescale_result() uses for predictions and
        # apply them to GT too, so both end up in original-image space.
        resized_h, resized_w = img_show.shape[:2]
        scale_x = W / resized_w
        scale_y = H / resized_h

        _, pred_polys = rescale_result(img_show, contours, H, W)
        pred_polys = [p for p in pred_polys if cv2.contourArea(p) > 0]

        label_tag = meta['label_tag'][0].int().cpu().numpy()
        gt_polys = []
        for annot, n_annot, tag in zip(meta['annotation'][0], meta['n_annotation'][0], label_tag):
            if n_annot.item() > 0 and tag > 0:
                gt = annot[:n_annot].cpu().numpy().astype(np.float64).copy()
                gt[:, 0] *= scale_x
                gt[:, 1] *= scale_y
                gt_polys.append(gt)

        matched_pred = set()
        matched_gt = 0
        for gt in gt_polys:
            best_iou, best_pi = 0.0, -1
            for pi, pred in enumerate(pred_polys):
                if pi in matched_pred:
                    continue
                iou = polygon_iou(gt, pred)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_iou >= iou_thresh and best_pi != -1:
                matched_pred.add(best_pi)
                matched_gt += 1

        total_gt += len(gt_polys)
        total_pred += len(pred_polys)
        total_matched += matched_gt

        print(f'\r[{i + 1}/{len(val_loader)}] {meta["image_id"][0]}: '
              f'pred={len(pred_polys)} gt={len(gt_polys)} matched={matched_gt}',
              end='', flush=True)

    print()
    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, total_pred, total_gt, total_matched


def main():
    parser = argparse.ArgumentParser(description="Standalone P/R/F1 eval for MixNet on Custom_Text_mid data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/custom_data")
    parser.add_argument("--net", default="FSNet_M", choices=["FSNet_M", "FSNet_S", "FSNet_hor"])
    parser.add_argument("--mid", type=str2bool, default=True)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--test_size", type=int, nargs=2, default=[640, 1024])
    parser.add_argument("--dis_threshold", type=float, default=0.3)
    parser.add_argument("--cls_threshold", type=float, default=0.85)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--cuda", type=str2bool, default=True)
    args = parser.parse_args()

    build_config(args)
    model = load_model(args.checkpoint)

    valset = CustomText_mid(
        data_root=args.data_root, is_training=False,
        transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    )
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    precision, recall, f1, total_pred, total_gt, total_matched = evaluate(model, val_loader, args.iou_thresh)

    print(f"\n=== Results (IoU@{args.iou_thresh}, cls_threshold={cfg.cls_threshold}, "
          f"dis_threshold={cfg.dis_threshold}) ===")
    print(f"total_gt_boxes:   {total_gt}")
    print(f"total_pred_boxes: {total_pred}")
    print(f"matched:          {total_matched}")
    print(f"precision:        {precision:.4f}")
    print(f"recall:           {recall:.4f}")
    print(f"f1:               {f1:.4f}")


if __name__ == "__main__":
    main()