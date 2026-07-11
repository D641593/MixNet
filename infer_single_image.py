# -*- coding: utf-8 -*-
"""
infer_single_image.py
----------------------
Standalone single-image inference for MixNet
(https://github.com/D641593/MixNet).

Place this file at the ROOT of the cloned MixNet repo (next to
`eval_mixNet.py`, `cfglib/`, `network/`, `util/`, ...) and run it from there,
so the `cfglib`, `network`, `util`, `dataset` packages are importable.

Usage
-----
    python3 infer_single_image.py \
        --image path/to/image.jpg \
        --checkpoint model/Totaltext_mid/MixNet_FSNet_M_622.pth \
        --net FSNet_M --mid True \
        --test_size 640 1024 \
        --dis_threshold 0.3 --cls_threshold 0.85 \
        --output result.jpg --output_json result.json

Checkpoint / config combos taken straight from the repo's README `# Eval`
section (only the ones needed to reproduce these results, no dataset
loading involved):

    Total-Text (curved text): MixNet_FSNet_M_622.pth
        --net FSNet_M --mid True  --test_size 640 1024
        --dis_threshold 0.3 --cls_threshold 0.85

    CTW1500: MixNet_FSNet_hor_925.pth
        --net FSNet_hor --mid False --test_size 640 1024
        --dis_threshold 0.3 --cls_threshold 0.85

    MSRA-TD500: MixNet_FSNet_M_284.pth
        --net FSNet_M --mid True --test_size 640 1024
        --dis_threshold 0.3 --cls_threshold 0.85

    ICDAR-ArT: MixNet_FSNet_M_160.pth
        --net FSNet_M --mid True --test_size 960 2880
        --dis_threshold 0.4 --cls_threshold 0.8

Weights are not included in the repo — download them from the links in
README.md and place them under model/<exp_name>/ as described in
model/README.md.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from PIL import ImageFile, Image

from cfglib.config import config as cfg
from network.textnet import TextNet
from util.augmentation import BaseTransform
from util.misc import rescale_result

# NOTE: intentionally not importing from dataset.data_util / dataset package:
# dataset/__init__.py eagerly imports every dataset loader (Icdar, ArT,
# TD500HUST, ...), some of which require extra deps (e.g. mmcv) that aren't
# needed at all for plain single-image inference. pil_load_img is a 3-line
# function, so it's inlined here instead.
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def build_config(args):
    """Populate the global `cfg` object with everything the network code
    touches. The upstream repo normally does this via cfglib/option.py's
    argparse + cfglib/config.update_config(), but a few attributes
    (`pos`, `resume`) are referenced in network/textnet.py without ever
    being exposed as CLI flags or given defaults in cfglib/config.py, so
    they are set explicitly here to avoid AttributeError.
    """
    cfg.exp_name = args.exp_name
    cfg.net = args.net
    cfg.mid = args.mid
    cfg.embed = False
    cfg.know = False
    cfg.onlybackbone = False
    cfg.pos = False        # not exposed upstream; must default to False
    cfg.resume = None      # not exposed upstream; must default to None
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
    """Build TextNet and load weights.

    Works around an upstream bug in network/textnet.py: when cfg.embed is
    True, TextNet.__init__ builds `self.embed_head` as an nn.Sequential and
    then immediately *overwrites* it with a bare, unregistered tuple
    (`self.embed_head = nn.Conv2d(...),` -- note the trailing comma). That
    means cfg.embed=True is broken out of the box: the attribute never gets
    registered as a submodule, so strict state_dict loading fails with
    "Unexpected key(s): embed_head.0.weight, ...", and forward() would even
    try to call a tuple as if it were a layer.

    Some released checkpoints (e.g. Totaltext_mid) *were* trained with an
    embed head. We detect that from the checkpoint's own keys and rebuild +
    properly register the intended nn.Sequential ourselves before loading.
    """
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    ckpt_keys = state_dict["model"].keys()
    needs_embed = any(k.startswith("embed_head.") for k in ckpt_keys)

    # Construct with cfg.embed=False no matter what: with cfg.embed=True,
    # TextNet.__init__ itself raises TypeError on current PyTorch versions
    # (it tries to overwrite a registered nn.Sequential submodule with a
    # bare tuple). So we build without triggering that code path at all,
    # then attach the intended embed_head module ourselves afterwards, and
    # only flip cfg.embed on for forward() to actually use it.
    cfg.embed = False
    model = TextNet(backbone=cfg.net, is_training=False)

    if needs_embed:
        # Same architecture as the (correct) first assignment in
        # TextNet.__init__, re-applied so it's actually registered.
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


def preprocess(image_path, test_size):
    image = pil_load_img(image_path)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    H, W = image.shape[:2]

    transform = BaseTransform(size=test_size, mean=cfg.means, std=cfg.stds)
    resized_norm_img, _ = transform(image.copy())

    img_tensor = torch.from_numpy(
        resized_norm_img.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return img_tensor, resized_norm_img, H, W


@torch.no_grad()
def detect(model, image_path):
    """Run MixNet on a single image and return a list of polygons
    (each a list of [x, y] points) in the *original* image's pixel space.
    """
    img_tensor, resized_norm_img, H, W = preprocess(image_path, cfg.test_size)
    img_tensor = img_tensor.to(cfg.device)

    output_dict = model({"img": img_tensor})

    # py_preds is a list of successive contour-evolution iterations;
    # [-1] is the final, most refined set of polygons.
    contours = output_dict["py_preds"][-1].int().cpu().numpy()

    # Undo normalization just to get a same-shape reference for rescaling
    # contour coordinates back to the original image size (mirrors
    # eval_mixNet.py's `inference()`).
    img_show = ((resized_norm_img * cfg.stds + cfg.means) * 255)
    img_show = np.clip(img_show, 0, 255).astype(np.uint8)

    _, contours = rescale_result(img_show, contours, H, W)

    polygons = []
    for cont in contours:
        cont = np.stack([cont[:, 0], cont[:, 1]], axis=1)
        if cv2.contourArea(cont) <= 0:
            continue
        polygons.append(cont.tolist())

    return polygons


def visualize(image_path, polygons, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description="MixNet single-image text detection")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", required=True, help="Path to a MixNet .pth weight file")

    parser.add_argument("--net", default="FSNet_M", choices=["FSNet_M", "FSNet_S", "FSNet_hor"],
                         help="Backbone architecture, must match the checkpoint")
    parser.add_argument("--exp_name", default="Totaltext_mid",
                         help="Only affects internal padding logic for 'ArT'/'MLT2017'/'MLT2019'; "
                              "safe to leave at default for most checkpoints")
    parser.add_argument("--mid", type=str2bool, default=True,
                         help="True for *_mid checkpoints (Totaltext_mid, TD500HUST_mid, ArT_mid); "
                              "False for e.g. the CTW1500 checkpoint")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--test_size", type=int, nargs=2, default=[640, 1024],
                         help="min/max side used for resizing before inference, e.g. 640 1024")
    parser.add_argument("--dis_threshold", type=float, default=0.3)
    parser.add_argument("--cls_threshold", type=float, default=0.5)
    parser.add_argument("--cuda", type=str2bool, default=True)

    parser.add_argument("--output", default="result.jpg", help="Path to save the visualization image")
    parser.add_argument("--output_json", default=None, help="Optional path to dump detected polygons as JSON")

    args = parser.parse_args()

    build_config(args)
    model = load_model(args.checkpoint)
    polygons = detect(model, args.image)

    print(f"Detected {len(polygons)} text region(s).")

    visualize(args.image, polygons, args.output)
    print(f"Saved visualization to {args.output}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(polygons, f, ensure_ascii=False)
        print(f"Saved polygons to {args.output_json}")


if __name__ == "__main__":
    main()