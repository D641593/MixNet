import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gc
import time
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset

from dataset import SynthText, TotalText, Ctw1500Text,\
    Mlt2017Text, TD500Text, TD500HUSTText, ArtTextJson, Mlt2019Text, TotalText_New, ArtText, MLTTextJson, TotalText_mid, Ctw1500Text_mid, TD500HUSTText_mid, ALLTextJson, ArtTextJson_mid
from dataset.Custom_Text_mid import CustomText_mid
from network.loss import TextLoss, knowledge_loss
from network.textnet import TextNet
from util.augmentation import Augmentation, BaseTransform
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device, rescale_result
from cfglib.option import BaseOptions
from util.visualize import visualize_network_output
from util.summary import LogSummary
from util.shedule import FixLR
import cv2
from shapely.geometry import Polygon
# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True)

lr = None
train_step = 0


def save_model(model, epoch, lr, optimzer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'MixNet_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict()
        # 'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError as e:
        print("Missing key in state_dict, try to load with strict = False")
        model.load_state_dict(state_dict['model'], strict = False)
        print(e)

def polygon_iou(poly1, poly2):
    """IoU between two polygons (Nx2 point arrays), robust to self-intersecting
    contours (which sometimes come out of the network / raw annotations)."""
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


@torch.no_grad()
def validate(model, val_loader, iou_thresh=0.5):
    """Lightweight periodic validation: greedy IoU@iou_thresh matching between
    predicted and ground-truth polygons -> precision / recall / F1.

    NOTE: this is a simplified proxy metric for monitoring training progress,
    not a reimplementation of the official per-dataset benchmark protocols
    (DetEval / TIoU / ...) used in util/eval.py. Use it to see whether the
    model is trending in the right direction between checkpoints; for a
    rigorous final number, run eval_mixNet.py's dataset-specific evaluator.
    """
    was_training = model.training
    model.eval()

    total_gt, total_pred, total_matched = 0, 0, 0

    for image, meta in val_loader:
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
            if n_annot.item() > 0 and tag > 0:  # skip padding rows and "#" / ignore regions
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

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print('  [validate] total_pred_boxes={} total_gt_boxes={} matched={} '
          '(cls_threshold={}, dis_threshold={})'.format(
              total_pred, total_gt, total_matched, cfg.cls_threshold, cfg.dis_threshold))

    if was_training:
        model.train()
    return precision, recall, f1


def _parse_data(inputs):
    input_dict = {}
    inputs = list(map(lambda x: to_device(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]
    if cfg.embed:
        input_dict['edge_field'] = inputs[9]
    if cfg.mid:
        input_dict['gt_mid_points'] = inputs[9]
        input_dict['edge_field'] = inputs[10]

    return input_dict


def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()

    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))

    for i, inputs in enumerate(train_loader):

        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
        loss = loss_dict["total_loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and epoch % 8 == 0:
            visualize_network_output(output_dict, input_dict, mode='train')

        if i % cfg.display_freq == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for (k, v) in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)

    if cfg.exp_name == 'Synthtext' or cfg.exp_name == 'ALL' or cfg.exp_name == "preSynthMLT" or cfg.exp_name == "preALL":
        print("save checkpoint for pretrain weight. ")
        if epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif cfg.exp_name == 'MLT2019' or cfg.exp_name == 'ArT' or cfg.exp_name == 'MLT2017' or cfg.exp_name == 'Custom_mid':
        if epoch < 10 and cfg.max_epoch >= 200:
            if epoch % (2*cfg.save_freq) == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
        else:
            if epoch % cfg.save_freq == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
    else:
        if epoch % cfg.save_freq == 0 and epoch > 150:
            save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))

def knowledgetrain(model, knowledge, train_loader, criterion, know_criterion, scheduler, optimizer, epoch):

    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    # scheduler.step()

    print('Epoch: {} : LR = {}'.format(epoch, scheduler.get_lr()))

    for i, inputs in enumerate(train_loader):

        data_time.update(time.time() - end)
        train_step += 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        output_know = knowledge(input_dict, knowledge = True)
        loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
        loss = loss_dict["total_loss"]

        know_loss = know_criterion(output_dict["image_feature"], output_know["image_feature"])
        loss = loss + know_loss
        loss_dict["know_loss"] = know_loss
        # backward
        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0) and epoch % 8 == 0:
            visualize_network_output(output_dict, input_dict, mode='train')

        if i % cfg.display_freq == 0:
            gc.collect()
            print_inform = "({:d} / {:d}) ".format(i, len(train_loader))
            for (k, v) in loss_dict.items():
                print_inform += " {}: {:.4f} ".format(k, v.item())
            print(print_inform)

    if cfg.exp_name == 'Synthtext' or cfg.exp_name == 'ALL' or cfg.exp_name == "preSynthMLT" or cfg.exp_name == "preALL":
        print("save checkpoint for pretrain weight. ")
        if epoch % cfg.save_freq == 0:
            save_model(model, epoch, scheduler.get_lr(), optimizer)
    elif cfg.exp_name == 'MLT2019' or cfg.exp_name == 'ArT' or cfg.exp_name == 'MLT2017' or cfg.exp_name == 'Custom_mid':
        if epoch < 10 and cfg.max_epoch >= 200:
            if epoch % (2*cfg.save_freq) == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
        else:
            if epoch % cfg.save_freq == 0:
                save_model(model, epoch, scheduler.get_lr(), optimizer)
    else:
        if epoch % cfg.save_freq == 0 and epoch > 150:
            save_model(model, epoch, scheduler.get_lr(), optimizer)

    print('Training Loss: {}'.format(losses.avg))


def main():

    global lr
    if cfg.exp_name == 'Totaltext':
        trainset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    
    elif cfg.exp_name == 'Totaltext_mid':
        trainset = TotalText_mid(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Synthtext':
        trainset = SynthText(
            data_root='../FAST/data/SynthText',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500':
        trainset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'Ctw1500_mid':
        trainset = Ctw1500Text_mid(
            data_root='data/ctw1500',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500HUST':
        trainset = TD500HUSTText(
            data_root='data/',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'TD500HUST_mid':
        trainset = TD500HUSTText_mid(
            data_root='data/',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'ArT':
        trainset = ArtTextJson(
            data_root='data/ArT',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None
    
    elif cfg.exp_name == 'ArT_mid':
        trainset = ArtTextJson_mid(
            data_root='data/ArT',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'preSynthMLT':
        trainset = MLTTextJson(
            is_training=True,
            load_memory=False,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'preALL':
        trainset = ALLTextJson(
            is_training=True,
            load_memory=False,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        valset = None

    elif cfg.exp_name == 'ALL':
        trainset_SynthMLT = MLTTextJson(
            is_training=True,
            load_memory=False,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )

        trainset_SynthText = SynthText(
            data_root='../FAST/data/SynthText',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        trainset_totaltext = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        # trainset_ctw1500 = Ctw1500Text(
        #     data_root='data/ctw1500',
        #     is_training=True,
        #     load_memory=cfg.load_memory,
        #     transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        # )
        trainset_TD500 = TD500HUSTText(
            data_root='data/',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )

        trainset = ConcatDataset([trainset_SynthText, trainset_SynthMLT,trainset_totaltext,trainset_TD500])
        valset = None

    elif cfg.exp_name == 'Custom_mid':
        trainset = CustomText_mid(
            data_root='data/custom_data',
            is_training=True,
            load_memory=cfg.load_memory,
            transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
        )
        try:
            valset = CustomText_mid(
                data_root='data/custom_data',
                is_training=False,
                transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
            )
        except FileNotFoundError:
            print("[Custom_mid] no val_det.txt found -- training without periodic validation.")
            valset = None

    else:
        print("dataset name is not correct")

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers,
                                   pin_memory=True, generator=torch.Generator(device=cfg.device)
    )

    val_loader = None
    if valset is not None:
        val_loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    if cfg.exp_name == 'Synthtext' or cfg.exp_name == 'ALL' or cfg.exp_name == "preSynthMLT":
        print("save checkpoint for pretrain weight. ")
        
    # Model
    model = TextNet(backbone=cfg.net, is_training=True)
    model = model.to(cfg.device)
    if cfg.know:
        know_model = TextNet(backbone=cfg.knownet, is_training=False)
        load_model(know_model, cfg.know_resume)
        know_model.eval()
        know_model.requires_grad = False

    if cfg.exp_name == 'TD500HUST' or cfg.exp_name == "Ctw1500":
        criterion = TextLoss_ctw()
    else:       
        criterion = TextLoss()

    if cfg.mgpu:
        model = nn.DataParallel(model)
    if cfg.cuda:
        cudnn.benchmark = True
    if cfg.resume:
        load_model(model, cfg.resume)

    lr = cfg.lr
    moment = cfg.momentum
    if cfg.optim == "Adam" or cfg.exp_name == 'Synthtext':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)

    if cfg.exp_name == 'Synthtext':
        scheduler = FixLR(optimizer)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    print('Start training MixNet.')
    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        scheduler.step()
        if cfg.know:
            know_criterion = knowledge_loss(T = 5)
            knowledgetrain(model, know_model, train_loader, criterion, know_criterion, scheduler, optimizer, epoch)
        else:
            train(model, train_loader, criterion, scheduler, optimizer, epoch)

        if val_loader is not None and epoch % cfg.val_freq == 0:
            precision, recall, f1 = validate(model, val_loader)
            print('Validation @ epoch {}: precision={:.4f} recall={:.4f} f1={:.4f}'.format(
                epoch, precision, recall, f1))

    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()