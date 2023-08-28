# -*- coding: utf-8 -*-
# @Time    : 10/1/21
# @Author  : GXYM
import torch
import torch.nn as nn
from cfglib.config import config as cfg
# from network.Seg_loss import SegmentLoss
from network.Reg_loss import PolyMatchingLoss
import torch.nn.functional as F
from .emb_loss import EmbLoss_v2
from .overlap_loss import overlap_loss
import pytorch_ssim

import cv2

class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.PolyMatchingLoss = PolyMatchingLoss(cfg.num_points, cfg.device)
        if cfg.mid:
            self.midPolyMatchingLoss = PolyMatchingLoss(cfg.num_points // 2, cfg.device)
        # self.embed_loss = EmbLoss_v2()
        self.ssim = pytorch_ssim.SSIM()
        self.overlap_loss = overlap_loss()

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss/batch_size

    def cls_ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()
        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):

        # norm loss
        gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        norm_loss = weight_matrix * torch.mean((pred_flux - gt_flux) ** 2, dim=1)*train_mask
        norm_loss = norm_loss.sum(-1).mean()
        # norm_loss = norm_loss.sum()

        # angle loss
        mask = train_mask * mask
        pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        # angle_loss = weight_matrix * (torch.acos(torch.sum(pred_flux * gt_flux, dim=1))) ** 2
        # angle_loss = angle_loss.sum(-1).mean()
        angle_loss = (1 - torch.cosine_similarity(pred_flux, gt_flux, dim=1))
        angle_loss = angle_loss[mask].mean()

        return norm_loss, angle_loss

    @staticmethod
    def get_poly_energy(energy_field, img_poly, ind, h, w):
        img_poly = img_poly.clone().float()
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

        batch_size = energy_field.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), energy_field.size(1), img_poly.size(1)]).to(img_poly.device)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(0)
            gcn_feature[ind == i] = torch.nn.functional.grid_sample(energy_field[i:i + 1], poly)[0].permute(1, 0, 2)
        return gcn_feature

    def loss_energy_regularization(self, energy_field, img_poly, inds, h, w):
        energys = []
        for i, py in enumerate(img_poly):
            energy = self.get_poly_energy(energy_field.unsqueeze(1), py, inds, h, w)
            energys.append(energy.squeeze(1).sum(-1))

        regular_loss = torch.tensor(0.)
        energy_loss = torch.tensor(0.)
        for i, e in enumerate(energys[1:]):
            regular_loss += torch.clamp(e - energys[i], min=0.0).mean()
            energy_loss += torch.where(e <= 0.01, torch.tensor(0.), e).mean()

        return (energy_loss+regular_loss)/len(energys[1:])

    def dice_loss(self, x, target, mask):
        b = x.shape[0]
        x = torch.sigmoid(x)

        x = x.contiguous().reshape(b, -1)
        target = target.contiguous().reshape(b, -1)
        mask = mask.contiguous().reshape(b, -1)

        x = x * mask
        target = target.float()
        target = target * mask

        a = torch.sum(x * target, 1)
        b = torch.sum(x * x, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)

        loss = 1 - d
        loss = torch.mean(loss)
        return loss

    def forward(self, input_dict, output_dict, eps=None):
        """
          calculate boundary proposal network loss
        """
        # tr_mask = tr_mask.permute(0, 3, 1, 2).contiguous()
        fy_preds = output_dict["fy_preds"]

        if not cfg.onlybackbone:
            py_preds = output_dict["py_preds"]
            inds = output_dict["inds"]

        train_mask = input_dict['train_mask'].float()
        tr_mask = input_dict['tr_mask'] > 0
        distance_field = input_dict['distance_field']
        direction_field = input_dict['direction_field']
        weight_matrix = input_dict['weight_matrix']
        gt_tags = input_dict['gt_points']
        instance = input_dict['tr_mask'].long()
        conf = tr_mask.float()
        
        if cfg.scale > 1:
            train_mask = F.interpolate(train_mask.float().unsqueeze(1),
                                       scale_factor=1/cfg.scale, mode='bilinear').squeeze().bool()
            tr_mask = F.interpolate(tr_mask.float().unsqueeze(1),
                                    scale_factor=1/cfg.scale, mode='bilinear').squeeze().bool()

            distance_field = F.interpolate(distance_field.unsqueeze(1),
                                           scale_factor=1/cfg.scale, mode='bilinear').squeeze()
            direction_field = F.interpolate(direction_field,
                                            scale_factor=1 / cfg.scale, mode='bilinear')
            weight_matrix = F.interpolate(weight_matrix.unsqueeze(1),
                                          scale_factor=1/cfg.scale, mode='bilinear').squeeze()

        # pixel class loss
        cls_loss = self.BCE_loss(fy_preds[:, 0, :, :],  conf)
        cls_loss = torch.mul(cls_loss, train_mask).mean()

        # distance field loss
        dis_loss = self.MSE_loss(fy_preds[:, 1, :, :], distance_field)
        dis_loss = torch.mul(dis_loss, train_mask)
        dis_loss = self.single_image_loss(dis_loss, distance_field)

        # # direction field loss
        train_mask = train_mask > 0
        norm_loss, angle_loss = self.loss_calc_flux(fy_preds[:, 2:4, :, :], direction_field, weight_matrix, tr_mask, train_mask)

        if cfg.onlybackbone:
            alpha = 1.0; beta = 3.0; theta=0.5
            loss = alpha*cls_loss + beta*(dis_loss) + theta*(norm_loss + angle_loss)
            loss_dict = {
                'total_loss': loss,
                'cls_loss': alpha*cls_loss,
                'distance loss': beta*dis_loss,
                'dir_loss': theta*(norm_loss + angle_loss),
                'norm_loss': theta*norm_loss,
            }
            return loss_dict

        # boundary point loss
        point_loss = self.PolyMatchingLoss(py_preds[1:], gt_tags[inds])
        if cfg.mid:
            midline = output_dict["midline"]
            gt_midline = input_dict['gt_mid_points']
            midline_loss = 0.5*self.midPolyMatchingLoss(midline, gt_midline[inds])

        if cfg.embed:# or cfg.mid:
            embed = output_dict["embed"]
            # kernel_field = distance_field > cfg.dis_threshold
            # embed_loss = self.embed_loss(embed, instance, kernel_field, train_mask.float(), reduce=True)
            # print(input_dict.keys())
            edge_field = input_dict['edge_field'].float()
            # print(edge_field.shape)
            # ssim_loss = 1 - self.ssim(embed, edge_field.unsqueeze(1))
            # embed_loss = self.BCE_loss(embed[:,0,:,:],  edge_field)
            # embed_loss = torch.mul(embed_loss, train_mask).mean() + ssim_loss
            embed_loss = self.overlap_loss(embed, conf, instance, edge_field, inds)
            # if cfg.mid:
            #     embed_loss = embed_loss * 0.1


        #  Minimum energy loss regularization
        h, w = distance_field.size(1) * cfg.scale, distance_field.size(2) * cfg.scale
        energy_loss = self.loss_energy_regularization(distance_field, py_preds, inds[0], h, w)

        alpha = 1.0; beta = 3.0; theta=0.5; embed_ratio = 0.5
        if eps is None:
            gama = 0.05; 
        else:
            gama = 0.1*torch.sigmoid(torch.tensor((eps - cfg.max_epoch)/cfg.max_epoch))
        loss = alpha*cls_loss + beta*(dis_loss) + theta*(norm_loss + angle_loss) + gama*(point_loss + energy_loss)
        
        if cfg.mid:
            loss = loss + gama*midline_loss
        if cfg.embed: # or cfg.mid:
            loss = loss + embed_ratio*embed_loss

        loss_dict = {
            'total_loss': loss,
            'cls_loss': alpha*cls_loss,
            'distance loss': beta*dis_loss,
            'dir_loss': theta*(norm_loss + angle_loss),
            'norm_loss': theta*norm_loss,
            'angle_loss': theta*angle_loss,
            'point_loss': gama*point_loss,
            'energy_loss': gama*energy_loss,
        }

        if cfg.embed: # or cfg.mid:
            loss_dict['embed_loss'] = embed_ratio*embed_loss
            # loss_dict['ssim_loss'] = ssim_loss
        if cfg.mid:
            loss_dict['midline_loss'] = gama*midline_loss

        return loss_dict

class knowledge_loss(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.KLDloss = torch.nn.KLDivLoss(size_average = False)
        self.T = T
    def forward(self, pred, know):
        log_pred = F.log_softmax(pred / self.T, dim = 1)
        sftknow = F.softmax(know / self.T, dim=1)
        kldloss = self.KLDloss(log_pred, sftknow)
        # print(pred.shape)
        kldloss = kldloss * (self.T**2) / (pred.shape[0] * pred.shape[2] * pred.shape[3])
        return kldloss    