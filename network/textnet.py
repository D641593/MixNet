import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import math
import cv2

from cfglib.config import config as cfg
from network.layers.model_block import FPN
from network.layers.Transformer import Transformer
from network.layers.gcn_utils import get_node_feature
from util.misc import get_sample_point, get_cosine_map
# import cc_torch
from .midline import midlinePredictor

class Evolution(nn.Module):
    def __init__(self, node_num, seg_channel, is_training=True, device=None):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.seg_channel = seg_channel
        self.device = device
        self.is_training = is_training
        self.clip_dis = 100

        self.iter = 3
        for i in range(self.iter):
            evolve_gcn = Transformer(seg_channel, 128, num_heads=8, dim_feedforward=1024, drop_rate=0.0, if_resi=True, block_nums=3)
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        if not is_training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):

        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > cfg.dis_threshold
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                # 50 for MLT2017 and ArT (or DCN is used in backone); else is all 150;
                # just can set to 50, which has little effect on the performance
                if np.sum(text_mask) < 50/(cfg.scale*cfg.scale) or confidence < cfg.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                
                poly = get_sample_point(text_mask, cfg.num_points,
                                        cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

        return init_polys, inds, confidences

    # def get_boundary_proposal_eval_cuda(self, input=None, seg_preds=None):
    #     # print ("using cuda ccl")
    #     cls_preds = seg_preds[:, 0, :, :].detach()
    #     dis_preds = seg_preds[:, 1, :, :].detach()

    #     inds = []
    #     init_polys = []
    #     confidences = []
    #     for bid, dis_pred in enumerate(dis_preds):
    #         dis_mask = dis_pred > cfg.dis_threshold
    #         dis_mask = dis_mask.type(torch.cuda.ByteTensor)
    #         labels = cc_torch.connected_components_labeling(dis_mask)
    #         key = torch.unique(labels, sorted = True)
    #         for l in key:
    #             text_mask = labels == l
    #             confidence = round(torch.mean(cls_preds[bid][text_mask]).item(), 3)
    #             if confidence < cfg.cls_threshold or torch.sum(text_mask) < 50/(cfg.scale*cfg.scale):
    #                 continue
    #             confidences.append(confidence)
    #             inds.append([bid, 0])
                
    #             text_mask = text_mask.cpu().numpy()
    #             poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
    #             init_polys.append(poly)

    #     if len(inds) > 0:
    #         inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
    #         init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
    #     else:
    #         init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
    #         inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

    #     return init_polys, inds, confidences
        
    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        num_point = i_it_poly.shape[1]
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2)*cfg.scale, cnn_feature.size(3)*cfg.scale
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:,:num_point]
        if self.is_training:
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt", embed = None):
        if self.is_training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            # init_polys, inds, confidences = self.get_boundary_proposal_eval_cuda(input=input, seg_preds=seg_preds - embed)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter+1)], inds, confidences

        py_preds = [init_polys, ]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences


class TextNet(nn.Module):

    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, is_training=(not cfg.resume and is_training))

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

        if cfg.embed:
            self.embed_head = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
                nn.PReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
                nn.PReLU(),
                nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
            )
            self.embed_head = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        if not cfg.onlybackbone:
            if cfg.mid:
                self.BPN = midlinePredictor(seg_channel=32+4)
            elif cfg.pos:
                self.BPN = Evolution(cfg.num_points, seg_channel=32+4+2, is_training=is_training, device=cfg.device)
            else:
                self.BPN = Evolution(cfg.num_points, seg_channel=32+4, is_training=is_training, device=cfg.device)

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(cfg.device))
        self.load_state_dict(state_dict['model'], strict=(not self.is_training))

    def forward(self, input_dict, test_speed=False, knowledge = False):
        output = {}
        b, c, h, w = input_dict["img"].shape
        # print(b,c,h,w)
        if self.is_training or cfg.exp_name in ['ArT', 'MLT2017', "MLT2019"] or test_speed:
            image = input_dict["img"]
        else:
            # image = input_dict["img"]

            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[1]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]

        up1 = self.fpn(image)
        if cfg.know or knowledge:
            output["image_feature"] = up1
        if knowledge:
            return output
        preds = self.seg_head(up1)

        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)

        if cfg.onlybackbone:
            output["fy_preds"] = fy_preds
            return output

        cnn_feats = torch.cat([up1, fy_preds], dim=1)
        if cfg.embed: #or cfg.mid:
            embed_feature = self.embed_head(up1)
            # embed_feature = self.overlap_head(up1)
            # if not self.training:
                # andpart = embed_feature[0][0] * embed_feature[0][1]

        if cfg.mid:
            py_preds, inds, confidences, midline = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        else:
            py_preds, inds, confidences = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["inds"] = inds
        output["confidences"] = confidences
        if cfg.mid:
            output["midline"] = midline
        if cfg.embed : # or cfg.mid:
            output["embed"] = embed_feature

        # print(py_preds)
        return output
