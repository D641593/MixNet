import torch
import torch.nn as nn
from cfglib.config import config as cfg
import torch.nn.functional as F

class overlap_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.inst_loss = torch.nn.MSELoss(reduction = 'sum')

    def forward(self, preds, conf, inst, overlap, inds):
        p1 = preds[:,0]
        p2 = preds[:,1]
        and_preds = p1 * p2

        and_loss = self.BCE_loss(and_preds, overlap)

        or_preds = torch.maximum(p1,p2)
        or_loss = self.BCE_loss(or_preds, conf)

        and_overlap = and_preds * overlap
        op1 = torch.maximum(p1, and_overlap)
        op2 = torch.maximum(p2, and_overlap)

        inst_loss = torch.tensor(0)
        b, h, w = p1.shape
        for i in range(b):
            bop1 = op1[i]
            bop2 = op2[i]
            inst_label = inst[i]
            keys = torch.unique(inst_label)
            # print(keys.shape)
            tmp = torch.tensor(0)
            for k in keys:
                inst_map = (inst_label == k).float()
                suminst = torch.sum(inst_map)
                d1 = self.inst_loss(bop1 * inst_map, inst_map) / suminst
                d2 = self.inst_loss(bop2 * inst_map, inst_map) / suminst
                tmp = tmp + torch.min(d1,d2) - torch.max(d1,d2) + 1
            inst_loss = inst_loss + ( tmp / keys.shape[0] ) 
        # print(and_loss[conf == 1].mean(), and_loss[conf == 0].mean())
        and_loss = and_loss[conf == 1].mean() + and_loss[conf == 0].mean()
        or_loss = or_loss.mean()
        inst_loss = inst_loss / b
        # print("and_loss : ",and_loss.item(), "or_loss : ",or_loss.item(),"inst_loss : ",inst_loss.item())
        # print("or_loss : ",or_loss.item())
        # print("inst_loss : ",inst_loss.item())
        loss = 0.5 * and_loss +  0.25 * or_loss + 0.25 * inst_loss
        return loss