
import os
import time

import numpy as np
import torch
from torchvision import transforms

def Eval_Smeasure(self,gt,pred):
    alpha, avg_q, img_num = 0.5, 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        pred = trans(pred).cuda()
        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                            torch.min(pred) + 1e-20)
        gt = trans(gt).cuda()

        gt = trans(gt)
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            Q = alpha * self._S_object(
                pred, gt) + (1 - alpha) * self._S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
                
def _S_object(self, pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = self._object(fg, gt)
    o_bg = self._object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q

def _object(self, pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score

def _S_region(self, pred, gt):
    X, Y = self._centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
    p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
    Q1 = self._ssim(p1, gt1)
    Q2 = self._ssim(p2, gt2)
    Q3 = self._ssim(p3, gt3)
    Q4 = self._ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

def _centroid(self, gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if self.cuda:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if self.cuda:
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
    return X.long(), Y.long()