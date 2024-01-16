import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy_loss(pred, target):
    """
    :param pred: [B, N=1024, 1] output for patch/location prediction
    :param target: label list [(index, ty, tx)], len(target) = B
    :return: loss value
    """
    pred_input = pred.squeeze(2)  # [B N]
    label_index = torch.zeros(pred_input.shape[0], dtype=torch.long)  # [B]
    for i in range(len(target)):
        label_index[i] = target[i][0]
    loss = F.cross_entropy(pred_input, label_index.to('cuda'))
    return loss

def regression_loss(pred, target):
    """
    :param pred: [B, N=1024, 2] prediction for ty and tx
    :param target: label list [(index, ty, tx)], len(target) = B
    :return: regression loss
    """
    label_txy = torch.zeros(pred.shape[0], 2).to('cuda')  # [B, 2]
    pred_input = torch.zeros(pred.shape[0], 2).to('cuda')  # [B, 2]
    print(pred.shape)
    for i in range(len(target)):
        print(i)
        idx, ty, tx = target[i]
        print(idx, ty, tx)
        label_txy[i] = torch.tensor([ty, tx])
        pred_input[i] = pred[i, int(idx), :]
    loss = F.mse_loss(pred_input, label_txy)
    return loss
