import torch
from torch.nn.functional import mse_loss
import sys

def MSE_loss(x,y,mask):
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    return ((x_masked-y_masked)**2).sum().item()

def MAE_loss(x,y,mask):
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    MAE = abs(x_masked-y_masked).sum().item()
    return MAE