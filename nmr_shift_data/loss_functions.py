import torch
from torch.nn.functional import mse_loss
import sys

def MSE_loss(x,y,mask):
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    return mse_loss(x_masked, y_masked)

def MAE_loss(x,y,mask):
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    MAE = float(abs(x_masked-y_masked).sum())
    return MAE