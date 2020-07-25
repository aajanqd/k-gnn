import torch
import torch.nn as nn
import sys

def MSE_loss(x,y,mask):
	print(x.shape, mask.shape)
	sys.stdout.flush()
    x_masked = x[mask>0].reshape(-1, 1)
    y_masked = y[mask>0].reshape(-1, 1)
    return nn.MSELoss(x_masked, y_masked)

def MAE_loss(x,y,mask):
    x_masked = x[mask>0].reshape(-1, 1)
    y_masked = y[mask>0].reshape(-1, 1)
    l = y[mask>0].reshape(-1).size()[0]
    MAE = float(abs(t1-t2).sum()/l)
    return MAE