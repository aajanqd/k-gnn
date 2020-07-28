import torch
from torch.nn.functional import mse_loss
import sys

def MSE_loss(x,y,mask):
    print("x, y, mask sizes: " +str(pred.size())+","+str(target.size())+","+str(mask.size()))
    sys.stdout.flush()
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    return mse_loss(x_masked, y_masked)

def MAE_loss(x,y,mask):
	print("x, y, mask sizes: " +str(pred.size())+","+str(target.size())+","+str(mask.size()))
    sys.stdout.flush()
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    l = y[mask>0].size()[0]
    MAE = float(abs(x_masked-y_masked).sum()/l)
    return MAE