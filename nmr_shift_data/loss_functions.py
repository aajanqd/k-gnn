import torch
from torch.nn.functional import mse_loss
import sys

def MSE_loss(x,y,mask):
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    print("x, y masked : " +str(x_masked)+","+str(y_masked))
    sys.stdout.flush()
    return mse_loss(x_masked, y_masked)

def MAE_loss(x,y,mask):
    # print("x, y, mask sizes: " +str(x.size())+","+str(y.size())+","+str(mask.size()))
    # sys.stdout.flush()
    x_masked = x[mask>0]
    y_masked = y[mask>0]
    l = y[mask>0].size()[0]
    MAE = float(abs(x_masked-y_masked).sum()/l)
    return MAE