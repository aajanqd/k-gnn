import rdkit
import os.path as osp
import graph_conv_many_nuc_util
from graph_conv_many_nuc_util import move
import argparse
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv
import sys
from loader_processing import process
import loss_functions

infile = '/scratch/aqd215/k-gnn/nmr_shift_data/graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle'
                                                               
train_loader, test_loader = process(infile)

print('train loaders in 1-nmr')
sys.stdout.flush()
print(train_loader)
sys.stdout.flush()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        M_in, M_out = 37, 128
        nn1 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 256
        nn2 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 512
        nn3 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1) #64x64

    def forward(self, data):
        x = data.x #4096x37
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr)) #4096x128
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr)) #4096x256
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr)) #4096x512

        # x = scatter_mean(x, data.batch, dim=0) #4096x512 -> 64x512 (aggregates across molecules)

        x = F.elu(self.fc1(x)) #4096x256
        x = F.elu(self.fc2(x)) #4096x128
        x = self.fc3(x) #4096x1
        return x.flatten() #4096


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)

def train(epoch):
    model.train()
    loss_all = 0
    total = 0

    # note that the number of atoms exceeds the number of carbons, and therefore there will be many zeros
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        target = torch.FloatTensor(data.y)
        target = target.reshape(target.size()[0]*target.size()[1]).flatten().to(device)

        mask = torch.FloatTensor(data.mask)
        mask = mask.reshape(mask.size()[0]*mask.size()[1]).flatten().to(device)

        pred = model(data)

        loss = loss_functions.MSE_loss(pred, target, mask)
        loss.backward()
        loss_all += loss
        optimizer.step()
        total += 1
    return float(loss_all) / total


def test(loader):
    model.eval()
    error = 0
    total = 0

    for data in loader:
        data = data.to(device)

        target = torch.FloatTensor(data.y)
        target = target.reshape(target.size()[0]*target.size()[1]).flatten().to(device)

        mask = torch.FloatTensor(data.mask)
        mask = mask.reshape(mask.size()[0]*mask.size()[1]).flatten().to(device)

        pred = model(data)

        error += loss_functions.MAE_loss(pred, target, mask)  # MAE
        total += 1
    return float(error) / total

for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    avg_train_loss = train(epoch)
    test_error = test(test_loader)
    scheduler.step(test_error)

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, avg_train_loss, test_error))
    sys.stdout.flush()