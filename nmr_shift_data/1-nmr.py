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
        nn1 = Sequential(Linear(4, 512), ReLU(), Linear(512, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 256
        nn2 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 256
        nn3 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x = data.x
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))

        x = scatter_mean(x, data.batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    # note that the number of atoms exceeds the number of carbons, and therefore there will be many zeros
    # 
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # print(type(data.y), type(data.y[0]), len(data.y), data.y[0].shape, data.y[0])
        # sys.stdout.flush()
        # print(torch.FloatTensor(data.y).size(), torch.FloatTensor(data.y))
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss * 64
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += ((model(data) * std[target].cuda()) -
                  (data[5] * std[target].cuda())).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 301):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    # val_error = test(val_loader)
    # scheduler.step(val_error)

    if epoch%50==0:
        test_error = test(test_loader)
        best_val_error = val_error
        print(
            'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}'
            'Test MAE: {:.7f}, '
            'Test MAE norm: {:.7f}'.format(epoch, lr, loss,
                                           test_error,
                                           test_error / std[target].cuda()))
        sys.stdout.flush()
    else:
        print('Epoch: {:03d}'.format(epoch))
        sys.stdout.flush()
