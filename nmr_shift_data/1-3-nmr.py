import rdkit
import graph_conv_many_nuc_util
from graph_conv_many_nuc_util import move
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import NNConv
import sys
from loader_processing import process
import loss_functions
from k_gnn import GraphConv, DataLoader, avg_pool
from k_gnn import ConnectedThreeMalkin

infile = '/scratch/aqd215/k-gnn/nmr_shift_data/graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle'
                                                               
train_loader, val_loader, test_loader, num_i_3 = process(infile, kgnn = True)

print('train loaders in 1-nmr')
sys.stdout.flush()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        M_in, M_out = 37, 64
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv6 = GraphConv(64 + num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(data.x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.flatten()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.00001)

def train(epoch):
    model.train()
    loss_all = 0
    total_atoms = 0

    # note that the number of atoms exceeds the number of carbons, and therefore there will be many zeros
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        atoms = data.mask.sum().item()
        pred = model(data)

        loss = loss_functions.MSE_loss(pred, data.y, data.mask)
        loss.backward()
        loss_all += loss
        optimizer.step()
        total_atoms += atoms
    return float(loss_all) / total_atoms

def test(loader):
    model.eval()
    error = 0
    loss = 0
    total_atoms = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            atoms = data.mask.sum().item()
            pred = model(data)

            loss += loss_functions.MSE_loss(pred, data.y, data.mask)
            error += loss_functions.MAE_loss(pred, data.y, data.mask)
            total_atoms += atoms

        return float(error) / total_atoms, float(loss) / total_atoms

for epoch in range(1500):
    # torch.cuda.empty_cache()
    lr = scheduler.optimizer.param_groups[0]['lr']
    avg_train_loss = train(epoch)
    val_error, val_loss = test(val_loader)
    scheduler.step(val_error)
    test_error, test_loss = test(test_loader)
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Val Loss: {:.7f}, Test Loss: {:.7f}, Test MAE: {:.7f}'.format(epoch, lr, avg_train_loss, val_loss, test_loss, test_error))
    sys.stdout.flush()