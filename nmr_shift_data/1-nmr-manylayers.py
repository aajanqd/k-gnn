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

infile = '/scratch/aqd215/k-gnn/nmr_shift_data/graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle'
                                                               
train_loader, val_loader, test_loader = process(infile)

print('train loaders in 1-nmr')
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

        M_in, M_out = M_out, 512
        nn4 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv4 = NNConv(M_in, M_out, nn4)

        M_in, M_out = M_out, 512
        nn5 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv5 = NNConv(M_in, M_out, nn5)

        M_in, M_out = M_out, 512
        nn6 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv6 = NNConv(M_in, M_out, nn6)

        M_in, M_out = M_out, 512
        nn7 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv7 = NNConv(M_in, M_out, nn7)

        M_in, M_out = M_out, 512
        nn8 = Sequential(Linear(4, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv8 = NNConv(M_in, M_out, nn8)

        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 16)
        self.fc6 = torch.nn.Linear(16, 1)

        self.initialize_weights()

    def forward(self, data):
        x = data.x #4096x37
        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr)) #4096x128
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr)) #4096x256
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr)) #4096x512
        x = F.elu(self.conv4(x, data.edge_index, data.edge_attr)) #4096x512

        x = F.elu(self.fc1(x)) #4096x256
        x = F.elu(self.fc2(x)) #4096x128
        x = self.fc3(x) #4096x1
        return x.flatten() #4096
    
    def initialize_weights(self):
        for m in self.modules():
#             print(m)
            if isinstance(m, Sequential):
                for elem in m:
                    if isinstance(elem, Linear):
                        torch.nn.init.kaiming_uniform_(elem.weight)
            elif isinstance(m, Linear):
                torch.nn.init.kaiming_uniform_(elem.weight)


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
