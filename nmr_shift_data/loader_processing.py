import rdkit

import graph_conv_many_nuc_util
import torch
import torch_geometric.transforms as T
from dataloader import DataLoader
import sys
from k_gnn import ConnectedThreeMalkin
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)

class knnGraph(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph, self).__init__(root, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'temp.pt'

    @property
    def processed_file_names(self):
        return 'whole.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d[0],
                atom_types=d[1],
                edge_index=d[2],
                edge_attr=d[3],
                mask=d[4],
                y=d[5],
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def process(infile, kgnn = False):
    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS
    ds = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, dataset_hparams)
    print('made datasets')
    sys.stdout.flush()

    torch.save(ds, '/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/raw/temp.pt')
    print('saved temp files')
    sys.stdout.flush()

    dataset = knnGraph(root='/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/')
    print('made dataset. dataset length: ' +str(dataset))
    sys.stdout.flush()

    if kgnn == True:
        dataset.data = ConnectedThreeMalkin()(dataset.data)

        #gets all the unique values of iso_type_3 and arranges them
        #then returns the index of the values in the original iso_type_3 in the unique ordered iso_type_3
        #reassigns this vector of indices (same size as original iso_type_3) to iso_type_3
        dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]

        #gets max of iso_type_3 + 1. This is the max index + 1
        num_i_3 = dataset.data.iso_type_3.max().item() + 1

        dataset.data.iso_type_3 = F.one_hot(dataset.data.iso_type_3, num_classes=num_i_3).to(torch.float)

    train_split = int(len(dataset)*0.6)
    val_split = int(len(dataset)*0.2)
    train_dataset = dataset[:train_split]
    val_dataset = dataset[train_split:train_split+val_split]
    test_dataset = dataset[train_split+val_split:]

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=1)
    print('created data loaders')
    sys.stdout.flush()

    if kgnn == False:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader, num_i_3