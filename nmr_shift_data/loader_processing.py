import rdkit
import graph_conv_many_nuc_util
import torch
import torch_geometric.transforms as T
from dataloader import DataLoader
import sys
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
                edge_index=d[1],
                edge_attr=d[2],
                mask=d[3],
                y=d[4],
                ) for d in raw_data_list
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def process(infile):
    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS
    ds = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, dataset_hparams)
    print('made datasets')
    sys.stdout.flush()

    torch.save(ds, '/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/raw/temp.pt')
    print('saved temp files')
    sys.stdout.flush()

    dataset = knnGraph(root='/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/')
    print('made dataset')
    sys.stdout.flush()

    train_split = int(len(dataset)*0.6)
    val_split = int(len(dataset)*0.2)
    train_dataset = dataset[:train_split]
    val_dataset = dataset[train_split:train_split+val_split]
    test_dataset = dataset[train_split+val_split:]

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=1)
    print('created data loaders')
    sys.stdout.flush()

    return train_loader, val_loader, test_loader