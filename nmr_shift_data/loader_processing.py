import rdkit

import os.path as osp
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
        return 'train_temp.pt'

    @property
    def processed_file_names(self):
        return 'train.pt'
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        data_list = [
            Data(
                x=d['x'],
                edge_index=d['edge_index'],
                edge_attr=d['edge_attr'],
                y=d['y'],
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
	ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, dataset_hparams)
	torch.save(ds_train, '/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/train_temp.pt')
	torch.save(ds_test, '/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/test_temp.pt')

	dataset = knnGraph(root='/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/')
	train_loader = DataLoader(dataset, batch_size=64, num_workers=1)

	for i, t in enumerate(train_loader):
		print(t.x.size())
		if i >5:
			break

process('/scratch/aqd215/k-gnn/nmr_shift_data/graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle')
