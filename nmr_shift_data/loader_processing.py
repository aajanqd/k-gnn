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
                 dset,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(knnGraph, self).__init__(root, dset, transform, pre_transform, pre_filter)
        self.type = type
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.dset = dset

    @property
    def raw_file_names(self):
    	if self.dset == 'train':
    		return 'train_temp.pt'
    	elif self.dset == 'test':
    		return 'test_temp.pt'
    	else:
    		print('not working')
    		sys.stdout.flush()

    @property
    def processed_file_names(self):
    	if self.dset == 'train':
    		return 'train.pt'
    	elif self.dset == 'test':
    		return 'test.pt'
    	else:
    		print('not working')
    		sys.stdout.flush()
    
    def download(self):
        pass

    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        print('raw_data_list type: ' + str(type(raw_data_list)) + ';', 'raw_data_list type: ' + str(type(raw_data_list[0])) + ';')
        sys.stdout.flush()
        data_list = [
            Data(
                x=d[0],
                edge_index=d[1],
                edge_attr=d[2],
                y=d[3],
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
	print('made datasets')
	sys.stdout.flush()
	torch.save(ds_train, '/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/raw/train_temp.pt')
	torch.save(ds_test, '/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/raw/test_temp.pt')
	print('saved temp files')
	sys.stdout.flush()
	train_dataset = knnGraph(root='/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/', dset = 'train')
	test_dataset = knnGraph(root='/scratch/aqd215/k-gnn/nmr_shift_data/temp_files/', dset = 'test')
	print('made dataset')
	sys.stdout.flush()
	train_loader = DataLoader(train_dataset, batch_size=64, num_workers=1)
	test_loader = DataLoader(test_dataset, batch_size=64, num_workers=1)
	print('created data loader')

	for i, t in enumerate(train_loader):
		print(t.x.size())
		if i >5:
			break