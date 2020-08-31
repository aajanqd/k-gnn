import torch

import netdataio
import pickle
import copy

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
        

default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17] #filtering for H, C, N, O, F, Cl, etc

### Create datasets and data loaders

default_feat_vect_args = dict(feat_atomicno_onehot=default_atomicno, 
                              feat_pos=False, feat_atomicno=True,
                              feat_valence=True, aromatic=True, hybridization=True, 
                              partial_charge=False, formal_charge=True,  # WE SHOULD REALLY USE THIS 
                              r_covalent=False,
                              total_valence_onehot=True, 
                              
                              r_vanderwals=False, default_valence=True, rings=True)

default_feat_mat_args = dict(feat_distances = False, 
                             feat_r_pow = None)

default_split_weights = [1, 1.5, 2, 3]

default_adj_args = dict(edge_weighted=False, 
                        norm_adj=True, add_identity=True, 
                        split_weights=default_split_weights)

alt_adj_args = dict(edge_weighted=True, 
                        norm_adj=True, add_identity=True, 
                        split_weights=None)


DEFAULT_DATA_HPARAMS = {'feat_vect_args' : default_feat_vect_args,       #These are the arugments for the feature vectors
                       'feat_mat_args' : default_feat_mat_args,          #These are the arguments for the (adjacency) matrices
                       'adj_args' : default_adj_args}

WEIGHTED_ADJ_DATA_HPARAMS = {'feat_vect_args' : default_feat_vect_args,       #These are the arugments for the feature vectors
                       'feat_mat_args' : default_feat_mat_args,          #These are the arguments for the (adjacency) matrices
                       'adj_args' : alt_adj_args}


def make_datasets(exp_config, hparams, train_sample=0):
    """
    """


    d = pickle.load(open(exp_config['filename'], 'rb'))
    train_df = d['train_df']
    if train_sample > 0:
        print("WARNING, subsampling training data to ", train_sample)
        train_df = train_df.sample(train_sample, random_state=0) 

    tgt_nucs = d['tgt_nucs']
    test_df = d['test_df'] #.sample(10000, random_state=10)

    MAX_N = d['MAX_N']

    datasets = {}

    df = train_df.append(test_df)
    # for phase, df in [('train', train_df), 
    #                   ('test', test_df)]:
                       

    ds = netdataio.MoleculeDatasetMulti(df.rdmol.tolist(), 
                                        df.value.tolist(),  
                                        MAX_N, len(tgt_nucs), 
                                        hparams['feat_vect_args'], 
                                        hparams['feat_mat_args'], 
                                        hparams['adj_args'])        
    # datasets[phase] = ds

        

    # return datasets['train'], datasets['test']
    return ds