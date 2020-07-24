import numpy as np
import torch
import pandas as pd
import atom_features
import molecule_features
import torch.utils.data
import sys

class MoleculeDatasetMulti(torch.utils.data.Dataset):

    def __init__(self, mols, pred_vals, MAX_N, 
                 PRED_N = 1, 
                 feat_vert_args = {}, 
                 feat_edge_args = {}, 
                 adj_args = {}, combine_mat_vect = None, 
        ):
        self.mols = mols
        self.pred_vals = pred_vals
        self.MAX_N = MAX_N
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        #self.mask_zeroout_prob = mask_zeroout_prob
        self.PRED_N = PRED_N

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):

        mol = self.mols[idx]
        
        #each value in pred_vals is a dictionary containing key value pairs of atom numbers and chem shift vals
        # pred_val returns the dictionary for the appropriate index (molecule)
        pred_val = self.pred_vals[idx] 

        conf_idx = np.random.randint(mol.GetNumConformers()) #returns random number between 0 and numconformers; this is used to randomly select a conformation
        
        #f_vect is a 2d tensor containing atom features; inner tensors represent one atom and contain features
        #shape of f_vect is num_atomsxnum_features
        vect_feat = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
                                                
        DATA_N = vect_feat.shape[0] #number of atoms
                        
        edge_index, edge_attr = molecule_features.get_edge_attr_and_ind(mol)

        #pred_val is a dictionary containing key value pairs of atom numbers and chem shift vals
        vals = np.zeros((DATA_N, 1), dtype=np.float32)
        for pn in range(self.PRED_N):
            for k, v in pred_val[pn].items():
                vals[int(k), pn] = v

        v = (vect_feat, edge_index, edge_attr, vals)
        print(vect_feat.shape, edge_index.shape, edge_attr.shape, vals.shape)
        sys.stdout.flush()

        return v