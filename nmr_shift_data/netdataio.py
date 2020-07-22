import numpy as np
import torch
import pandas as pd
import atom_features
import molecule_features
import torch.utils.data

class MoleculeDatasetNew(torch.utils.data.Dataset):

    def __init__(self, mols, pred_vals , MAX_N, 
                 feat_vert_args = {}, 
                 feat_edge_args = {}, 
                 adj_args = {}, combine_mat_vect = None, 
                 mask_zeroout_prob=0.0):
        self.mols = mols
        self.pred_vals = pred_vals
        self.MAX_N = MAX_N
        self.cache = {}
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        self.mask_zeroout_prob = mask_zeroout_prob

    def __len__(self):
        return len(self.mols)
    
    def mask_sel(self, v):
        if self.mask_zeroout_prob == 0.0: # not self.single_value:
            return v
        else:
            mask = v[4].copy()
            a = np.sum(mask)
            mask[np.random.rand(*mask.shape) < self.mask_zeroout_prob] = 0.0
            b = np.sum(mask)
            out = list(v)
            out[4] = mask
            return out

    def cache_key(self, idx, conf_idx):
        return (idx, conf_idx)

    def __getitem__(self, idx):

        mol = self.mols[idx]
        pred_val = self.pred_vals[idx]

        conf_idx = np.random.randint(mol.GetNumConformers())

        if self.cache_key(idx, conf_idx) in self.cache:
            return self.mask_sel(self.cache[self.cache_key(idx, conf_idx)])
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
                                                
        DATA_N = f_vect.shape[0]
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect

        f_mat = molecule_features.feat_tensor_mol(mol, conf_idx=conf_idx,
                                                  **self.feat_edge_args) 

        if self.combine_mat_vect:
            MAT_CHAN = f_mat.shape[2] + vect_feat.shape[1]
        else:
            MAT_CHAN = f_mat.shape[2]

        mat_feat = np.zeros((self.MAX_N, self.MAX_N, MAT_CHAN), dtype=np.float32)
        # do the padding
        mat_feat[:DATA_N, :DATA_N, :f_mat.shape[2]] = f_mat  
        
        if self.combine_mat_vect == 'row':
            # row-major
            for i in range(DATA_N):
                mat_feat[i, :DATA_N, f_mat.shape[2]:] = f_vect
        elif self.combine_mat_vect == 'col':
            # col-major
            for i in range(DATA_N):
                mat_feat[:DATA_N, i, f_mat.shape[2]:] = f_vect

        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad

        edge_index, edge_attr = molecule_features.get_edge_attr_and_ind(m)
                        
        # create mask and preds 
        
        mask = np.zeros(self.MAX_N, dtype=np.float32)
        vals = np.zeros(self.MAX_N, dtype=np.float32)
        for k, v in pred_val.items():
            mask[k] = 1.0
            vals[k] = v

        v = (vect_feat, vals, edge_index, edge_attr)
        
        
        self.cache[self.cache_key(idx, conf_idx)] = v
        return self.mask_sel(v)


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
        self.cache = {}
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        #self.mask_zeroout_prob = mask_zeroout_prob
        self.PRED_N = PRED_N

    def __len__(self):
        return len(self.mols)
    
    def mask_sel(self, v):
        return v

    def cache_key(self, idx, conf_idx):
        return (idx, conf_idx)

    def __getitem__(self, idx):

        mol = self.mols[idx]
        pred_val = self.pred_vals[idx]

        conf_idx = np.random.randint(mol.GetNumConformers()) #returns random number between 0 and numconformers; this is used to randomly select a conformation

        if self.cache_key(idx, conf_idx) in self.cache:
            return self.mask_sel(self.cache[self.cache_key(idx, conf_idx)])
        
        #f_vect is a 2d tensor containing atom features; inner tensors represent one atom and contain features
        #shape of f_vect is num_atomsxnum_features
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
                                                
        DATA_N = f_vect.shape[0] #number of atoms
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32) #2d numpy array with same shape as f_vect
        vect_feat[:DATA_N] = f_vect #fill in vect_feat with data from f_vect; if number of atoms <64, unfilled items will be 0

        #f_mat is an NxNxnum_features matrix, where N is the number of atoms in the molecule, and num_features is the number of features per atom
        #incorporates coordinates
        f_mat = molecule_features.feat_tensor_mol(mol, conf_idx=conf_idx,
                                                  **self.feat_edge_args)

        if self.combine_mat_vect:
            MAT_CHAN = f_mat.shape[2] + vect_feat.shape[1]
        else:
            MAT_CHAN = f_mat.shape[2]
        if MAT_CHAN == 0: # Dataloader can't handle tensors with empty dimensions
            MAT_CHAN = 1
        mat_feat = np.zeros((self.MAX_N, self.MAX_N, MAT_CHAN), dtype=np.float32)

        #fill in mat_feat with data from f_mat; if number of atoms <64, unfilled items will be 0
        #resulting matrix is 64x64xnum_features, filled in up to num_atomsxnum_atomsxnum_features
        mat_feat[:DATA_N, :DATA_N, :f_mat.shape[2]] = f_mat  
        
        #DON'T WORRY ABOUT THIS CODE
        #below, they're filling in the values along the third dimension with feat_vect if they haven't already been filled
        #this will only change things if combine_mat_vect = True; hence the name "combine mat vect"
        if self.combine_mat_vect == 'row':
            # row-major
            for i in range(DATA_N):
                mat_feat[i, :DATA_N, f_mat.shape[2]:] = f_vect
        elif self.combine_mat_vect == 'col':
            # col-major
            for i in range(DATA_N):
                mat_feat[:DATA_N, i, f_mat.shape[2]:] = f_vect

        #code below returns padded adjacency matrix
        #padding is the same as the madding for mat_feat
        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad
                        
        # create mask and preds 
        
        mask = np.zeros((self.MAX_N, self.PRED_N), 
                        dtype=np.float32)
        vals = np.zeros((self.MAX_N, self.PRED_N), 
                        dtype=np.float32)

        edge_index, edge_attr = molecule_features.get_edge_attr_and_ind(mol)

        #each value in pred_val is a dictionary containing key value pairs of atom numbers and chem shift vals
        for pn in range(self.PRED_N):
            for k, v in pred_val[pn].items():
                mask[int(k), pn] = 1.0
                vals[int(k), pn] = v

        v = (vect_feat, edge_index, edge_attr, vals)
        
        
        # self.cache[self.cache_key(idx, conf_idx)] = v

        #returns tuple containing tensors for adj matx, feature vects, feature matxs, targets, and mask
        #each one of these things exists for every atom
        return v
