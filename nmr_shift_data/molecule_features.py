import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
from rdkit.Chem import AllChem  # noqa
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType
from util import get_nos_coords

def coalesce(index, value):
    n = index.max().item() + 1
    row, col = index
    unique, inv = torch.unique(row * n + col, sorted=True, return_inverse=True)

    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    index = torch.stack([row[perm], col[perm]], dim=0)
    value = value[perm]

    return index, value

def feat_tensor_mol(mol, feat_distances=True, feat_r_pow = None, 
                    MAX_POW_M = 2.0, conf_idx = 0):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    
    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)

    if feat_distances:
        pos = coords
        a = pos.T.reshape(1, 3, -1) #turns Nx3 matx into 1x3xN
        b = (a - a.T)
        c = np.swapaxes(b, 2, 1)
        res_mats.append(c) #appends a new NxNx3 matrix
    if feat_r_pow is not None:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = (a - a.T)**2
        c = np.swapaxes(b, 2, 1)
        d = np.sqrt(np.sum(c, axis=2)) #sum along third axis to get NxN matrix, sqrt entries
        e = (np.eye(d.shape[0]) + d)[:, :, np.newaxis] #add identity, return NxNx1 matx

                       
        for p in feat_r_pow:
            e_pow = e**p #square each entry in NxNx1 matx
            if (e_pow > MAX_POW_M).any():
                print("WARNING: max(M) = {:3.1f}".format(np.max(e_pow)))
                e_pow = np.minimum(e_pow, MAX_POW_M)

            res_mats.append(e_pow)
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)
    else: # Empty matrix
        M = np.zeros((ATOM_N, ATOM_N, 0), dtype=np.float32)

    return M 

def mol_to_nums_adj(m, MAX_ATOM_N=None):# , kekulize=False):
    """
    molecule to symmetric adjacency matrix
    added in edge attributes and index
    """

    m = Chem.Mol(m)

    # m.UpdatePropertyCache()
    # Chem.SetAromaticity(m)
    # if kekulize:
    #     Chem.rdmolops.Kekulize(m)

    ATOM_N = m.GetNumAtoms()
    if MAX_ATOM_N is None:
        MAX_ATOM_N = ATOM_N

    adj = np.zeros((MAX_ATOM_N, MAX_ATOM_N))
    atomic_nums = np.zeros(MAX_ATOM_N)

    assert ATOM_N <= MAX_ATOM_N

    for i in range(ATOM_N):
        a = m.GetAtomWithIdx(i)
        atomic_nums[i] = a.GetAtomicNum()

    for b in m.GetBonds():
        head = b.GetBeginAtomIdx() #first atom in bond
        tail = b.GetEndAtomIdx() #second atom in bond
        order = b.GetBondTypeAsDouble() # order of bond
        adj[head, tail] = order 
        adj[tail, head] = order

    #adj returns adjacency matrix with bond orders as entries
    #note that adj is SYMMETRIC
    #shape is 64x64, diagonals are all are 0
    return atomic_nums, adj

def get_edge_attr_and_ind(m):
    m = Chem.Mol(m)
    # print(m.GetNumBonds())
    row, col, single, double, triple, aromatic = [], [], [], [], [], []

    for bond in m.GetBonds():
        head = bond.GetBeginAtomIdx() #first atom in bond
        tail = bond.GetEndAtomIdx() #second atom in bond

        row.append(head)
        col.append(tail)

        row.append(tail)
        col.append(head)

        bond_type = bond.GetBondType()
        single.append(1 if bond_type == BondType.SINGLE else 0)
        single.append(single[-1])
        double.append(1 if bond_type == BondType.DOUBLE else 0)
        double.append(double[-1])
        triple.append(1 if bond_type == BondType.TRIPLE else 0)
        triple.append(triple[-1])
        aromatic.append(1 if bond_type == BondType.AROMATIC else 0)
        aromatic.append(aromatic[-1])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor([single, double, triple, aromatic],
                             dtype=torch.float).t().contiguous()

    # assert tuple(edge_index.size()) == (2,140)
    # assert tuple(edge_attr.size()) == (4,140)

    # edge_index, edge_attr = coalesce(edge_index, edge_attr)

    return edge_index, edge_attr

def feat_mol_adj(mol, edge_weighted=True, add_identity=False, 
                 norm_adj=False, split_weights = None):
    """
    Compute the adjacency matrix for this molecule

    If split-weights == [1, 2, 3] then we create separate adj matrices for those
    edge weights

    NOTE: We do not kekulize the molecule, we assume that has already been done

    """
    
    atomic_nos, adj = mol_to_nums_adj(mol) #returns atomic numbers and adj matx
    ADJ_N = adj.shape[0] #num_atoms
    adj = torch.Tensor(adj) #make tensor from 2d numpy array, 
    
    if edge_weighted and split_weights is not None:
        raise ValueError("can' have both weighted edies and split the weights")
    
    if split_weights is None:
        adj = adj.unsqueeze(0) #adj now 1x64x64
        if edge_weighted:
            pass # already weighted
        else:
            adj[adj > 0] = 1.0
    # list [1,1.5,2,3] passed in as argument
    else:
        split_adj = torch.zeros((len(split_weights), ADJ_N, ADJ_N ))
        for i in range(len(split_weights)):
            split_adj[i] = (adj == split_weights[i])
        adj = split_adj
        
    if norm_adj and not add_identity:
        raise ValueError()
        
    if add_identity:
        adj = adj + torch.eye(ADJ_N)

    if norm_adj:
        res = []
        for i in range(adj.shape[0]):
            a = adj[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))

            s1 = D_12.reshape(ADJ_N, 1)
            s2 = D_12.reshape(1, ADJ_N)
            adj_i = s1 * a * s2 
            res.append(adj_i)
        adj = torch.stack(res)
    return adj

