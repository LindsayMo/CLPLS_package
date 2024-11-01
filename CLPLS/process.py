import scanpy as sc
import torch
import numpy as np
from .util import *

def process_st(st_adata):
    spatial_locs = st_adata.obsm['spatial']
    sadj = graph_alpha(spatial_locs, n_neighbors=10)
    graph_neigh = torch.FloatTensor(sadj.copy().toarray() + np.eye(sadj.shape[0]))
    
    if 'label_CSL' not in st_adata.obsm.keys():    
        add_contrastive_label(st_adata)
    label_CSL = torch.FloatTensor(st_adata.obsm['label_CSL'])
    return sadj, graph_neigh, label_CSL

def process_sc(sc_adata, key):
    
    ctype = sc_adata.obs[key]
    temp=[*ctype.unique()]
    ctype_dict = dict(zip(range(len(temp)), temp))

    ctype_dict_r = dict((y, x) for x, y in ctype_dict.items())
    ctype_lab = np.asarray(
        [ctype_dict_r[ii] for ii in ctype])
    ctype_lab = torch.LongTensor(np.array(ctype_lab))
   
    fadj = sc_adata.obsp['connectivities']
    fadj = disconnect_dif_label_nodes(ctype_lab,fadj)
    return ctype_lab, fadj

def process_activity(activity_adata, key):
    ctype_activity = activity_adata.obs[key]

    temp_activity=[*ctype_activity.unique()]
    ctype_dict_activity = dict(zip(range(len(temp_activity)), temp_activity))

    ctype_dict_r_activity = dict((y, x) for x, y in ctype_dict_activity.items())
    ctype_lab_activity = np.asarray(
        [ctype_dict_r_activity[ii] for ii in ctype_activity])
    ctype_lab_activity = torch.LongTensor(np.array(ctype_lab_activity))

    activity_adj = activity_adata.obsp['connectivities']
    activity_adj = disconnect_dif_label_nodes(ctype_lab_activity,activity_adj)
    return ctype_lab_activity, activity_adj
    