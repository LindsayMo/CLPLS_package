import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from .model import Encoder, MLP, GraphEncoder
from tqdm import tqdm
from torch import nn
import scanpy as sc
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from .util import *




def train(dim_input, hidden_channels, dim_output,nclass,st_adata,graph_neigh,spatial_graph,sc_adata,snn,ctype_lab,label_CSL,epochs,beta):
    expr = st_adata.X.todense() if type(st_adata.X).__module__ != np.__name__ else st_adata.X
    expr = torch.tensor(expr).float()
    edge_list = sparse_mx_to_torch_edge_list(spatial_graph)
    print(edge_list)

    expr_sc = sc_adata.X.todense() if type(sc_adata.X).__module__ != np.__name__ else sc_adata.X
    expr_sc = torch.tensor(expr_sc).float()
    edge_list_sc = sparse_mx_to_torch_edge_list(snn)
    edge_weight_sc = sparse_mx_to_torch_edge_weight(snn)
    print(edge_list_sc)


    model = Encoder(dim_input, hidden_channels, dim_output, graph_neigh)
    mlp_model = MLP(nfeat=dim_output, nhid=25, nclass=nclass)
    loss_CSL = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0005, weight_decay=5e-4)                                    
    
    print('Begin to train ST data...')
    model.train()
    mlp_model.train()
    pos_loss = []
    class_loss = []
    total_loss = []
    for epoch in tqdm(range(epochs)): 
        model.train()
            
        expr_a, _ = corruption(expr,edge_list)
        z_t, ret, ret_a = model(expr, expr_a, edge_list, None)
        z_c = model(expr_sc, None, edge_list_sc,edge_weight_sc)
        output = mlp_model(z_c)
        
        predicted_labels = torch.argmax(output, dim=1)
        correct = (predicted_labels == ctype_lab).sum().item()
        accuracy = correct / len(ctype_lab)
        loss_class =  F.nll_loss(output, ctype_lab)
        loss_sl_1 = loss_CSL(ret, label_CSL)
       
        loss =  loss_class + beta*loss_sl_1
        
        optimizer.zero_grad()
        mlp_optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        mlp_optimizer.step()

        pos_loss.append(loss_sl_1.detach().cpu().item())
        class_loss.append(loss_class.detach().cpu().item())
        total_loss.append(loss.detach().cpu().item())

    model.eval()
    ## try mlp prediction
    mlp_model.eval()
    z_t, ret, ret_a = model(expr, expr_a, edge_list, None)
    z_c = model(expr_sc, None, edge_list_sc,edge_weight_sc)

    embedding_st = z_t.cpu().detach().numpy()
    embedding_sc = z_c.cpu().detach().numpy()
  
    print("Optimization finished for ST data!")
    # np.savetxt(os.path.join(args.save_path,args.dataset,f'embedding_st_epoch{epochs}_beta{beta}.tsv'), embedding_st[:, :], delimiter="\t")
    # np.savetxt(os.path.join(args.save_path,args.dataset,f'embedding_rna_epoch{epochs}_beta{beta}.tsv'), embedding_sc[:, :], delimiter="\t")
    
    print("Training complete!\nEmbedding is saved ")

    return embedding_st, embedding_sc

def train_multiomics(rna_dim_input, activity_dim_input, dim_input, hidden_channels, dim_output,activity_adata,st_adata,graph_neigh,spatial_graph,sc_adata,snn,activity_adj,ctype_lab_rna,ctype_lab_activity,label_CSL,epochs,beta,alpha,theta):
    expr = st_adata.X.todense() if type(st_adata.X).__module__ != np.__name__ else st_adata.X
    expr = torch.tensor(expr).float()
    edge_list = sparse_mx_to_torch_edge_list(spatial_graph)
    print(edge_list)

    expr_sc = sc_adata.X.todense() if type(sc_adata.X).__module__ != np.__name__ else sc_adata.X
    expr_sc = torch.tensor(expr_sc).float()
    edge_list_sc = sparse_mx_to_torch_edge_list(snn)
    edge_weight_sc = sparse_mx_to_torch_edge_weight(snn)
    print(edge_list_sc)

    expr_activity = activity_adata.X.todense() if type(activity_adata.X).__module__ != np.__name__ else activity_adata.X
    expr_activity = torch.tensor(expr_activity).float()
    edge_list_activity = sparse_mx_to_torch_edge_list(activity_adj)
    edge_weight_activity = sparse_mx_to_torch_edge_weight(activity_adj)
   

    model = Encoder(dim_input, hidden_channels, dim_output, graph_neigh)

    rna_model = GraphEncoder(rna_dim_input, hidden_channels, dim_output)
    activity_model = GraphEncoder(activity_dim_input, hidden_channels, dim_output)
    mlp_model = MLP(nfeat=dim_output, nhid=25, nclass=7)
    mlp_model_activity = MLP(nfeat=dim_output, nhid=25, nclass=8)
   
    loss_CSL = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.0005, weight_decay=5e-4)
    mlp_optimizer_activity = torch.optim.Adam(mlp_model_activity.parameters(), lr=0.0005, weight_decay=5e-4)                                     
    
    print('Begin to train ST data...')
    model.train()
    mlp_model.train()
    mlp_model_activity.train()
    pos_loss = []
    neg_loss = []
    class_loss = []
    class_loss_activity = []
    total_loss = []
    for epoch in tqdm(range(epochs)): 
        model.train()
            
        expr_a, _ = corruption(expr,edge_list)
        z_t, ret, ret_a = model(expr, expr_a, edge_list, None)
        ##令rna和activity的卷积网络参数与ST的DGI中的卷积网络参数分块共享
        rna_model.conv.lin.weight.data = model.graph_encoder.conv.lin.weight.data[:, 0:rna_dim_input]
        rna_model.conv2.lin.weight.data = model.graph_encoder.conv2.lin.weight.data
        activity_model.conv.lin.weight.data = model.graph_encoder.conv.lin.weight.data[:, 246:]
        activity_model.conv2.lin.weight.data = model.graph_encoder.conv2.lin.weight.data

        z_c = rna_model(expr_sc, edge_list_sc,edge_weight_sc)
        z_activity = activity_model(expr_activity, edge_list_activity,edge_weight_activity)
        output = mlp_model(z_c)
        output_activity = mlp_model_activity(z_activity)
    
        loss_class =  F.nll_loss(output, ctype_lab_rna)
        loss_class_activity = F.nll_loss(output_activity, ctype_lab_activity)
        loss_sl_1 = loss_CSL(ret, label_CSL)
    
        loss =  alpha*loss_class + beta*loss_sl_1 + theta*loss_class_activity
        
        optimizer.zero_grad()
        mlp_optimizer.zero_grad()
        mlp_optimizer_activity.zero_grad()
        loss.backward() 
        optimizer.step()
        mlp_optimizer.step()
        mlp_optimizer_activity.step()

        pos_loss.append(loss_sl_1.detach().cpu().item())
    
        class_loss.append(loss_class.detach().cpu().item())
        class_loss_activity.append(loss_class_activity.detach().cpu().item())
        total_loss.append(loss.detach().cpu().item())


    model.eval()
    rna_model.eval()
    activity_model.eval()
    z_t, ret, ret_a = model(expr, expr_a, edge_list, None)
  
    z_c = rna_model(expr_sc, edge_list_sc,edge_weight_sc)
    z_activity = activity_model(expr_activity, edge_list_activity,edge_weight_activity)

    embedding_st = z_t.cpu().detach().numpy()
    embedding_sc = z_c.cpu().detach().numpy()
    embedding_activity = z_activity.cpu().detach().numpy()
    print("Optimization finished for ST data!")
    # np.savetxt(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_inter/combine_embedding_st_epoch{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.tsv', embedding_st[:, :], delimiter="\t")
    # np.savetxt(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_inter/combine_embedding_rna_epoch{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.tsv', embedding_sc[:, :], delimiter="\t")
    # np.savetxt(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_inter/combine_embedding_activity_epoch{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.tsv', embedding_activity[:, :], delimiter="\t")

    print("Training complete!\nEmbedding is saved ")

    return embedding_st, embedding_sc, embedding_activity
    
def PLSR(sc_adata, st_adata,embedding_sc,embedding_st, component,label):
    pls = PLSRegression(n_components=component)
    pls.fit(embedding_sc.T, embedding_st.T)
    pls.coef_.shape
    map = F.softmax(torch.tensor(pls.coef_), dim=1).numpy()
    map_matrix = map.T

    # construct cell type matrix
    matrix_cell_type = construct_cell_type_matrix(sc_adata,label)
    matrix_cell_type = matrix_cell_type.values
        
    # projection by spot-level
    matrix_projection = map_matrix.dot(matrix_cell_type)

    # rename cell types
    cell_type = list(sc_adata.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    #cell_type = [s.replace(' ', '_') for s in cell_type]
    df_projection = pd.DataFrame(matrix_projection, index=st_adata.obs_names, columns=cell_type)  # spot x cell type

    #normalize by row (spot)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)
    return  df_projection
   


def construct_cell_type_matrix(adata_sc,label):
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    #res = mat.sum()
    return mat

def extract_top_value(map_matrix, retain_percent = 0.1): 
    '''\
    Filter out cells with low mapping probability

    Parameters
    ----------
    map_matrix : array
        Mapped matrix with m spots and n cells.
    retain_percent : float, optional
        The percentage of cells to retain. The default is 0.1.

    Returns
    -------
    output : array
        Filtered mapped matrix.

    '''

    #retain top 1% values for each spot
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    
    return output 
