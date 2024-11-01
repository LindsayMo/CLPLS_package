import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import time
import pickle
import random
import numpy as np
from model import Encoder, GraphEncoder, MLP
from tqdm import tqdm
from torch import nn
import scanpy as sc
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from util import *

# class MLP(nn.Module):
#     def __init__(self, nfeat,nhid, nclass):
#         super(MLP,self).__init__()
#         self.fc1 = nn.Linear(nfeat, nhid)
#         self.fc2 = nn.Linear(nhid, nclass)
#         self.relu = nn.ReLU()
#         self.softmax = nn.LogSoftmax(dim=1)
#     def forward(self,v):
#         fc1_out = self.relu(self.fc1(v))
#         fc2_out = self.fc2(fc1_out)
#         output = self.softmax(fc2_out)
#         return output
    

def train(rna_dim_input, activity_dim_input, dim_input, hidden_channels, dim_output,activity_adata,st_adata,graph_neigh,spatial_graph,sc_adata,snn,activity_adj,ctype_lab_rna,ctype_lab_activity,label_CSL,epochs,beta,alpha,theta):
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
        print(model.graph_encoder.conv.lin.weight.data.shape)
        print(model.graph_encoder.conv2.lin.weight.data.shape)
        rna_model.conv.lin.weight.data = model.graph_encoder.conv.lin.weight.data[:, 0:rna_dim_input]
        rna_model.conv2.lin.weight.data = model.graph_encoder.conv2.lin.weight.data
        activity_model.conv.lin.weight.data = model.graph_encoder.conv.lin.weight.data[:, 246:]
        activity_model.conv2.lin.weight.data = model.graph_encoder.conv2.lin.weight.data

        z_c = rna_model(expr_sc, edge_list_sc,edge_weight_sc)
        z_activity = activity_model(expr_activity, edge_list_activity,edge_weight_activity)
        output = mlp_model(z_c)
        output_activity = mlp_model_activity(z_activity)
        print(output)
        print(ctype_lab_rna)
        loss_class =  F.nll_loss(output, ctype_lab_rna)
        loss_class_activity = F.nll_loss(output_activity, ctype_lab_activity)
        loss_sl_1 = loss_CSL(ret, label_CSL)
        # loss_sl_2 = loss_CSL(ret_a, label_CSL)
        
        # loss =  loss_class + beta*loss_sl_1 + theta*loss_sl_2
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
        print(f"Epoch {epoch}, Loss: {str(loss)}")


    plot_fitting_process_2mlp(pos_loss, class_loss,class_loss_activity,total_loss,epochs,beta,theta,alpha)

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
    np.savetxt(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_inter/combine_embedding_st_epoch{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.tsv', embedding_st[:, :], delimiter="\t")
    np.savetxt(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_inter/combine_embedding_rna_epoch{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.tsv', embedding_sc[:, :], delimiter="\t")
    np.savetxt(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_inter/combine_embedding_activity_epoch{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.tsv', embedding_activity[:, :], delimiter="\t")

    print("Training complete!\nEmbedding is saved ")

    return embedding_st, embedding_sc
    

if __name__ == "__main__":
    random_seed = 11
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    #read data
    st_adata = sc.read_h5ad('/home/nas2/biod/myy/project/SNPLS_new/data/ISSAAC_renamed_label/combine_inter/adata_st.h5ad') 
    spatial_locs = st_adata.obsm['spatial']
    sadj = graph_alpha(spatial_locs, n_neighbors=10)

    # construct_interaction(st_adata)
    graph_neigh = torch.FloatTensor(sadj.copy().toarray() + np.eye(sadj.shape[0]))
    if 'label_CSL' not in st_adata.obsm.keys():    
        add_contrastive_label(st_adata)
    label_CSL = torch.FloatTensor(st_adata.obsm['label_CSL'])

    sc_adata = sc.read_h5ad('/home/nas2/biod/myy/project/SNPLS_new/data/ISSAAC_renamed_label/combine_inter/adata_rna.h5ad')
   
    activity_adata = sc.read_h5ad("/home/nas2/biod/myy/project/SNPLS_new/data/ISSAAC_renamed_label/combine_inter/adata_activity.h5ad")
    #类别信息
    ctype_rna = sc_adata.obs["Annotation"]
  
    temp_rna=[*ctype_rna.unique()]
    ctype_dict_rna = dict(zip(range(len(temp_rna)), temp_rna))

    ctype_dict_r_rna = dict((y, x) for x, y in ctype_dict_rna.items())
    ctype_lab_rna = np.asarray(
        [ctype_dict_r_rna[ii] for ii in ctype_rna])
    ctype_lab_rna = torch.LongTensor(np.array(ctype_lab_rna))
    
    ctype_activity = activity_adata.obs["ATAC_Cluster"]

    temp_activity=[*ctype_activity.unique()]
    ctype_dict_activity = dict(zip(range(len(temp_activity)), temp_activity))

    ctype_dict_r_activity = dict((y, x) for x, y in ctype_dict_activity.items())
    ctype_lab_activity = np.asarray(
        [ctype_dict_r_activity[ii] for ii in ctype_activity])
    ctype_lab_activity = torch.LongTensor(np.array(ctype_lab_activity))

    fadj = sc_adata.obsp['connectivities']
    fadj = disconnect_dif_label_nodes(ctype_lab_rna,fadj)
    activity_adj = activity_adata.obsp['connectivities']
    activity_adj = disconnect_dif_label_nodes(ctype_lab_activity,activity_adj)

    embedding_st, embedding_sc = train(rna_dim_input=319, activity_dim_input=378, dim_input=624,hidden_channels=256,dim_output=50,activity_adata=activity_adata,st_adata=st_adata,graph_neigh=graph_neigh,spatial_graph=sadj,sc_adata=sc_adata,snn=fadj,activity_adj=activity_adj,ctype_lab_rna=ctype_lab_rna,ctype_lab_activity=ctype_lab_activity,label_CSL=label_CSL,epochs=200,beta=0.1,theta=1,alpha=1)
    
    
