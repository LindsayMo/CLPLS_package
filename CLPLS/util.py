# -*- coding: utf-8 -*-
import torch
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import gudhi
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import torch.nn.functional as F
import os

#input:A
#ouput:(D^-1/2)*A*(D^-1/2)
def normalize_new(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))                  # 对每一行求和
    r_inv = np.power(rowsum, -0.5).flatten()          #-1/2次方
    r_inv[np.isinf(r_inv)] = 0.                   # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)                    # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)                      #用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx

def plot_fitting_process(pos_loss,class_loss,total_loss,epochs,args):
    plt.subplot(221, title = "pos_loss")
    plt.plot(pos_loss)
    plt.xlabel("Epochs")

    plt.subplot(222, title = "class_loss")
    plt.plot(class_loss)
    plt.xlabel("Epochs")
    
    plt.subplot(223, title = "total_loss")
    plt.plot(total_loss)
    plt.xlabel("Epochs")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    save_path = os.path.join(args.save_path,args.dataset,'figure')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    #plt.savefig(os.path.join(save_path, f'Loss_epochs{epochs}_beta{args.beta}_alpha{args.alpha}_{args.label}_withoutbatchnorm.png'))
    plt.savefig(os.path.join(save_path, f'Loss_epochs{epochs}_beta{args.beta}.png'))
   
def plot_fitting_process_2mlp(pos_loss,class_loss,class_loss_activity,total_loss,epochs,beta,theta,alpha):


    plt.subplot(221, title = "pos_loss")
    plt.plot(pos_loss)
    plt.xlabel("Epochs")
   
    plt.subplot(222, title = "class_loss_rna")
    plt.plot(class_loss)
    plt.xlabel("Epochs")
    plt.subplot(223, title = "class_loss_activity")
    plt.plot(class_loss_activity)
    plt.xlabel("Epochs")
    plt.subplot(224, title = "total_loss")
    plt.plot(total_loss)
    plt.xlabel("Epochs")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    #plt.savefig(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/integrate/ISSAAC_Loss_{epochs}_beta{beta}.png')
    plt.savefig(f'/home/nas2/biod/myy/project/SNPLS_new/result/ISSAAC_renamed_label/combine_new_order/ISSAAC_Loss_{epochs}_beta{beta}_theta{theta}_alpha{alpha}_withoutbatchnorm.png')

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def sparse_mx_to_torch_edge_weight(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_weight = torch.from_numpy(sparse_mx.data)
    return edge_weight

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def graph_alpha(spatial_locs, n_neighbors=10):
        """
        Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
        :type n_neighbors: int, optional, default: 10
        :return: a spatial neighbor graph
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
        estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
        spatial_locs_list = spatial_locs.tolist()
        n_node = len(spatial_locs_list)
        alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
        skeleton = simplex_tree.get_skeleton(1)
        initial_graph = nx.Graph()
        initial_graph.add_nodes_from([i for i in range(n_node)])
        for s in skeleton:
            if len(s[0]) == 2:
                initial_graph.add_edge(s[0][0], s[0][1])
        extended_graph = nx.Graph()
        extended_graph.add_nodes_from(initial_graph)
        extended_graph.add_edges_from(initial_graph.edges)
        # Remove self edges
        for i in range(n_node):
            try:
                extended_graph.remove_edge(i, i)
            except:
                pass
        print(extended_graph)
        #sadj=nx.to_scipy_sparse_matrix(extended_graph, format='coo')
        sadj=nx.to_scipy_sparse_array(extended_graph, format='coo')
        print(sadj.row[:30])
        print(sadj.col)
        # sadj = sparse_mx_to_torch_sparse_tensor(sadj)        
        return sadj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_graph(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return  nfadj, fadj


###参考GraphST
def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL



def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    # 计算距离矩阵
    # 使用 np.linalg.norm 函数计算两点之间的欧几里得距离
    distance_matrix = np.zeros((len(position), len(position)))

    for i in range(len(position)):
        for j in range(len(position)):
            distance_matrix[i, j] = np.linalg.norm(position[i] - position[j])
    # distance_matrix = euclidean_distance(position, position)
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)  

def disconnect_dif_label_nodes(ctype_lab,fadj):
    fadj = fadj.tocoo().astype(np.float32)
    edge_num = len(fadj.col)
    edge_index = []
    for i in range(edge_num):
        point_a = fadj.col[i]
        point_b = fadj.row[i]
        if ctype_lab[point_a] != ctype_lab[point_b]:
            # 找到要删除的边的索引
            edge_index.append(i)
    fadj.row = np.delete(fadj.row, edge_index)
    fadj.col = np.delete(fadj.col, edge_index)
    fadj.data = np.delete(fadj.data, edge_index)
    return fadj


def nomalize_input(X):
    ave_cols = X.mean(axis=0)
    std_cols = X.std(axis=0)
    X_norm = (X-ave_cols)/std_cols
    return X_norm

