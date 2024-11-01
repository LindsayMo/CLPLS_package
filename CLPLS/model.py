import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv

class MLP(nn.Module):
    def __init__(self, nfeat,nhid, nclass):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,v):
        fc1_out = self.relu(self.fc1(v))
        fc2_out = self.fc2(fc1_out)
        output = self.softmax(fc2_out)
        return output
    
class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,hidden2):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden2, cached=False)
        self.norm2 = nn.BatchNorm1d(hidden2)
        self.prelu2 = nn.PReLU(hidden2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
       # x = self.norm1(x)
        x = self.prelu(x)
        x = self.conv2(x, edge_index, edge_weight)
        #x = self.norm2(x)
        x = self.prelu2(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    
class Encoder(Module):
    def __init__(self, in_features, hidden_channels, out_features, graph_neigh):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        
        self.disc = Discriminator(self.out_features)
        self.graph_encoder = GraphEncoder(in_features, hidden_channels, out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
  

    def forward(self, expr, expr_a, edge_list,edge_weight):
        z = self.graph_encoder(expr, edge_list, edge_weight)   
        hiden_emb = z
        
        if expr_a is None:
            return hiden_emb
        z_a = self.graph_encoder(expr_a, edge_list, edge_weight)
    
        
        g = self.read(z, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(z_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, z, z_a)  
        ret_a = self.disc(g_a, z_a, z) 
        
        return hiden_emb, ret, ret_a