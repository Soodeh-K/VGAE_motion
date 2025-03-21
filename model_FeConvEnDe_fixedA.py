#########################################################################
# Script for Variational Graph Autoencoder (VGAE) network with 
# Purpose: Nodal Embedding Extraction
# Author: Soodeh Kalaie
# Date: October 2022
#########################################################################

import torch
import numpy as np
import torch.nn as nn
#from scipy.sparse import csr_matrix
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import knn_graph
from torch_geometric.nn import FeaStConv, GCNConv, GATConv
#from torch_geometric.utils import from_scipy_sparse_matrix ,to_scipy_sparse_matrix
#from src import clip_by_tensor,Convert_Adj_EdgeList
#from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_scatter import scatter
from torch_scatter import scatter_add
#from torch_geometric.utils import from_scipy_sparse_matrix ,to_scipy_sparse_matrix

class Smoothing_Block(nn.Module):
    def __init__(self, gamma, D, loss):
        super(Smoothing_Block, self).__init__()
        
        #self.gamma = 0.01*gamma
        #self.gamma = 0.1*gamma
        self.gamma = gamma
        print( 'gamma :',self.gamma)
        self.D = D
        self.loss = loss
        



    def forward(self, h, edge_indexT):
         
        #print('input',h.shape)
        
        #Loss_h = 10000
        counter = 0 
        hp = 0

        #print ('h',h)
        #while Loss_h > 0.01:
        for e in range(2):

#                start_time = time.time()
#                D = D.to(device=device)
                # ###############################################
                # #       hp
                # ###############################################
#                f1 = (h + args.gamma * scatter_add(h[edge_indexT[0]], edge_indexT[1],dim=0))  ## Dense
#                f2 = (torch.ones_like(D) + args.gamma * D)  ##Sparse

            f1 = (h + self.gamma * scatter_add(h[edge_indexT[0]], edge_indexT[1],dim=0))  ## Dense
            f2 = (torch.ones_like(self.D) + self.gamma * self.D)  ##Sparse
                #f2 = f2.to(device=device)
                
            f2_inv = 1.0 / f2  ## inverse of a diagonal matrix

            hp = torch.sparse.mm(torch.diag(f2_inv), f1)
#                hp = hp.to(device=device)
#                hp = Variable(hp, requires_grad=True)
                # print(' hp max:', hp.max())
            #print('hp:', hp)
            # print(' hp max:', hp.max())
            #print('hp:', hp)
            
            Loss_h = self.loss(h, hp)
            #print('loss h:', Loss_h)
            h = torch.clone(hp)
            #print('h:',h)

            

        return hp


#class Smoothing_Block(nn.Module):
#    def __init__(self, D, loss):
#        super(Smoothing_Block, self).__init__()
#        
#        self.D = D
#        self.loss = loss
#        
#         ## Liver
##        print('#######################################')
##        print(' Liver ')
##        print('#######################################')
#        #self.gamma =torch.Tensor(torch.round(1025 / D.max()))  ##Liver
#        
#        ## LV
#        print('#######################################')
#        print(' LV ')
#        print('#######################################')
#        ## LV  refine: 0.001*gamma with one iteration
#        ## LV VGAE-Geo-Norm  : gamma with 2 iteration
#        
#        self.gamma = 10*torch.Tensor(torch.round(1093 / D.max()))  
#        #self.gamma =0.001     ### close to 0 : very sharp edge /  close to 1 : too smooth
#        #self.D = D
#        
#
##        print('#######################################')
##        print(' Gamma ',self.gamma)
##        print('#######################################')
#
#
#    def forward(self, h, edge_indexT):
#        #print('input',h)
#        #Loss_h = 10000
#        counter = 0 
#        hp = 0
#
#        #print ('h',h)
#        #while Loss_h > 0.01:
#        for e in range(2):
#
##                start_time = time.time()
##                D = D.to(device=device)
#                # ###############################################
#                # #       hp
#                # ###############################################
##                f1 = (h + args.gamma * scatter_add(h[edge_indexT[0]], edge_indexT[1],dim=0))  ## Dense
##                f2 = (torch.ones_like(D) + args.gamma * D)  ##Sparse
#
#            f1 = (h + self.gamma * scatter_add(h[edge_indexT[0]], edge_indexT[1],dim=0))  ## Dense
#            f2 = (torch.ones_like(self.D) + self.gamma * self.D)  ##Sparse
#                #f2 = f2.to(device=device)
#                
#            f2_inv = 1.0 / f2  ## inverse of a diagonal matrix
#
#            hp = torch.sparse.mm(torch.diag(f2_inv), f1)
##                hp = hp.to(device=device)
##                hp = Variable(hp, requires_grad=True)
#                # print(' hp max:', hp.max())
#            #print('hp:', hp)
#            # print(' hp max:', hp.max())
#            #print('hp:', hp)
#            
#            Loss_h = self.loss(h, hp)
#            #print('loss h:', Loss_h)
#            h = torch.clone(hp)
#            #print('h:',h)
#
#            
#
#        return hp



class FeaStNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, heads, t_inv=True):
        super(FeaStNet, self).__init__()

        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = FeaStConv(16, 32, heads=heads, t_inv=t_inv)
        self.conv2 = FeaStConv(32, 64, heads=heads, t_inv=t_inv)
        self.conv3 = FeaStConv(64, 128, heads=heads, t_inv=t_inv)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#########################################################################################################################################################
    ################################################################
    ##   EencoderBlock  V
    ################################################################
class EncoderBlock_fixedA(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, heads):
        super(EncoderBlock_fixedA, self).__init__()
        
#        self.register_parameter(name='gamma', param=torch.nn.Parameter(torch.ones(1)))
        #self.gamma =  torch.tensor(1)

        ########################### 3layer #######################
#        self.gc1 = FeaStConv(input_feat_dim, hidden_dim1,heads) # 3-->64
#        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim1)
#        self.gc3 = FeaStConv(hidden_dim1, hidden_dim2,heads) # 64-->128
#        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim2)
#        self.relu = F.relu 
#        ## mean ,var
#        self.gc5 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 128-->128
#        self.gc6 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 128-->128
# 
#
#    def encode(self, x, edge_index):
#        x = self.relu(self.bn1(self.gc1(x, edge_index)))
#        x = self.relu(self.bn3(self.gc3(x, edge_index)))
#
#        return self.gc5(x, edge_index),self.gc6(x, edge_index)

#         ########################### Deep layer Faust Dataset #######################
#        self.fc0 = nn.Linear(input_feat_dim,  hidden_dim1) # 3-->8
#        self.gc1 = FeaStConv( hidden_dim1, hidden_dim1,heads) # 8-->8
#        self.gc2 = FeaStConv(hidden_dim1, 2*hidden_dim1,heads) # 8-->16
#        self.gc3 = FeaStConv(2*hidden_dim1, hidden_dim2,heads) # 16-->32
#        #self.gc3 = FeaStConv(hidden_dim2, 2*hidden_dim2,heads) # 16-->32
#        self.relu = F.relu 
#
#        ## mean ,var
#        self.gc4 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 32-->32
#        self.gc5 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 32-->32
#
#    def encode(self, x, edge_index):
#        x = self.relu(self.fc0(x))
#        x = self.relu(self.gc1(x, edge_index))
#        x = self.relu(self.gc2(x, edge_index))
#        x = self.relu(self.gc3(x, edge_index))
#        
#
#        return self.gc4(x, edge_index),self.gc5(x, edge_index)
        
        ########################### Deep 5 layer #######################
        self.gc1 = FeaStConv(input_feat_dim, hidden_dim1,heads) # 3-->64
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim1)
        self.gc2 = FeaStConv(hidden_dim1, hidden_dim1,heads) # 64-->64
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim1)
        self.gc3 = FeaStConv(hidden_dim1, hidden_dim2,heads) # 64-->128
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim2)
        self.gc4 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 128-->128
        self.bn4 = nn.BatchNorm1d(num_features=hidden_dim2)
        self.relu = F.relu 
        ## mean ,var
        self.gc5 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 128-->128
        self.gc6 = FeaStConv(hidden_dim2, hidden_dim2,heads) # 128-->128
        
        
        print('new model w 1 :',  self.gc1.c )

    def encode(self, x, edge_index):
        x = self.relu(self.bn1(self.gc1(x, edge_index)))
        x = self.relu(self.bn2(self.gc2(x, edge_index)))
        x = self.relu(self.bn3(self.gc3(x, edge_index)))
        x = self.relu(self.bn4(self.gc4(x, edge_index)))

        return self.gc5(x, edge_index),self.gc6(x, edge_index)




###############################################################################################

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)

            # eps * std + mu
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        # print('mu shape:',mu.shape)
        # print('logvar shape:', logvar.shape)
        # print(mu.max())
        # print(logvar.type())
        z = self.reparameterize(mu, logvar)
        # print('z 1 shape:', z.type())
        # print('z shape:', z.shape)
        return z, mu, logvar

    ################################################################
    ##   DecoderBlock  V , A
    ################################################################
class DecoderBlock_fixedA(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, heads):
        super(DecoderBlock_fixedA, self).__init__()

        ########################### 3 layer #######################)
#        self.gc8 = FeaStConv(hidden_dim2, hidden_dim1,heads)  #128-->64
#        self.bn8 = nn.BatchNorm1d(num_features=hidden_dim1)
#        self.gc10 = FeaStConv(hidden_dim1, input_feat_dim,heads) #64-->3
#        self.bn10 = nn.BatchNorm1d(num_features=input_feat_dim)
#        self.relu = F.relu 
#        ## mean ,var
#        self.gc11 = FeaStConv(input_feat_dim, input_feat_dim,heads) #3-->3
#        self.gc12 = FeaStConv(input_feat_dim, input_feat_dim,heads) #3-->3
#
#
#    def decode(self,  z, edge_index):
#        z = self.relu(self.bn8(self.gc8(z, edge_index)))
#        z = self.relu(self.bn10(self.gc10(z, edge_index)))
#
#        return self.gc11(z, edge_index), self.gc12(z, edge_index)



#        ########################### Deep layer Faust #######################
#        self.gc6 = FeaStConv(hidden_dim2, 2*hidden_dim1,heads)  #32-->16
#        self.gc7 = FeaStConv(2*hidden_dim1, hidden_dim1,heads)  #16-->8
#        self.gc8 = FeaStConv(hidden_dim1, hidden_dim1,heads) #8-->8
#
#        self.relu = F.relu 
#        ## mean ,var
#        self.fc1 = nn.Linear(hidden_dim1, input_feat_dim) 
#        self.fc2 =  nn.Linear(hidden_dim1, input_feat_dim)  #8-->3
#
#
#    def decode(self,  z, edge_index):
#        z = self.relu(self.gc6(z, edge_index))
#        z = self.relu(self.gc7(z, edge_index))
#        z = self.relu(self.gc8(z, edge_index))
#
#        return self.fc1(z), self.fc2(z)

        ########################### Deep 5 layer #######################
        self.gc7 = FeaStConv(hidden_dim2, hidden_dim2,heads)  #128-->128
        self.bn7 = nn.BatchNorm1d(num_features=hidden_dim2)
        self.gc8 = FeaStConv(hidden_dim2, hidden_dim1,heads)  #128-->64
        self.bn8 = nn.BatchNorm1d(num_features=hidden_dim1)
        self.gc9 = FeaStConv(hidden_dim1, hidden_dim1,heads) #64-->64
        self.bn9 = nn.BatchNorm1d(num_features=hidden_dim1)
        self.gc10 = FeaStConv(hidden_dim1, input_feat_dim,heads) #64-->3
        self.bn10 = nn.BatchNorm1d(num_features=input_feat_dim)
        self.relu = F.relu 
        ## mean ,var
        self.gc11 = FeaStConv(input_feat_dim, input_feat_dim,heads) #3-->3
        self.gc12 = FeaStConv(input_feat_dim, input_feat_dim,heads) #3-->3


    def decode(self,  z, edge_index):
        z = self.relu(self.bn7(self.gc7(z, edge_index)))
        z = self.relu(self.bn8(self.gc8(z, edge_index)))
        z = self.relu(self.bn9(self.gc9(z, edge_index)))
        z = self.relu(self.bn10(self.gc10(z, edge_index)))

        return self.gc11(z, edge_index), self.gc12(z, edge_index)


    ################################################################
    ##   Decoding  V
    ################################################################
    def forward(self, z,edge_index):

        de_mu, de_logvar = self.decode(z, edge_index)
        # print('de_mu',de_mu.shape)
        # print('de_logvar',de_logvar.shape)
        v = self.reparameterize(de_mu, de_logvar)
        # return z,A,mu, logvar,v,de_mu, de_logvar
        # print('v',v.shape)
        return v, de_mu, de_logvar




    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)

            # eps * std + mu
            return eps.mul(std).add_(mu)
        else:
            return mu