import torch
import torch.nn as nn
from src.model_FeConvEnDe_fixedA import EncoderBlock_fixedA, DecoderBlock_fixedA 
#from src import Convert_Adj_EdgeList, convert_Edg_to_face



################################################################
##   Decoding A
################################################################
class GraphVAE_A_GCNEnDe(nn.Module):

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(GraphVAE_A_GCNEnDe, self).__init__()
        self.encoder = EncoderBlock_A(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
        self.decoder = DecoderBlock_A()


    def forward(self, x, edge_index):
        z, mu, logvar = self.encoder(x,edge_index)
        A = self.decoder(z)
        # print('Edge probability:',A)
        A_de = torch.bernoulli(A)
        # print('A decod:',A_de)
        # A_de = self.decoder(x, z)
        # print('Adec:',A_de)
        return z, A_de, mu, logvar

################################################################

class GraphVAE_A_FeConvEnDe(nn.Module):

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, heads):
        super(GraphVAE_A_FeConvEnDe, self).__init__()
        self.encoder = EncoderBlock(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)
        # self.decoder = DecoderBlock_A()
        self.decoder = DecoderBlock_A1(input_feat_dim=input_feat_dim, hidden_dim2=hidden_dim2, heads=heads)


    def forward(self, x, edge_index):
        z, mu, logvar = self.encoder(x,edge_index)
        # A_de = self.decoder(z)
        A_de = self.decoder(x, z)
        # print('Adec:',A_de)
        return z, A_de, mu, logvar

################################################################
##   Decoding V
################################################################
# class GraphVAE_FeConvEnDe(nn.Module):

#     def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, heads):
#         super(GraphVAE_FeConvEnDe, self).__init__()
#         self.encoder = EncoderBlock(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)
#         self.decoder = DecoderBlock(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)


#     def forward(self, x, edge_index):
#         z, mu, logvar = self.encoder(x,edge_index)
#         v, A_de , de_mu, de_logvar = self.decoder(z,edge_index)
#         # print("Final output shape: ", v.shape)
#         return v,z,A_de, mu, logvar


################################################################
##   Decoding A and V
################################################################
class GraphVAE_FeConvEnDe(nn.Module):
    
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, heads):
        super(GraphVAE_FeConvEnDe, self).__init__()
        # self.A_decoder = A_Decoder(n_node=1024)
        print('----Predicting V and A(KNN)----')
        self.encoder = EncoderBlock(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)
        self.decoder = DecoderBlock(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)



    def forward(self, x, edge_index):

        z, mu, logvar = self.encoder(x,edge_index)
        v, A_de , de_mu, de_logvar = self.decoder(z)
        # print("Final output shape: ", v.shape)
        return v,z,A_de, mu, logvar

    # def forward(self, x, edge_index):
    #     z, mu, logvar = self.encoder(x,edge_index)
    #     print(z.cpu().detach())
    #     A_dec = self.A_decoder(z.cpu().detach())
    #     print('A dec:',A_dec.max() )
    #     print('A dec:',A_dec.min() )
    #     edge_index_dec, edgeSet_Dec = Convert_Adj_EdgeList(A_dec)
    #     v, de_mu, de_logvar = self.decoder(z,edge_index_dec)

    #     # v, A_de ,edge_index_dec,  de_mu, de_logvar = self.decoder(z)
    #     # print("Final output shape: ", v.shape)
    #     return v,z , A_dec, edge_index_dec, mu, logvar

################################################################
##   Decoding  V, fixedA
################################################################
class GraphVAE_FeConvEnDe_fixedA(nn.Module):
    
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, heads):
        super(GraphVAE_FeConvEnDe_fixedA, self).__init__()
        # self.A_decoder = A_Decoder(n_node=1024)
        print('----Predicting V ; A(Fixed)----')
        self.encoder = EncoderBlock_fixedA(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)
        self.decoder = DecoderBlock_fixedA(input_feat_dim=input_feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, heads=heads)



    def forward(self, x, edge_index):

        z, mu, logvar = self.encoder(x,edge_index)
        v,  de_mu, de_logvar = self.decoder(z, edge_index)
        # print("Final output shape: ", v.shape)
        return v, z, mu, logvar
