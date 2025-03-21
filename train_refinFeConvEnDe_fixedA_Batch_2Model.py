##########################################################################################################
# Training script for Variational Graph Autoencoder (VGAE) network 
# Purpose: Nodal Embedding Extraction and Establishing Vertex Correspondences
# Author: Soodeh Kalaie
# Date: October 2022
##########################################################################################################

import torch
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
from src import VarSizDataset, VTKObject, Create_unstructuredGrid
from src import GraphVAE_FeConvEnDe_fixedA, Smoothing_Block
from src import parameter_parser
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
### Libraries =================================================================
import torch
import torch.utils
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
import itertools
from tvtk.api import tvtk, write_data  ## conda install mayaviimport os
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import gc
# import deepspeed
import copy
from torch_scatter import scatter
from torch_scatter import scatter_add
from torch.nn import Parameter
import pytorch3d
print('pytorch3d version:', pytorch3d.__version__)
# from pytorch3d.utils import ico_sphere
# from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
# from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


# from numpy.core.umath_tests import inner1d
# def HausdorffDist(A, B):
#    # A: Source
#    # B: Target
#
#    # Hausdorf Distance: Compute the Hausdorff distance between two point
#    # clouds.
#    # Let A and B be subsets of metric space (Z,dZ),
#    # The Hausdorff distance between A and B, denoted by dH(A,B),
#    # is defined by:
#    # dH(A,B) = max(h(A,B),h(B,A)),
#    # where h(A,B) = max(min(d(a,b))
#    # and d(a,b) is a L2 norm
#    # dist_H = hausdorff(A,B)
#    # A: First point sets (MxN, with M observations in N dimension)
#    # B: Second point sets (MxN, with M observations in N dimension)
#    # ** A and B may have different number of rows, but must have the same
#    # number of columns.
#    #
#    # Edward DongBo Cui; Stanford University; 06/17/2014
#
#    # Find pairwise distance
#    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
#    # Find DH
#    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
#    return (dH), D_mat, np.min(D_mat, axis=0)

# from .src import Create_unstructuredGrid, Create_pointcloud
# from src import VGAEModel
# from src import GraphVAE, EncoderBlock, DecoderBlock, GraphVAE_FeConv, GraphVAE_FeConvEnDe


torch.cuda.empty_cache()
gc.collect()


# ###############################################
def Normalize(data, mean, std):
    Normdata = (data - mean) / std
    return Normdata


# ###############################################
## Computing Degree matrix in a Graph
def Degree(edge_index, edge_weight=None, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        # print(edge_weight)
    row, col = edge_index[0], edge_index[1]
    # print(row)
    # print(col)
    # print(edge_weight)
    return scatter_add(edge_weight, row, dim=0)


############################################################

class InnerProduct(torch.nn.Module):
    def __init__(self, requires_grad: bool = True):
        super(InnerProduct, self).__init__()

        self.requires_grad = requires_grad
        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))
        # self.register_buffer('beta', torch.Tensor(1000))

        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            # self.beta.data.fill_(1)
            self.beta.data.fill_(args.beta)

    def forward(self, z, G):
        torch.cuda.empty_cache()

        # sim = torch.zeros(z.shape[0],G.shape[0])
        # for i in range(z.shape[0]):
        #   # print(i)
        #   sim[i] = F.cosine_similarity(G, z[i].reshape((1,len(z[i]))))
        #   # print(sim[i])
        # print(sim)
        # print(z.norm(dim=1, keepdim=True).shape)
        # z /= z.norm(dim=1, keepdim=True)

        z = z / z.norm(dim=1, keepdim=True)
        # print('z max',z.max())
        # print('z min',z.min())
        # print('z',z.shape)
        # print(G.norm(dim=1, keepdim=True).shape)
        # G /= G.norm(dim=1, keepdim=True)  ### As the error states, you’ve modified a variable by an inplace operation by using x += ....
        #### If you want to sum the conv output to x, you should write out the operation as:

        G = G / G.norm(dim=1, keepdim=True)
        # print('G',G.shape)
        # print('G max',G.max())
        # print('G min',G.min())
        # z_norm = F.normalize(z, p=2., dim=-1)
        # G_norm = F.normalize(G, p=2., dim=-1)

        if torch.isnan(G) == 'True':
            print('input has NaN:', torch.isnan(G))

        sim = torch.matmul(z, G.t())
        # print('sim  ', sim)
        # print('sim', sim.shape)
        # print('norm sim max', sim.max())
        # print('norm sim min', sim.min())
        # print('sim:',sim)
        # sim = (torch.matmul(z_norm, G_norm.t()))
        # print('sim:',sim)
        # print(sim.shape)
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        # sim = cos(z, G)
        # m = nn.Softmax(dim=1) ##along with dimension M
        # print('sim', sim)
        # print(sim.shape)

        # e = F.leaky_relu(self.beta * sim, negative_slope=0.2)
        # attention = F.softmax(e, dim=1)

        # attention = F.softmax(e, dim=1)

        # print('beta:',self.beta)

        # print('std:',torch.std(sim))
        # self.beta = nn.Parameter(1/torch.std(sim))

        # print('beta:',self.beta)
        # attention = F.gumbel_softmax(self.beta *sim, tau=1, dim=1)
        self.attention = F.softmax(self.beta * sim, dim=0)
        # print('sum atten',torch.sum(self.attention,dim=0))

        #        import matplotlib.pyplot as plt

        #        plt.plot(attention.cpu().detach().numpy(),'o')
        #        plt.savefig(args.Resultdir + "att.png")

        # a=m(sim)
        # print('atten', attention)
        # atten= atten.to(device=device)
        # print('aG', torch.matmul(attention, G))
        # print(sparse.csr_matrix(attention.detach().numpy()))

        # return sparse.csr_matrix(attention.detach().numpy())
        return self.attention.to_sparse()


######################################## Train/ Validation Phase #########################################################################
# Main function for training
def init_train(args, modelName, modelVAE, modelSmooth, dataT, optimizer, criterion, lr_scheduler, train_loader,
               val_loader):
               
               
               
#    print('==============Continue training model ================')
#
#    weight_prefix_VAE = os.path.join(args.Modeldir, 'RefineDeep5LayerVGAE_GeoNorm_FeConvEnDe_Batch1_fixedA_lr_sch_liver_L1_LNorm_KL_chamf_epoch2000.pt') 
#    checkpoint_VAE = torch.load(weight_prefix_VAE)
#    modelVAE.load_state_dict(checkpoint_VAE['model_state_dict'])
##    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
##    epoch = checkpoint['epoch']
##    loss = checkpoint['loss']
#
#
#    weight_prefix_Sm = os.path.join(args.Modeldir, 'SmoothRefineDeep5LayerVGAE_GeoNorm_FeConvEnDe_Batch1_fixedA_lr_sch_liver_L1_LNorm_KL_chamf_epoch2000.pt') 
#    checkpoint_Sm = torch.load(weight_prefix_Sm)
#    modelSmooth.load_state_dict(checkpoint_Sm['model_state_dict'])
#    
#    ########################################################
    
    
    modelVAE.to(device)
    modelSmooth.to(device)
    N_batch_train = len(train_loader)
    N_batch_val = len(val_loader)

    losstrain_collection = []
    lossval_collection = []
    writer = SummaryWriter()
    ########################################################
    #### Mu data ======
    ########################################################
    # Initialize tensors
    rowT = np.array([x[0] for x in dataT.edges], dtype=np.double)
    colT = np.array([x[1] for x in dataT.edges], dtype=np.double)

    edge_indexT = torch.tensor(np.array([rowT, colT]), dtype=torch.long)
    edge_indexT = edge_indexT.to(device=device)

    Min_cor = np.min(dataT.points, axis=0)
    dataT.points -= Min_cor

    Max_cor = np.max(dataT.points, axis=0)
    dataT.points /= Max_cor

    print('normal mu :',  torch.tensor(dataT.normals, dtype=torch.float))
    
    x_inT = torch.cat((torch.tensor(dataT.points, dtype=torch.float), torch.tensor(dataT.normals, dtype=torch.float)),1)
    x_inT = x_inT.to(device=device)

    faces1 = dataT.triangles
    facesT = np.int32(faces1)
    
    x_outT, z_T, mu_zT, logvar_zT = modelVAE(x_inT, edge_indexT)

    mesh_in = tvtk.PolyData(points=x_inT[:, 0:3].cpu().detach().numpy(), polys=facesT)
    write_data(mesh_in, os.path.join(args.Resultdir, 'mu_in'+ ".vtk" ))

            #            x_outT *= Max_cor.cpu().detach().numpy()
            #            x_outT += Min_cor.cpu().detach().numpy()
    mesh_out = tvtk.PolyData(points=x_outT[:, 0:3].cpu().detach().numpy(), polys=facesT)
    write_data(mesh_out, os.path.join(args.Resultdir, 'mu_out'+ ".vtk" ))
    
    

    for epoch in range(args.epochs):
        start_time = time.time()
        ### Train =================================================================
        # Init trainining phase
        modelVAE.train()
        modelSmooth.train()

        epoch_train_mse_x = 0
        epoch_train_lossnorm = 0
        epoch_train_mse_hp = 0
        epoch_train_klz = 0
        epoch_train_loss = 0

        for ind, (data) in enumerate(train_loader):
            # print(data.name)
            data = data.to(device=device)
            #            x_in = data.x.to(device=device)
            x_in = torch.cat((data.x, data.normals), 1)

            x_out, z, mu_z, logvar_z = modelVAE(x_in, data.edge_index)

            ##            x_in *= data.std.to(device=device)
            ##            x_in += data.mean.to(device=device)
            #            mesh_in = tvtk.PolyData(points=x_in.cpu().detach().numpy(), polys=data.face)
            #            write_data(mesh_in, os.path.join(args.Resultdir,'in_'+data.name[0]))
            #
            ##            x_out *= data.std.to(device=device)
            ##            x_out += data.mean.to(device=device)
            #            mesh_in = tvtk.PolyData(points=x_out.cpu().detach().numpy(), polys=data.face)
            #            write_data(mesh_in, os.path.join(args.Resultdir,'out_'+data.name[0]))
            #### Mu data ==========================================================

            x_outT, z_T, mu_zT, logvar_zT = modelVAE(x_inT, edge_indexT)
            # z_T = z_T.to(device=device)

            #            print(Min_cor)
            #            print(Min_cor[0])

            #            x_in = np.transpose(x_inT[0].cpu().detach().numpy())
            #            x_out = np.transpose(x_outT[0].cpu().detach().numpy())
            #
            #            min_cor = min_cor[0].cpu().detach().numpy()
            #            max_cor = max_cor[0].cpu().detach().numpy()

            #            x_inT *= Max_cor.cpu().detach().numpy()
            #            x_inT += Min_cor.cpu().detach().numpy()
            mesh_in = tvtk.PolyData(points=x_inT[:, 0:3].cpu().detach().numpy(), polys=facesT)
            write_data(mesh_in, os.path.join(args.Resultdir, 'mu_in'+ args.fixedmeshName[0:-4]+ ".vtk"))

            #            x_outT *= Max_cor.cpu().detach().numpy()
            #            x_outT += Min_cor.cpu().detach().numpy()
            mesh_in = tvtk.PolyData(points=x_outT[:, 0:3].cpu().detach().numpy(), polys=facesT)
            write_data(mesh_in, os.path.join(args.Resultdir, 'mu_out' + args.fixedmeshName[0:-4] +".vtk"))
            # # ##############################################
            # print('z_T shape:', z_T.shape)
            # print('x_T shape:', Avgshape.shape)
            # # print('start first epoch  z_T:', z_T)
            # faces = dataT.triangles
            # facesT = np.int32(faces)

            # ###############################################
            # #       Mesh Regression using attention
            # ###############################################
            a = InnerProduct()
            a = a.to(device=device)
            # print(z)
            # print(z_T)
            attention = a(z, z_T)
            # print('x_M max:', x_in)

            h = torch.sparse.mm(attention.t(), x_in[:, 0:3])  ##.cuda()

            # print('x_in shape', x_in.shape)
            # print('h shape', h.shape)

            # Create_unstructuredGrid(h.cpu().detach().numpy(), facesT, args.Resultdir,'h_{}'.format(epoch)+data.name[0])

            #            mesh_in = tvtk.PolyData(points=h.cpu().detach().numpy(), polys=facesT)
            #            write_data(mesh_in, os.path.join(args.Resultdir,'h'))
            # Create_unstructuredGrid(h.cpu().detach().numpy(), facesT, args.Resultdir + 'corresp/','B{}_h_{}'.format(args.beta, names_all[ind]))

            hp = modelSmooth(h, edge_indexT)

            #            mesh_in = tvtk.PolyData(points=hp.cpu().detach().numpy(), polys=facesT)
            #            write_data(mesh_in, os.path.join(args.Resultdir,'hp_{}'.format(epoch)+data.name[0]))

            #############################################################################################################################
            mesh_out = Meshes(verts=[x_out[:, 0:3]], faces=[data.face.T])
            #############################################################################################################################
            loss, mse_loss_x, loss_normal, mse_loss_hp, klz_loss = criterion(x_in, x_out, mesh_out, hp, mu_z, logvar_z)

            # a = list(modelSmooth.parameters())[0].clone()
            # a = list(modelVAE.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            # b = list(modelSmooth.parameters())[0].clone()
            # b = list(modelVAE.parameters())[0].clone()
            # print(torch.equal(a.data, b.data))

            # print('gamma grad:',list(modelSmooth.parameters())[0].grad)

            # print('gamma grad:',list(modelSmooth.parameters())[0])

            #            #mse_loss_hp.retain_grad()
            #            loss.backward()
            ##            print('loss x grad', mse_loss_x.grad)
            ##            print('loss hp grad', mse_loss_hp.grad)
            ##            print('loss  grad', loss.grad)
            #            #print(' hppp',hp.grad_fn)
            #            #print('gamma : ', model.gamma.grad.data)
            #            print('encoder.gc1.c : ', model.encoder.gc1.c.grad.data)
            #            optimizer.step()

            torch.nn.utils.clip_grad_norm_(modelVAE.parameters(), 10)

            epoch_train_mse_x += mse_loss_x.item()
            epoch_train_lossnorm += loss_normal.item()
            epoch_train_mse_hp += mse_loss_hp.item()
            epoch_train_klz += klz_loss.item()
            epoch_train_loss += loss.item()

        epoch_train_mse_x /= N_batch_train
        epoch_train_lossnorm /= N_batch_train
        epoch_train_mse_hp /= N_batch_train
        epoch_train_klz /= N_batch_train
        epoch_train_loss /= N_batch_train

        losstrain_collection.append(epoch_train_loss)

        writer = SummaryWriter()

        writer.add_scalar("epoch_train_mse_x", epoch_train_mse_x, epoch)
        writer.add_scalar("epoch_train_lossnorm", epoch_train_lossnorm, epoch)
        writer.add_scalar("epoch_train_mse_hp", epoch_train_mse_hp, epoch)
        writer.add_scalar("epoch_train_klz", epoch_train_klz, epoch)
        writer.add_scalar("epoch_train_loss", epoch_train_loss, epoch)

        ### Val =================================================================
        # Init validation phase
        modelVAE.eval()
        modelSmooth.eval()
        with torch.no_grad():
            epoch_val_mse_x = 0
            epoch_val_lossnorm = 0
            epoch_val_mse_hp = 0
            epoch_val_klz = 0
            epoch_val_loss = 0

            for ind, (data) in enumerate(val_loader):
                data = data.to(device=device)
                x_in = torch.cat((data.x, data.normals), 1)
                # print( 'X',x_in)
                # print( 'Edge',edge_index)
                # print( 'face',data.face)
                x_out, z, mu_z, logvar_z = modelVAE(x_in, data.edge_index)
                #### Mu data ==========================================================
                x_outT, z_T, mu_zT, logvar_zT = modelVAE(x_inT, edge_indexT)
                # ###############################################
                # #       Mesh Regression using attention
                # ###############################################
                a = InnerProduct()
                a = a.to(device=device)
                attention = a(z, z_T)
                h = torch.sparse.mm(attention.t(), x_in[:, 0:3])  ##.cuda()
                # Create_unstructuredGrid(h.cpu().detach().numpy(), facesT, args.Resultdir + 'corresp/','B{}_h_{}'.format(args.beta, names_all[ind]))

                hp = modelSmooth(h, edge_indexT)

                #############################################################################################################################
                mesh_out = Meshes(verts=[x_out[:, 0:3]], faces=[data.face.T])
                # print('mesh_out:', mesh_out)
                #############################################################################################################################

                loss, mse_loss_x, loss_norm, mse_loss_hp, klz_loss = criterion(x_in, x_out, mesh_out, hp, mu_z,
                                                                               logvar_z)

                epoch_val_mse_x += mse_loss_x.item()
                epoch_val_lossnorm += loss_norm.item()
                epoch_val_mse_hp += mse_loss_hp.item()
                epoch_val_klz += klz_loss.item()
                epoch_val_loss += loss.item()

            epoch_val_mse_x /= N_batch_val
            epoch_val_lossnorm /= N_batch_val
            epoch_val_mse_hp /= N_batch_val
            epoch_val_klz /= N_batch_val
            epoch_val_loss /= N_batch_val

            lossval_collection.append(epoch_val_loss)

            writer.add_scalar("epoch_val_mse_x", epoch_val_mse_x, epoch)
            writer.add_scalar("epoch_val_lossnorm", epoch_val_lossnorm, epoch)
            writer.add_scalar("epoch_val_mse_hp", epoch_val_mse_hp, epoch)
            writer.add_scalar("epoch_val_klz", epoch_val_klz, epoch)
            writer.add_scalar("epoch_val_loss", epoch_val_loss, epoch)

        end_time = time.time()
        ## lr update ==============================================================
        current_lr = lr_scheduler.optimizer.param_groups[0]['lr']

        if epoch % 50 == 0:  ###can give an output every 50 epochs

            ### ===========================================================================
            ### Save weight ===============================================================
            save_checkpoint(args, modelVAE, modelName, optimizer, loss)
            save_checkpoint(args, modelSmooth, 'Smooth' + modelName, optimizer, loss)

            ### write as vtk ==========================================================

            print(data.name[0].upper())
            print('Loss L2:', mse_loss_x)
            print('Loss Chamfer:', mse_loss_hp)

            input = x_in[:, 0:3]
            input *= data.std
            input += data.mean

            pred = x_out[:, 0:3]
            pred *= data.std
            pred += data.mean

            mesh_in = tvtk.PolyData(points=input.cpu().detach().numpy(), polys=data.face.cpu().detach().numpy())
            write_data(mesh_in, os.path.join(args.Resultdir, 'in_' + data.name[0]+ ".vtk"))

            mesh_in = tvtk.PolyData(points=pred.cpu().detach().numpy(), polys=data.face.cpu().detach().numpy())
            write_data(mesh_in, os.path.join(args.Resultdir, 'out_' + data.name[0]+ ".vtk"))

            mesh_in = tvtk.PolyData(points=h.cpu().detach().numpy(), polys=facesT)
            write_data(mesh_in, os.path.join(args.Resultdir, 'h_{}'.format(data.name[0])+ ".vtk"))

            mesh_in = tvtk.PolyData(points=hp.cpu().detach().numpy(), polys=facesT)
            write_data(mesh_in, os.path.join(args.Resultdir, 'hp_{}'.format(data.name[0])+ ".vtk" ))

        print(
            "Epoch: {} \t Mse_loss_x: {:.7f} \t loss_normal: {:.7f} \t Mse_loss_hp: {:.7f} \t Klz_loss: {:.7f} \t Total Loss: {:.7f} \t lr_rate: {:.5f} \t Time: {:.2f}".format(
                epoch, epoch_train_mse_x, epoch_train_lossnorm, epoch_train_mse_hp, epoch_train_klz, epoch_train_loss,
                current_lr, end_time - start_time))

        #        import pandas as pd
        #
        #        df = pd.DataFrame(epoch_train_loss)
        #        df.to_csv(args.figdir + 'Loss train_Deep5-4LayerVGAE_FeConvEnDe_Batch_fixedA_lr_sch_Liver_L1_2e-3_KL_{}epochs.csv'.format(args.epochs), index=False)

        plot1 = plt.figure(1)
        plt.plot(lossval_collection, '-b', linewidth=1,
                 label='Validation' if epoch == 0 else "")  # linewidth=1 (default)
        plt.plot(losstrain_collection, '-r', linewidth=1, label='Train' if epoch == 0 else "")
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.xlabel("Epochs")
        plt.xticks(range(0, args.epochs, 50))
        plt.ylabel("Loss")
        plt.legend(loc='upper right', numpoints=1, frameon=False)
        title = 'Bathc_Sch_lr_4KLV_(L2+KL)'
        plt.title(title)
        # save image
        plt.savefig(args.figdir + title + "_VGAEfixedA_{}epochs.png".format(args.epochs))

        #        import pandas as pd
        #        df = pd.DataFrame(losstrain_collection)
        #        df.to_csv(args.figdir + modelName+'losstrain_collection(L2+KL).csv', index=False)

        lr_scheduler.step()

    writer.flush()
    writer.close()

    ### ===========================================================================
    ### Save weight ===============================================================

    save_checkpoint(args, modelVAE, modelName, optimizer, loss)
    save_checkpoint(args, modelSmooth, 'Smooth' + modelName, optimizer, loss)
    #    weight_prefix = os.path.join(args.Modeldir, modelName)
    ##    torch.save(model.state_dict(), weight_prefix)
    #    torch.save({
    #            'epoch': args.epochs,
    #            'model_state_dict': model.state_dict(),
    #            'optimizer_state_dict': optimizer.state_dict(),
    #            'loss': loss,
    #            }, weight_prefix)

    print('==============')
    print('Well Done!')
    print('==============')

    return


def save_checkpoint(args, model, modelName, optimizer, loss):
    weight_prefix = os.path.join(args.Modeldir, modelName)
    #    torch.save(model.state_dict(), weight_prefix)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, weight_prefix)


##############################################  Test Phase  #########################################################################
def test_main(args, modelName, modelVAE, modelSmooth, dataT, testLoader):
    # ###############################################
    # ### Mu data
    # ###############################################
    # Initialize tensors

    rowT = np.array([x[0] for x in dataT.edges], dtype=np.double)
    colT = np.array([x[1] for x in dataT.edges], dtype=np.double)
    # print(torch.tensor(row))
    # print(torch.tensor(col))
    edge_indexT = torch.tensor(np.array([rowT, colT]), dtype=torch.long)
    edge_indexT = edge_indexT.to(device=device)

    Min_cor = np.min(dataT.points, axis=0)
    dataT.points -= Min_cor

    Max_cor = np.max(dataT.points, axis=0)
    dataT.points /= Max_cor

    x_inT = torch.cat((torch.tensor(dataT.points, dtype=torch.float), torch.tensor(dataT.normals, dtype=torch.float)),
                      1)
    x_inT = x_inT.to(device=device)

    faces1 = dataT.triangles
    facesT = np.int32(faces1)
    ###############################################
    modelVAE.load_state_dict(torch.load(os.path.join(args.Modeldir, modelName))['model_state_dict'])
    modelSmooth.load_state_dict(torch.load(os.path.join(args.Modeldir, 'Smooth' + modelName))['model_state_dict'])

    modelVAE.eval()
    modelVAE = modelVAE.to(device)
    modelSmooth.eval()
    modelSmooth = modelSmooth.to(device)

    with torch.no_grad():
        for ind, (data) in enumerate(testLoader):
            data = data.to(device=device)
            x_in = torch.cat((data.x, data.normals), 1)
            
            

            # x_in = data.x.to(device=device)
            # mean = data.mean.to(device=device)
            # std = data.std.to(device=device)
            # edge_index = data.edge_index.to(device=device)

            x_out, z, mu_z, logvar_z = modelVAE(x_in, data.edge_index)
            #### Mu data ==========================================================
            x_outT, z_T, mu_zT, logvar_zT = modelVAE(x_inT, edge_indexT)
            # ###############################################
            # #       Mesh Regression using attention
            # ###############################################
            a = InnerProduct()
            a = a.to(device=device)
            attention = a(z, z_T)

            input = x_in[:, 0:3]
            input *= data.std
            input += data.mean

            #h = torch.sparse.mm(attention.t(), input)  ##.cuda()
           
            h = torch.sparse.mm(attention.t(), x_in[:, 0:3])  ##.cuda()



            hp = modelSmooth(h, edge_indexT)
            
            ### Denormalize Output ==========================================================
            # input = x_in[:, 0:3]
            # input *= data.std
            # input += data.mean

            pred = x_out[:, 0:3]
            pred *= data.std
            pred += data.mean


            #############################################################################################################################
            mesh_out = Meshes(verts=[x_out[:, 0:3]], faces=[data.face])
            #mesh_out = Meshes(verts=[pred], faces=[data.face])
            # print('mesh_out:', mesh_out)
            #############################################################################################################################
            loss, mse_loss_x, loss_norm, mse_loss_hp, klz_loss = criterion(x_in, x_out, mesh_out, hp, mu_z, logvar_z)


            ### write as vtk ==========================================================
            mesh_in = tvtk.PolyData(points=input.cpu().detach().numpy(), polys=data.face.cpu().detach().numpy())
            mesh_out = tvtk.PolyData(points=pred.cpu().detach().numpy(), polys=data.face.cpu().detach().numpy())
            mesh_h = tvtk.PolyData(points=h.cpu().detach().numpy(), polys=facesT)
            mesh_hp = tvtk.PolyData(points=hp.cpu().detach().numpy(), polys=facesT)

            print(data.name[0].upper())
            print('Loss L2:', mse_loss_x)
            print('Loss chamfer:', mse_loss_hp)

#            write_data(mesh_in, args.Resultdir + "in_" + data.name[0] + ".vtk")
#            write_data(mesh_out, args.Resultdir + "out_" + data.name[0] + ".vtk")
#            write_data(mesh_h, args.Resultdir + "h_" + data.name[0] + ".vtk")
#            write_data(mesh_hp, args.Resultdir + "hp_" + data.name[0] + ".vtk")

            print('==============')
            print('Well Done!')
            print('==============')


if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is :', device)

    args = parameter_parser()
    print('Data Dir:', args.directory)
    print('Result dir:', args.Resultdir)
    print('Model dir:', args.Modeldir)
    print('fixedmeshName:' , args.fixedmeshName)


    lr_rate = 1e-4
    lr_step_size = 500
    lr_gamma = 0.5  # 0.5
    print('lr_rate:' , lr_rate)
    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries', 'first')):
        os.mkdir(os.path.join('summaries', 'first'))

    ### Read data =================================================================

    data = os.listdir(args.directory)
    # data =data[1:10]
    # N_subj = 3
    # data = data[0:10]
    # print(data)
    N_subj = len(data)
    #    datapaths = gather_paths(args.directory ,N_subj )
    #    pointclouds, mean, std = gather_data(datapaths)

    trainRatio = 0.7
    valRatio = 0.2
    testRatio = 0.1
    nBatch = 1
    print("number of Batch: ", nBatch)

    dataset_list = VarSizDataset(args.directory, data)

    train_size = int(0.99 * len(dataset_list))
    test_size = len(dataset_list) - train_size
    trainDataset, testDataset = torch.utils.data.random_split(dataset_list, [train_size, test_size])

    N_batch_train = len(trainDataset)
    N_batch_test = len(testDataset)
    print("number of Training Data: ", N_batch_train)
    print("number of Test Data: ", N_batch_test)

    trainLoader = DataLoader(trainDataset, batch_size=nBatch, num_workers=4)
    # valLoader = DataLoader( valDataset, batch_size = nBatch, num_workers = 0 )
    testLoader = DataLoader(testDataset, batch_size=1, num_workers=4)

    #loss_likelihood = torch.nn.MSELoss()


    loss_likelihood = torch.nn.L1Loss()
    ### ===========================================================================
    ### Loss_function =============================================================
    class Loss_function(nn.Module):
        def __init__(self):
            super(Loss_function, self).__init__()
            self.image_sigma = 1.0
            self.hp_sigma = 1.0
            self.klz_coef = 1
            # self.klz_coef = 2e-6
            # self.klz_coef = 2e-3
            # self.klz_coef = 1e-3
            self.w_normal = 0.01
            print('klz_coef:', self.klz_coef)

        def forward(self, y_true, y_pred, mesh_out, hp_pred, Z_mu, Z_lvar):
            # Mse_loss = 1.0 / (self.image_sigma ** 2) * torch.mean(torch.square(y_true - y_pred))
            # print( 1.0 / (self.image_sigma ** 2) * torch.mean(torch.square(y_true - y_pred)))
            Mse_loss_x = 1.0 / (self.image_sigma ** 2) * loss_likelihood(y_true, y_pred)
            # Mse_loss_hp = 1.0 / (self.hp_sigma ** 2)* loss_likelihood(y_true , hp_pred)
#            print(y_pred.shape)
#            print(y_true[:, 0:3].shape)
#            print(hp_pred.shape)
            loss_chamfer, _ = chamfer_distance(y_true[:, 0:3].unsqueeze(0), hp_pred.unsqueeze(0))
            #            print(y_true.shape)
            #            print(hp_pred.shape)
            #            loss_chamfer, _ = chamfer_distance(y_true, hp_pred)

            print('loss_chamfer:',loss_chamfer)
            # dh, D_mat, md = HausdorffDist(y_true.cpu().detach().numpy().astype('double'), hp_pred.cpu().detach().numpy().astype('double'))
            Mse_loss_hp = 1.0 / (self.hp_sigma ** 2) * loss_chamfer

            loss_normal = mesh_normal_consistency(mesh_out)
            loss_normal *= self.w_normal

            Klz_loss = -0.5 * torch.mean(1 + Z_lvar - Z_mu.pow(2) - Z_lvar.exp())
            Klz_loss *= (1 / y_true.shape[0])
            Klz_loss *= self.klz_coef

            Loss = Mse_loss_x + loss_normal + Mse_loss_hp + Klz_loss
            return Loss, Mse_loss_x, loss_normal, Mse_loss_hp, Klz_loss


    #### Mu data ======
    dataT = VTKObject(filename=os.path.join(args.Tdirectory, args.fixedmeshName))  # 'liv-mesh-0.vtu'
    # Initialize tensors
    rowT = np.array([x[0] for x in dataT.edges], dtype=np.double)
    colT = np.array([x[1] for x in dataT.edges], dtype=np.double)
    edge_indexT = torch.tensor(np.array([rowT, colT]), dtype=torch.long)
    D = Degree(edge_indexT, dtype=torch.float)
    # D = torch.tensor(100)
    D = D.to(device=device)

    print('##################################################################')
    print(' Max number of neighbourhood:', D.max())
    print('##################################################################')
    #####################################################
    # Define Model
    #####################################################


    # model_pretrained = GraphVAE_FeConvEnDe_fixedA(input_feat_dim=3, hidden_dim1=64, hidden_dim2=128, heads=8)
    # model = GraphVAE_FeConvEnDe(input_feat_dim=3, hidden_dim1=16, hidden_dim2=128, heads=8)

    ### ===========================================================================
    modelVAE = GraphVAE_FeConvEnDe_fixedA(input_feat_dim=6, hidden_dim1=64, hidden_dim2=128, heads=8)
    modelSmooth = Smoothing_Block(D, torch.nn.MSELoss())
    ### ===========================================================================

    print('==============load pretrained model for Initilizing weights weights================')
    
    #### Liver ########################################################################
    #modelName_pretrained = 'Deep5Layer_Geo_Normal_VGAE_FeConvEnDe_Batch1_fixedA_lrfixed_LV_L1_1KL_LNo_liver_M8_epoch2000.pt'
    

    #### LV ########################################################################
    
    modelName_pretrained =  'Deep5Layer_Geo_Normal_VGAE_FeConvEnDe_Batch1_fixedA_lrfixed_LV_L1_1KL_LNo_500LV_M8_epoch1000.pt'
    
    path_modelName_pretrained = '/usr/not-backed-up/scsk/Code/LiverVGAlayers-Final/VGA-fixedA-Geo-Norm/saved_models/'
    # print(modelName_pretrained)
    
    weight_prefix = os.path.join(path_modelName_pretrained, modelName_pretrained) 
    checkpoint = torch.load(weight_prefix)
    modelVAE.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    epoch = checkpoint['epoch']
#    loss = checkpoint['loss']





    ### ===========================================================================
    ### Create model ==============================================================
    criterion = Loss_function().to(device)
    optimizer = torch.optim.Adam(list(modelVAE.parameters()) + list(modelSmooth.parameters()), lr=lr_rate, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma, last_epoch=-1)
    writer = SummaryWriter()

    ### print model details =======================================================
    # print(net)
    # print("number of parameters: ", sum([param.numel() for param in model.parameters()]))

    print(modelVAE.__repr__())
    print(modelSmooth.__repr__())
    import prettytable
    from prettytable import PrettyTable


    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


    # count_parameters(model_pretrained)
    # count_parameters(modelVAE)
    # count_parameters(modelSmooth)

    #####################################################
    ### Liver
    #modelName = 'RefineDeep5LayerVGAE_GeoNorm_FeConvEnDe_Batch{}_fixedA_lr_sch_liver_L1_LNorm_KL_chamf_epoch{}.pt'.format(nBatch, args.epochs)
    

    #####################################################
    ### LV
    modelName = 'RefineDeep5LayerVGAE_GeoNorm_FeConvEnDe_Batch{}_fixedA_lr_sch_LV_L1_LNorm_KL_chamf_epoch{}.pt'.format(nBatch, args.epochs)
    
    print('modelName:', modelName)
    
#    modelName = 'RefineDeep5LayerVGAE_GeoNorm_FeConvEnDe_Batch{}_fixedA_lr_sch_liver_L1_LNorm_KL_chamf_epoch{}.pt'.format(nBatch, args.epochs)
#    print('modelName:', modelName)

    #    init_train(args, modelName, model_pretrained, optimizer, criterion,lr_scheduler, trainLoader, testLoader, mean, std)
    #init_train(args, modelName, modelVAE, modelSmooth, dataT, optimizer, criterion, lr_scheduler, trainLoader,testLoader)

    #####################################################
    # Test Model
    #####################################################
    # model_path = os.path.join(args.Modeldir, modelName)
    # print(model_path)
    test_main(args, modelName, modelVAE, modelSmooth, dataT, testLoader)







