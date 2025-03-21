import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Run VGAE for Liver Graphs")
    parser.add_argument('--gamma', type=float, default=50)
    #parser.add_argument("--pytorch", action="store_true")
    parser.add_argument('--hidden1', type=int, default=8, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument("--fixedmeshName", dest = "fixedmeshName", default='1000363_LV.vtk', help='Fixed mesh for registration, default is liv-mesh-9.vtk.') ##LV_MU_e0_g50.vtu 1000363_LV.vtk   'liv-mesh-9.vtk'
    parser.add_argument("--MultifixedmeshName", dest = "MultifixedmeshName",nargs='+', default =['liv-mesh-9.vtk','liv-mesh-3.vtk'])
    parser.add_argument('--batch', type=int, default=1)
#    parser.add_argument('--alpha', type=float, default= 1e-6)
#    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default= 0.001, help= 'Coeff of KL Loss.') #
    parser.add_argument('--beta', type=float, default=10000, help= 'Coeff of Reconstruction Loss.') ###default=10000
    parser.add_argument('--Lambda', type=float, default=0.001, help= 'Coeff of L1/L2 Regularization Loss.')
    parser.add_argument("--epochs",
                        dest = "epochs",
                        type = int,
                        default = 400, #300
	                help = "Number of gradient descent iterations. Default is 300.")

    parser.add_argument("--learning_rate",
                        dest = "learning_rate",
                        type = float,
                        default = 0.001,
	                help = "Gradient descent learning rate. Default is 0.01.")

    # parser.add_argument("--neurons",
    #                     dest = "num_neurons",
    #                     type = int,
    #                     default = 2000, #32
	#                 help = "Number of neurons by hidden layer. Default is 32.")
    #
    # parser.add_argument("--dataset",
    #                     dest ="dataset",
    #                     default = 'liver3_decimated_0.9.vtp', #karate_club
	#                 help = "Name of the dataset. Default is karate_club.")
    parser.add_argument("--figdir",
                        dest ="figdir",
	                      default =  '/usr/not-backed-up/scsk/Code/MIL-refineVGAE-Geo-Norm/figs/',
                        help = "Output figures directory.")
    parser.add_argument("--directory",
                        dest ="directory",
	                      default =   '/usr/not-backed-up/scsk/Data/cpd_liver_mesh_139_dec0.9_0.6/vtk/',
#                        default =   '/usr/not-backed-up/scsk/Data/LV_cpd_vtk_4k/', 
                        #default =   '/usr/not-backed-up/scsk/Data/faust-dec0.9-vtk/',
                  help = "Name of data's container. Default is data.")  ##'/content/drive/My Drive/Data/liver_mesh_simp/'  ####Heart-4k-Registered-cpd/
#    parser.add_argument("--zdirectory",
#                        dest ="zdirectory",
#	                      default =  '/usr/not-backed-up/scsk/Data/Zcloud_Liver_VGAE_5Layer/',  ###livermesh_1024points  ####zcloud  ###Zcloud_Heart_Registered
#                  help = "Name of data's container. Default is data.")  ##'/content/drive/My Drive/Data/liver_mesh_simp/'


    parser.add_argument("--Resultdir",
                        dest ="Resultdir",
                        #default = '/usr/not-backed-up/scsk/Data/Corresp_1KLV_VGAE_Geo_Norm(0.1gamma)/',
                        #default = '/usr/not-backed-up/scsk/Data/Corresp_1KLV_VGAE_Geo_Norm(0.01gamma)/',
                        #default = '/usr/not-backed-up/scsk/Data/Corresp_139liver_Refin-VGAE-Geo-Norm-gamma/',
                        #default = '/usr/not-backed-up/scsk/Data/Corresp_Faust_Refin-VGAE-Geo-Norm-gamma/',
                        #default = '/usr/not-backed-up/scsk/Data/Corresp_139liver_Refin-VGAE-Geo-gamma/',
                        #default = '/usr/not-backed-up/scsk/Code/MIL-refineVGAE-Geo-Norm/results-LV/',
                        #default = '/usr/not-backed-up/scsk/Code/MIL-refineVGAE-Geo-Norm/results-liver/', 
                        default = '/usr/not-backed-up/scsk/Code/MIL-refineVGAE-Geo-Norm/results-liver-mu1/', 
	                help = "Name of data's container. Default is data.") 
                                 
    parser.add_argument("--Tdirectory",
                        dest ="Tdirectory",
	                      #default =  '/usr/not-backed-up/scsk/Data/Target/',  ### 1000215.vtu
                        default =   '/usr/not-backed-up/scsk/Data/cpd_liver_mesh_139_dec0.9_0.6/vtk/',
                  help = "Name of data's container. Default is data.")   
    
    
    parser.add_argument("--Modeldir",
                        dest ="Modeldir",
                        default = '/usr/not-backed-up/scsk/Code/MIL-refineVGAE-Geo-Norm/saved_models/',
                        #default = '/usr/not-backed-up/scsk/Code/MIL-refineVGAE/saved_models/',
	                help = "Name of data's container. Default is data.") ## '/content/drive/MyDrive/code/LiverVGAE/saved_models/'
#    parser.add_argument("--test_size",
#                        dest = "test_size",
#                        type = float,
#                        default = 0.10,
#	                help = "Size of test dataset. Default is 10%.")
    
    return parser.parse_args()



