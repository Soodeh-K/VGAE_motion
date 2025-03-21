
import vtk
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch.nn import Parameter
#from src import parameter_parser
#
#args = parameter_parser()


def colormap_mesh(points,faces,err,out_path,name):

    #c_map_post = (err[faces[:,0]] + err[faces[:,1]] + err[faces[:,2]]) / 3
    c_map_post = err
    # c_map_post[c_map_post>1] = 1
    from tvtk.api import tvtk, write_data 
    mesh = tvtk.PolyData(points=points, polys=faces)

    mesh.cell_data.scalars = c_map_post
    mesh.cell_data.scalars.name = 'Model Output'
    write_data(mesh,  os.path.join(out_path, name)+ '.vtk')
    #write_data(mesh, 'output.vtk')
    

def gather_paths(directory, N_subj):
    '''
        Gather all the paths associated to mesh files
    '''
    data = os.listdir(directory)
    data = data[0:N_subj]
    # print(data)
    datapaths = []
    for i in range(len(data)):
        # print(os.path.join(directory, data[i]))
        datapaths.append(os.path.join(directory, data[i]))
    return datapaths

####################################################################################

def gather_data(datapaths):
    vertices = []
    minx = []
    miny = []
    minz = []
    faces = []
    edges = []
    volumes = []

    for i, mesh_filename in enumerate(datapaths):
        # print(i)
        # print(mesh_filename)
        mesh = VTKObject(filename = mesh_filename)  # Load mesh
        
        row = np.array([x[0] for x in mesh.edges], dtype=np.double)
        col = np.array([x[1] for x in mesh.edges], dtype=np.double)
        edge_index = np.array([row, col])
        # print('edge_{}:{} '.format(ind,edge_index))
#        face = mesh.triangles
#        face = np.int32(face)
#        data_volume = np.array(data.volume)
        
        
        vertices.append(mesh.points)
        
#        volumes.append(data_volume)
#        faces.append(face)
#        edges.append(edge_index)
        
        minx.append(min(mesh.points[:, 0]))
        miny.append(min(mesh.points[:, 1]))
        minz.append(min(mesh.points[:, 2]))
    # print(minx)
    # print(miny)
    # print(minz)
    mean = np.array([min(minx), min(miny), min(minz)])
    print( np.array(vertices))
    print(mean)
    moved_FO = np.array(vertices) - mean  ### shifts all the hearts to the first octant in the Cartesian coordinate system where x, y, and z values are all positive.
    std = (moved_FO).max()  ### finds maximum coordinate value among all shifted hearts to the first octant
    
    # return np.array(vertices), mean, std
    return np.array(vertices), mean, std 


####################################################################################
class VTKObject( ) :

    def __init__ (self , filename=None ) :

        self.filename = filename
        self.n_points = None
        self.n_cells = None
        self.points = None
        self.edges = [ ]
        self.triangles = [ ]
        self.neighbors_dict = {}
        self.normals = None


        if filename is not None :
            # self.reader = vtk.vtkPolyDataReader() ### vtkPolyDataReader reads legacy .vtk files
            # self.reader =vtk.vtkXMLPolyDataReader() ###Use vtkXMLPolyDataReader to read .vtp files.
            path , extension = os.path.splitext( self.filename )
            # print('path:',path)
            # print('extension:', extension)
            extension = extension.lower( )
            # print('extension lower:', extension)
            if extension == ".vtp" :
                self.reader = vtk.vtkXMLPolyDataReader( )
            elif extension == ".ply" :
                self.reader = vtk.vtkPLYReader( )
            elif extension == ".obj" :
                self.reader = vtk.vtkOBJReader( )
            elif extension == ".stl" :
                self.reader = vtk.vtkSTLReader( )
            elif extension == ".vtk" :
                self.reader = vtk.vtkPolyDataReader( )
            elif extension == ".vtu" :
                self.reader = vtk.vtkXMLUnstructuredGridReader( )
            elif extension == ".g" :
                self.reader = vtk.vtkBYUReader( )
            else :
                self.reader = None
            self.reader.SetFileName( self.filename )
            self.reader.Update( )

        data = self.reader.GetOutput( )

        ### Data Points ===========================================================
        self.n_points = data. GetNumberOfPoints
        # self.points = np.array([data.GetPoints(i) for i in range(self.n_points)])
        points = data.GetPoints( )
        data_point = points.GetData( )
        self.points = np.array( data_point )

        ### Edge and Faces  ===========================================================
        self.n_cells = data.GetNumberOfCells( )
        for i in range(self.n_cells):
            pts_cell = data.GetCell(i)
            # print('pts_cell',pts_cell)
            # print ('pts_cell edges',pts_cell.GetEdge())
            for j in range(pts_cell.GetNumberOfEdges()):
                self.edges.append(([int(pts_cell.GetEdge(j).GetPointId(i)) for i in (0, 1)]))
        # print(len(edges))
        self.triangles = [ [ int( data.GetCell( j ).GetPointId( i ) ) for i in (0 , 1 , 2) ] for j in range( self.n_cells ) ]
        
        ### Data Normal Points ===========================================================
#'''     
#Default setting for vtk.vtkPolyDataNormals( )
#
#https://github.com/Kitware/VTK/blob/master/Filters/Core/vtkPolyDataNormals.cxx
#
#//-----------------------------------------------------------------------------
#// Construct with feature angle=30, splitting and consistency turned on,
#// flipNormals turned off, and non-manifold traversal turned on.
#vtkPolyDataNormals::vtkPolyDataNormals()
#{
#  this->FeatureAngle = 30.0;
#  this->Splitting = 1;
#  this->Consistency = 1;
#  this->FlipNormals = 0;
#  this->ComputePointNormals = 1;
#  this->ComputeCellNormals = 0;
#  this->NonManifoldTraversal = 1;
#  this->AutoOrientNormals = 0;
#  // some internal data
#  this->NumFlips = 0;
#  this->OutputPointsPrecision = vtkAlgorithm::DEFAULT_PRECISION;
#  this->CosAngle = 0.0;
#}
#'''
        normfilter = vtk.vtkPolyDataNormals( )
        normfilter.SetInputData( data )
        normfilter.ComputePointNormalsOn( )
        #normfilter.ComputeCellNormalsOff
        normfilter.ComputeCellNormalsOn( )
        normfilter.SplittingOff( )
        normfilter.Update( )
        varray = normfilter.GetOutput( ).GetPointData( ).GetNormals( )
        self.normals = np.array(varray)
        
        #print('just point normal:', np.array(varray))
        
#        normfilter = vtk.vtkPolyDataNormals( )
#        normfilter.SetInputData( data )
#        normfilter.SplittingOff( )
#        normfilter.Update( )
#        varray = normfilter.GetOutput( ).GetPointData( ).GetNormals( )
#        
#        print('general normal:', np.array(varray))
        


class VarSizDataset(Dataset):
    def __init__(self, path,dataset):
        self.dataset = dataset
        self.path =path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
    
#    for index in len(self.dataset):

        #ID_sub = self.dataset[index]
        # print(ID_sub )
        mesh = VTKObject( filename = os.path.join( self.path,self.dataset[index] ) )  # Load mesh
        Data_point = mesh.points
        row = np.array([x[0] for x in mesh.edges], dtype=np.double)
        col = np.array([x[1] for x in mesh.edges], dtype=np.double)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        
        #print (edge_index.type())
        
        #print('mesh norm:',mesh.normals)     
        faces = mesh.triangles
        #print(faces)        
        #face1 = np.int32(faces)
        
        #face1 = np.array(mesh.triangles, dtype=np.int32)
        #face1 =  torch.tensor(np.int32(faces))
        
        
        face1 =  torch.tensor(np.array(mesh.triangles).T, dtype=torch.int32)
        #print('face shape', face1.shape)
        Min_cor = np.min(Data_point, axis= 0)
        Data_point -= Min_cor

        Max_cor = np.max(Data_point, axis= 0)
        Data_point /= Max_cor
        #print('data shape', Data_point.shape)
#        normals = mesh.normals        
#        print(normals)
#        print(Data_point)
        
        #Data_point = torch.Tensor(Data_point.T.astype(float))
        Data_point = torch.tensor(Data_point , dtype=torch.float)
        name_kwargs = {'name': self.dataset[index].replace(self.dataset[index][-4 :], "")}
        #print(name_kwargs)
        #print( self.dataset[index])
        #name_kwargs = {'name': self.dataset[index].replace(".vtu", "")}
        #face_kwargs = {'face': face}
        mean_kwargs = {'mean': torch.tensor(Min_cor , dtype=torch.float)}
        std_kwargs = {'std': torch.tensor(Max_cor , dtype=torch.float)}
        normals_kwargs = {'normals': torch.tensor(mesh.normals , dtype=torch.float)}
        
        # print('graph:{}---name:{}---nodes:{}'.format([index],self.dataset[index].upper(), Data_point.shape[0]))
        
#        from tvtk.api import tvtk, write_data       ## conda install mayaviimport os
#        mesh_in = tvtk.PolyData(points=Data_point.cpu().detach().numpy(), polys=face)             
#        write_data(mesh_in, '/usr/not-backed-up/scsk/Code/LiverVGAlayers-Final/VGAE-fixdA/result-faust-M16/' + "in_" + self.dataset[index][0] + ".vtk")
        #print('face insisd',face1) 
#        print('============data loader:====================' )
#        print('data.x shape:', Data_point.shape)
#        print('data.face shape:', face1.shape)   ##[3, num_faces] 
#        print('============================================')   
        '''
        # ================================= DATA HANDLING OF GRAPHS ========================================

        # A single graph in PyTorch Geometric is decribed by torch_geometric.data.Data.
        # This Data object holds all information needed to define an arbitrary graph.
        # There are already some predefined attributes:
        #   data.x - Node feature matrix with shape [num_nodes, num_node_features]
        #   data.edge_index - graph connectivity in COO format with shape [2, num_edges] and type torch.long
        #   data.edge_attr - Edge feature matrix with shape [num_edges, num_edge_features]
        #   data.y - target to train against (may have arbitrary shape)
        #   data.pos - Node position matrix with shape [num_nodes, num_dimensions]

        # None of these attributes is required. In fact, the Data object is not even restricted to these
        # attributes. We can, e.g., extend it by data.face to save the connectivity of triangles from
        # a 3D mesh in a tensor with shape [3, num_faces] and type torch.long.
        ''' 
        data = Data( x=Data_point,edge_index=edge_index, face=face1, **name_kwargs, **mean_kwargs, **std_kwargs, **normals_kwargs)
        #print('new data')
        #data = Data( x=Data_point,edge_index=edge_index, **face_kwargs, **name_kwargs, **mean_kwargs, **std_kwargs, **normals_kwargs)
                
        return data
        
        
class VarSizDataset_atlas(Dataset):
    def __init__(self, path, dataset, pathT, nameT):
        self.dataset = dataset
        self.path = path
        self.nameT = nameT
        self.pathT = pathT

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
    
#    for index in len(self.dataset):

        #ID_sub = self.dataset[index]
        # print(ID_sub )

        ########################################################################
        ### Atlas
        ########################################################################
        meshT = VTKObject( filename =  os.path.join( self.pathT,self.nameT ) )  # Load mesh
        Data_pointT = meshT.points
        rowT = np.array([x[0] for x in meshT.edges], dtype=np.double)
        colT = np.array([x[1] for x in meshT.edges], dtype=np.double)
        edge_indexT = torch.tensor(np.array([rowT, colT]), dtype=torch.long)

        facesT = meshT.triangles
        face1T =  torch.tensor(np.array(meshT.triangles).T, dtype=torch.int32)
        #print('face shape', face1.shape)
        Min_corT = np.min(Data_pointT, axis= 0)
        Data_pointT -= Min_corT

        Max_corT = np.max(Data_pointT, axis= 0)
        Data_pointT /= Max_corT

        Data_pointT = torch.tensor(Data_pointT , dtype=torch.float)
        name_kwargsT = {'name': self.nameT.replace(self.nameT[-4 :], "")}
        mean_kwargsT = {'mean': torch.tensor(Min_corT , dtype=torch.float)}
        std_kwargsT = {'std': torch.tensor(Max_corT , dtype=torch.float)}
        normals_kwargsT = {'normals': torch.tensor(meshT.normals , dtype=torch.float)}
        
        dataT = Data(x=Data_pointT,edge_index=edge_indexT, face=face1T, **name_kwargsT, **mean_kwargsT, **std_kwargsT, **normals_kwargsT)
        
        ########################################################################
        mesh = VTKObject( filename = os.path.join( self.path,self.dataset[index] ) )  # Load mesh
        Data_point = mesh.points
        row = np.array([x[0] for x in mesh.edges], dtype=np.double)
        col = np.array([x[1] for x in mesh.edges], dtype=np.double)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        
        #print (edge_index.type())
        

#        faces = mesh.triangles
#        #print(faces)        
#        face = np.int32(faces)
        
        #face1 = np.array(mesh.triangles, dtype=np.int32)
        #face1 =  torch.tensor(np.int32(faces))
        
        
        face1 =  torch.tensor(np.array(mesh.triangles).T, dtype=torch.int32)
        #print('face shape', face1.shape)
        Min_cor = np.min(Data_point, axis= 0)
        Data_point -= Min_cor

        Max_cor = np.max(Data_point, axis= 0)
        Data_point /= Max_cor
        #print('data shape', Data_point.shape)
#        normals = mesh.normals        
#        print(normals)
#        print(Data_point)
        
        #Data_point = torch.Tensor(Data_point.T.astype(float))
        Data_point = torch.tensor(Data_point , dtype=torch.float)
        name_kwargs = {'name': self.dataset[index].replace(self.dataset[index][-4 :], "")}
        #print(name_kwargs)
        #print( self.dataset[index])
        #name_kwargs = {'name': self.dataset[index].replace(".vtu", "")}
        #face_kwargs = {'face': face}
        mean_kwargs = {'mean': torch.tensor(Min_cor , dtype=torch.float)}
        std_kwargs = {'std': torch.tensor(Max_cor , dtype=torch.float)}
        normals_kwargs = {'normals': torch.tensor(mesh.normals , dtype=torch.float)}
        dataT_kwargs = {'dataT': dataT}
        
        # print('graph:{}---name:{}---nodes:{}'.format([index],self.dataset[index].upper(), Data_point.shape[0]))
        
#        from tvtk.api import tvtk, write_data       ## conda install mayaviimport os
#        mesh_in = tvtk.PolyData(points=Data_point.cpu().detach().numpy(), polys=face)             
#        write_data(mesh_in, '/usr/not-backed-up/scsk/Code/LiverVGAlayers-Final/VGAE-fixdA/result-faust-M16/' + "in_" + self.dataset[index][0] + ".vtk")
        #print('face insisd',face1) 
               
        data = Data( x=Data_point,edge_index=edge_index, face=face1, **name_kwargs, **mean_kwargs, **std_kwargs, **normals_kwargs, **dataT_kwargs)
        #print('new data')
        #data = Data( x=Data_point,edge_index=edge_index, **face_kwargs, **name_kwargs, **mean_kwargs, **std_kwargs, **normals_kwargs)
        

                
        return data ,dataT



class VarSizDataset_multiatlas(Dataset):
    def __init__(self, path, dataset, pathT, nameT):
        self.dataset = dataset
        self.path = path
        self.nameT = nameT
        self.pathT = pathT

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
    
#    for index in len(self.dataset):

        #ID_sub = self.dataset[index]
        # print(ID_sub )

        ########################################################################
        ### Atlas
        ########################################################################
        meshT = VTKObject( filename =  os.path.join( self.pathT,self.nameT ) )  # Load mesh
        Data_pointT = meshT.points
        rowT = np.array([x[0] for x in meshT.edges], dtype=np.double)
        colT = np.array([x[1] for x in meshT.edges], dtype=np.double)
        edge_indexT = torch.tensor(np.array([rowT, colT]), dtype=torch.long)

        facesT = meshT.triangles
        face1T =  torch.tensor(np.array(meshT.triangles).T, dtype=torch.int32)
        #print('face shape', face1.shape)
        Min_corT = np.min(Data_pointT, axis= 0)
        Data_pointT -= Min_corT

        Max_corT = np.max(Data_pointT, axis= 0)
        Data_pointT /= Max_corT

        Data_pointT = torch.tensor(Data_pointT , dtype=torch.float)
        name_kwargsT = {'name': self.nameT.replace(self.nameT[-4 :], "")}
        mean_kwargsT = {'mean': torch.tensor(Min_corT , dtype=torch.float)}
        std_kwargsT = {'std': torch.tensor(Max_corT , dtype=torch.float)}
        normals_kwargsT = {'normals': torch.tensor(meshT.normals , dtype=torch.float)}
        
        dataT = Data(x=Data_pointT,edge_index=edge_indexT, face=face1T, **name_kwargsT, **mean_kwargsT, **std_kwargsT, **normals_kwargsT)
        
        ########################################################################
        mesh = VTKObject( filename = os.path.join( self.path,self.dataset[index] ) )  # Load mesh
        Data_point = mesh.points
        row = np.array([x[0] for x in mesh.edges], dtype=np.double)
        col = np.array([x[1] for x in mesh.edges], dtype=np.double)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        
        #print (edge_index.type())
        

#        faces = mesh.triangles
#        #print(faces)        
#        face = np.int32(faces)
        
        #face1 = np.array(mesh.triangles, dtype=np.int32)
        #face1 =  torch.tensor(np.int32(faces))
        
        
        face1 =  torch.tensor(np.array(mesh.triangles).T, dtype=torch.int32)
        #print('face shape', face1.shape)
        Min_cor = np.min(Data_point, axis= 0)
        Data_point -= Min_cor

        Max_cor = np.max(Data_point, axis= 0)
        Data_point /= Max_cor
        #print('data shape', Data_point.shape)
#        normals = mesh.normals        
#        print(normals)
#        print(Data_point)
        
        #Data_point = torch.Tensor(Data_point.T.astype(float))
        Data_point = torch.tensor(Data_point , dtype=torch.float)
        name_kwargs = {'name': self.dataset[index].replace(self.dataset[index][-4 :], "")}
        #print(name_kwargs)
        #print( self.dataset[index])
        #name_kwargs = {'name': self.dataset[index].replace(".vtu", "")}
        #face_kwargs = {'face': face}
        mean_kwargs = {'mean': torch.tensor(Min_cor , dtype=torch.float)}
        std_kwargs = {'std': torch.tensor(Max_cor , dtype=torch.float)}
        normals_kwargs = {'normals': torch.tensor(mesh.normals , dtype=torch.float)}
        dataT_kwargs = {'dataT': dataT}
        
        # print('graph:{}---name:{}---nodes:{}'.format([index],self.dataset[index].upper(), Data_point.shape[0]))
        
#        from tvtk.api import tvtk, write_data       ## conda install mayaviimport os
#        mesh_in = tvtk.PolyData(points=Data_point.cpu().detach().numpy(), polys=face)             
#        write_data(mesh_in, '/usr/not-backed-up/scsk/Code/LiverVGAlayers-Final/VGAE-fixdA/result-faust-M16/' + "in_" + self.dataset[index][0] + ".vtk")
        #print('face insisd',face1) 
               
        data = Data( x=Data_point,edge_index=edge_index, face=face1, **name_kwargs, **mean_kwargs, **std_kwargs, **normals_kwargs, **dataT_kwargs)
        #print('new data')
        #data = Data( x=Data_point,edge_index=edge_index, **face_kwargs, **name_kwargs, **mean_kwargs, **std_kwargs, **normals_kwargs)
        

                
        return data ,dataT


def Convert_Adj_EdgeList(Adj):
  if torch.is_tensor(Adj):
   Adj=Adj.cpu().detach().numpy()
  #  print ("array A:",Adj)
   num_nodes = Adj.shape[0] + Adj.shape[1]
   num_edge = 0
   edgeSet = set()
   for row in range(Adj.shape[0]):
       for column in range(Adj.shape[1]):
        # print(row)
        # print(column)
        # print(A.item((0, 1)))
          #  if Adj.item(row,column) >= 0.5 and (column,row) not in edgeSet: #get rid of repeat edge >=
          if Adj.item(row,column) == 1 and (column,row) not in edgeSet:
              num_edge += 1
              edgeSet.add((row,column))
  #  print ('edge Set:', edgeSet)
   row=np.array([x[0] for x in edgeSet],dtype=np.double)
   col=np.array([x[1] for x in edgeSet],dtype=np.double)

  #  print(torch.tensor(row))
  #  print(torch.tensor(col))
   edge_index = torch.tensor([row, col], dtype=torch.long)
   return edge_index,edgeSet


from collections import defaultdict

def convert_Edg_to_face(edgeSet):
    edge_lookup = defaultdict(set)
    for a, b in edgeSet:
        edge_lookup[a] |= {b}
        edge_lookup[b] |= {a}
    # print(edge_lookup)
    faces = set()
    for a in range(len(edge_lookup)):
        for b in edge_lookup[a]:
            for c in edge_lookup[a]:
                if b in edge_lookup[c]:
                    faces.add(frozenset([a, b, c]))
    faces = [list(x) for x in faces]  ##list 
    return faces


def clip_by_tensor(t,t_min,t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t=t.float()
        # t_min=t_min.float()
        # t_max=t_max.float()
    
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result




def Create_unstructuredGrid(points,faces,out_path,name):

    from pyevtk.hl import unstructuredGridToVTK
    from pyevtk.vtk import VtkTriangle, VtkQuad

    x = np.ascontiguousarray(points[:, 0])
    y = np.ascontiguousarray(points[:, 1])
    z = np.ascontiguousarray(points[:, 2])

    ctype = np.zeros(len(faces))
    # print('ctype', ctype.shape)
    for i in range(len(faces)):
        ctype[i - 1] = VtkTriangle.tid
    # ctype[0], ctype[1] = VtkTriangle.tid, VtkTriangle.tid
    # ctype[2] = VtkTriangle.tid
    # print('ctype', ctype)

    offset = np.zeros(len(faces))
    for i in range(len(faces)):
        offset[i] = offset[i - 1] + 3.0
    # print((offset))

    '''
           unstructuredGridToVTK: 
           Export unstructured grid and associated data.

                path: name of the file without extension where data should be saved.
                x, y, z: 1D arrays with coordinates of the vertices of cells. It is assumed that each element
                         has diffent number of vertices.
                connectivity: 1D array that defines the vertices associated to each element. 
                              Together with offset define the connectivity or topology of the grid. 
                              It is assumed that vertices in an element are listed consecutively.
                offsets: 1D array with the index of the last vertex of each element in the connectivity array.
                         It should have length nelem, where nelem is the number of cells or elements in the grid.
                cell_types: 1D array with an integer that defines the cell type of each element in the grid.
                            It should have size nelem. This should be assigned from evtk.vtk.VtkXXXX.tid, where XXXX represent
                            the type of cell. Please check the VTK file format specification for allowed cell types.                       
                cellData: Dictionary with variables associated to each line.
                          Keys should be the names of the variable stored in each array.
                          All arrays must have the same number of elements.        
                pointData: Dictionary with variables associated to each vertex.
                           Keys should be the names of the variable stored in each array.
                           All arrays must have the same number of elements.                   
    '''
    # img_dir = os.path.dirname(filename)
    # if len(img_dir) != 0 and not os.path.exists(img_dir):
    #     os.mkdir(img_dir)
    
#    from tvtk.api import tvtk, write_data   ## conda install mayavi
#
#    mesh = tvtk.PolyData(points=points, polys=faces)
#    write_data(mesh,  os.path.join(out_path, name)+ '.vtk')

    unstructuredGridToVTK(os.path.join(out_path, name), x, y, z, connectivity=faces.flatten(), offsets=offset,
                          cell_types=ctype,
                          cellData=None, pointData=None)
    return

def Create_pointcloud(points,out_path,name):
    from pyevtk.hl import pointsToVTK

    x = np.ascontiguousarray(points[:, 0])
    y = np.ascontiguousarray(points[:, 1])
    z = np.ascontiguousarray(points[:, 2])
    pointsToVTK(os.path.join(out_path, name), x, y, z, data=None)
    return
