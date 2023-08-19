import os,time,sys
from typing import Optional, Union, Callable, List
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn
import torch_geometric.transforms as transforms
from torch_geometric.loader import DataLoader
from models import GNN_TopK, EdgeAggregation

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_mnist_dataset(path_to_train, path_to_valid, path_to_ei, path_to_pos, device):
    
    # Load data
    mnist_train_dataset = np.load(path_to_train)
    mnist_train_5_6 = mnist_train_dataset['data']
    labels_train_5_6 = mnist_train_dataset['labels'] 

    mnist_test_dataset = np.load(path_to_valid)
    mnist_test_5_6 = mnist_test_dataset['data']
    labels_test_5_6 = mnist_test_dataset['labels']

    # Load edge index (adjacency)
    edge_index = np.loadtxt(path_to_ei, dtype=np.compat.long).T

    # Load node positions
    pos = np.loadtxt(path_to_pos, dtype=np.float32)

    # Get number of cells and number of edges
    N_cells = mnist_train_5_6.shape[0]
    N_edges = edge_index.shape[1]

    N_train = mnist_train_5_6.shape[1]
    N_test = mnist_test_5_6.shape[1]

    # Get edge attributes 
    data_ref = Data( pos = torch.tensor(pos), edge_index = torch.tensor(edge_index) )
    cart = transforms.Cartesian(norm=False, max_value = None, cat = False)
    dist = transforms.Distance(norm = False, max_value = None, cat = True)
    cart(data_ref) # adds cartesian/component-wise distance
    dist(data_ref) # adds euclidean distance
    edge_attr = np.array(data_ref.edge_attr)

    # Standardize edge attr
    eps = 1e-10
    edge_attr_mean = edge_attr.mean()
    edge_attr_std = edge_attr.std()
    edge_attr = (edge_attr - edge_attr_mean)/(edge_attr_std + eps)

    # Convert to torch tensor
    mnist_train_5_6 = torch.tensor(mnist_train_5_6)
    mnist_test_5_6 = torch.tensor(mnist_test_5_6)
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    pos = torch.tensor(pos)

    # Coalesce
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # create pygeom datasets 
    data_train_list = []
    print('Gathering PyGeom training data (%d graphs)...' %(N_train))
    for i in range(N_train):
        # print('Train %d/%d' %(i+1, N_train))
        data_temp = Data(   x = mnist_train_5_6[:,i].reshape(-1,1),
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos = pos,
                            y = labels_train_5_6)

        data_temp = data_temp.to(device)

        data_train_list.append(data_temp)

    data_test_list = []
    print('Gathering PyGeom testing data (%d graphs)...' %(N_test))
    for i in range(N_test):
        # print('Test %d/%d' %(i+1, N_test))
        data_temp = Data(   x = mnist_test_5_6[:,i].reshape(-1,1),
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos = pos,
                            y = labels_test_5_6)
        data_temp = data_temp.to(device)
        data_test_list.append(data_temp)

    return data_train_list, data_test_list


def test(model, loader, criterion):
    model.eval()
    mse_total = 0.0 
    count = 0 
    with torch.no_grad():
        for data in loader: 
            out,  _ = model(data.x, data.edge_index, data.edge_attr, data.pos, batch=data.batch, return_mask=False)
            mse = criterion(out, data.x)
            mse_total += mse.detach().item() 
            count += 1
        mse_total = mse_total / count 
    return mse_total

def train(N_epochs, model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=3, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)
    criterion = torch.nn.MSELoss()

    train_hist = np.zeros(N_epochs)
    test_hist = np.zeros(N_epochs)
    update_iter = 0

    for epoch in range(N_epochs):
        time_epoch = time.time()
        model.train()
        batch_count = 0 
        train_mse = 0 

        for step, data in enumerate(train_loader):
            out, _ = model(data.x, data.edge_index, data.edge_attr, data.pos, batch=data.batch, return_mask=False)
            loss = criterion(out, data.x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_mse += loss.item()
            batch_count += 1
            update_iter += 1 
            if (step%50 == 0): 
                print('Batches complete: %d' %(step+50))


        train_mse = train_mse/batch_count
        
        test_mse = test(model, test_loader, criterion)

        time_epoch = time.time() - time_epoch 

        print(f'Epoch: {epoch:04d},\tTrain mse: {train_mse:.8f},\tTest mse: {test_mse:.8f},\tTime: {time_epoch:.8f}s')

        # Accumulate accuracies 
        train_hist[epoch] = train_mse
        test_hist[epoch] = test_mse

        # Step scheduler
        scheduler.step(test_mse)

    # Save model:
    model.to('cpu')
    model_savepath = 'model.tar'
    save_dict = {   'state_dict' : model.state_dict(), 
                    'input_dict' : model.input_dict(), 
                    'train_loss' : train_hist, 
                    'test_loss': test_hist
                } 
    torch.save(save_dict, model_savepath) 



if __name__ == '__main__': 
    
    # ~~~~ Training inputs:
    batch_size = 8
    epochs = 10
    device = 'cuda'

    # ~~~~ Create dataloader 
    path_to_train = 'dataset/unstructured_mnist_train_56.npz'
    path_to_valid = 'dataset/unstructured_mnist_valid_56.npz'
    path_to_ei = 'dataset/edge_index.txt'
    path_to_pos = 'dataset/pos.txt'

    train_dataset, valid_dataset = get_mnist_dataset(path_to_train, path_to_valid, path_to_ei, path_to_pos, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False) 

    # ~~~~ Create model
    # Model inputs for Model 2 (decoder coarsening only) used in Sec. 4.1:
    sample = train_dataset[0]

    in_channels_node = sample.x.shape[1]                # num input node channels
    in_channels_edge = sample.edge_attr.shape[1]        # num input edge channels
    hidden_channels = 32                                # num hidden node/edge channels 
    out_channels = in_channels_node                     # num output node channels
    n_mlp_encode = 3                                    # num MLP layers in node/edge attr encoder and decoder 
    n_mlp_mp = 2                                        # num MLP layers used in message passing operations
    n_mp_down_topk = [1,1]                              # num MMP blocks per Top-K level in encoder. Length = number of topk levels.
    n_mp_up_topk = [1,1]                                # num MMP blocks per Top-K level in decoder. Length = number of topk levels.
    pool_ratios = [1./4.]                               # pool ratio used in Top-K pooling (1/RF). Length = number of topk levels - 1

    n_mp_down_enc = [4]                                 # number of message passing layers used in each coarsening level in 
                                                        # downward pass for ENCODER MMP layers. 
                                                        # Length = number of coarsening levels in each encoder MMP layer. 

    n_mp_up_enc = []                                    # number of message passing layers used in each coarsening level in 
                                                        # upward pass for the ENCODER MMP layers. 
                                                        # Length = number of coarsening levels - 1. If length = 0, coarsening in MMP
                                                        # layer is not used

    lengthscales_enc = []                               # list of lengthscales used in voxel clustering algorithm for ENCODER MMP layers. Length = len(n_mp_up_enc) 

    n_mp_down_dec = [2,2,4]                             # same as "n_mp_down_enc", but for DECODER MMP layers 
    n_mp_up_dec = [2,2]                                 # same as "n_mp_up_enc", but for DECODER MMP layers
    lengthscales_dec = [0.5,1.0]                        # same as "lengthscales_enc", but for DECODER MMP layers
    interp = 'learned'                                  # interpolation mode used in MMP layer ("learned", "knn", "pc")
    act = F.elu                                         # activation function used in MLPs
    param_sharing = False                               # parameter sharing option for MMP layers

    # Create bounding box: needed for voxel clustering if applicable 
    bounding_box = []
    if len(lengthscales_dec) > 0:
        x_lo = sample.pos[:,0].min() - lengthscales_dec[0]/2
        x_hi = sample.pos[:,0].max() + lengthscales_dec[0]/2
        y_lo = sample.pos[:,1].min() - lengthscales_dec[0]/2
        y_hi = sample.pos[:,1].max() + lengthscales_dec[0]/2
        bounding_box = [x_lo.cpu().item(), x_hi.cpu().item(), y_lo.cpu().item(), y_hi.cpu().item()]

    # Build model
    model = GNN_TopK(
                in_channels_node,
                in_channels_edge,
                hidden_channels,
                out_channels,
                n_mlp_encode,
                n_mlp_mp,
                n_mp_down_topk,
                n_mp_up_topk,
                pool_ratios,
                n_mp_down_enc,
                n_mp_up_enc,
                n_mp_down_dec,
                n_mp_up_dec,
                lengthscales_enc,
                lengthscales_dec,
                bounding_box,
                interp,
                act,
                param_sharing,
                name='gnn_topk')
    model.to(device)

    # ~~~~ Train model
    train(epochs, model, train_loader, valid_loader)
