import os, time
from typing import Optional, Union, Callable, List
import numpy as np
import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils.repeat import repeat
from torch_geometric.utils.to_dense_batch import to_dense_batch
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)

from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x

class Multiscale_MessagePassing_Layer(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int, 
                 n_mlp_mp : int, 
                 n_mp_down : List[int], 
                 n_mp_up : List[int], 
                 n_repeat_mp_up : int,
                 lengthscales : List[float], 
                 bounding_box : List[float],
                 act : Optional[Callable] = F.elu, 
                 interpolation_mode : Optional[str] = 'learned',
                 name : Optional[str] = 'mmp_layer'):
        super().__init__()
        self.edge_aggregator = EdgeAggregation()
        self.hidden_channels = hidden_channels
        self.act = act
        self.interpolation_mode = interpolation_mode
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing blocks
        self.n_mp_down = n_mp_down # number of message passing blocks in downsampling path 
        self.n_mp_up = n_mp_up # number of message passing blocks in upsampling path  
        self.n_repeat_mp_up = n_repeat_mp_up # number of times to repeat each upward MP layer 
        self.depth = len(n_mp_up) # depth of u net 
        self.lengthscales = lengthscales # lengthscales needed for voxel grid clustering
        self.l_char = [1.0] + self.lengthscales

        if not bounding_box:
            self.x_lo = None
            self.x_hi = None
            self.y_lo = None
            self.y_hi = None
        else:
            self.x_lo = bounding_box[0]
            self.x_hi = bounding_box[1]
            self.y_lo = bounding_box[2]
            self.y_hi = bounding_box[3]
        self.name = name

        assert(len(self.lengthscales) == self.depth), "size of lengthscales must be equal to size of n_mp_up"

        # ~~~~ DOWNWARD Message Passing
        # Edge updates: 
        self.edge_down_mps = torch.nn.ModuleList() 
        self.edge_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_down_mps.append(edge_mp)
            self.edge_down_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_down_mps = torch.nn.ModuleList()
        self.node_down_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_down)):
            n_mp = n_mp_down[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*2
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_down_mps.append(node_mp)
            self.node_down_norms.append(node_mp_norm)

        # ~~~~ UPWARD Message Passing
        self.edge_up_mps = torch.nn.ModuleList() 
        self.edge_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_up)):
            n_mp = n_mp_up[m]
            edge_mp = torch.nn.ModuleList()
            edge_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*3
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                edge_mp.append(temp)
                edge_mp_norm.append( nn.LayerNorm(output_features) )

            self.edge_up_mps.append(edge_mp)
            self.edge_up_norms.append(edge_mp_norm)

        # Node updates: 
        self.node_up_mps = torch.nn.ModuleList()
        self.node_up_norms = torch.nn.ModuleList()

        # Loop through levels: 
        for m in range(len(n_mp_up)):
            n_mp = n_mp_up[m]
            node_mp = torch.nn.ModuleList()
            node_mp_norm = torch.nn.ModuleList()

            # Loop through message passing steps per level 
            for i in range(n_mp):
                temp = torch.nn.ModuleList()

                # Loop through layers in MLP
                for j in range(self.n_mlp_mp):
                    if j == 0:
                        input_features = hidden_channels*2
                        output_features = hidden_channels 
                    else:
                        input_features = hidden_channels
                        output_features = hidden_channels
                    temp.append( nn.Linear(input_features, output_features) )
                node_mp.append(temp)
                node_mp_norm.append( nn.LayerNorm(output_features) )

            self.node_up_mps.append(node_mp)
            self.node_up_norms.append(node_mp_norm)

        # For learned interpolations:
        self.edge_encoder_f2c_mlp = torch.nn.ModuleList()
        self.downsample_mlp = torch.nn.ModuleList()
        self.upsample_mlp = torch.nn.ModuleList()
        self.downsample_norm = []
        self.upsample_norm = []

        if (self.interpolation_mode == 'learned' and self.depth > 0):

            # encoder for fine-to-coarse edge features 
            for j in range(self.n_mlp_mp):
                if j == 0: 
                    input_features = 2 # 2-dimensional distance vector 
                    output_features = hidden_channels
                else:
                    input_features = hidden_channels
                    output_features = hidden_channels
                self.edge_encoder_f2c_mlp.append( nn.Linear(input_features, output_features) )

            # downsample mlp  
            for j in range(self.n_mlp_mp):
                if j == 0:
                    input_features = hidden_channels*2 # 2*hidden_channels for encoded f2c edges and sender node attributes 
                    output_features = hidden_channels 
                else:
                    input_features = hidden_channels
                    output_features = hidden_channels
                self.downsample_mlp.append( nn.Linear(input_features, output_features) ) 
            self.downsample_norm = nn.LayerNorm(output_features) 

            # upsample mlp
            for j in range(self.n_mlp_mp):
                if j == 0:
                    input_features = hidden_channels*3 # 3 for encoded edge + sender and receiver node
                    output_features = hidden_channels
                else:
                    input_features = hidden_channels
                    output_features = hidden_channels
                self.upsample_mlp.append( nn.Linear(input_features, output_features) )
            self.upsample_norm = nn.LayerNorm(output_features)

        # Reset params 
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pos, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # ~~~~ INITIAL MESSAGE PASSING ON FINE GRAPH (m = 0)
        m = 0 # level index 
        n_mp = self.n_mp_down[m] # number of message passing blocks 
        for i in range(n_mp):
            x_own = x[edge_index[0,:], :]
            x_nei = x[edge_index[1,:], :]
            edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

            # edge update mlp
            for j in range(self.n_mlp_mp):
                edge_attr_t = self.edge_down_mps[m][i][j](edge_attr_t) 
                if j < self.n_mlp_mp - 1:
                    edge_attr_t = self.act(edge_attr_t)

            edge_attr = edge_attr + edge_attr_t
            edge_attr = self.edge_down_norms[m][i](edge_attr)

            # edge aggregation 
            edge_agg = self.edge_aggregator(x, edge_index, edge_attr)

            x_t = torch.cat((x, edge_agg), axis=1)

            # node update mlp
            for j in range(self.n_mlp_mp):
                x_t = self.node_down_mps[m][i][j](x_t)
                if j < self.n_mlp_mp - 1:
                    x_t = self.act(x_t) 
            
            x = x + x_t
            x = self.node_down_norms[m][i](x)

            
        xs = [x] 
        edge_indices = [edge_index]
        edge_attrs = [edge_attr]
        positions = [pos]
        batches = [batch]
        clusters = []
        edge_attrs_f2c = []
       
        # ~~~~ Downward message passing 
        for m in range(1, self.depth + 1):
            # Run voxel clustering
            cluster = tgnn.voxel_grid(pos = pos,
                                      size = self.lengthscales[m-1],
                                      batch = batch,
                                      start = [self.x_lo, self.y_lo], 
                                      end = [self.x_hi, self.y_hi])

            if self.interpolation_mode == 'learned':
                pos_f = pos.clone()
                edge_index, edge_attr, batch, pos, cluster, perm = avg_pool_mod_no_x(
                                                                                cluster,
                                                                                edge_index, 
                                                                                edge_attr,
                                                                                batch, 
                                                                                pos)
                
                n_nodes = x.shape[0]
                edge_index_f2c = torch.concat( (torch.arange(0, n_nodes, dtype=torch.long, device=x.device).view(1,-1), cluster.view(1,-1)), axis=0 )
                
                pos_c = pos
                edge_attr_f2c = (pos_c[edge_index_f2c[1,:]] - pos_f[edge_index_f2c[0,:]])/self.l_char[m-1]

                # encode the edge attributes with MLP
                for j in range(self.n_mlp_mp):
                    edge_attr_f2c = self.edge_encoder_f2c_mlp[j](edge_attr_f2c)
                    if j < self.n_mlp_mp - 1:
                        edge_attr_f2c = self.act(edge_attr_f2c)
                 
                # append list
                edge_attrs_f2c += [edge_attr_f2c]

                # Concatenate
                temp_ea = torch.cat((edge_attr_f2c, x), axis=1)

                # Apply downsample MLP
                for j in range(self.n_mlp_mp):
                    temp_ea = self.downsample_mlp[j](temp_ea)
                    if j < self.n_mlp_mp - 1:
                        temp_ea = self.act(temp_ea)
                
                temp_ea = edge_attr_f2c + temp_ea
                temp_ea = self.downsample_norm(temp_ea)

                x = self.edge_aggregator( (pos_f, pos_c), edge_index_f2c, temp_ea )  
                
            else:
                x, edge_index, edge_attr, batch, pos, cluster, perm = avg_pool_mod(
                                                                                cluster, 
                                                                                x, 
                                                                                edge_index, 
                                                                                edge_attr,
                                                                                batch, 
                                                                                pos)
            
            # Append lists
            positions += [pos]
            batches += [batch]
            clusters += [cluster]

            # Do message passing on coarse graph
            for i in range(self.n_mp_down[m]):
                x_own = x[edge_index[0,:], :]
                x_nei = x[edge_index[1,:], :]
                edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

                for j in range(self.n_mlp_mp):
                    edge_attr_t = self.edge_down_mps[m][i][j](edge_attr_t) 
                    if j < self.n_mlp_mp - 1:
                        edge_attr_t = self.act(edge_attr_t)
                
                edge_attr = edge_attr + edge_attr_t
                edge_attr = self.edge_down_norms[m][i](edge_attr)
                edge_agg = self.edge_aggregator(x, edge_index, edge_attr)
                x_t = torch.cat((x, edge_agg), axis=1)

                for j in range(self.n_mlp_mp):
                    x_t = self.node_down_mps[m][i][j](x_t)
                    if j < self.n_mlp_mp - 1:
                        x_t = self.act(x_t) 
                
                x = x + x_t
                x = self.node_down_norms[m][i](x)

            
            # If there are coarser levels, append the fine-level lists
            if m < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_attrs += [edge_attr]

        # ~~~~ Upward message passing
        for m in range(self.depth):
            fine = self.depth - 1 - m
            res = xs[fine]
            edge_index = edge_indices[fine]
            edge_attr = edge_attrs[fine]
            
            if self.interpolation_mode == 'pc':
                x = x[clusters[fine]] + res

            elif self.interpolation_mode == 'knn':
                x = tgnn.knn_interpolate(x = x,
                                         pos_x = positions[fine+1],
                                         pos_y = positions[fine],
                                         batch_x = batches[fine+1],
                                         batch_y = batches[fine],
                                         k = 4) 
                x += res

            elif self.interpolation_mode == 'learned':
                edge_attr_c2f = -edge_attrs_f2c[fine]
    
                # coarse node attributes upsampled using pc interp 
                x = x[clusters[fine]]
                x = torch.cat((edge_attr_c2f, x, res), axis=1)

                # apply MLP: interpolation
                for j in range(self.n_mlp_mp):
                    x = self.upsample_mlp[j](x)
                    if j < self.n_mlp_mp - 1:
                        x = self.act(x)

                x = self.upsample_norm(x)
            else:
                raise ValueError('Invalid input to interpolation_mode: %s' %(self.interpolation_mode)) 

            # Message passing on new upsampled graph
            for i in range(self.n_mp_up[m]):
                for r in range(self.n_repeat_mp_up):
                    x_own = x[edge_index[0,:], :]
                    x_nei = x[edge_index[1,:], :]
                    edge_attr_t = torch.cat((x_own, x_nei, edge_attr), axis=1)

                    for j in range(self.n_mlp_mp):
                        edge_attr_t = self.edge_up_mps[m][i][j](edge_attr_t) 
                        if j < self.n_mlp_mp - 1:
                            edge_attr_t = self.act(edge_attr_t)
                    edge_attr = edge_attr + edge_attr_t

                    edge_attr = self.edge_up_norms[m][i](edge_attr)
                    edge_agg = self.edge_aggregator(x, edge_index, edge_attr)
                    x_t = torch.cat((x, edge_agg), axis=1)

                    for j in range(self.n_mlp_mp):
                        x_t = self.node_up_mps[m][i][j](x_t)
                        if j < self.n_mlp_mp - 1:
                            x_t = self.act(x_t) 
                    x = x + x_t
                    x = self.node_up_norms[m][i](x)
        return x, edge_attr 

    def input_dict(self):
        a = { 'edge_aggregator' : self.edge_aggregator, 
                'hidden_channels' : self.hidden_channels, 
                'act' : self.act, 
                'interpolation_mode' : self.interpolation_mode,
                'n_mlp_mp' : self.n_mlp_mp, 
                'n_mp_down' : self.n_mp_down, 
                'n_mp_up' : self.n_mp_up, 
                'n_repeat_mp_up' : self.n_repeat_mp_up, 
                'depth' : self.depth, 
                'lengthscales' : self.lengthscales, 
                'x_lo' : self.x_lo, 
                'x_hi' : self.x_hi,
                'y_lo' : self.y_lo, 
                'y_hi' : self.y_hi,
                'name' : self.name }
        return a

    def reset_parameters(self):
        # Down Message passing, edge update 
        for modulelist_level in self.edge_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Down Message passing, node update 
        for modulelist_level in self.node_down_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Up message passing, edge update 
        for modulelist_level in self.edge_up_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()

        # Up message passing, node update 
        for modulelist_level in self.node_up_mps:
            for modulelist_mp in modulelist_level:
                for module in modulelist_mp: 
                    module.reset_parameters()


        # learned interpolations 
        if self.interpolation_mode == 'learned':
            for module in self.downsample_mlp:
                module.reset_parameters()
            for module in self.upsample_mlp:
                module.reset_parameters()
            for module in self.edge_encoder_f2c_mlp:
                module.reset_parameters()

class GNN_TopK(torch.nn.Module):
    def __init__(self, 
                 in_channels_node : int, 
                 in_channels_edge : int, 
                 hidden_channels : int, 
                 out_channels : int, 
                 n_mlp_encode : int, 
                 n_mlp_mp : int, 
                 n_mp_down_topk : List[int], 
                 n_mp_up_topk : List[int], 
                 pool_ratios : List[float],
                 n_mp_down_enc : List[int], 
                 n_mp_up_enc : List[int], 
                 n_mp_down_dec : List[int],
                 n_mp_up_dec : List[int], 
                 lengthscales_enc : List[float], 
                 lengthscales_dec : List[float],
                 bounding_box : List[float], 
                 interp: Optional[str] = 'learned',
                 act: Optional[Callable] = F.elu, 
                 param_sharing: Optional[bool] = False,
                 name: Optional[str] = 'gnn_topk'):

        super().__init__()

        self.in_channels_node = in_channels_node
        self.in_channels_edge = in_channels_edge
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act
        self.n_mlp_encode = n_mlp_encode    # number of MLP layers in node/edge encoding stage 
        self.n_mlp_decode = n_mlp_encode    # number of MLP layers in node/edge decoding stage
        self.n_mlp_mp = n_mlp_mp # number of MLP layers in node/edge update functions used in message passing
        self.n_mp_down_topk = n_mp_down_topk # number of message passing blocks in downsampling path 
        self.n_mp_up_topk = n_mp_up_topk # number of message passing blocks in upsampling path  
        self.depth = len(n_mp_up_topk) - 1
        self.pool_ratios = pool_ratios
        self.param_sharing = param_sharing
        self.name = name

        assert(len(self.n_mp_up_topk) == len(self.n_mp_down_topk)), "n_mp_up and n_mp_down must be same length"

        # For multiscale gnn used in decoding stage  
        self.interp = interp
        self.n_mp_down_enc = n_mp_down_enc
        self.n_mp_up_enc = n_mp_up_enc
        self.n_mp_down_dec = n_mp_down_dec 
        self.n_mp_up_dec = n_mp_up_dec 
        self.lengthscales_enc = lengthscales_enc 
        self.lengthscales_dec = lengthscales_dec 
        self.bounding_box = bounding_box

        # ~~~~ Node encoder
        self.node_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_node 
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_encode.append( nn.Linear(input_features, output_features) )
        self.node_encode_norm = nn.LayerNorm(output_features)
       
        # ~~~~ Edge encoder 
        self.edge_encode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_encode): 
            if i == 0:
                input_features = in_channels_edge
                output_features = hidden_channels 
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.edge_encode.append( nn.Linear(input_features, output_features) )
        self.edge_encode_norm = nn.LayerNorm(output_features)

        # ~~~~ DOWNWARD Message Passing
        n_repeat_mp_up_enc = 1
        if not self.param_sharing: 
            self.down_mps = torch.nn.ModuleList() 
            for m in range(len(n_mp_down_topk)):
                n_mp = n_mp_down_topk[m]
                down_mp = torch.nn.ModuleList()
                for i in range(n_mp): 
                    down_mp.append(Multiscale_MessagePassing_Layer(hidden_channels,
                                                            n_mlp_mp,
                                                            n_mp_down_enc,
                                                            n_mp_up_enc,
                                                            n_repeat_mp_up_enc,
                                                            lengthscales_enc,
                                                            bounding_box,
                                                            act=act,
                                                            interpolation_mode=interp,
                                                            name=self.name + '_down_mp_%d' %(i)))
                self.down_mps.append(down_mp)
        else:
            self.down_mps = Multiscale_MessagePassing_Layer(hidden_channels,
                                                            n_mlp_mp,
                                                            n_mp_down_enc,
                                                            n_mp_up_enc,
                                                            n_repeat_mp_up_enc,
                                                            lengthscales_enc,
                                                            bounding_box,
                                                            act=act,
                                                            interpolation_mode=interp,
                                                            name=self.name + '_down_mp')
        
        # ~~~~ POOLING  
        self.pools = torch.nn.ModuleList() # for pooling 
        for i in range(self.depth):
            self.pools.append(TopKPooling_Mod(hidden_channels, self.pool_ratios[i]))


        # ~~~~ UPWARD Message Passing
        n_repeat_mp_up_dec = 1
        if not self.param_sharing: 
            self.up_mps = torch.nn.ModuleList()
            for m in range(len(n_mp_up_topk)):
                n_mp = n_mp_up_topk[m]
                up_mp = torch.nn.ModuleList()
                for i in range(n_mp):
                    up_mp.append(Multiscale_MessagePassing_Layer(hidden_channels,
                                                            n_mlp_mp,
                                                            n_mp_down_dec,
                                                            n_mp_up_dec,
                                                            n_repeat_mp_up_dec,
                                                            lengthscales_dec,
                                                            bounding_box,
                                                            act=act,
                                                            interpolation_mode=interp,
                                                            name=self.name + '_up_mp_%d' %(i)))
                self.up_mps.append(up_mp)
        else:
            self.up_mps = Multiscale_MessagePassing_Layer(hidden_channels,
                                                            n_mlp_mp,
                                                            n_mp_down_dec,
                                                            n_mp_up_dec,
                                                            n_repeat_mp_up_dec,
                                                            lengthscales_dec,
                                                            bounding_box,
                                                            act=act,
                                                            interpolation_mode=interp,
                                                            name=self.name + '_up_mp')

        # ~~~~ Node-wise decoder
        self.node_decode = torch.nn.ModuleList() 
        for i in range(self.n_mlp_decode): 
            if i == self.n_mlp_decode - 1:
                input_features = hidden_channels 
                output_features = out_channels
            else:
                input_features = hidden_channels 
                output_features = hidden_channels 
            self.node_decode.append( nn.Linear(input_features, output_features) )

        # ~~~~ Reset params upon initialization
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, pos, batch=None, return_mask=True):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if return_mask: 
            mask = x.new_zeros(x.size(0))
        else:
            mask = None

        # ~~~~ Node Encoder: 
        for i in range(self.n_mlp_encode):
            x = self.node_encode[i](x) 
            if i < self.n_mlp_encode - 1:
                x = self.act(x)
            else:
                x = x
        x = self.node_encode_norm(x)

        # ~~~~ Edge Encoder: 
        for i in range(self.n_mlp_encode):
            edge_attr = self.edge_encode[i](edge_attr)
            if i < self.n_mlp_encode - 1:
                edge_attr = self.act(edge_attr)
            else:
                edge_attr = edge_attr
        edge_attr = self.edge_encode_norm(edge_attr)

        m = 0 # level index 
        n_mp = self.n_mp_down_topk[m] # number of message passing blocks 
        for i in range(n_mp):
            if not self.param_sharing: 
                x, edge_attr = self.down_mps[m][i](x, edge_index, edge_attr, pos, batch=batch)
            else:
                x, edge_attr = self.down_mps(x, edge_index, edge_attr, pos, batch=batch)

        xs = [x] 
        positions = [pos]
        edge_indices = [edge_index]
        edge_attrs = [edge_attr]
        batches = [batch]
        perms = []
        edge_masks = []

        for m in range(1, self.depth + 1):
            # Pooling: returns new x and edge_index for coarser grid 
            x, edge_index, edge_attr, batch, perm, edge_mask, _ = self.pools[m - 1](x, edge_index, edge_attr, batch)
            pos = pos[perm]

            # Append the permutation list for node upsampling
            perms += [perm]

            # Append the edge mask list for edge upsampling
            edge_masks += [edge_mask]

            # Append the positions list for upsampling
            positions += [pos]

            # append the batch list for upsampling
            batches += [batch]

            for i in range(self.n_mp_down_topk[m]):
                if not self.param_sharing:
                    x, edge_attr = self.down_mps[m][i](x, edge_index, edge_attr, pos, batch=batch)
                else:
                    x, edge_attr = self.down_mps(x, edge_index, edge_attr, pos, batch=batch)
            
            if m < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_attrs += [edge_attr]

        # ~~~~ Fill node mask:
        if return_mask: 
            print('Filling mask')
            perm_global = perms[0] 
            mask[perm_global] = 1
            for i in range(1,self.depth):
                perm_global = perm_global[perms[i]]
                mask[perm_global] = i+1 

        # ~~~~ Upward message passing (decoder)
        m = 0
        for i in range(self.n_mp_up_topk[m]):
            if not self.param_sharing:
                x, edge_attr = self.up_mps[m][i](x, edge_index, edge_attr, pos, batch=batch)
            else:
                x, edge_attr = self.up_mps(x, edge_index, edge_attr, pos, batch=batch)
        
        # upward cycle
        for m in range(self.depth):
            # Get the fine level index
            fine = self.depth - 1 - m

            # Get the batch
            batch = batches[fine]

            # Get node features and edge features on fine level
            res = xs[fine]
            pos = positions[fine]
            res_edge = edge_attrs[fine]

            # Get edge index on fine level
            edge_index = edge_indices[fine]

            # Upsample edge features
            edge_mask = edge_masks[fine]
            up_edge = torch.zeros_like(res_edge)
            up_edge[edge_mask] = edge_attr
            edge_attr = up_edge

            # Upsample node features
            # get node assignments on fine level
            perm = perms[fine]
            up = torch.zeros_like(res)
            up[perm] = x
            x = up

            # Message passing on new upsampled graph
            for i in range(self.n_mp_up_topk[m+1]):
                for r in range(1):
                    if not self.param_sharing:
                        x, edge_attr = self.up_mps[m+1][i](x, edge_index, edge_attr, pos, batch=batch)
                    else:
                        x, edge_attr = self.up_mps(x, edge_index, edge_attr, pos, batch=batch)


        # ~~~~ Node decoder
        for i in range(self.n_mlp_decode):
            x = self.node_decode[i](x) 
            if i < self.n_mlp_decode - 1:
                x = self.act(x)
            else:
                x = x


        return x, mask 

    def input_dict(self):
        a = { 'in_channels_node' : self.in_channels_node, 
                'in_channels_edge' : self.in_channels_edge,
                'hidden_channels' : self.hidden_channels, 
                'out_channels' : self.out_channels, 
                'n_mlp_encode' : self.n_mlp_encode, 
                'n_mlp_mp' : self.n_mlp_mp, 
                'n_mp_down_topk' : self.n_mp_down_topk, 
                'n_mp_up_topk' : self.n_mp_up_topk, 
                'pool_ratios' : self.pool_ratios, 
                'n_mp_down_enc' : self.n_mp_down_enc,
                'n_mp_up_enc' : self.n_mp_up_enc,
                'n_mp_down_dec' : self.n_mp_down_dec,
                'n_mp_up_dec' : self.n_mp_up_dec,
                'lengthscales_enc' : self.lengthscales_enc, 
                'lengthscales_dec' : self.lengthscales_dec, 
                'bounding_box' : self.bounding_box, 
                'interp' : self.interp,
                'depth' : self.depth, 
                'act' : self.act, 
                'param_sharing' : self.param_sharing,
                'name' : self.name }

        return a

    def reset_parameters(self):
        # Node encoding
        for module in self.node_encode:
            module.reset_parameters()

        # Edge encoding 
        for module in self.edge_encode:
            module.reset_parameters()

        # Node decoder: 
        for module in self.node_decode:
            module.reset_parameters()

        # Pooling: 
        for module in self.pools:
            module.reset_parameters()


class EdgeAggregation(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

