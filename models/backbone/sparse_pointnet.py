import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_scatter
import functools


class Mlp(nn.Module):

    def __init__(self, num_input_features, num_output_features=None, act=nn.ReLU, num_hidden=1, hidden_dim=32):
        super().__init__()
        layers = [
            nn.Linear(num_input_features, hidden_dim),
            act(inplace=True),
        ] 
        if num_hidden > 0:
            for i in range(num_hidden):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    act(inplace=True),
                ]
        if num_output_features is not None: 
            layers += [
                nn.Linear(hidden_dim, num_output_features),
            ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """*c
        """
        return self.net(x) 
    

# Copied from shapeformer.
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

def sparse_pooling(x, segment_idx=None, reduce='max'):
    out = torch_scatter.scatter(x, segment_idx, dim=0, reduce=reduce)  
    return out


class SparsePointnetEncoder(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''
    def __init__(self, c_dim=128, dim=3, hidden_dim=128,  num_blocks=2, pooling="max"):
        super().__init__()
       
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        

        self.blocks = nn.Sequential()
        self.num_blocks = num_blocks
        for _i in range(num_blocks):
            self.blocks.add_module(
                "block_{}".format(_i), ResnetBlockFC(2*hidden_dim, hidden_dim),
            )
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.pooling = pooling
        self.num_blocks = num_blocks 

    def forward(self, p, segment_idx, pooling=True):
        """
        p: nc
        segment_idx: n
        """
        pooling_func = functools.partial(
            sparse_pooling,
            segment_idx=segment_idx, 
            reduce="max"
            )  
        net = self.fc_pos(p)
        for i in range(self.num_blocks -1):
            net = self.blocks[i](net)
            pooled = pooling_func(net)[segment_idx]
            net = torch.cat([net, pooled], dim=-1) 
        net = self.blocks[self.num_blocks-1](net)
        # Reduce to  B x F
        if pooling == True:
            net = pooling_func(net)
            c = self.fc_c(self.actvn(net))
        else:
            c = net
        return c


class SparsePointnetDecoder(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''
    def __init__(self, c_dim=128, dim=3, hidden_dim=128,  num_blocks=1, pooling="max"):
        super().__init__()
       
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2*hidden_dim) 

        self.blocks = nn.Sequential()
        self.num_blocks = num_blocks
        for _i in range(num_blocks):
            self.blocks.add_module(
                "block_{}".format(_i), ResnetBlockFC(2*hidden_dim, hidden_dim),
            )
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.pooling = pooling
        self.num_blocks = num_blocks
        
        

    def forward(self, p, segment_feat, segment_idx, pooling=False):
        """
        p: nc
        segment_idx: n
        """
        pooling_func = functools.partial(
            sparse_pooling,
            segment_idx=segment_idx, 
            reduce="max"
            )
        pts_feat = segment_feat[segment_idx]
        pts_feat = torch.cat([pts_feat, p], dim=-1)
        net = self.fc_pos(pts_feat)

        for i in range(self.num_blocks -1):
            net = self.blocks[i](net)
            pooled = pooling_func(net)[segment_idx]
            net = torch.cat([net, pooled], dim=-1)  
        net = self.blocks[self.num_blocks-1](net)
        net =self.fc_c(net)
        return net