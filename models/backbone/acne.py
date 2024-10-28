import torch 
from torch import nn
import torch_scatter
import torch.nn.functional as F 


class Mlp(nn.Module):

    def __init__(self, num_input_features, num_output_features=None, act=nn.LeakyReLU, num_hidden=1, hidden_dim=32):
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

class ACN(nn.Module):
    def __init__(self, inc, atten_opt="sigmoid_softmax", eps=1e-3):
        super().__init__()
        self.atten_opt = atten_opt
        self.eps = eps
        self.att_layer = nn.Linear(inc, 1)

    def forward(self, x, sparse_idx):
        # x: NC; segment_idx: N 
        ret_a = None
        segment_idx, weight_pri = sparse_idx 
        att = self.att_layer(x)
        a = torch_scatter.scatter_softmax(att, segment_idx, dim=0) 

        # include weight prior
        a = a * weight_pri[:, None]
        a = a / (torch_scatter.scatter(a, segment_idx, dim=0, reduce="sum")[segment_idx]).clamp(min=1e-3)

        mean = torch_scatter.scatter(x * a, segment_idx, dim=0, reduce="sum")
        out = x - mean[segment_idx]
        std = torch.sqrt(
            torch_scatter.scatter(a*out**2, segment_idx, dim=0, reduce="sum") + self.eps
            )
        out = out / std[segment_idx] # BCN1
        return out, a 


class ConvLayer(nn.Module):
    def __init__(self, inc, outc, cn_type="ACN", bn="gn"):
        super(ConvLayer, self).__init__()

        # Conv layer
        self.conv = nn.Linear(inc, outc)

        # Context norm layer
        if cn_type == "ACN":
            self.context_norm = ACN(outc)
        elif cn_type == "None":
            self.context_norm = nn.Identity()
        else:
            raise NotImplementedError

        # Batch norm layer
        if bn == "gn":
            self.bn_norm = nn.GroupNorm(32, outc)
        elif bn == "bn":
            self.bn_norm = nn.BatchNorm2d(outc)
        elif bn == "None":
            self.bn_norm = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, sparse_idx):
        x = self.conv(x)
        x, ret_a = self.context_norm(x, sparse_idx)
        x = self.bn_norm(x)
        x = F.relu(x)
        return x, ret_a


class ResBlock(nn.Module):

    def __init__(self, inc, outc, num_inner=2, cn_type="ACN", bn="gn"):
        super(ResBlock, self).__init__()
        if inc != outc:
            self.pass_through = ConvLayer(
                inc, outc, cn_type="None", bn=bn)
        else:
            self.pass_through = None

        self.conv_layers = nn.Sequential()
        self.num_inner = num_inner
        for _i in range(num_inner):
            self.conv_layers.add_module(
                "conv-{}".format(_i), ConvLayer(outc, outc, cn_type=cn_type, bn=bn)
            )

    def forward(self, x, sparse_idx):
        if self.pass_through is not None:
            x = self.pass_through(x)
        x_in = x
        ret_a = []
        for i in range(self.num_inner):
            x_in, a_i = self.conv_layers[i](x_in, sparse_idx)
            ret_a += [a_i]
        out = x_in + x
        return out, ret_a

class Acne(nn.Module):
    def __init__(self, inc, outc, opt):
        super().__init__()
        num_layer = getattr(opt, "acne_num_layer" , 6)
        num_inner = getattr(opt, "acne_num_inner", 2)
        self.num_layer = num_layer

        hidden_dim = getattr(opt, "hidden_dim", 128)
        cn_type = getattr(opt, "cn_type", 'ACN')
        bn_type = getattr(opt, "bn_type", 'None')

        # input layer: Conv layer
        # no relu activation in original ACNe implementation
        self.input_layer = Mlp(
            inc, hidden_dim, num_hidden=3, hidden_dim=hidden_dim)
        # self.input_layer = nn.Linear(inc, hidden_dim)
        inc = hidden_dim
        # intermediate layers for embedding the points
        self.layers = torch.nn.Sequential()
        for i in range(num_layer):
            self.layers.add_module(
                "arb-{}".format(i),
                ResBlock(inc, hidden_dim, num_inner, cn_type=cn_type, bn=bn_type))
            inc = hidden_dim
        self.outlayer_layer = Mlp(
            inc, outc, num_hidden=0, hidden_dim=outc)

    def forward(self, x, sparse_idx):
        """
        """
        x = F.leaky_relu(self.input_layer(x)) 
        ret_a = [] 
        for i in range(self.num_layer):
            x, a_i = self.layers[i](x, sparse_idx)
            ret_a += [a_i]  
        x = self.outlayer_layer(x)
        
        return x, ret_a[-1][-1]
