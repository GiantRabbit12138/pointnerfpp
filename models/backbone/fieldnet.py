import torch
from torch import nn
import torch.nn.functional as F
import functools
import torch_scatter

def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts

def get_subpixidx(ih, iw, grid):
    """Implements grid_sample2d with double-differentiation support.
    """
    # bs, nc, ih, iw = image.shape
    # _, h, w, _ = grid.shape 

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (iw - 1)
    iy = ((iy + 1) / 2) * (ih - 1)

    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    ix_nw = torch.clamp(ix_nw.long(), 0, iw - 1)
    iy_nw = torch.clamp(iy_nw.long(), 0, ih - 1)

    ix_ne = torch.clamp(ix_ne.long(), 0, iw - 1)
    iy_ne = torch.clamp(iy_ne.long(), 0, ih - 1)

    ix_sw = torch.clamp(ix_sw.long(), 0, iw - 1)
    iy_sw = torch.clamp(iy_sw.long(), 0, ih - 1)

    ix_se = torch.clamp(ix_se.long(), 0, iw - 1)
    iy_se = torch.clamp(iy_se.long(), 0, ih - 1)

    nw_subidx = iy_nw * iw + ix_nw
    ne_subidx = iy_ne * iw + ix_ne
    sw_subidx = iy_sw * iw + ix_sw
    se_subidx = iy_se * iw + ix_se
    subidx = torch.stack([nw_subidx, ne_subidx, sw_subidx, se_subidx], -1)
    subw = torch.stack([nw, ne, sw, se], -1)
    return subidx, subw 

@torch.jit.script
def grid_sample2d(image, grid):
    """Implements grid_sample2d with double-differentiation support.
    Equivalent to F.grid_sample(..., mode='bilinear',
                                padding_mode='border', align_corners=True).
    """
    bs, nc, ih, iw = image.shape
    _, h, w, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (iw - 1)
    iy = ((iy + 1) / 2) * (ih - 1)

    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    ix_nw = torch.clamp(ix_nw.long(), 0, iw - 1)
    iy_nw = torch.clamp(iy_nw.long(), 0, ih - 1)

    ix_ne = torch.clamp(ix_ne.long(), 0, iw - 1)
    iy_ne = torch.clamp(iy_ne.long(), 0, ih - 1)

    ix_sw = torch.clamp(ix_sw.long(), 0, iw - 1)
    iy_sw = torch.clamp(iy_sw.long(), 0, ih - 1)

    ix_se = torch.clamp(ix_se.long(), 0, iw - 1)
    iy_se = torch.clamp(iy_se.long(), 0, ih - 1)

    image = image.view(bs, nc, ih * iw)

    nw_val = torch.gather(image, 2,
                          (iy_nw * iw + ix_nw).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))
    ne_val = torch.gather(image, 2,
                          (iy_ne * iw + ix_ne).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))
    sw_val = torch.gather(image, 2,
                          (iy_sw * iw + ix_sw).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))
    se_val = torch.gather(image, 2,
                          (iy_se * iw + ix_se).view(bs, 1,
                                                    h * w).expand(-1, nc, -1))

    out_val = (nw_val.view(bs, nc, h, w) * nw.view(bs, 1, h, w) +
               ne_val.view(bs, nc, h, w) * ne.view(bs, 1, h, w) +
               sw_val.view(bs, nc, h, w) * sw.view(bs, 1, h, w) +
               se_val.view(bs, nc, h, w) * se.view(bs, 1, h, w))

    return out_val

def positional_encoding(positions, freqs=1, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts

class PosEncoder(nn.Module):
    def __init__(self, d, c):
        super().__init__()   
        # gaussian matrix: dxc
        gauss_mat = nn.Parameter(
            torch.FloatTensor(1, d, c), requires_grad=False)
        gauss_mat.data.normal_()
        self.register_parameter("gauss_mat", gauss_mat)
    def forward(self, x):
        """x: bnd. """
        b, n, d = x.shape
        x = x * torch.pi * 2
        x = torch.matmul(x, self.gauss_mat.expand(b, -1, -1))
        pos_emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return pos_emb  


class RgbLayer(nn.Module): 
    def __init__(self, num_input_features, num_output_features, num_hidden=0):
        super().__init__()

        hidden_dim = 32 
        layers = [
            nn.Linear(num_input_features, hidden_dim),
            nn.Softplus(),
        ] 
        if num_hidden > 0:
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Softplus(),
            ]
        layers += [
            nn.Linear(hidden_dim, num_output_features),
        ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        x = x * (1 + 2 * 0.001) - 0.001
        return x

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


class SkyModel(nn.Module):
    """Similar to PointNerf
    """
    def __init__(self, opt):
        super().__init__()
        hidden = 128 
        self.opt = opt
        if opt.skymodel_opt == "mlp":
            self.net = Mlp(
                3, 3, num_hidden=5, hidden_dim=hidden)
            self.color_act = torch.nn.Sigmoid()
        else:
            import tinycudann as tcnn
            self.net = tcnn.Network(
            n_input_dims=3,
            n_output_dims=3,
            # following mars
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 128, 
                "n_hidden_layers": 4, 
            },
        )
            
    
    def forward(self, ray_dir):
        if self.opt.skymodel_opt == "mlp":
            color = self.net(ray_dir)
            color = self.color_act(color)
            color = color * (1 + 2 * 0.001) - 0.001
        else:
            ray_dir_flat = ray_dir.reshape(-1, 3) 
            ray_dir_flat = ray_dir_flat / 2.0 + 0.5
            color = self.net(ray_dir_flat).float()
            color = color.reshape(*ray_dir.shape)
        return color

class Aggregator(nn.Module):
    """Similar to PointNerf
    """
    def __init__(self, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_viewdir_freqs = 4
        self.num_loc_freqs = 5
        self.num_feat_freqs = 3
        loc_freq_len = self.num_loc_freqs * 2 * 3
        feat_freq_len = feat_dim * self.num_feat_freqs * 2

        self.output_feat = 256 
        self.block1 = Mlp(
            feat_dim+loc_freq_len+feat_freq_len, 
            None, num_hidden=4, hidden_dim=self.output_feat)

        self.color_block = Mlp(
            self.output_feat+3*2*self.num_viewdir_freqs, 3, num_hidden=2, hidden_dim=128) 
        
        
        self.sem_feat_dim = cfg.sem_feat_dim 
        if self.sem_feat_dim > 0:
            self.semantic_block = Mlp(
                self.output_feat, self.sem_feat_dim, num_hidden=2, hidden_dim=256)
        else:
            self.semantic_block = None
        self.sigma_linear = nn.Linear(
            self.output_feat, 1)
        self.density_super_act = torch.nn.Softplus()
        self.color_act = torch.nn.Sigmoid()
        self.outdim = 1+3+self.sem_feat_dim

    
    def forward(self, rel_loc, feat, mask, viewdir, request_agg_output=[]):
        """
        Args: 
            viewdir: n3
            rel_loc: nk3
            feat: nkc
            mask: nk
        Outputs:
            radiance_feat:  
        """ 
        outputs = {}
        # agg_weights
        dist = rel_loc.norm(dim=-1)

        # weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)

        agg_weights = (1.0 / dist.clamp(min=1e-8)) * mask.float()
        # agg_weights = mask.float()
        agg_weights = agg_weights / agg_weights.sum(
            dim=-1, keepdim=True).clamp(min=1e-8)
        n, k, _ = rel_loc.shape
        radiance_feat = torch.full((n, k, 1+self.output_feat), 0.0).to(feat)
        feat = feat[mask]
        rel_loc = rel_loc[mask]

        # print(rel_loc.shape)
        rel_loc_freq = positional_encoding(rel_loc, self.num_loc_freqs)
        feat_freq = positional_encoding(feat, self.num_feat_freqs) # 
        feat = torch.cat([feat, feat_freq, rel_loc_freq], dim=-1) 
        feat = self.block1(feat)
        if self.cfg.use_sdf:
            density_or_distance = self.sigma_linear(feat)
        else:
            density_or_distance = self.density_super_act(
                self.sigma_linear(feat) - 1)
    
        radiance_feat_ = torch.cat([density_or_distance, feat], dim=-1)
        radiance_feat[mask] = radiance_feat_ # nxkx(1+c)
        radiance_feat = torch.matmul(
            agg_weights.unsqueeze(-2), radiance_feat).squeeze(-2) # n(1+c)  
        density_or_distance = radiance_feat[..., :1] 
        radiance_feat = radiance_feat[..., 1:]
        color_feat = torch.cat(
            [radiance_feat, positional_encoding(
                viewdir if viewdir is not None else torch.zeros((n,3)).to(rel_loc), 
                self.num_viewdir_freqs)
            ], dim=-1)
        color = self.color_block(color_feat)
        color = self.color_act(color)
        color = color * (1 + 2 * 0.001) - 0.001
        semantic = self.semantic_block(radiance_feat) \
            if self.semantic_block is not None else None
        
        if semantic is None:
            return torch.cat([density_or_distance, color], dim=-1) 
        else:
            return torch.cat([density_or_distance, color, semantic], dim=-1) 

class RadianceNet(nn.Module):
    """Similar to PointNerf
    """
    def __init__(self, feat_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_feat = 256 
        self.block1 = Mlp(
            feat_dim, 
            None, num_hidden=4, hidden_dim=self.output_feat)
        self.num_viewdir_freqs = 4

        self.color_block = Mlp(
            self.output_feat+3*2*self.num_viewdir_freqs, 3, num_hidden=2, hidden_dim=128) 
        
        self.sem_feat_dim = cfg.sem_feat_dim 

        self.semantic_block = Mlp(
            self.output_feat, self.sem_feat_dim, num_hidden=2, hidden_dim=256)
        self.sigma_linear = nn.Linear(
            self.output_feat, 1)
        self.density_super_act = torch.nn.Softplus()
        self.color_act = torch.nn.Sigmoid()
        self.outdim = 1+3+self.sem_feat_dim
    
    def forward(self, feat, viewdir=None, request_agg_output=[]):
        """
        Args: 
            feat: bnc
            mask: bn
        Outputs:
            radiance_feat:  
        """ 
        outputs = {}
        # agg_weights
        feat = self.block1(feat)
        if self.cfg.use_sdf:
            density_or_distance = self.sigma_linear(feat)
        else:
            density_or_distance = self.density_super_act(
                self.sigma_linear(feat) - 1) 
        radiance_feat = feat 
        color_feat = torch.cat(
            [radiance_feat, positional_encoding(
                viewdir if viewdir is not None else torch.zeros((*feat.shape[:-1],3)).to(feat), 
                self.num_viewdir_freqs)
            ], dim=-1)
        color = self.color_block(color_feat)
        color = self.color_act(color)
        color = color * (1 + 2 * 0.001) - 0.001
        semantic = self.semantic_block(radiance_feat)
        return torch.cat([density_or_distance, color, semantic], dim=-1) 

class HierLocalRadianceNet(nn.Module):
    """
    """
    def __init__(self, feat_dim_list=[128], cfg=None):
        super().__init__()
        self.cfg = cfg
        self.output_feat = 256  
        self.num_loc_freqs = 5
        self.num_feat_freqs = 3

        self.loc_encoder = functools.partial(
            positional_encoding, 
            freqs=self.num_loc_freqs, ori=True)
        loc_enc_len = (self.num_loc_freqs * 2 + 1)* 3 
        self.feat_dim_list = feat_dim_list

        self.feat_fourier_encoder = functools.partial(
            positional_encoding, 
            freqs=self.num_feat_freqs, ori=False)

        for idx, feat_dim in enumerate(feat_dim_list):
            feat_freq_len = feat_dim * self.num_feat_freqs * 2
            setattr(
                self,
                f'block1_{idx}_{feat_dim}',
                Mlp(feat_dim+loc_enc_len+feat_freq_len, 
                None, num_hidden=4, hidden_dim=self.output_feat)
                )
        self.num_viewdir_freqs = 4
        self.viewdir_encoder = functools.partial(
            positional_encoding,
            freqs=self.num_viewdir_freqs
        ) 
        view_enc_len = 3*2*self.num_viewdir_freqs

        self.color_block = Mlp(
            self.output_feat+view_enc_len, 3, num_hidden=2, hidden_dim=128)  
        self.sem_feat_dim = cfg.sem_feat_dim 
        if self.sem_feat_dim > 0:
            self.semantic_block = Mlp(
                self.output_feat, self.sem_feat_dim, num_hidden=2, hidden_dim=256)
        else:
            self.semantic_block = None
        self.sigma_linear = nn.Linear(
            self.output_feat, 1)
        self.density_super_act = torch.nn.Softplus()
        self.color_act = torch.nn.Sigmoid()
        self.outdim = 1+3+self.sem_feat_dim
    
    def forward(self, idx, loc, feat, viewdir=None, request_agg_output=[]):
        """
        Args: 
            feat: bnc
            mask: bn
        Outputs:
            radiance_feat:  
        """ 
        outputs = {}
        # agg_weights
        in_dim = feat.shape[-1]
        rel_loc_freq = self.loc_encoder(loc)
        feat_freq = self.feat_fourier_encoder(feat) # 
        feat = torch.cat([rel_loc_freq, feat, feat_freq], dim=-1)

        feat = getattr(self, f'block1_{idx}_{in_dim}')(feat)

        if self.cfg.use_sdf:
            density_or_distance = self.sigma_linear(feat)
        else:
            density_or_distance = self.density_super_act(
                self.sigma_linear(feat) - 1) 
        radiance_feat = feat 
        color_feat = torch.cat(
            [radiance_feat, self.viewdir_encoder(
                viewdir if viewdir is not None else torch.zeros((*feat.shape[:-1],3)).to(feat)
                )
            ], dim=-1)
        color = self.color_block(color_feat)
        color = self.color_act(color)
        color = color * (1 + 2 * 0.001) - 0.001

        semantic = self.semantic_block(radiance_feat) \
            if self.semantic_block is not None else None
        
        if semantic is None:
            return torch.cat([density_or_distance, color], dim=-1) 
        else:
            return torch.cat([density_or_distance, color, semantic], dim=-1) 


class SparseHierRadianceNet(nn.Module):
    """
    """
    def __init__(self, feat_dim_list=[128], cfg=None):
        super().__init__()
        self.cfg = cfg
        self.output_feat = 256  
        self.num_loc_freqs = 5
        self.num_feat_freqs = 3

        self.loc_encoder = functools.partial(
            positional_encoding, 
            freqs=self.num_loc_freqs, ori=True)
        loc_enc_len = (self.num_loc_freqs * 2 + 1)* 3 
        self.feat_dim_list = feat_dim_list

        self.feat_fourier_encoder = functools.partial(
            positional_encoding, 
            freqs=self.num_feat_freqs, ori=False)

        for idx, feat_dim in enumerate(feat_dim_list):
            feat_freq_len = feat_dim * self.num_feat_freqs * 2
            setattr(
                self,
                f'block1_{idx}_{feat_dim}',
                Mlp(feat_dim+loc_enc_len+feat_freq_len, 
                None, num_hidden=4, hidden_dim=self.output_feat)
                )
        self.num_viewdir_freqs = 4
        self.viewdir_encoder = functools.partial(
            positional_encoding,
            freqs=self.num_viewdir_freqs
        ) 
        view_enc_len = 3*2*self.num_viewdir_freqs

        self.color_block = Mlp(
            self.output_feat+view_enc_len, 3, num_hidden=2, hidden_dim=128)  
        self.sem_feat_dim = cfg.sem_feat_dim 
        if self.sem_feat_dim > 0:
            self.semantic_block = Mlp(
                self.output_feat, self.sem_feat_dim, num_hidden=2, hidden_dim=256)
        else:
            self.semantic_block = None
        self.sigma_linear = nn.Linear(
            self.output_feat, 1)
        self.density_super_act = torch.nn.Softplus()
        self.color_act = torch.nn.Sigmoid()
        self.outdim = 1+3+self.sem_feat_dim
    
    def forward(self, level, loc, feat, segment_idx, viewdir=None, request_agg_output=[]):
        """
        Args: 
            feat: bnc
            mask: bn
        Outputs:
            radiance_feat:  
        """ 
        outputs = {}
        # agg_weights
        in_dim = feat.shape[-1]
        rel_loc_freq = self.loc_encoder(loc)
        feat_freq = self.feat_fourier_encoder(feat) # 
        feat = torch.cat([rel_loc_freq, feat, feat_freq], dim=-1)

        feat = getattr(self, f'block1_{level}_{in_dim}')(feat)

        if self.cfg.use_sdf:
            density_or_distance = self.sigma_linear(feat)
        else:
            density_or_distance = self.density_super_act(
                self.sigma_linear(feat) - 1) 
        
        # spase aggregation
        with torch.no_grad():
            unique_idx, reverse_idx = segment_idx.unique(return_inverse=True)
        radiance_feat = torch_scatter.scatter(feat, reverse_idx, dim=0, reduce='mean')
        density_or_distance = torch_scatter.scatter(
            density_or_distance, reverse_idx, dim=0, reduce='mean') 

        if viewdir is None:
            viewdir = torch.zeros((*radiance_feat.shape[:-1],3)).to(radiance_feat)
        else:
            viewdir = viewdir[unique_idx]
            
        color_feat = torch.cat(
            [radiance_feat, self.viewdir_encoder(
                viewdir if viewdir is not None else torch.zeros((*feat.shape[:-1],3)).to(feat)
                )
            ], dim=-1)
        color = self.color_block(color_feat)
        color = self.color_act(color)
        color = color * (1 + 2 * 0.001) - 0.001

        semantic = self.semantic_block(radiance_feat) \
            if self.semantic_block is not None else None
        
        if semantic is None:
            return torch.cat([density_or_distance, color], dim=-1), unique_idx 
        else:
            return torch.cat([density_or_distance, color, semantic], dim=-1), unique_idx


def set_parameters(module, tensor, var_name):     
    para = torch.nn.Parameter(tensor.clone().detach(), requires_grad=True)
    module.register_parameter(var_name, para)

def set_buffer(module, tensor, var_name): 
    module.register_buffer(var_name, tensor)

class PointConvs(nn.Module):
    def __init__(self, loc_enc_len, num_weights, indim, outdim, cfg):
        super().__init__()
        num_layers = getattr(cfg, 'point_conv_num_layer', 1) 
        for i_layer in range(num_layers):
            setattr(
                self,
                f'score_net_{i_layer}',
                Mlp(loc_enc_len, 
                num_weights, num_hidden=4, hidden_dim=32)
            )
            set_parameters(
                self, 
                nn.init.kaiming_normal_(torch.empty(num_weights, indim * outdim)).contiguous(),
                f"weights_bank_{i_layer}")
            setattr(
                self, 
                f'global_mlp_{i_layer}',
                Mlp(outdim, outdim, hidden_dim=outdim, num_hidden=1)
            )
            indim = outdim * 2

        self.outdim = outdim
        self.num_layers = num_layers
        self.cfg = cfg

    def forward(self, rel_loc_feat, feat, segment_idx=None, reduction='mean'):
        if segment_idx is None:
            assert self.num_layers == 1 
        indim = feat.shape[-1]
        for i_layer in range(self.num_layers): 
            # rel_loc_feat = torch.cat([rel_loc_freq], dim=-1)
            weight_coeff = getattr(self, f'score_net_{i_layer}')(rel_loc_feat)
            weight_coeff = torch.softmax(weight_coeff, dim=-1)
            weights_bank = getattr(self, f'weights_bank_{i_layer}')
            weights = torch.matmul(weight_coeff, weights_bank).reshape(
                weight_coeff.shape[0], indim, self.outdim)
            feat = torch.matmul(feat[:, None], weights).squeeze(1) 
            # import pdb; pdb.set_trace()

            # reduced_feat = torch_scatter.scatter(feat, segment_idx, dim=0, reduce=reduction)
            # global_mlp = getattr(self, f'global_mlp_{i_layer}')
            # reduced_feat = global_mlp(reduced_feat)
            
            # if self.num_layers > 1 and i_layer < self.num_layers - 1:
            #     feat = torch.cat([feat, reduced_feat[segment_idx]], dim=-1)
            #     # feat = F.relu(feat) # add nonlinearity.
            # else:
            #     feat = reduced_feat
    
            # indim = self.outdim * 2
                 
        return feat        

class SparseHierConv(nn.Module):
    """
    """
    def __init__(self, feat_dim_list=[128], cfg=None):
        super().__init__()
        self.cfg = cfg
        self.output_feat = 32 
        self.num_loc_freqs = 5
        self.num_feat_freqs = 3

        self.loc_encoder = functools.partial(
            positional_encoding, 
            freqs=self.num_loc_freqs, ori=True)
        loc_enc_len = (self.num_loc_freqs * 2 + 1)* 3 
        self.feat_dim_list = feat_dim_list

        self.feat_fourier_encoder = functools.partial(
            positional_encoding, 
            freqs=self.num_feat_freqs, ori=False)

        num_weights = cfg.num_weights 
        for idx, feat_dim in enumerate(feat_dim_list): 
            setattr(
                self,
                f'conv_{idx}_{feat_dim}', 
                PointConvs(loc_enc_len, num_weights, feat_dim, self.output_feat, cfg)
            )
            # setattr(
            #     self,
            #     f'block1_{idx}_{feat_dim}',
            #     Mlp(loc_enc_len, 
            #     num_weights, num_hidden=4, hidden_dim=32)
            # )
            # utils.set_paramters(
            #     self, 
            #     torch.rand(num_weights, feat_dim * self.output_feat), 
            #     f"weights_bank_{idx}_{feat_dim}")

        self.num_viewdir_freqs = 4
        self.viewdir_encoder = functools.partial(
            positional_encoding,
            freqs=self.num_viewdir_freqs
        ) 
        view_enc_len = 3*2*self.num_viewdir_freqs

        self.color_block = Mlp(
            self.output_feat+view_enc_len, 3, num_hidden=2, hidden_dim=128)  
        self.sem_feat_dim = cfg.sem_feat_dim 
        if self.sem_feat_dim > 0:
            self.semantic_block = Mlp(
                self.output_feat, self.sem_feat_dim, num_hidden=2, hidden_dim=256)
        else:
            self.semantic_block = None
        self.sigma_linear = nn.Linear(
            self.output_feat, 1)
        self.density_super_act = torch.nn.Softplus()
        self.color_act = torch.nn.Sigmoid()
        self.outdim = 1+3+self.sem_feat_dim
    
    def forward(self, level, loc, feat, segment_idx, viewdir=None, request_agg_output=[]):
        """
        Args: 
            feat: bnc
            mask: bn
        Outputs:
            radiance_feat:  
        """ 
        outputs = {}
        # agg_weights
        in_dim = feat.shape[-1]
        rel_loc_freq = self.loc_encoder(loc)
        # feat_freq = self.feat_fourier_encoder(feat) # 
        rel_loc_feat = torch.cat([rel_loc_freq], dim=-1)

        with torch.no_grad():
            unique_idx, reverse_idx = segment_idx.unique(return_inverse=True)

        radiance_feat = getattr(self, f'conv_{level}_{in_dim}')(
            rel_loc_feat, feat, reverse_idx,reduction='sum')

        # spase aggregation
        return radiance_feat, unique_idx
    
    def get_radiance(self, radiance_feat, viewdir):

        if self.cfg.use_sdf:
            density_or_distance = self.sigma_linear(radiance_feat)
        else:
            density_or_distance = self.density_super_act(
                self.sigma_linear(radiance_feat) - 1) 
        if viewdir is None:
            viewdir = torch.zeros((*radiance_feat.shape[:-1],3)).to(radiance_feat) 
                    
        color_feat = torch.cat(
            [radiance_feat, self.viewdir_encoder(viewdir)
            ], dim=-1)
        color = self.color_block(color_feat)
        color = self.color_act(color)
        color = color * (1 + 2 * 0.001) - 0.001

        semantic = self.semantic_block(radiance_feat) \
            if self.semantic_block is not None else None
        
        if semantic is None:
            return torch.cat([density_or_distance, color], dim=-1) 
        else:
            return torch.cat([density_or_distance, color, semantic], dim=-1) 
        
        

class TriplanarDecoder(nn.Module):

    def __init__(
            self, num_input_features, num_output_features, opt, 
            hidden_dim = 32, num_hidden=0, pose_encoder="pos"): 
        super().__init__()
        self.num_input_features = num_input_features
        if pose_encoder == "pos":
            self.pos_encoder = PosEncoder(3, 16)
            self.pos_emb_net = nn.Sequential(
                nn.Linear(num_input_features+32, num_input_features),
                nn.LeakyReLU(),
            )
        else:
            self.pos_encoder = None

        layers = [
            nn.Linear(num_input_features, hidden_dim),
            nn.LeakyReLU(),
        ] 
        if num_hidden > 0:
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            ]
        layers += [
            nn.Linear(hidden_dim, num_output_features),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, xy=None, xz=None, yz=None, coords=None, requires_double_backward=True):
        assert xy.shape[1] == self.num_input_features
        assert xz.shape[1] == self.num_input_features
        assert yz.shape[1] == self.num_input_features

        if requires_double_backward:
            # Use custom grid sample with double differentiation support
            e1 = grid_sample2d(xy, coords[..., [0, 1]])
            e2 = grid_sample2d(xz, coords[..., [0, 2]])
            e3 = grid_sample2d(yz, coords[..., [1, 2]])
        else:
            e1 = F.grid_sample(xy,
                               coords[..., [0, 1]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)
            e2 = F.grid_sample(xz,
                               coords[..., [0, 2]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)
            e3 = F.grid_sample(yz,
                               coords[..., [1, 2]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)

        x = (e1 + e2 + e3) / 3
        # x = x.view(x.shape[0], self.num_input_features, -1).transpose(-2, -1)
        x = x.moveaxis(1, -1)
        if self.pos_encoder is not None:
            # cat pos_emb and linear layer to reduce dimension.
            pos_emb = self.pos_encoder(coords.reshape(coords.shape[0], -1, 3)).reshape(*coords.shape[:3], -1) 
            x = self.pos_emb_net(torch.cat([x, pos_emb], dim=-1))
        x = self.net(x)
        return x 
def set_parameters(module, tensor, var_name, grad=True):     
    para = torch.nn.Parameter(tensor.clone().detach(), requires_grad=grad)
    module.register_parameter(var_name, para)   


class TriplaneField(nn.Module):

    def __init__(self, opt, img_res=512, num_output_features=None):
        super().__init__()
        self.opt = opt 

        self.img_res = 512
        self.img_feat_dim = self.opt.point_features_dim
        if self.opt.dataset_name == "kitti360" and self.opt.global_nerf_t == "pca": 
            x_res = 512
            y_res = 2048 
            z_res = 128
            set_parameters(
                self, (torch.rand(1, self.img_feat_dim, x_res, y_res).to() - 0.5) * 0.01, 
                f"xy") 
            set_parameters(
                self, (torch.rand(1, self.img_feat_dim, x_res, z_res).to() - 0.5) * 0.01, 
                f"xz") 
            set_parameters(
                self, (torch.rand(1, self.img_feat_dim, y_res, z_res).to() - 0.5) * 0.01, 
                f"yz")  
        else:
            set_parameters(
                self, (torch.rand(3, self.img_feat_dim, img_res, img_res).to() - 0.5) * 0.01, 
                f"planes") 
            
        self.num_pos_freq = self.opt.dist_xyz_freq
        self.pos_enc_dim = 2 * abs(self.num_pos_freq) * 3 
        self.num_input_features = self.img_feat_dim + self.pos_enc_dim 

        hidden_dim = self.opt.shading_feature_num
        if num_output_features is None: 
            num_output_features = self.opt.shading_feature_num
        num_hidden = self.opt.shading_feature_mlp_layer3

        layers = [
            nn.Linear(self.num_input_features, hidden_dim),
            nn.LeakyReLU(),
        ] 
        if num_hidden > 0:
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            ]
        layers += [
            nn.Linear(hidden_dim, num_output_features),
        ]
        self.net = nn.Sequential(*layers)

    def sample_from_planes(self, xy=None, xz=None, yz=None, coords=None, requires_double_backward=False):
        if requires_double_backward:
            # Use custom grid sample with double differentiation support
            e1 = grid_sample2d(xy, coords[..., [0, 1]])
            e2 = grid_sample2d(xz, coords[..., [0, 2]])
            e3 = grid_sample2d(yz, coords[..., [1, 2]])
        else:
            e1 = F.grid_sample(xy,
                               coords[..., [0, 1]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)
            e2 = F.grid_sample(xz,
                               coords[..., [0, 2]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)
            e3 = F.grid_sample(yz,
                               coords[..., [1, 2]],
                               mode='bilinear',
                               padding_mode='border',
                               align_corners=True)

        x = (e1 + e2 + e3) / 3
        x = x.moveaxis(1, -1)
        return x 

    # # Jax version. 
    # def contract(x):
    #     """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    #     eps = jnp.finfo(jnp.float32).eps
    #     # Clamping to eps prevents non-finite gradients when x == 0.
    #     x_mag_sq = jnp.maximum(eps, jnp.sum(x**2, axis=-1, keepdims=True))
    #     z = jnp.where(x_mag_sq <= 1, x, ((2 * jnp.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    #     return z 

    def contract(self, x): 
        eps = torch.tensor(1e-8)  # or any small positive value you choose
        x_mag_sq = torch.maximum(eps, torch.sum(x**2, dim=-1, keepdim=True))
        z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
        return z

    def forward(self, x, mean, scale, pca_rot, pca_t, pca_scale, format="bnr", act_func=None):
        """x: bnr3
        """ 
        if format == "nr":
            x = x[None]
        assert x.shape[-1] == 3
        b, n, r, _ = x.shape
        if self.opt.dataset_name == "kitti360" and self.opt.global_nerf_t == "pca": 
            # pcd_aligned = torch.matmul(points_xyz[None], rot.transpose(2, 1)) + t.transpose(2, 1)
            rot = pca_rot.transpose(2, 1)[:, None].expand(b, n, -1, -1)
            t = pca_t.transpose(2, 1)[:, None].expand(b, n, r, -1)
            s = pca_scale[None, None].expand(b, n, r, -1)
            x = x @ rot + t 
            x = x /s
        else:
            x = (x - mean) / scale # 

        if self.opt.contract_opt == "mip360":
            # print(f"before contract: min: {x.min()}, max: {x.max()}")
            x = self.contract(x) # (-2, 2)
            # print(f"After contract: min: {x.min()}, max: {x.max()}")
            x = x / 2.0
            # print(f"Input of triplane: min: {x.min()}, max: {x.max()}")

    
        if self.opt.dataset_name == "kitti360" and self.opt.global_nerf_t == "pca": 
            xy, xz, yz = self.xy, self.xz, self.yz
        else:
            xy, xz, yz = self.planes[0:1], self.planes[1:2], self.planes[2:3]

        x_pos = positional_encoding(x, freqs=self.num_pos_freq)
        feat = self.sample_from_planes(xy, xz, yz, x)
        feat = torch.cat([x_pos, feat], dim=-1)
        feat = self.net(feat)

        if format == "nr":
            feat = feat.squeeze(0)
        if act_func is not None:
            feat = act_func(feat) 
        return feat
        
        
# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out        

def pca_aligner(pcd):
    # pcd: B3N
    # pcd_can: canonicalized pcd with LRF from PCA
    pcd_mean = torch.mean(pcd, dim=2, keepdim=True) 
    pcd_centered = pcd - pcd_mean # B3N

    cov = torch.matmul(
        pcd_centered, pcd_centered.transpose(2, 1))
    U, _, _ = torch.svd(cov.cpu())
    U = U.cuda()
    # estimate local reference frame
    z = U[:, :, -1] # Get the normal: B3

    # sign disambiguity of normal
    pcd_proj_z = torch.matmul(pcd_centered.transpose(2, 1), z[:, :, None]) # BN1
    indicator = (torch.sum(pcd_proj_z, dim=1) > 0).float() # B1
    indicator = 1 - 2 * indicator
    z = indicator * z
    pcd_proj_z = indicator[:, :, None] * pcd_proj_z # BN1

    pcd_proj_plane = pcd_centered.transpose(2, 1) - pcd_proj_z * z[:, None]# BxNx3
    cov_y = torch.matmul(pcd_proj_plane.transpose(2, 1), pcd_proj_plane)# B33
    U_y, _, _ = torch.svd(cov_y.cpu())
    y = U_y.cuda()[:, :, 0] # Principle curvature: B3

    # sign disambiguity of principal curvature
    # indicator = ((pcd_proj_plane.sum(dim=1) * y).sum(dim=1, keepdim=True) > 0).float() # B1
    # indicator = 1 - 2 * indicator
    # y = indicator * y

    x = cross_product(z, y)# B3
    rots = torch.stack([x, y, z], dim=1) # 
    pcd_can = torch.matmul(rots, pcd_centered)
    return pcd_can, [rots, - torch.matmul(rots, pcd_mean)] 
    

class MlpField(nn.Module):

    def __init__(self, opt, img_res=512):
        super().__init__()
        self.opt = opt 
        self.num_pos_freq = self.opt.dist_xyz_freq
        self.pos_enc_dim = 2 * abs(self.num_pos_freq) * 3 + 3 
        self.num_input_features = self.pos_enc_dim 

        hidden_dim = self.opt.shading_feature_num
        num_output_features = self.opt.shading_feature_num
        num_hidden = self.opt.shading_feature_mlp_layer3

        layers = [
            nn.Linear(self.num_input_features, hidden_dim),
            nn.LeakyReLU(),
        ] 
        if num_hidden > 0:
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            ]
        layers += [
            nn.Linear(hidden_dim, num_output_features),
        ]
        self.net = nn.Sequential(*layers)

    def contract(self, x): 
        eps = torch.tensor(1e-8)  # or any small positive value you choose
        x_mag_sq = torch.maximum(eps, torch.sum(x**2, dim=-1, keepdim=True))
        z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
        return z

    def forward(self, x, mean, scale, pca_rot, pca_t, pca_scale):
        """x: bnr3
        """ 
        assert x.shape[-1] == 3
        b, n, r, _ = x.shape
        if self.opt.dataset_name == "kitti360" and self.opt.global_nerf_t == "pca": 
            # pcd_aligned = torch.matmul(points_xyz[None], rot.transpose(2, 1)) + t.transpose(2, 1)
            rot = pca_rot.transpose(2, 1)[:, None].expand(b, n, -1, -1)
            t = pca_t.transpose(2, 1)[:, None].expand(b, n, r, -1)
            s = pca_scale[None, None].expand(b, n, r, -1)
            x = x @ rot + t 
            x = x /s
        else:
            x = (x - mean) / scale # 

        if self.opt.contract_opt == "mip360":
            # print(f"before contract: min: {x.min()}, max: {x.max()}")
            x = self.contract(x) # (-2, 2)
            # print(f"After contract: min: {x.min()}, max: {x.max()}")
            x = x / 2.0
            # print(f"Input of triplane: min: {x.min()}, max: {x.max()}")

        feat = positional_encoding(x, freqs=self.num_pos_freq, ori=True)
        feat = self.net(feat)
        return feat