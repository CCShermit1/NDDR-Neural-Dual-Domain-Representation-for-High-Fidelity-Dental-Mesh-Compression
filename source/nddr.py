from __future__ import annotations

import meshio
import logging
import numpy as np
import torch
import torch.nn as nn

from typing import Callable
    
class SIREN(nn.Module):
    """SIREN (Sinusoidal Representation Networks) for high-frequency function fitting"""
    def __init__(self, ffin: int, hidden_dim: int = 64, num_layers: int = 3, w0: float = 30.0) -> None:
        super(SIREN, self).__init__()
        
        self.w0 = w0
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(ffin, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, 3))
        
        # Initialize weights according to SIREN paper
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights according to SIREN paper recommendations"""
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer: uniform distribution in [-1/n, 1/n] where n is input size
                n = layer.weight.shape[1]
                layer.weight.data.uniform_(-1/n, 1/n)
            else:
                # Hidden layers: uniform distribution in [-sqrt(6/n)/w0, sqrt(6/n)/w0]
                n = layer.weight.shape[1]
                bound = np.sqrt(6.0 / n) / self.w0
                layer.weight.data.uniform_(-bound, bound)
            
            if layer.bias is not None:
                layer.bias.data.zero_()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.sin(self.w0 * x)  # SIREN activation function
        x = self.layers[-1](x)  # No activation on output layer
        return x
    
    def stream(self):
        """Convert SIREN network to byte stream"""
        weights = [layer.weight.data.cpu() for layer in self.layers]
        biases = [layer.bias.data.cpu() for layer in self.layers]

        bytestream = b''
        for w in weights:
            bytestream += w.shape[0].to_bytes(4, 'little')
            bytestream += w.shape[1].to_bytes(4, 'little')
            bytestream += w.numpy().astype('float32').tobytes()

        for b in biases:
            bytestream += b.shape[0].to_bytes(4, 'little')
            bytestream += b.numpy().astype('float32').tobytes()

        return bytestream


# Positional encoding
def positional_encoding(vector: torch.Tensor, extras: list[torch.Tensor], levels: int) -> torch.Tensor:
    # result = extras
    # for i in range(levels):
    #     k = 2 ** i
    #     result += [torch.sin(k * vector), torch.cos(k * vector)]

    if not hasattr(positional_encoding, '_rff_B'):
        device = vector.device
        in_dim = vector.shape[-1]          
        m = levels * in_dim                
        sigma = 4.0                        
        positional_encoding._rff_B = torch.normal(
            mean=0.0,
            std=sigma,
            size=(m, in_dim),
            device=device
        )
    
    B = positional_encoding._rff_B         # [m, d]
    proj = 2 * np.pi * (vector @ B.t())    # [N, m]
    result += [torch.sin(proj), torch.cos(proj)]
    return torch.cat(result, dim=-1)

class NDDR:
    def __init__(self,
                 points: torch.Tensor,
                 features: torch.Tensor,
                 complexes: torch.Tensor,
                 fflevels: int,
                 jittering: bool,
                 normals: bool,
                 mlp=None) -> None:
        self.points = points
        self.features = features
        self.complexes = complexes
        self.fflevels = fflevels
        self.jittering = jittering
        self.normals = normals

        self.ffin = self.features.shape[-1] + 3 * 2 * self.fflevels
        
        
        self.mlp = SIREN(self.ffin, hidden_dim=64, num_layers=3, w0=10.0).cuda()#siren - SIREN network for high-frequency fitting


        if mlp is not None:
            self.mlp.load_state_dict(mlp.state_dict())

        self.sampler = self.sample_uniform
        if self.jittering:
            self.sampler = self.sample_jittered

        # Caches
        self.uv_cache = {}
        self.uv_mask = {}

        # Log details
        logging.info('Instantiated neural geometry field with properties:')
        logging.info(f'     Vertex count: {self.points.shape[0]}')
        logging.info(f'     Quad count:   {self.complexes.shape[0]}')
        logging.info(f'     Feature size: {self.features.shape[-1]}')
        logging.info(f'     FF levels:    {fflevels}')
        logging.info(f'     Jittering:    {jittering}')
        logging.info(f'     Normals:      {normals}')

    # List of parameters
    def parameters(self):
        return list(self.mlp.parameters()) + [self.points, self.features]

    # Number of patches
    def patches(self) -> int:
        return self.complexes.shape[0]

    @staticmethod
    def interpolate(attrs, complexes, U, V):
        attr_size = attrs.shape[1]
        cattrs = attrs[complexes]

        Up, Um = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        Vp, Vm = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = cattrs[:, 0, :].unsqueeze(1) * Um * Vm
        lp01 = cattrs[:, 1, :].unsqueeze(1) * Up * Vm
        lp10 = cattrs[:, 3, :].unsqueeze(1) * Um * Vp
        lp11 = cattrs[:, 2, :].unsqueeze(1) * Up * Vp

        return (lp00 + lp01 + lp10 + lp11).reshape(-1, attr_size)

    def eval(self, *uvs):
        lp = NDDR.interpolate(self.points, self.complexes, *uvs)
        lf = NDDR.interpolate(self.features, self.complexes, *uvs)
        lin = positional_encoding(lp, [lf], self.fflevels)
        return lp + self.mlp(lin)

    def base(self, rate):
        uvs = self.sample_uniform(rate)
        return NDDR.interpolate(self.points, self.complexes, *uvs)

    # Sampling functions
    def sample_uniform(self, rate: int):
        if rate in self.uv_cache:
            return self.uv_cache[rate]

        U = torch.linspace(0.0, 1.0, steps=rate, device='cuda')
        V = torch.linspace(0.0, 1.0, steps=rate, device='cuda')
        U, V = torch.meshgrid(U, V, indexing='ij')

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        self.uv_cache[rate] = (U, V)

        return U, V

    def sample_jittered(self, rate: int):
        U, V = self.sample_uniform(rate)

        if rate in self.uv_mask:
            UV_interior = self.uv_mask[rate]
        else:
            U_plus, U_minus = U, (1.0 - U)
            V_plus, V_minus = V, (1.0 - V)
            UV_interior = (U_minus * U_plus * V_minus * V_plus) > 0
            self.uv_mask[rate] = UV_interior

        delta = 0.45/(rate - 1)

        rtheta = 2 * np.pi * torch.rand(*U.shape, device='cuda')
        rr = torch.rand(*U.shape, device='cuda').sqrt()
        ru = delta * rr * rtheta.cos() * UV_interior
        rv = delta * rr * rtheta.sin() * UV_interior

        return U + ru, V + rv

    def save(self, filename):
        """Save into a PyTorch (PT) file"""
        torch.save({
            'points': self.points,
            'features': self.features,
            'complexes': self.complexes,
            'fflevels': self.fflevels,
            'jittering': self.jittering,
            'normals': self.normals,
            'model': self.mlp,
        }, filename)

    def stream(self):
        """Convert neural geometry field to byte stream"""
        sizes = [self.complexes.shape[0], self.points.shape[0], self.features.shape[-1]]
        size_bytes = np.array(sizes, dtype=np.int32).tobytes()

        with torch.no_grad():
            points_bytes = self.points.cpu().numpy().tobytes()
            features_bytes = self.features.cpu().numpy().tobytes()
            complexes_bytes = self.complexes.cpu().numpy().tobytes()

        mlp_bytes = self.mlp.stream()

        return size_bytes + points_bytes + features_bytes + complexes_bytes + mlp_bytes

    @staticmethod
    def from_base(path: str, normalizer: Callable, features: int, config: dict = dict()) -> NDDR:
        mesh = meshio.read(path)
        points = torch.from_numpy(mesh.points)
        complexes = torch.from_numpy(mesh.cells_dict['quad'])

        points = normalizer(points.float().cuda())
        features = torch.zeros((points.shape[0], features)).cuda()
        complexes = complexes.int().cuda()

        points.requires_grad = True
        features.requires_grad = True

        return NDDR(points, features, complexes,
                   config.setdefault('fflevels', 8),
                   config.setdefault('jittering', True),
                   config.setdefault('normals', True))

    @staticmethod
    def from_pt(path: str) -> NDDR:
        data = torch.load(path)
        return NDDR(data['points'],
                   data['features'],
                   data['complexes'],
                   data['fflevels'],
                   data['jittering'],
                   data['normals'],
                   mlp=data['model'])
