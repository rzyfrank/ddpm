import time
from inspect import isfunction
from torch import nn
import torch
import math

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim,dim, kernel_size=3, padding=1)
    )


def Downsample(dim):
    return nn.Conv2d(dim, dim, 3, 2, 1)

# def Upsample(dim):
#     return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
#
# def Downsample(dim):
#     return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  return '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))