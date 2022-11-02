import torch
from torch import nn
from .tool import default, SinusoidalPositionEmbeddings, Residual, Upsample, Downsample, exists
from .attension import Attention
from .resnet import ResnetBlock
from functools import partial


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# class Unet(nn.Module):
#     def __init__(
#             self,
#             dim,
#             init_dim=None,
#             out_dim=None,
#             dim_mults=(1, 2, 4, 8),
#             channels=3,
#             with_time_emb=True,
#             resnet_block_groups=8,
#             use_convnext=True,
#             convnext_mult=2,
#     ):
#         super().__init__()
#
#         # determine dimensions
#         self.channels = channels
#
#         init_dim = default(init_dim, dim)
#         self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
#
#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))
#
#         if use_convnext:
#             block_klass = partial(ConvNextBlock, mult=convnext_mult)
#         else:
#             block_klass = partial(ResnetBlock, groups=resnet_block_groups)
#
#         # time embeddings
#         if with_time_emb:
#             time_dim = dim * 4
#             self.time_mlp = nn.Sequential(
#                 SinusoidalPositionEmbeddings(dim),
#                 nn.Linear(dim, time_dim),
#                 nn.GELU(),
#                 nn.Linear(time_dim, time_dim),
#             )
#         else:
#             time_dim = None
#             self.time_mlp = None
#
#         # layers
#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)
#
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)
#
#             self.downs.append(
#                 nn.ModuleList(
#                     [
#                         block_klass(dim_in, dim_out, time_emb_dim=time_dim),
#                         block_klass(dim_out, dim_out, time_emb_dim=time_dim),
#                         Residual(PreNorm(dim_out, LinearAttention(dim_out))),
#                         Downsample(dim_out) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )
#
#         mid_dim = dims[-1]
#         self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
#         self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
#
#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 1)
#
#             self.ups.append(
#                 nn.ModuleList(
#                     [
#                         block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
#                         block_klass(dim_in, dim_in, time_emb_dim=time_dim),
#                         Residual(PreNorm(dim_in, LinearAttention(dim_in))),
#                         Upsample(dim_in) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )
#
#         out_dim = default(out_dim, channels)
#         self.final_conv = nn.Sequential(
#             block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
#         )
#
#     def forward(self, x, time):
#         x = self.init_conv(x)
#
#         t = self.time_mlp(time) if exists(self.time_mlp) else None
#
#         h = []
#
#         # downsample
#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             x = block2(x, t)
#             x = attn(x)
#             h.append(x)
#             x = downsample(x)
#
#         # bottleneck
#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)
#
#         # upsample
#         for block1, block2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim=1)
#             x = block1(x, t)
#             x = block2(x, t)
#             x = attn(x)
#             x = upsample(x)
#
#         return self.final_conv(x)
class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, Attention(dim_in))) if dim_out == 128 else nn.Identity(),
                        # Residual(PreNorm(dim_out, Attention(dim_out))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )
        # print(self.downs)
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, Attention(dim_out))) if dim_out == 128 else nn.Identity(),
                        # Residual(PreNorm(dim_in, Attention(dim_in))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        # print(self.ups)
        out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, out_dim, 1)
        # self.final_conv = nn.Sequential(
        #     block_klass(dim * 2, dim, time_emd_dim=time_dim),
        #     nn.Conv2d(dim, out_dim, 1)
        # )

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = attn(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = attn(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


if __name__ == '__main__':
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4, 8)
    )
