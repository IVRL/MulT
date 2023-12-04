# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.runner import force_fp32
from mmseg.ops import resize
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .decode_head import BaseDecodeHead
from ..builder import HEADS, build_loss
from ..losses import accuracy


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, coa, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.coa = coa
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if coa:
            self.v_se = nn.Linear(2 * dim, dim, bias=qkv_bias)
            self.v_de = nn.Linear(2 * dim, dim, bias=qkv_bias)
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.v_se = nn.Linear(dim, dim, bias=qkv_bias)
            self.qkv_de = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_se = nn.Linear(dim, dim)
        self.proj_de = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if self.coa:
            se, de, co = x
        else:
            se, de = x
        B_, N, C = se.shape
        if self.coa:
            v_se = self.v_se(torch.cat([se, co], dim=-1)).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,
                                                                                                                      2,
                                                                                                                      1,
                                                                                                                      3)
            v_de = self.v_de(torch.cat([de, co], dim=-1)).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,
                                                                                                                      2,
                                                                                                                      1,
                                                                                                                      3)
            qk = self.qk(co).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        else:
            v_se = self.v_se(se).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            qkv_de = self.qkv_de(de).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v_de = qkv_de[0], qkv_de[1], qkv_de[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        se = (attn @ v_se).transpose(1, 2).reshape(B_, N, C)
        de = (attn @ v_de).transpose(1, 2).reshape(B_, N, C)
        se = self.proj_se(se)
        de = self.proj_de(de)
        se = self.proj_drop(se)
        de = self.proj_drop(de)
        return se, de


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, coa, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.coa = coa
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_se = norm_layer(dim)
        self.norm1_de = norm_layer(dim)
        self.attn = WindowAttention(
            coa, dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_se = norm_layer(dim)
        self.norm2_de = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_se = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_de = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        if self.coa:
            se, de, co = x
        else:
            se, de = x
        B, L, C = se.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut_se = se
        shortcut_de = de
        se = self.norm1_se(se)
        de = self.norm1_de(de)
        se = se.view(B, H, W, C)
        de = de.view(B, H, W, C)
        if self.coa:
            co = co.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        se = F.pad(se, (0, 0, pad_l, pad_r, pad_t, pad_b))
        de = F.pad(de, (0, 0, pad_l, pad_r, pad_t, pad_b))
        if self.coa:
            co = F.pad(co, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = se.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_se = torch.roll(se, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_de = torch.roll(de, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.coa:
                shifted_co = torch.roll(co, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_se = se
            shifted_de = de
            if self.coa:
                shifted_co = co
            attn_mask = None

        # partition windows
        se_windows = window_partition(shifted_se, self.window_size)  # nW*B, window_size, window_size, C
        de_windows = window_partition(shifted_de, self.window_size)  # nW*B, window_size, window_size, C
        se_windows = se_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        de_windows = de_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if self.coa:
            co_windows = window_partition(shifted_co, self.window_size)  # nW*B, window_size, window_size, C
            co_windows = co_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.coa:
            x_windows = (se_windows, de_windows, co_windows)
        else:
            x_windows = (se_windows, de_windows)
        se_windows, de_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        se_windows = se_windows.view(-1, self.window_size, self.window_size, C)
        de_windows = de_windows.view(-1, self.window_size, self.window_size, C)
        shifted_se = window_reverse(se_windows, self.window_size, Hp, Wp)  # B H' W' C
        shifted_de = window_reverse(de_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            se = torch.roll(shifted_se, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            de = torch.roll(shifted_de, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            se = shifted_se
            de = shifted_de

        if pad_r > 0 or pad_b > 0:
            se = se[:, :H, :W, :].contiguous()
            de = de[:, :H, :W, :].contiguous()

        se = se.view(B, H * W, C)
        de = de.view(B, H * W, C)

        # FFN
        se = shortcut_se + self.drop_path(se)
        de = shortcut_de + self.drop_path(de)
        se = se + self.drop_path(self.mlp_se(self.norm2_se(se)))
        de = de + self.drop_path(self.mlp_de(self.norm2_de(de)))

        return se, de


class Upsampler(nn.Module):
    """ Upsampling Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, align_corners, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.align_corners = align_corners
        self.reduction = nn.Linear(dim, dim // 2, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W, nextHW):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.transpose(1, 2).view(B, C, H, W)
        x = resize(
            x,
            size=nextHW,
            mode='bilinear',
            align_corners=self.align_corners)
        x = x.view(B, C, nextHW[0] * nextHW[1]).transpose(1, 2)  # B H*2*W*2 C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 align_corners,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                coa=True if i == 0 else False,
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample_se = upsample(dim=dim, align_corners=align_corners, norm_layer=norm_layer)
            self.upsample_de = upsample(dim=dim, align_corners=align_corners, norm_layer=norm_layer)
        else:
            self.upsample_se = None

    def forward(self, xHW, coHW, nextHW):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        se, de, H, W = xHW
        co, coH, coW = coHW
        assert H == coH and W == coW, "input and context have different HW"
        assert torch.equal(torch.Tensor([*se.shape]),
                           torch.Tensor([*co.shape])), "input and context have different shape"

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=se.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk_enum, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            x = (se, de, co) if blk_enum == 0 else (se, de)
            if self.use_checkpoint:
                se, de = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                se, de = blk(x, attn_mask)
        if self.upsample_se is not None:
            se_down = self.upsample_se(se, H, W, nextHW)
            de_down = self.upsample_de(de, H, W, nextHW)
            Wh, Ww = nextHW
            return se, de, H, W, se_down, de_down, Wh, Ww
        else:
            return se, de, H, W, se, de, H, W


def to_2D(x, H, W):
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    x = x.transpose(1, 2).view(B, C, H, W)  # B C H*W -> B C H W
    return x


def to_1D(x):
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)  # B C H*W -> B H*W C
    return x, H, W


@HEADS.register_module()
class SwinTransformerDEA2(BaseDecodeHead):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 depths=[2, 2, 2, 2],
                 num_heads=[24, 12, 6, 3],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.num_layers = len(depths)
        self.depths = depths

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.in_channels[-1 - i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                align_corners=self.align_corners,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Upsampler if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.conv_de = nn.Conv2d(self.channels, 1, kernel_size=1)

        self.loss_dep = build_loss(dict(type='DepthLoss', dataset='NYU', height=464, max_depth=10.0, loss_weight=1.0))

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        # TODO: normal_init conv_seg & conv_de?

    def cls_de(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_de(feat)
        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)
        inputs = [to_1D(input) for input in inputs]

        x, Wh, Ww = inputs[-1]
        B, L, C = x.shape
        se, de = x, x
        assert L == Wh * Ww, "input feature has wrong size"
        assert len(inputs) == self.num_layers, "wrong number of skip connections/layers"
        for i in range(self.num_layers):
            layer = self.layers[i]
            se_out, de_out, H, W, se, de, Wh, Ww = layer((se, de, Wh, Ww), inputs[-1 - i],
                                                         inputs[-2 - i][1:] if (i < self.num_layers - 1) else (2 * e for
                                                                                                               e in
                                                                                                               inputs[
                                                                                                                   0][
                                                                                                               1:]))

        output_se = to_2D(se_out, H, W)
        output_de = to_2D(de_out, H, W)

        output_se = self.cls_seg(output_se)
        output_de = self.cls_de(output_de)

        return output_se, output_de

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, output, gt):
        """Compute segmentation loss."""
        seg_logit, dep_out = output[0], output[1]
        seg_label, dep_map = gt[0], gt[1]

        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        dep_out = resize(
            input=dep_out,
            size=dep_map.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            dep_weight = self.sampler.sample(dep_out, dep_map)
        else:
            seg_weight = None
            dep_weight = None
        loss['loss_dep'] = self.loss_dep(
            dep_out,
            dep_map,
            weight=dep_weight,
            ignore_index=self.ignore_index)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
