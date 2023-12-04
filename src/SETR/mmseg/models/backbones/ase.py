import math
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from .helpers import load_pretrained
from .layers import DropPath, to_2tuple, trunc_normal_
from ..builder import BACKBONES


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_base_p16_224-4e355ebd.pth',
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
    'deit_base_distilled_path16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, checkpoint=True,
    ),
}


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.v_de = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_se = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_se = nn.Linear(dim, dim)
        self.proj_de = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, se, de):
        B, N, C = se.shape
        v_de = self.v_de(de).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_se, k_se, v_se = self.qkv_se(se).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q_se @ k_se.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        se = (attn @ v_se).transpose(1, 2).reshape(B, N, C)
        se = self.proj_se(se)
        se = self.proj_drop(se)

        de = (attn @ v_de).transpose(1, 2).reshape(B, N, C)
        de = self.proj_de(de)
        de = self.proj_drop(de)
        return se, de


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1_se = norm_layer(dim)
        self.norm1_de = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_se = norm_layer(dim)
        self.mlp_se = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2_de = norm_layer(dim)
        self.mlp_de = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        se, de = x[0], x[1]
        se, de = self.attn(self.norm1_se(se), self.norm1_de(de))
        se, de = x[0] + self.drop_path(se), x[1] + self.drop_path(de)
        se = se + self.drop_path(self.mlp_se(self.norm2_se(se)))
        de = de + self.drop_path(self.mlp_de(self.norm2_de(de)))
        return se, de


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
                      (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Conv_MLA(nn.Module):
    def __init__(self, in_channels=1024, mla_channels=256, norm_cfg=None):
        super(Conv_MLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.mla_p5 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, res2, res3, res4, res5):
        res2 = self.to_2D(res2)
        res3 = self.to_2D(res3)
        res4 = self.to_2D(res4)
        res5 = self.to_2D(res5)

        mla_p5_1x1 = self.mla_p5_1x1(res5)
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_plus + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4, mla_p5


@BACKBONES.register_module()
class Ase(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024,
                 depth=24,
                 num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1,
                 attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                 pos_embed_interp=False, random_init=False, align_corners=False, mla_channels=256,
                 mla_index=(5, 11, 17, 23), **kwargs):
        super(Ase, self).__init__(**kwargs)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners
        self.mla_channels = mla_channels
        self.mla_index = mla_index

        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=self.img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])

        self.mla_se = Conv_MLA(in_channels=self.embed_dim,
                               mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)
        self.mla_de = Conv_MLA(in_channels=self.embed_dim,
                               mla_channels=self.mla_channels, norm_cfg=self.norm_cfg)

        self.norm_se_0 = norm_layer(self.embed_dim)
        self.norm_se_1 = norm_layer(self.embed_dim)
        self.norm_se_2 = norm_layer(self.embed_dim)
        self.norm_se_3 = norm_layer(self.embed_dim)
        self.norm_de_0 = norm_layer(self.embed_dim)
        self.norm_de_1 = norm_layer(self.embed_dim)
        self.norm_de_2 = norm_layer(self.embed_dim)
        self.norm_de_3 = norm_layer(self.embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        # nn.init.normal_(self.pos_embed, std=0.02)
        # nn.init.zeros_(self.cls_token)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if not self.random_init:
            self.default_cfg = default_cfgs[self.model_name]

            if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans,
                                pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners,
                                filter_fn=self._conv_filter)
            else:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans,
                                pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners,
                                filter_fn=self.map_state_dict)
        else:
            print('Initialize weight randomly')

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def map_state_dict(self, init_dict: dict) -> dict:
        ret_dict = {}
        for k, v in init_dict.items():
            if k[:7] == 'blocks.':
                ks = k.split('.')
                if ks[2] in ['norm1', 'norm2', 'mlp']:
                    ks[2] = ks[2] + '_se'
                    ret_dict['.'.join(ks)] = v
                    ks[2] = ks[2][:-3] + '_de'
                    ret_dict['.'.join(ks)] = v
                elif ks[2] == 'attn':
                    if ks[3] == 'proj':
                        ks[3] = 'proj_se'
                        ret_dict['.'.join(ks)] = v
                        ks[3] = 'proj_de'
                        ret_dict['.'.join(ks)] = v
                    elif ks[3] == 'qkv':
                        ks[3] = 'qkv_se'
                        ret_dict['.'.join(ks)] = v
                        ks[3] = 'v_de'
                        ret_dict['.'.join(ks)] = v[-1024:]
                    else:
                        ret_dict[k] = v
                else:
                    ret_dict[k] = v
            else:
                ret_dict[k] = v
        return ret_dict

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = x[:, 1:]
        x = self.pos_drop(x)

        se, de = x, x

        outs_se = []
        outs_de = []
        for i, blk in enumerate(self.blocks):
            se, de = blk((se, de))
            if i in self.out_indices:
                outs_se.append(se)
                outs_de.append(de)

        c6_se = self.norm_se_0(outs_se[self.mla_index[0]])
        c12_se = self.norm_se_1(outs_se[self.mla_index[1]])
        c18_se = self.norm_se_2(outs_se[self.mla_index[2]])
        c24_se = self.norm_se_3(outs_se[self.mla_index[3]])
        c6_de = self.norm_de_0(outs_de[self.mla_index[0]])
        c12_de = self.norm_de_1(outs_de[self.mla_index[1]])
        c18_de = self.norm_de_2(outs_de[self.mla_index[2]])
        c24_de = self.norm_de_3(outs_de[self.mla_index[3]])

        p6_se, p12_se, p18_se, p24_se = self.mla_se(c6_se, c12_se, c18_se, c24_se)
        p6_de, p12_de, p18_de, p24_de = self.mla_de(c6_de, c12_de, c18_de, c24_de)

        return (p6_se, p12_se, p18_se, p24_se), (p6_de, p12_de, p18_de, p24_de)
