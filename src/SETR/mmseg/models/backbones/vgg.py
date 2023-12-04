from typing import Union, List, Dict, Any, cast, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.runner import load_state_dict
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.utils import get_root_logger

from .helpers import load_state_dict_from_url
from ..builder import BACKBONES

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_cfgs = {
    'vgg11': 'A',
    'vgg13': 'B',
    'vgg16': 'D',
    'vgg19': 'E',
    'vgg11_bn': 'A',
    'vgg13_bn': 'B',
    'vgg16_bn': 'D',
    'vgg19_bn': 'E',
}


def map_keys(init_dict: dict, mapping: dict) -> dict:
    ret_dict = {}
    for k, v in init_dict.items():
        if k in mapping.keys():
            ret_dict[mapping[k]] = v
        else:
            ret_dict[k] = v
    return ret_dict


def get_mapping(cfg, root='features') -> dict:
    ret_dict = {}
    cnt_o = 0
    for cnt, l in enumerate(cfg):
        cnt_i = 0
        for e in l:
            if e == 'M':
                cnt_i += 1
                cnt_o += 1
            else:
                ret_dict[root + '.' + str(cnt_o) + '.weight'] = root + '.' + str(cnt) + '.' + str(cnt_i) + '.weight'
                ret_dict[root + '.' + str(cnt_o) + '.bias'] = root + '.' + str(cnt) + '.' + str(cnt_i) + '.bias'
                cnt_i += 1
                cnt_o += 1
                ret_dict[root + '.' + str(cnt_o) + '.weight'] = root + '.' + str(cnt) + '.' + str(cnt_i) + '.weight'
                ret_dict[root + '.' + str(cnt_o) + '.bias'] = root + '.' + str(cnt) + '.' + str(cnt_i) + '.bias'
                ret_dict[root + '.' + str(cnt_o) + '.running_mean'] = root + '.' + str(cnt) + '.' + str(
                    cnt_i) + '.running_mean'
                ret_dict[root + '.' + str(cnt_o) + '.running_var'] = root + '.' + str(cnt) + '.' + str(
                    cnt_i) + '.running_var'
                cnt_i += 2
                cnt_o += 2
    return ret_dict


@BACKBONES.register_module()
class VGG(nn.Module):

    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        outs = []
        for layer in self.features:
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            state_dict = load_state_dict_from_url(model_urls[pretrained])
            state_dict = map_keys(state_dict, get_mapping(cfgs[model_cfgs[pretrained]]))
            load_state_dict(self, state_dict, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layer(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg: List[List[Union[str, int]]], batch_norm: bool = False) -> nn.Module:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        layers += [make_layer(v, batch_norm=batch_norm, in_channels=in_channels)]
        in_channels = v[-2]
    return nn.ModuleList(layers)


cfgs: Dict[str, List[List[Union[str, int]]]] = {
    'A': [[64, 'M', 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'B': [[64, 64, 'M', 128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'D': [[64, 64, 'M', 128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
    'E': [[64, 64, 'M', 128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']]
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
