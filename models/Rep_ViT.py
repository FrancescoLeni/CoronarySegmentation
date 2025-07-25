
# all adapted from edgeSAM

import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from models.utility_blocks import SE3D



__all__ = ['RepViT', 'LayerNorm2d']

# I think they used m1
m1_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 48, 1, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 96, 0, 0, 2],
    [3, 2, 96, 1, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 192, 0, 1, 2],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 384, 0, 1, 2],
    [3, 2, 384, 1, 1, 1],
    [3, 2, 384, 0, 1, 1]
]

m2_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1]
]

m3_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 1, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 64, 0, 0, 1],
    [3, 2, 128, 0, 0, 2],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 1, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 128, 0, 0, 1],
    [3, 2, 256, 0, 1, 2],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 1, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 256, 0, 1, 1],
    [3, 2, 512, 0, 1, 2],
    [3, 2, 512, 1, 1, 1],
    [3, 2, 512, 0, 1, 1]
]


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def resize(
        x: torch.Tensor,
        size: any or None = None,
        scale_factor=None,
        mode: str = "bicubic",
        align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in ["bilinear", "bicubic"]:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in ["nearest", "area"]:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class UpSampleLayer(nn.Module):
    def __init__(
            self,
            mode="bicubic",
            size=None,
            factor=2,
            align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class OpSequential(nn.Module):
    def __init__(self, op_list):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Conv3d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class RepVGGDW3D(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv3d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv3d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, skip_downsample=False):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            if skip_downsample:
                stride = 1
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTBlock3D(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, skip_downsample=False):
        super(RepViTBlock3D, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            if skip_downsample:
                stride = 1
            self.token_mixer = nn.Sequential(
                Conv3d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SE3D(inp, 0.25) if use_se else nn.Identity(),
                Conv3d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv3d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv3d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW3D(inp),
                SE3D(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv3d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv3d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViT(nn.Module):
    arch_settings = {
        'm1': m1_cfgs,
        'm2': m2_cfgs,
        'm3': m3_cfgs
    }

    def __init__(self, arch, img_size=1024, fuse=False, freeze=False,
                 load_from=None, use_rpn=False, out_indices=['stem', 'stage0', 'stage1', 'final'], upsample_mode='bicubic'):
        super(RepViT, self).__init__()

        self.input_ch = 2  # also graph dim
        # setting of inverted residual blocks
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.fuse = fuse
        self.freeze = freeze
        self.use_rpn = use_rpn
        self.out_indices = out_indices

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(self.input_ch, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            skip_downsample = False
            if not self.fuse and c in [384, 512]:
                skip_downsample = True
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, skip_downsample))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)

        if self.fuse:
            stage2_channels = _make_divisible(self.cfgs[self.stage_idx[2]][2], 8)
            stage3_channels = _make_divisible(self.cfgs[self.stage_idx[3]][2], 8)
            self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode=upsample_mode),
            ])
            neck_in_channels = 256
        else:
            neck_in_channels = output_channel
        self.neck = nn.Sequential(
            nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        if load_from is not None:
            state_dict = torch.load(load_from)['model']
            new_state_dict = dict()
            use_new_dict = False
            for key in state_dict:
                if key.startswith('image_encoder.'):
                    use_new_dict = True
                    new_key = key[len('image_encoder.'):]
                    new_state_dict[new_key] = state_dict[key]
            if use_new_dict:
                state_dict = new_state_dict
            print(self.load_state_dict(state_dict, strict=True))

    # def train(self, mode=True):
    #     super(RepViT, self).train(mode)
    #     if self.freeze:
    #         self.features.eval()
    #         self.neck.eval()
    #         for param in self.parameters():
    #             param.requires_grad = False

    def forward(self, x):
        counter = 0
        output_dict = dict()
        # patch_embed
        x = self.features[0](x)
        output_dict['stem'] = x
        # stages
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1

        if self.fuse:
            x = self.fuse_stage2(output_dict['stage2']) + self.fuse_stage3(output_dict['stage3'])

        x = self.neck(x)
        output_dict['final'] = x

        if self.use_rpn:
            if self.out_indices is None:
                self.out_indices = [len(output_dict) - 1]
            out = [output_dict[k] for k in output_dict.keys() if k in self.out_indices]
            return tuple(out)
        return x


class Conv2d_BN_UP(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.ConvTranspose2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Conv3d_BN_UP(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.ConvTranspose3d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm3d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepViTUP(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [2]

        self.token_mixer = nn.Sequential(
            Conv2d_BN_UP(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
        )
        self.channel_mixer = Residual(nn.Sequential(
            # pw
            Conv2d_BN(oup, 2 * oup, 1, 1, 0),
            nn.GELU() if use_hs else nn.GELU(),
            # pw-linear
            Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTupCat(nn.Module):
    def __init__(self, c):
        super().__init__()
        c_ = c//2
        self.up = RepViTUP(c, c_, 3, 2, True, True)
        self.pw = nn.Conv2d(c, c_, 1)

    def forward(self, x):
        x, x_cat = x
        x = self.up(x)
        x = torch.cat([x, x_cat], dim=1)
        x = self.pw(x)
        return x


class RepViTUP3D(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [2]

        self.token_mixer = nn.Sequential(
            Conv3d_BN_UP(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
            SE3D(inp, 0.25) if use_se else nn.Identity(),
            Conv3d_BN(inp, oup, ks=1, stride=1, pad=0)
        )
        self.channel_mixer = Residual(nn.Sequential(
            # pw
            Conv3d_BN(oup, 2 * oup, 1, 1, 0),
            nn.GELU() if use_hs else nn.GELU(),
            # pw-linear
            Conv3d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
        ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class RepViTupCat3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        c_ = c//2
        self.up = RepViTUP3D(c, c_, 3, 2, True, True)
        self.pw = nn.Conv3d(c, c_, 1)

    def forward(self, x):
        x, x_cat = x
        x = self.up(x)
        x = torch.cat([x, x_cat], dim=1)
        x = self.pw(x)
        return x


class RepViTDecoder(nn.Module):
    def __init__(self, n_classes, upsample_stem=True):
        super().__init__()

        self.stages = nn.ModuleList([RepViTBlock(128, 256, 128, 3, 1, True, True),
                                     RepViTBlock(64, 128, 64, 3, 1, True, True),
                                     RepViTBlock(32, 64, 32, 3, 1, True, True),
                                     RepViTBlock(8, 16, 8, 3, 1, True, True)])

        self.ups = nn.ModuleList([RepViTupCat(256),
                                  RepViTupCat(128),
                                  RepViTupCat(64),
                                  RepViTUP(32,8,3,2,True,True)])

        self.head = nn.Conv2d(8, n_classes, 1)

        if upsample_stem:
            self.up_stem = RepViTUP(64,32,3,2,True,True)
        else:
            self.ups[2] = RepViTUP(64,32,3,2,True,True)
            self.up_stem = None


    def forward(self, x):
        # RepViT encoder use_rpn = True
        x_stem, x0, x1, x_f = x

        x = self.stages[0](self.ups[0]((x_f, x1)))
        x = self.stages[1](self.ups[1]((x, x0)))
        x = self.ups[2]((x, self.up_stem(x_stem))) if self.up_stem else self.ups[2](x)
        x = self.stages[2](x)
        x = self.stages[3](self.ups[3](x))

        return self.head(x)


class RepViTUnet(nn.Module):
    # use_rpn is == True to allow for decoder
    def __init__(self, arch='m2', n_classes=2, img_size=1024, fuse=False, freeze=False,
                 load_from=None, use_rpn=True, out_indices=['stem', 'stage0', 'stage1', 'final'], upsample_mode='bicubic', upsample_stem=True):
        super().__init__()

        self.name = 'RepViT'

        self.encoder = RepViT(arch, img_size, fuse, freeze, load_from, use_rpn, out_indices, upsample_mode)
        self.decoder = RepViTDecoder(n_classes, upsample_stem)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class RepViT3D(nn.Module):
    arch_settings = {
        'm1': m1_cfgs,
        'm2': m2_cfgs,
        'm3': m3_cfgs
    }

    def __init__(self, arch, img_size=1024, fuse=False, freeze=False,
                 load_from=None, use_rpn=False, out_indices=['stem', 'stage0', 'stage1', 'final'], upsample_mode='bicubic'):
        super(RepViT3D, self).__init__()

        self.input_ch = 2  # also graph dim
        # setting of inverted residual blocks
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.fuse = fuse
        self.freeze = freeze
        self.use_rpn = use_rpn
        self.out_indices = out_indices

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv3d_BN(self.input_ch, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv3d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock3D
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            skip_downsample = False
            if not self.fuse and c in [384, 512]:
                skip_downsample = True
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, skip_downsample))
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)

        if self.fuse:
            stage2_channels = _make_divisible(self.cfgs[self.stage_idx[2]][2], 8)
            stage3_channels = _make_divisible(self.cfgs[self.stage_idx[3]][2], 8)
            self.fuse_stage2 = nn.Conv3d(stage2_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv3d(stage3_channels, 256, kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode=upsample_mode),
            ])
            neck_in_channels = 256
        else:
            neck_in_channels = output_channel
        self.neck = nn.Sequential(
            nn.Conv3d(neck_in_channels, 256, kernel_size=1, bias=False),
            LayerNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(256),
        )

        if load_from is not None:
            state_dict = torch.load(load_from)['model']
            new_state_dict = dict()
            use_new_dict = False
            for key in state_dict:
                if key.startswith('image_encoder.'):
                    use_new_dict = True
                    new_key = key[len('image_encoder.'):]
                    new_state_dict[new_key] = state_dict[key]
            if use_new_dict:
                state_dict = new_state_dict
            print(self.load_state_dict(state_dict, strict=True))

    # def train(self, mode=True):
    #     super(RepViT, self).train(mode)
    #     if self.freeze:
    #         self.features.eval()
    #         self.neck.eval()
    #         for param in self.parameters():
    #             param.requires_grad = False

    def forward(self, x):
        counter = 0
        output_dict = dict()
        # patch_embed
        x = self.features[0](x)
        output_dict['stem'] = x
        # stages
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f'stage{counter}'] = x
                counter += 1

        if self.fuse:
            x = self.fuse_stage2(output_dict['stage2']) + self.fuse_stage3(output_dict['stage3'])

        x = self.neck(x)
        output_dict['final'] = x

        if self.use_rpn:
            if self.out_indices is None:
                self.out_indices = [len(output_dict) - 1]
            out = [output_dict[k] for k in output_dict.keys() if k in self.out_indices]
            return tuple(out)
        return x


class RepViTDecoder3D(nn.Module):
    def __init__(self, n_classes, upsample_stem=True):
        super().__init__()

        self.stages = nn.ModuleList([RepViTBlock3D(128, 256, 128, 3, 1, True, True),
                                     RepViTBlock3D(64, 128, 64, 3, 1, True, True),
                                     RepViTBlock3D(32, 64, 32, 3, 1, True, True),
                                     RepViTBlock3D(8, 16, 8, 3, 1, True, True)])

        self.ups = nn.ModuleList([RepViTupCat3D(256),
                                  RepViTupCat3D(128),
                                  RepViTupCat3D(64),
                                  RepViTUP3D(32,8,3,2,True,True)])

        self.head = nn.Conv3d(8, n_classes, 1)

        if upsample_stem:
            self.up_stem = RepViTUP3D(64,32,3,2,True,True)
        else:
            self.ups[2] = RepViTUP3D(64,32,3,2,True,True)
            self.up_stem = None


    def forward(self, x):
        # RepViT encoder use_rpn = True
        x_stem, x0, x1, x_f = x

        x = self.stages[0](self.ups[0]((x_f, x1)))
        x = self.stages[1](self.ups[1]((x, x0)))
        x = self.ups[2]((x, self.up_stem(x_stem))) if self.up_stem else self.ups[2](x)
        x = self.stages[2](x)
        x = self.stages[3](self.ups[3](x))

        return self.head(x)


class RepViTUnet3D(nn.Module):
    # use_rpn is == True to allow for decoder
    def __init__(self, arch='m2', n_classes=2, img_size=1024, fuse=False, freeze=False,
                 load_from=None, use_rpn=True, out_indices=['stem', 'stage0', 'stage1', 'final'], upsample_mode='bicubic', upsample_stem=True):
        super().__init__()

        self.name = 'RepViT3D'

        self.encoder = RepViT3D(arch, img_size, fuse, freeze, load_from, use_rpn, out_indices, upsample_mode)
        self.decoder = RepViTDecoder3D(n_classes, upsample_stem)

    def forward(self, x):
        return self.decoder(self.encoder(x))