import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple, Sequence

import torch
from torch import nn, Tensor
from itertools import repeat
import collections


__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _make_ntuple(x: Any, n: int):
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        output_padding: Optional[int] = 0,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ):

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            # else:
            #     _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
            #     kernel_size = _make_ntuple(kernel_size, _conv_dim)
            #     dilation = _make_ntuple(dilation, _conv_dim)
            #     padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        if conv_layer == torch.nn.ConvTranspose2d:
            conv = conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            conv = conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
                
        layers = [
            conv
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            relu_series = [nn.ReLU, nn.ReLU6]
            params = {} if inplace not in relu_series else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        output_padding: Optional[int] = 0,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            conv_layer
        )
        
# necessary for backwards compatibility
class _DeprecatedConvBNAct(Conv2dNormActivation):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The ConvBNReLU/ConvBNActivation classes are deprecated since 0.12 and will be removed in 0.14. "
            "Use torchvision.ops.misc.Conv2dNormActivation instead.",
            FutureWarning,
        )
        if kwargs.get("norm_layer", None) is None:
            kwargs["norm_layer"] = nn.BatchNorm2d
        if kwargs.get("activation_layer", None) is None:
            kwargs["activation_layer"] = nn.ReLU6
        super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, padding: int = None, norm_layer: Optional[Callable[..., nn.Module]] = None, conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, padding=padding,  activation_layer=nn.ReLU6, conv_layer=conv_layer)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    padding=padding,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                    conv_layer=conv_layer
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class MobileNetV2(nn.Module):
    def __init__(
        self,
        last_channel: int = 1024,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        img_channel: int = 3,
        is_decoder: bool = False,
        use_ufdn: bool = False,
        code_dim: int = 2,
        variational: bool = False,
        fixed: bool = False,
        fc_hidden_dim: int = 128,
        fc_n_hidden: int = 2,
    ):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.LayerNorm

        input_channel = 32
        # last_channel = 1024  #*
        self.is_decoder = is_decoder
        self.variational = variational
        self.fixed = fixed

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            if is_decoder:
                inverted_residual_setting.reverse()

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        
        if not is_decoder:  # encoder
            # building first layer 
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            features: List[nn.Module] = [
                Conv2dNormActivation(img_channel, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            ]
            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, 
                                          norm_layer=norm_layer))         
                    input_channel = output_channel
            # building last several layers
            features.append(
                Conv2dNormActivation(
                    input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
                )
            )
        
        else:
            # building first layer
            decoder_last_channel = _make_divisible(input_channel * width_mult, round_nearest)
            embedding_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            if use_ufdn: embedding_channel *= 2
            
            _, input_channel, _, _ =  inverted_residual_setting[0]
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            if use_ufdn:
                features: List[nn.Module] = [
                    Conv2dNormActivation(embedding_channel + code_dim, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6, conv_layer=nn.ConvTranspose2d)
                ]
            else:
                features: List[nn.Module] = [
                    Conv2dNormActivation(embedding_channel, input_channel, stride=2, padding=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, conv_layer=nn.ConvTranspose2d)
                ]
            
            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting[1:]:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, output_padding=1, expand_ratio=t, 
                                          norm_layer=norm_layer, 
                                          conv_layer=nn.ConvTranspose2d))
                    input_channel = output_channel
                    
            # building last several layers
            features.append(
                Conv2dNormActivation(input_channel, decoder_last_channel, stride=2, output_padding=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, conv_layer=nn.ConvTranspose2d)
            )
            features.append(
                Conv2dNormActivation(
                    decoder_last_channel, img_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6, conv_layer=nn.Conv2d
                )
            )
            
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # self.dropout = nn.Dropout(p=dropout)
        
        if self.variational:
            self._fc_layers = []
            fc_hidden_sizes = [fc_hidden_dim] * fc_n_hidden
            input_size = self.last_channel
            for hidden_size in fc_hidden_sizes:
                output_size = hidden_size
                self._fc_layers.append(nn.Linear(input_size, output_size))
                self._fc_layers.append(nn.LayerNorm(output_size))
                    # self._fc_layers.append(nn.BatchNorm1d(output_size))
                self._fc_layers.append(nn.ReLU(True))
                input_size = output_size

            self._fc_layers = nn.Sequential(*self._fc_layers)

            self._mu_layer = nn.Sequential(
                nn.Linear(input_size, self.last_channel),
            )

            self._logvar_layer = nn.Sequential(
                nn.Linear(input_size, self.last_channel),
            )

        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
    
        feat = self.features(x)
        if not self.is_decoder:
            # if not self.use_ufdn:
            feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))  # not used for ufdn
            feat = torch.flatten(feat, 1)  # not used for ufdn
            # feat = self.dropout(feat)  # add dropout
            # x = self.mlp(x)
        
        if self.variational:
            if self.fixed:
                feat = feat.detach()
            
            feat = feat.view(feat.size(0), -1)
            feat = self._fc_layers(feat)

            mu = self._mu_layer(feat)
            logvar = self._logvar_layer(feat)

            # reparameterize
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            encoding = eps.mul(std).add_(mu)
            return encoding, (mu, logvar)
        else:
            return feat
