import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple
import math
import torch.nn.functional as F

def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        # nn.init.normal_(module.weight, 0, 0.01)
        nn.init.kaiming_normal_(
            module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(module.bias)
    

class Film(nn.Module):
    def __init__(self, content_dim, feature_dim):
        super(Film, self).__init__()
        self.gamma = nn.Linear(content_dim, feature_dim)
        self.beta = nn.Linear(content_dim, feature_dim)
    
    def forward(self, feature, content):
        """
        feature: action
        content: state
        """
        return self.gamma(content) * feature + self.beta(content) 
    

def mlp(dims, 
        activation=nn.ReLU, 
        output_activation=None, 
        squeeze_output=False, 
        use_layer_norm=False,
        dropout_rate=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if use_layer_norm:  #* add
        layers.append(nn.LayerNorm(dims[-1]))
        # layers.append(nn.BatchNorm1d(dims[-1]))
    if dropout_rate is not None:  #* add
        layers.append(nn.Dropout(dropout_rate))
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SimpleConvs(nn.Module):
    def __init__(self, 
                 output_dim=512, 
                 output_type="flatten", 
                 use_dropout=False):
        super(SimpleConvs, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=output_dim,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
        ]
        if output_type == "flatten":
            feature_output = 12 * 12 * output_dim
            layers.extend([
                Flatten(),
                nn.Linear(feature_output, output_dim)
            ])
        elif output_type == "avg_pool":
            layers.extend([
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1)
            ])
        if use_dropout:
            layers.append(nn.Dropout())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class Residual(nn.Module):
    
    def __init__(self,
                 in_channels,
                 residual_hidden_dim,
                 pre_act=False,
                 out_activation='relu'):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=residual_hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=residual_hidden_dim,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
        )
        self._pre_act = pre_act  #*
        self._out_activation = out_activation

    def forward(self, x):
        if self._pre_act:
            return (x + self._block(F.relu(x)))
        
        if self._out_activation == 'relu':
            return F.relu(x + self._block(x))
        else:
            return x + self._block(x)
        
class ResidualStack(nn.Module):
    
    def __init__(self,
                 in_channels,
                 num_residual_layers,
                 residual_hidden_dim,
                 pre_act=False,
                 out_activation='relu'):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers

        layers = []
        for i in range(self._num_residual_layers):
            if i == self._num_residual_layers - 1:
                out_activation_i = out_activation
            else:
                out_activation_i = 'relu'

            layer = Residual(in_channels=in_channels,
                             residual_hidden_dim=residual_hidden_dim,
                             out_activation=out_activation_i,
                             pre_act=pre_act)
            layers.append(layer)

        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class ResnetStack(nn.Module):
    num_ch: int
    num_blocks: int
    use_max_pooling: bool = True

    def _xavier_init(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            
    def __init__(self, num_ch: int, num_blocks: int, use_max_pooling: bool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_ch,
            kernel_size=(3, 3),
            strides=1,
            padding_mode='SAME'
        )
        
    
    # @nn.compact
    def forward(self, observations):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_ch,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME'
        )(observations)

        if self.use_max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2)
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_ch, kernel_size=(3, 3), strides=1,
                padding='SAME',
                kernel_init=initializer)(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_ch, kernel_size=(3, 3), strides=1,
                padding='SAME', kernel_init=initializer
            )(conv_out)
            conv_out += block_input

        return conv_out
    
class ImpalaEncoder(nn.Module):
    """Implement in Pytorch based on codes from ReViND"""
    def __init__(self, nn_scale: int = 1, num_residual_layers=2, use_max_pooling=True):
        super().__init__()
        in_channels = 3
        stack_sizes = [16, 32, 32]
        self.stack_blocks = []
        for out_channels in stack_sizes:
            out_channels *= nn_scale
            one_block = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                ResidualStack(
                    in_channels=out_channels,
                    num_residual_layers=num_residual_layers,
                    residual_hidden_dim=out_channels,
                    pre_act=True,
                )
            ]
            
            if use_max_pooling:
                one_block.insert(1, nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1, 
                ))
            # one_block = nn.ModuleList(one_block)
            self.stack_blocks.extend(one_block)
            in_channels = out_channels
            
        self.stack_blocks = nn.Sequential(*self.stack_blocks)
        # self.stack_blocks = [
        #     ResnetStack(
        #         num_ch=stack_sizes[0] * nn_scale,
        #         num_blocks=2),
        #     ResnetStack(
        #         num_ch=stack_sizes[1] * nn_scale,
        #         num_blocks=2),
        #     ResnetStack(
        #         num_ch=stack_sizes[2] * nn_scale,
        #         num_blocks=2),
        # ]
        self.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            
    # @nn.compact
    def forward(self, x, train=True):
        conv_out = x
        
        conv_out = self.stack_blocks(conv_out)
        conv_out = F.relu(conv_out)
        # print("out:", conv_out.reshape((*x.shape[:-3], -1)).shape)
        return conv_out.reshape((*x.shape[:-3], -1))


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate=None):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.
        Args:
            query (torch.Tensor): Query tensor (#agent, time1, size).
            key (torch.Tensor): Key tensor (#agent, time2, size).
            value (torch.Tensor): Value tensor (#agent, time2, size).
        Returns:
            torch.Tensor: Transformed query tensor, size
                (#agent, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#agent, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#agent, n_head, time2, d_k).
        """
        n_agent = query.size(0)
        q = self.linear_q(query).view(n_agent, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_agent, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_agent, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (agent, head, time1, d_k)
        k = k.transpose(1, 2)  # (agent, head, time2, d_k)
        v = v.transpose(1, 2)  # (agent, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.
        Args:
            value (torch.Tensor): Transformed value, size
                (#agent, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#agent, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#agent, 1, time2) or
                (#agent, time1, time2); (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Transformed value (#agent, time1, d_model)
                weighted by the attention score (#agent, time1, time2).
        """

        n_agent = value.size(0)
        # print(mask.shape)
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (agent, 1, *, time2)  [50, 1, 40, 40]
            # print(mask.shape)
            # print(scores.shape)  # [40, 4, 50, 50]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (agent, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (agent, head, time1, time2)

        # p_attn = self.dropout(attn)
        x = torch.matmul(attn, value)  # (agent, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_agent, -1,
                                                 self.h * self.d_k)
             )  # (agent, time1, d_model)

        return self.linear_out(x)  # (agent, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                ) -> Tuple[torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#agent, time1, size).
            key (torch.Tensor): Key tensor (#agent, time2, size).
            value (torch.Tensor): Value tensor (#agent, time2, size).
            mask (torch.Tensor): Mask tensor (#agent, 1, time2) or
                (#time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)
 
 