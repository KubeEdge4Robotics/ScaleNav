import numpy as np
import torch
import torchvision  # NOQA
from torch import nn
from utils.trainer_utils import get_numpy, from_numpy
from models.utils import _init_weights

class ObsEncoder(nn.Module):
    
    def encode(self, inputs):
        return self.forward(inputs)

    def encode_np(self, inputs, cont=True):
        assert cont is True
        return get_numpy(self.forward(from_numpy(inputs)))

    def encode_one_np(self, inputs, cont=True):
        inputs = inputs[None, ...]
        outputs = self.encode_np(inputs)
        return outputs[0]

    def decode_np(self, inputs, cont=True):
        raise NotImplementedError

    def decode_one_np(self, inputs, cont=True):
        raise NotImplementedError
    
class VqvaeVariationalObsEncoder(ObsEncoder):
    
    def __init__(
            self,
            output_dim,
            input_width=256,
            input_height=256,
            input_channels=3,

            vqvae=None,
            fixed=True,
            
            fc_hidden_dim=128,
            fc_n_hidden=2,
            normalize=False
    ):
        super(VqvaeVariationalObsEncoder, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_dim = (
            self.input_channels * self.input_width * self.input_height)

        self.fixed = fixed
        self.normalize = normalize
        if self.fixed:
            for param in vqvae.parameters():
                param.requires_grad = False

        self._vqvae = vqvae

        input_size = self._vqvae.representation_size


        self._fc_layers = []
        fc_hidden_sizes = [fc_hidden_dim] * fc_n_hidden
        for hidden_size in fc_hidden_sizes:
            output_size = hidden_size
            self._fc_layers.append(nn.Linear(input_size, output_size))
            if normalize:
                self._fc_layers.append(nn.LayerNorm(output_size))
                # self._fc_layers.append(nn.BatchNorm1d(output_size))
            self._fc_layers.append(nn.ReLU(True))
            input_size = output_size

        self._fc_layers = nn.Sequential(*self._fc_layers)

        self._mu_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )
        self.output_dim = output_dim
        self.representation_size = output_dim
        self.input_channels = self._vqvae.input_channels
        self.imsize = self._vqvae.imsize
        self.apply(_init_weights)
        

    def forward(self, inputs, training=True):

        precomputed_vqvae_encoding = (
            inputs.shape[-1] == self._vqvae.representation_size)

        if precomputed_vqvae_encoding:
            feat = inputs
        else:
            obs = inputs
            obs = obs.view(obs.shape[0],
                           self.input_channels,
                           self.input_height,
                           self.input_width)
            obs = obs.permute([0, 1, 3, 2])
            
            feat = self._vqvae.encode(obs, flatten=False)
            # print("feat1:", feat.shape)  # [128, 8, 64, 64]
        if self.fixed:
            feat = feat.detach()
        
        feat = feat.view(feat.size(0), -1)  # [128, 32768]

        
        feat = self._fc_layers(feat)
        
        
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        # reparameterize
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        encoding = eps.mul(std).add_(mu)
        return encoding, (mu, logvar)


    def encode_np(self, inputs, cont=True):
        assert cont is True
        return get_numpy(self.forward(from_numpy(inputs)))