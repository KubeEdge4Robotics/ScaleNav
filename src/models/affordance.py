import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.trainer_utils import randn, get_numpy
from .encoders.mobilenet import MobileNetV2
from utils.criterion import _kld_loss
from .utils import Film, mlp


class SimpleCcVae(nn.Module):

    def __init__(
            self,
            input_dim=128,
            z_dim=8,
            hidden_dim=128,
            use_film=False,
            diff_goal=False,
            bias=False, #! False is better than True
            n_hidden=2,
            z_var=1
    ):
        super(SimpleCcVae, self).__init__()

        self.representation_size = z_dim
        self.use_film = use_film
        self.diff_goal = diff_goal
        self.z_var = z_var
        self.use_rnn = False
        self.use_attention = False
        
        # ##### Encoder #####
        self.concat_enc = nn.Linear(
                            input_dim + input_dim,
                            hidden_dim,
                            bias=bias,
                        )
        enc_layers = [
            nn.ReLU(True),
        ]
        for _ in range(n_hidden):
            enc_layers.extend([
                nn.Linear(
                    hidden_dim,  
                    hidden_dim,
                    bias=bias,
                ),
                nn.ReLU(True),
                
            ])
        self._encoder = nn.Sequential(*enc_layers)
        
        ##### Decoder #####
        if use_film:
            self.concat_dec = Film(input_dim, z_dim)
        else:
            self.concat_dec = nn.Linear(
                                z_dim + input_dim,
                                hidden_dim,
                                bias=bias,
                            )
        dec_layers = [
            nn.ReLU(True),
        ]
        for i in range(n_hidden - 1):
            dec_layers.extend([
                nn.Linear(
                    z_dim if use_film and i == 0 else hidden_dim,
                    hidden_dim,
                    bias=bias,
                ),
                nn.ReLU(True),
                
            ])
        dec_layers.append(
            nn.Linear(
                    hidden_dim,  
                    hidden_dim,
                    bias=bias,
            ),
        )
        self._decoder = nn.Sequential(*dec_layers)
        
        self._mu_layer = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
        )
        self.apply(self._init_linear)
            
    
    def _init_linear(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.zeros_(module.bias)
    
    def encode(self, goal, cond):
        encoder_input = torch.cat((goal, cond), dim=1)
        hidden = self.concat_enc(encoder_input)
        feat = self._encoder(hidden)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        return mu, logvar


    def decode(self, z, cond):
        if self.use_film:
            hidden = self.concat_dec(z, cond)
            recon = self._decoder(hidden)
        else:
            decoder_input = torch.cat((z, cond), dim=1)
            hidden = self.concat_dec(decoder_input)
            recon = self._decoder(hidden)
        return recon

    def forward(self, goal, cond, stack_lengths=None):
        mu, logvar = self.encode(goal, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return (mu, logvar), z, recon

    def sample_prior(self, batch_size, device):
        z_s = randn(batch_size, self.representation_size, torch_device=device) * self.z_var
        return get_numpy(z_s)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
class RNNSimpleCcVae(SimpleCcVae):
    def __init__(self,
                input_dim=128,
                z_dim=8,
                hidden_dim=128,
                use_film=False,
                diff_goal=False,
                bias=False,
                n_hidden=2,
                z_var=1,
                batch_first=False):
        super().__init__(input_dim, z_dim, hidden_dim, use_film, diff_goal, bias, n_hidden, z_var)
        self.use_rnn = True
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=input_dim, batch_first=True)

    def rnn_features(self, stack_cond, stack_lengths):
        self.rnn.flatten_parameters()
        stack_cond_packed = pack_padded_sequence(stack_cond, stack_lengths.cpu(), 
                                batch_first=True, enforce_sorted=False)
        try:
            out_packed, cond = self.rnn(stack_cond_packed)
        except:
            print(stack_cond)
            print(stack_cond.shape, stack_cond_packed.shape)
            assert False
        # output_padded, output_lengths = pad_packed_sequence(out_packed, batch_first=True)
        cond = cond.squeeze(0)  # 1, batch, dim -> batch, dim
        return cond
    
    def forward(self, goal, stack_cond, stack_lengths):
        goal = goal.detach()
        stack_cond = stack_cond.detach()
        cond = self.rnn_features(stack_cond, stack_lengths)
        mu, logvar = self.encode(goal, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return (mu, logvar), z, recon
    

class CrossBlock(nn.Module):
    def __init__(self, hidden_dim, n_head, n_linear, batch_first):
        super().__init__()
        self.use_attention = True
        self.mhatt = nn.MultiheadAttention(hidden_dim, n_head, batch_first=batch_first)
        self.layernorm_in = nn.LayerNorm(hidden_dim)
        self.layernorm_out = nn.LayerNorm(hidden_dim)
        self.mlp = mlp([hidden_dim] * n_linear)
        
    def forward(self, goal, cond):
        goal = goal.unsqueeze(1)  # (B, D) -> (B, T1, D)
        cond = self.layernorm_in(cond)
        hidden, att_weights = self.mhatt(query=goal, key=cond, value=cond)  # (B, T1, D)
        hidden += goal
        hidden = self.layernorm_out(hidden)
        feat = self.mlp(hidden)  # (B, T1, D)
        feat = feat.squeeze(1)
        return feat
    

class AttSimpleCcVae(SimpleCcVae):
    def __init__(self,
                input_dim=128,
                z_dim=8,
                hidden_dim=128,
                use_film=False,
                diff_goal=False,
                bias=False,
                z_var=1,
                batch_first=False,
                n_head=2,
                device='cuda'
                ):
        super().__init__(input_dim, z_dim, hidden_dim, use_film, diff_goal, bias, z_var)
        
        self.cross_enc = CrossBlock(hidden_dim, n_head, n_linear=3, batch_first=True)

        # pe = _build_position_encoding(d_model=feature_dims, length=4).to(device)
        # self.register_buffer('pe', pe)
    
    
    def encode(self, goal, cond):
        # goal: (B, 1, D), cond:(B, L, D)
        feat = self.cross_enc(goal, cond)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)
        return mu, logvar
    
    
    def decode(self, z, cond):
        cond = cond[:, -1, :]
        if self.use_film:
            hidden = self.concat_dec(z, cond)
            recon = self._decoder(hidden)
        else:
            decoder_input = torch.cat((z, cond), dim=1)
            hidden = self.concat_dec(decoder_input)
            recon = self._decoder(hidden)
        return recon
        
