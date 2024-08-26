import numpy as np
import torch
import torch.nn as nn
from .encoders.mobilenet import MobileNetV2
from .encoders.vqvae_encoder import VqvaeVariationalObsEncoder as VqvaeEncoder
from .encoders.vqvae import VqVae
from .utils import SimpleConvs, ImpalaEncoder


class ImageEncoder(nn.Module):
    def __init__(self, encoder_dim, config):
        super().__init__()
        enc_type = config['IQL']['encode_type']
        self.enc_type = enc_type
        if  enc_type == 'mobilenet':
            self.enc = MobileNetV2(img_channel=3, # rgb channel
                                    last_channel=encoder_dim,
                                    variational=True,
                                    fixed=False,
                                    )
            ## load pretrained mobilenet 
            if config['Model']['pretrain_mobile'] is not None:
                model_dict = self.enc.state_dict()
                mobilenetv2_dict = torch.load(config['Model']['pretrain_mobile'])
                mobilenetv2_dict = {k: v for k, v in mobilenetv2_dict.items() 
                                    if k in model_dict.keys() and 'features.18' not in k}
                model_dict.update(mobilenetv2_dict)
                self.enc.load_state_dict(model_dict)
        elif enc_type == 'simpleconvs':
            self.enc = SimpleConvs(output_dim=encoder_dim, output_type='avg_pool', use_dropout=False) # 'avg_pool','flatten'
        elif enc_type == 'vqvae':
            hidden_dim = config['VQVAE']['hidden_dim']
            n_hidden = config['VQVAE']['n_hidden']
            vqvae_dim = config['VQVAE']['embedding_dim']
            imsize = config['VQVAE']['imsize']
            vqvae = VqVae(embedding_dim=vqvae_dim, imsize=imsize)
            fix = False
            if config['VQVAE']['use_pretrain']:
                pretrained_path = config['VQVAE']['pretrain_path'][config['env_name']]
                fix = config['VQVAE']['fix']
                state_dict = torch.load(pretrained_path)
                vqvae.load_state_dict(state_dict['model_state'])
                print('use pretraiend vqvae:', pretrained_path, "normalize:", config['use_ln'])
            self.enc = VqvaeEncoder(input_width=imsize, input_height=imsize,output_dim=encoder_dim, 
                                    vqvae=vqvae, fc_hidden_dim=hidden_dim, fc_n_hidden=n_hidden, 
                                    fixed=fix, normalize=config['use_ln'])
        elif enc_type == 'impala':
            self.enc = ImpalaEncoder()


    def forward(self, x):
        mu = logvar = None
        if self.enc_type == 'vqvae' or self.enc_type == 'mobilenet':
            x, (mu, logvar) = self.enc(x)
        else:
            x = self.enc(x)
            mu = x
    
        return x, (mu, logvar)