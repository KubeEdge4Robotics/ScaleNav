import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .utils import mlp, _init_weights

"""
policy:
1) from rlkit: Gaussian Policy: 
mean = nn.Tanh(mean)
self.log_std_logits = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
log_std = torch.sigmoid(self.log_std_logits)
log_std = self.min_log_std + log_std * (         #(-6, 0)
            self.max_log_std - self.min_log_std)
std = torch.exp(log_std)

2) from gwthomas: Gaussian Policy:
self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
std = torch.exp(self.log_std.clamp(self.MIN_LOG_STD, self.MAX_LOG_STD))
scale_tril = torch.diag(std)

3) from ReViND:
- NormalTanhPolicy:
log_stds = nn.Dense(self.action_dim,
                    kernel_init=default_init())(outputs)
log_stds = jnp.exp(jnp.clip(log_stds, self.log_std_min, self.log_std_max))
tanh()
- UnitStdNormalPolicy: 
scale_diag=jnp.ones_like(means)
"""

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super().__init__()
        hidden_dim = config['IQL']['hidden_dim']
        n_hidden = config['IQL']['n_hidden']
        self.MIN_LOG_STD = config['IQL']['MIN_LOG_STD']
        self.MAX_LOG_STD = config['IQL']['MAX_LOG_STD']
        
        self.image_encode = config['IQL']['image_encode']
        self.image_goal = config["image_goal"]
        self.diff_goal = config['diff_goal']
        self.use_film = config['use_film']
        self.is_revind = config['revind']
        encoder_dim = config['IQL']['encoder_dim']
        
        if self.image_encode:
            if self.image_goal:
                obs_dim = encoder_dim
        self.net = mlp([obs_dim + encoder_dim, *([hidden_dim] * n_hidden), act_dim], 
                       output_activation=None)
        
        if self.is_revind:
            in_dims = 32768
            self.pre_block = nn.Sequential(
                nn.Linear(in_dims, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.Tanh()
            )
            self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32), requires_grad=False) # in ReViND, std = np.ones_like(mean)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32), requires_grad=True)
        
        self.apply(_init_weights)

    def forward(self, obs, img_emb=None, goal_img_emb=None):
        if self.image_encode:
            assert img_emb is not None
            if self.is_revind:
                img_emb = self.pre_block(img_emb)
                goal_img_emb = self.pre_block(goal_img_emb)
                
            img_emb = img_emb.detach()
            goal_img_emb = goal_img_emb.detach()
            if self.diff_goal:
                goal_img_emb = goal_img_emb - img_emb
            if self.image_goal:
                obs = torch.cat([img_emb, goal_img_emb], dim=1)
            else:
                obs = torch.cat([obs, img_emb], dim=1)
        mean = self.net(obs)
        
        if self.is_revind:
            std = torch.exp(self.log_std).detach()  # unit std
        else:
            #! remember to regularize action range
            mean[:, 0] = torch.tanh(mean[:, 0]) * 2 # range [-2, 2] 
            mean[:, 1] = torch.tanh(mean[:, 1]) * 1  # angular: range [-1, 1] 
        
            std = torch.exp(self.MIN_LOG_STD + torch.sigmoid(self.log_std) * ( 
                    self.MAX_LOG_STD - self.MIN_LOG_STD))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)


    def act(self, obs, img=None, goal_img=None, deterministic=False,
        enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs, img, goal_img)
            return dist.mean if deterministic else dist.sample()
            
        
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super().__init__()
        hidden_dim = config['IQL']['hidden_dim']
        n_hidden = config['IQL']['n_hidden']
        self.is_custom = config['env_name'] == 'custom'
        self.image_encode = config['IQL']['image_encode']
        self.image_goal = config["image_goal"]
        self.diff_goal = config['diff_goal']
        if self.image_encode:
            encoder_dim = config['IQL']['encoder_dim']
            if self.image_goal:
                obs_dim = encoder_dim
        self.net = mlp([obs_dim + encoder_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)
        self.apply(_init_weights)

    def forward(self, obs, img_emb=None, goal_img_emb=None):
        if self.image_encode:
            assert img_emb is not None
            img_emb = img_emb.detach()
            goal_img_emb = goal_img_emb.detach()
            if self.diff_goal:
                goal_img_emb = goal_img_emb - img_emb
            if self.image_goal:
                obs = torch.cat([img_emb, goal_img_emb], dim=1)
            else:
                obs = torch.cat([obs, img_emb], dim=1)
        act = self.net(obs)
        return act

    def act(self, obs, img_emb=None, goal_img_emb=None, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs, img_emb, goal_img_emb)
        
