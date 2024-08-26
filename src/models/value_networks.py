import numpy as np
from math import pi
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .utils import mlp, _init_weights, Film


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        hidden_dim = config['IQL']['hidden_dim']
        n_hidden = config['IQL']['n_hidden']
        self.image_encode = config['IQL']['image_encode']
        self.image_goal = config['image_goal']
        self.diff_goal = config['diff_goal']
        self.use_film = config['use_film']
        self.is_revind = config['revind']
        encoder_dim = config['IQL']['encoder_dim']
        
        if self.is_revind:
            in_dims = 32768
            self.pre_block = nn.Sequential(
                nn.Linear(in_dims, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.Tanh()
            )
        if self.image_encode:
            if self.image_goal:
                state_dim = encoder_dim
        dims = [state_dim + action_dim + encoder_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, dropout_rate=None, squeeze_output=True) 
        self.apply(_init_weights)

    def forward(self, state, action, img_emb=None, goal_img_emb=None):
        if self.image_encode:
            assert img_emb is not None
            if self.is_revind:
                img_emb = self.pre_block(img_emb)
                goal_img_emb = self.pre_block(goal_img_emb)
            # img_emb = img_emb.detach()
            # goal_img_emb = goal_img_emb.detach()
            if self.diff_goal:
                goal_img_emb = goal_img_emb - img_emb
            if self.image_goal:
                sa = torch.cat([img_emb, goal_img_emb, action], dim=1)
            else:
                sa = torch.cat([state, action, img_emb], dim=1)
        else:
            sa = torch.cat([state, action], dim=1)
        return self.q1(sa)

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        hidden_dim = config['IQL']['hidden_dim']
        n_hidden = config['IQL']['n_hidden']
        self.image_encode = config['IQL']['image_encode']
        self.image_goal = config['image_goal']
        self.diff_goal = config['diff_goal']
        
        if self.image_encode:
            encoder_dim = config['IQL']['encoder_dim']
            if self.image_goal:
                state_dim = encoder_dim
        dims = [state_dim + action_dim + encoder_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)
        self.apply(_init_weights)

    def forward(self, state, action, img_emb=None, goal_img_emb=None):
        if self.image_encode:
            assert img_emb is not None
            if self.diff_goal:
                goal_img_emb = goal_img_emb - img_emb
            if self.image_goal:
                sa = torch.cat([img_emb, goal_img_emb, action], dim=1)
            else:
                sa = torch.cat([state, action, img_emb], dim=1)
        else:
            sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class ValueFunction(nn.Module):
    def __init__(self, state_dim, config):
        super().__init__()
        hidden_dim = config['IQL']['hidden_dim']
        n_hidden = config['IQL']['n_hidden']

        self.image_encode = config['IQL']['image_encode']
        self.image_goal = config['image_goal']
        self.diff_goal = config['diff_goal']
        self.use_film = config['use_film']
        self.is_revind = config['revind']
        encoder_dim = config['IQL']['encoder_dim']
        
        if self.is_revind:
            in_dims = 32768
            self.pre_block = nn.Sequential(
                nn.Linear(in_dims, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.Tanh()
            )
            
        if self.image_encode:
            if self.image_goal:
                state_dim = encoder_dim
        dims = [state_dim + encoder_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, dropout_rate=None, squeeze_output=True)#, use_layer_norm=config['use_ln'])
        self.apply(_init_weights)
            
    def forward(self, state, img_emb=None, goal_img_emb=None):
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
                state = torch.cat([img_emb, goal_img_emb], dim=1)
            else:
                state = torch.cat([state, img_emb], dim=1)
        out = self.v(state)
        return out