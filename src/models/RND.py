import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import copy

from models.utils import SimpleConvs, Film
from utils.trainer_utils import RunningMeanStd, from_numpy, get_numpy
    


class FilmSA(nn.Module):
    def __init__(self, img_enc_dim=256, hidden_dim=256, action_dim=2, out_dim=512, is_prior=True):
        super(FilmSA, self).__init__()
        self.enc = SimpleConvs(img_enc_dim)
        self.is_prior = is_prior
        if is_prior:
            self.concat = Film(img_enc_dim, hidden_dim) # content: image, feature: action
            self.linear1 = nn.Linear(action_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.concat = nn.Bilinear(action_dim, img_enc_dim, hidden_dim) # content: image, feature: action
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        
    
    def forward(self, s, a):
        s = self.enc(s)
        if self.is_prior:
            
            a = F.relu(self.linear1(a))
            a = F.relu(self.linear2(a))
            sa = F.relu(self.concat(a, s))  # concat at penultimate layer(last two)
            out = self.out(sa)  
        else:
            sa = F.relu(self.concat(a, s))  # concat at first layer
            sa = F.relu(self.linear1(sa))
            sa = F.relu(self.linear2(sa))
            out = self.out(sa)
        return out



class RND(nn.Module):
    def __init__(self, hidden_dim=512, 
                 n_hidden=3, 
                 enc_dim=None, 
                 pretrain_path=None):
        super(RND, self).__init__()
        layers = []
        input_dim = enc_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.prior = nn.Sequential(*layers)
        
        pred_layers = copy.deepcopy(layers)
        pred_layers.extend([  #* no additional layers
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.predictor = nn.Sequential(*pred_layers)
    

        self._running_obs = RunningMeanStd(epsilon=1e-4, shape=enc_dim)
        self._running_reward = RunningMeanStd(epsilon=1e-4)
        
        self.register_buffer('_running_obs_mean', torch.tensor(self._running_obs.mean))
        self.register_buffer('_running_obs_std', torch.tensor(self._running_obs.std))
        self.register_buffer('_running_reward_mean', torch.tensor(self._running_reward.mean))
        self.register_buffer('_running_reward_std', torch.tensor(self._running_reward.std))
        if pretrain_path is not None:
            state_dict = torch.load(pretrain_path)['model_state']
            self.load_state_dict(state_dict, strict=False)
            self.load_running_params()
        else:
            for p in self.modules():
                if isinstance(p, nn.Conv2d):
                    init.orthogonal_(p.weight, np.sqrt(2))
                    # init.kaiming_normal_(p.weight, mode="fan_out")
                    p.bias.data.zero_()

                if isinstance(p, nn.Linear):
                    # init.normal_(p.weight, 0, 0.01)
                    init.orthogonal_(p.weight, np.sqrt(2))
                    p.bias.data.zero_()
            
        for param in self.prior.parameters():
            param.requires_grad = False
        
        self.loss = nn.MSELoss(reduction='none')
    
    def save_running_params(self):
        self._running_obs_mean = torch.tensor(self._running_obs.mean)
        self._running_obs_std = torch.tensor(self._running_obs.std)
        self._running_reward_mean = torch.tensor(self._running_reward.mean)
        self._running_reward_std = torch.tensor(self._running_reward.std)
        print('save running params:', self.running_params)
        
    def load_running_params(self):
        self._running_obs.mean = get_numpy(self._running_obs_mean)
        self._running_obs.std = get_numpy(self._running_obs_std)
        self._running_reward.mean = get_numpy(self._running_reward_mean)
        self._running_reward.std = get_numpy(self._running_reward_std)
        print('load running params:', self.running_params)
    
    @property
    def running_params(self):
        return {
            '_running_obs_mean': self._running_obs_mean,
            '_running_obs_std': self._running_obs_std,
            '_running_reward_mean': self._running_reward_mean,
            '_running_reward_std': self._running_reward_std
        }
    
    def normalize_obs(self, obs):
        device = obs.device
        obs = (obs - from_numpy(self._running_obs.mean, device)) \
            / from_numpy(self._running_obs.std, device)
        obs = obs.clamp(-5, 5)
        return obs   
        
    def normalize_reward(self, reward):
        device = reward.device
        
        return reward / from_numpy(np.array(self._running_reward.std), device)
        
    def get_reward(self, s):
        s = self.normalize_obs(s)
        y_true = self.prior(s).detach()
        y_pred = self.predictor(s)
        reward = torch.pow(y_pred - y_true, 2).sum(1)
        self._running_reward.update(get_numpy(reward))
        normalized_reward = self.normalize_reward(reward)  # [0, 1]
        if self.training:
            reward = reward.mean()
            normalized_reward = normalized_reward.mean()
        return reward, normalized_reward
    
    
class RND_SA(nn.Module):
    def __init__(self, enc_dim=256, hidden_dim=256, pretrain_path=None):
        super().__init__() 
        self.prior = FilmSA(enc_dim, hidden_dim, is_prior=True)
        self.predictor = FilmSA(enc_dim, hidden_dim, is_prior=False)
        

        if pretrain_path is not None:
            print("load pretrained RND_SA:", pretrain_path)
            state_dict = torch.load(pretrain_path)['model_state']
            self.load_state_dict(state_dict)
        else:
            for p in self.modules():
                if isinstance(p, nn.Conv2d):
                    init.orthogonal_(p.weight, np.sqrt(2))
                    # init.kaiming_normal_(p.weight, mode="fan_out")
                    p.bias.data.zero_()

                if isinstance(p, nn.Linear):
                    # init.normal_(p.weight, 0, 0.01)
                    init.orthogonal_(p.weight, np.sqrt(2))
                    p.bias.data.zero_()
        
        for param in self.prior.parameters():
            param.requires_grad = False
    
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(),lr=0.0003)
            
            
    def get_reward(self, s, a):
        with torch.no_grad():
            y_true = self.prior(s, a).detach()
        y_pred = self.predictor(s, a)
        reward = self.criterion(y_pred, y_true)
        return reward

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
