import torch
import torch.nn as nn

'''three types of implementation for expectile loss'''
class ExpectileLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, vf_pred, q_pred, expectile): # from flap
        vf_diff = vf_pred - q_pred
        vf_sign = (vf_diff > 0).float()
        vf_weight = (1 - vf_sign) * expectile + \
            vf_sign * (1 - expectile)  # value function loss, expertile regression
        vf_loss = (vf_weight * (vf_diff ** 2))
        return vf_loss  # .mean()

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def _expectile_loss(diff, expectile=0.9): # from ReViND
    """diff = v - q"""
    weight = torch.where(diff > 0, 1 - expectile, expectile)
    return (weight * (diff**2)).mean()


def _kld_loss(mu, logvar):
    return - 0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
class WeightedMSELoss(nn.Module):
    def __init__(self, lambda_linear=1.0, lambda_angular=10.0):
        '''calculate weighted-sum loss of pose prediction.
            range of dx,dy: approximately in [-50, 50] since the maximum temporal 
            dist for positive samples is 50.
            range of dyaw: [-pi, pi).
            default weight: lambda_xy = 1, lambda_yaw = 10.
            P_loss: (#batch, 3)
            return: (#batch, 1)
        '''
        super().__init__()
        self.lambda_linear = lambda_linear
        self.lambda_angular = lambda_angular
    
    def forward(self, action_error):
        action_error[:, 0] = action_error[:, 0] * self.lambda_linear
        action_error[:, 1] = action_error[:, 1] * self.lambda_angular
        return action_error.mean()