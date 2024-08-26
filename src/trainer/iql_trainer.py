import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from collections import OrderedDict
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.encoder import ImageEncoder
from utils.trainer_utils import get_numpy, soft_update_from_to, is_main_device, _share_encoder
from utils.criterion import _expectile_loss, asymmetric_l2_loss, _kld_loss, WeightedMSELoss, ExpectileLoss


class IQLTrainer(nn.Module):
    def __init__(
            self,
            image_encoder,
            qf1,
            qf2,
            vf,
            policy,
            cfg,
            affordance=None,
            rnd=None,
            device='cuda',
            min_value=None,
            max_value=0.0,
            gradient_clip_value=None,
            
    ):
        super().__init__()
        """networks"""
        # self.qf = qf.to(device)
        # self.target_qf = copy.deepcopy(qf).requires_grad_(False).to(device)
        self.qf1 = qf1.to(device)
        self.target_qf1 = copy.deepcopy(qf1).requires_grad_(False).to(device)
        self.qf2 = copy.deepcopy(qf2).to(device)
        self.target_qf2 = copy.deepcopy(qf2).requires_grad_(False).to(device)
        self.vf = vf.to(device)
        # self.vf2 = copy.deepcopy(vf).to(device)
        
        self.image_encoder = image_encoder.to(device)

        self.policy = policy.to(device)
        if affordance is not None:
            self.affordance = affordance.to(device)
        if rnd is not None:
            self.rnd = rnd.to(device)
        
        if cfg['Value']['use_pretrain']:
            print("load pretrained value function:", cfg['Value']['pretrain_path'])
            pretrained_model = torch.load(cfg['Value']['pretrain_path'])
            self.vf.load_state_dict(pretrained_model['vf_model_state'])
            self.image_encoder.load_state_dict(pretrained_model['enc_model_state'])
            
        """criterion"""
        self.qf_criterion = eval(cfg['Train']['qf_criterion'])()
        self.vf_criterion = eval(cfg['Train']['vf_criterion'])()
        self.pred_loss_fn = nn.SmoothL1Loss(reduction='none').to(device) 
        self.expectile_loss = ExpectileLoss().to(device)
        
        
        """hyperparameters"""
        self.discount = cfg['IQL']['discount']
        self.reward_scale = cfg['IQL']['reward_scale']
        self.beta = cfg['IQL']['beta']
        self.max_exp_adv = cfg['IQL']['max_exp_adv']
        self.expectile = cfg['IQL']['expectile']
        self.q_update_period = cfg['Train']['q_update_period']
        self.soft_target_tau = cfg['Train']['soft_target_tau']
        self.target_update_period = cfg['Train']['target_update_period']
        self.policy_update_period = cfg['Train']['policy_update_period']
        self.plot_period = cfg['Train']['plot_period']
        self.max_train_steps =  cfg['Train']['max_train_steps']
        self.use_rnd_penalty = cfg['RND']['use_rnd_penalty']
        self.rnd_beta = cfg['RND']['rnd_beta']
        self.use_afford = cfg['Affordance']['use_afford']
        self.affordance_pred_weight = cfg['Affordance']['pred_weight']
        self.affordance_beta = cfg['Affordance']['beta']
        self.affordance_weight = cfg['Affordance']['loss_weight']
        self.rnn_afford = cfg['Affordance']['rnn_afford']
        self.att_afford = cfg['Affordance']['att_afford']
        self.use_vib = cfg['IQL']['use_vib']
        self.vib_weight = cfg['IQL']['vib_weight']
        self.use_plan_vf = cfg['Value']['use_plan_vf']
        
        self.use_temporal = cfg['Value']['use_temporal']
        self.fraction_negative_obs = cfg['Value']['fraction_negative_obs']
        self.fraction_negative_goal = cfg['Value']['fraction_negative_goal']
        self.use_fdm = cfg['VQVAE']['use_fdm']
        self.min_value = -cfg['Data']['goal_range_max'] - 1  
        self.max_value = 0.0
        self.gradient_clip_value = gradient_clip_value
        self.diff_goal = cfg['diff_goal']
        

        """optimizers"""
        optimizer_class = eval(cfg['Train']['optimizer_class'])
        
        self.all_parameters = (
            list(self.qf1.parameters()) +
            list(self.qf2.parameters()) +
            list(self.vf.parameters()) +
            list(self.policy.parameters()) +
            list(self.image_encoder.parameters())
        )
        if self.use_plan_vf:
            self.plan_vf = copy.deepcopy(vf).to(device)
            self.all_parameters += list(self.plan_vf.parameters())
        if self.use_afford:
            self.all_parameters += list(self.affordance.parameters())
            
        # Remove duplicated parameters.
        self.all_parameters = list(set(self.all_parameters))
        self.optimizer = optimizer_class(
            self.all_parameters,
            lr=cfg['Train']['policy_lr'],
        )
    
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, self.max_train_steps)
        self._n_train_steps = 0
        self.device = device

    
    def _compute_affordance_loss(self, h0, h1, stack_lengths):
        (u_mu, u_logvar), u, h1_pred = self.affordance(h1, h0, stack_lengths)

        afford_pred = self.pred_loss_fn(h1_pred, h1.detach()).mean()
        
        kld = _kld_loss(u_mu, u_logvar).mean()
        return afford_pred, kld


    def _compute_plan_vf_loss(self, obs, goal, target_value, pos_index, weights=None):
        target_value = target_value.detach()  #* 
        num_samples = pos_index.shape[0]
        

        obs = obs.detach()
        goal = goal.detach()
        pos_obs = torch.index_select(obs, 0, pos_index)
        pos_goal = torch.index_select(goal, 0, pos_index)

        replace_obs = (
            torch.rand(num_samples, 1) < self.fraction_negative_obs
        ).to(self.device)
        # sampled_obs = torch.randn(obs.size()).to(ptu.device)
        sampled_obs = torch.where(
            (torch.rand(num_samples, 1) < 0.5).to(self.device),
            torch.randn(pos_obs.size()).to(self.device),
            torch.flip(pos_obs, [0]))
        
        pos_obs = torch.where(
            replace_obs,
            sampled_obs,
            pos_obs)

        replace_goal = (
            torch.rand(num_samples, 1) < self.fraction_negative_goal
        ).to(self.device)
        # sampled_goal = torch.randn(goal.size()).to(ptu.device)
        sampled_goal = torch.where(
            (torch.rand(num_samples, 1) < 0.5).to(self.device),
            torch.randn(pos_goal.size()).to(self.device),
            torch.flip(pos_goal, [0]))
        
        pos_goal = torch.where(
            replace_goal,
            sampled_goal,
            pos_goal)

        obs[pos_index, :] = pos_obs
        goal[pos_index, :] = pos_goal
        
        
        pred_value = self.plan_vf(None, obs, goal)

        # replace_any = (replace_obs + replace_goal) > 0 
        replace_any = torch.zeros(obs.shape[0]).cuda()
        replace_any[pos_index] = 1.0
        replace_any = replace_any.bool()
    

        # if self.train_encoder:
        pos_plan_vf_loss = self.pred_loss_fn(pred_value, target_value)
        neg_plan_vf_loss = torch.clamp(pred_value - self.min_value, min=0.)
        plan_vf_loss = torch.where(
            replace_any,
            neg_plan_vf_loss,
            pos_plan_vf_loss)
        
        plan_vf_loss = plan_vf_loss.mean()

        extra = {
            'vf_pred': pred_value,
        }
        return plan_vf_loss, extra
    
    
    def forward(self, batch, image_encode=False, warm_up=False):
        rewards = batch.rewards.to(self.device)  # (B,)
        terminals = batch.terminals.to(self.device)
        obs = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        next_obs = batch.next_observations.to(self.device)
        next_actions = batch.next_actions.to(self.device)
        is_positive = batch.is_positive.to(self.device)
        pos_index = torch.nonzero(is_positive).squeeze()
        neg_index = (is_positive == 0).nonzero().squeeze()
        time_to_goals = batch.time_to_goals.to(self.device)
        raw_observations = batch.raw_observations.to(self.device)
        stack_imgs = batch.stack_imgs.to(self.device)
        stack_lengths = batch.stack_lengths.to(self.device)
        if image_encode:
            imgs = batch.image_observations.to(self.device)
            next_imgs = batch.next_image_observations.to(self.device)
            imgs_goal = batch.image_goals.to(self.device)
            imgs_emb, (imgs_mu, imgs_logvar) = self.image_encoder(imgs)
            next_imgs_emb, (next_imgs_mu, next_imgs_logvar) = self.image_encoder(next_imgs)
            imgs_goal_emb, (imgs_goal_mu, imgs_goal_logvar) = self.image_encoder(imgs_goal)
            assert imgs_emb is not None
            if imgs_mu is None:
                imgs_mu = imgs_emb
                next_imgs_mu = next_imgs_emb
                imgs_goal_mu = imgs_goal_emb
        else:
            imgs_emb = next_imgs_emb = imgs_goal_emb = None
        
        
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions, imgs_emb, imgs_goal_emb)
        q2_pred = self.qf2(obs, actions, imgs_emb, imgs_goal_emb)
        target_next_vf_pred = self.vf(next_obs, next_imgs_mu, imgs_goal_mu).detach()  # affordance
        target_next_vf_pred = torch.clamp(
            target_next_vf_pred,
            max=self.max_value,  # 0
        )
            
        if self.use_rnd_penalty:
            rnd_penalty = self.rnd.get_reward(next_imgs, next_actions).detach()
            target_next_vf_pred = target_next_vf_pred - self.rnd_beta * rnd_penalty
            
        q_target = self.reward_scale * rewards + \
                (1. - terminals) * self.discount * target_next_vf_pred  # q function loss
        q_target = q_target.detach()

        
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss
        """
        q_pred = torch.min(self.target_qf1(obs, actions, imgs_mu, imgs_goal_mu),
                            self.target_qf2(obs, actions, imgs_mu, imgs_goal_mu)).detach()
        q_pred = q_pred.clamp(
                              max=self.max_value)  #*
    
        vf_pred = self.vf(obs, imgs_emb, imgs_goal_emb)
        
        # expectile regression loss
        pos_vf_loss = self.expectile_loss(vf_pred, q_pred, self.expectile)
        
        neg_label = torch.tensor([self.min_value]).repeat(vf_pred.shape[0]).to(self.device)
        neg_vf_loss = self.pred_loss_fn(vf_pred, neg_label)
        # neg_vf_loss = (vf_pred - self.min_value)#.clamp(min=0) #** 2  # .clamp(min=0)
        vf_loss = torch.where(is_positive, pos_vf_loss, neg_vf_loss).mean()
        
        
        """
        Plan vf
        """
        if self.use_plan_vf:
            plan_vf_loss, plan_vf_extra = self._compute_plan_vf_loss(
                imgs_emb, imgs_goal_emb, vf_pred, pos_index)
            
        """
        Affordance Loss
        """
        if self.use_afford:
            if len(pos_index):
                if self.rnn_afford:
                    bn, seq_len = stack_imgs.shape[0], stack_imgs.shape[1]
                    stack_imgs = stack_imgs.reshape(bn * seq_len, *stack_imgs.shape[2:])
                    _, (stack_imgs_mu, _) = self.image_encoder(stack_imgs)
                    stack_imgs_mu = stack_imgs_mu.reshape(bn, seq_len, -1)
                    imgs_mu_pos = torch.index_select(stack_imgs_mu, 0, pos_index).detach()
                    stack_lengths = torch.index_select(stack_lengths, 0, pos_index).detach()
                else:
                    imgs_mu_pos = torch.index_select(imgs_mu, 0, pos_index).detach()
                    
                imgs_goal_mu_pos = torch.index_select(imgs_goal_mu, 0, pos_index).detach()
                
                afford_pred, afford_kld = self._compute_affordance_loss(
                    imgs_mu_pos, imgs_goal_mu_pos, stack_lengths)
                affordance_loss = self.affordance_pred_weight * afford_pred + \
                    self.affordance_beta * afford_kld
                affordance_loss = affordance_loss * self.affordance_weight
            else:
                affordance_loss = afford_pred = afford_kld = torch.tensor([0]).to(self.device)
        
        
        """
        Policy Loss
        """
        vf_baseline = self.vf(obs, imgs_mu, imgs_goal_mu).detach()  # affordance
        dist = self.policy(obs, imgs_emb.detach(), imgs_goal_emb.detach())
        policy_logpp = dist.log_prob(actions)  # (B,)
        adv = q_pred - vf_baseline
        if len(pos_index):  # update only for positive samples
            policy_logpp = torch.index_select(policy_logpp, 0, pos_index)
            adv = torch.index_select(adv, 0, pos_index)
            exp_adv = torch.exp(adv / self.beta)
            if self.max_exp_adv is not None:
                exp_adv = torch.clamp(exp_adv, max=self.max_exp_adv)

            adv_weight = exp_adv.detach()
            policy_loss = (-policy_logpp * adv_weight).mean()  # Advantage Weighted Regression: extract a policy net subject to a distribution constrain
            if self.use_rnd_penalty:
                policy_loss = policy_loss + self.rnd_beta * rnd_penalty
        else:
            policy_loss = torch.tensor([0]).to(self.device)
        

        """
        VIB Loss: used to alleviate overfitting issue.
        """
        if self.use_vib:
            vib_loss = _kld_loss(imgs_mu, imgs_logvar).mean() 
            #vib_loss = (vib_loss + _kld_loss(imgs_goal_mu, imgs_goal_logvar).mean()) / 2
            vib_loss = vib_loss * self.vib_weight
        
        """
        Update Networks
        """
        if self._n_train_steps % self.q_update_period == 0:
            losses = qf1_loss + qf2_loss + vf_loss + policy_loss
            if self.use_afford:
                losses += affordance_loss
            if self.use_plan_vf:
                losses += plan_vf_loss
            if self.use_vib:
                losses += vib_loss
                
            self.optimizer.zero_grad()
            losses.backward()
            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.all_parameters, self.gradient_clip_value)
            self.optimizer.step()
            self.lr_scheduler.step()


        """
        Soft Updates
        """
        if self._n_train_steps % self.target_update_period == 0:
            soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
            

        """
        Evaluate on training dataset and plot losses
        """
        info = {
            'Rewards': rewards.mean(),
            'QF1_loss': qf1_loss,
            'QF2_loss': qf2_loss,
            'VF_loss': vf_loss,
            'Policy_loss': policy_loss
        }
        extra = OrderedDict({
            'rnd_penalty': rnd_penalty if self.use_rnd_penalty else None,
            'afford': affordance_loss if self.use_afford else None, 
            'afford_pred': self.affordance_pred_weight * afford_pred if self.use_afford else None,
            'afford_kld': self.affordance_beta * afford_kld if self.use_afford else None,
            'plan_vf': plan_vf_loss if self.use_plan_vf else None,
            'vib': vib_loss if self.use_vib else None,
        })
            
        return rewards.mean(), qf1_loss, qf2_loss, vf_loss, policy_loss, extra

    
    @ property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf,
            self.image_encoder,
        ]
        if hasattr(self, 'affordance'):
            nets.append(self.affordance)
        if hasattr(self, 'rnd'):
            nets.append(self.rnd)
        return nets
    
    
    def train(self):
        for net in self.networks:
            net.train()
    
    def eval(self):
        for net in self.networks:
            net.eval()
            