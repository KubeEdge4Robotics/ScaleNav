"""Based on https://github.com/kuanfang/flap/blob/master/rlkit/planning/planner.py"""
import numpy as np
import torch
from torch import Tensor
import itertools
import copy
from utils import trainer_utils as u
from utils.trainer_utils import get_numpy
from dataset.dataset_utils import action2pose, get_rel_pose_change, get_new_pose, _norm_heading


def l2_distance(h_t, goal_state, start_dim=-3):
    h_t = torch.flatten(h_t, start_dim=start_dim)
    goal_state = torch.flatten(goal_state, start_dim=start_dim)
    return torch.sqrt(torch.sum((goal_state - h_t) ** 2, dim=-1))


def compute_value(vf, plans, init_states, goal_states=None,
                  batched=True, replace_last=True, reverse=False):

    assert vf is not None

    if batched:
        if replace_last:
            plans = plans[:, :-1]

        num_samples = plans.shape[0]
        num_steps = plans.shape[1] + 1
        h0 = torch.cat([init_states[:, None], plans], 1)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[1] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans, goal_states[:, None]], 1)
        
    else:
        if replace_last:
            plans = plans[:-1]

        num_steps = plans.shape[0] + 1
        h0 = torch.cat([init_states[None], plans], 0)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[0] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans[:-1], goal_states[None]], 1)


    obs = None
    h0 = h0.view(-1, h0.shape[-1])
    h1 = h1.view(-1, h1.shape[-1])
    values = vf(obs, h0, h1).detach()

    if batched:
        values = values.view(num_samples, num_steps)
    else:
        values = values.view(num_steps)

    return values


def compute_q_value(qf, plans, init_states, goal_states=None,
                    batched=True, replace_last=True, act_dim=2):

    assert qf is not None

    if batched:
        if replace_last:
            plans = plans[:, :-1]

        num_samples = plans.shape[0]
        num_steps = plans.shape[1] + 1
        h0 = torch.cat([init_states[:, None], plans], 1)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[1] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans, goal_states[:, None]], 1)

    else:
        if replace_last:
            plans = plans[:-1]

        num_steps = plans.shape[0] + 1
        h0 = torch.cat([init_states[None], plans], 0)

        if goal_states is None:
            assert not replace_last
            h1 = plans
        else:
            # Replace the last step with goal_state.
            assert plans.shape[0] > 1, 'The plan should be longer than 1 step.'
            h1 = torch.cat([plans[:-1], goal_states[None]], 1)

    qf_inputs = torch.cat([
        h0.view(-1, h0.shape[-1]),
        h1.view(-1, h1.shape[-1]),
    ], 1)
    batch_size = qf_inputs.shape[0]
    num_action_samples = 128
    min_action = -1.
    max_action = 1. 
    actions = min_action + (max_action - min_action) * torch.rand(  
        num_action_samples, batch_size, act_dim).to(u.device)
    max_linear = 3
    actions[:, :, 0] = (actions[:, :, 0] + 1) / 2 * max_linear
    obs = None
    _qf_inputs = qf_inputs[None].repeat((num_action_samples, 1, 1)) 
    _qf_inputs = _qf_inputs.view(num_action_samples * batch_size, 2, -1)
    actions = actions.view(num_action_samples * batch_size, -1)
    values = qf(obs, actions, _qf_inputs[:, 0], _qf_inputs[:, 1]).detach()
    values = values.view(num_action_samples, batch_size)

    values = values.mean(dim=0)

    if batched:
        values = values.view(num_samples, num_steps)
    else:
        values = values.view(num_steps)

    return values


def compute_value_v2(vf, plans, init_states, goal_states=None, batched=True):
    assert vf is not None
    if batched:
        num_samples = plans.shape[0]
        num_steps = plans.shape[1]
        h0 = torch.cat([init_states[:, None]], 1)
        h1 = torch.cat([plans[:, 0:1]], 1)
    else:
        num_steps = plans.shape[0]
        h0 = torch.cat([init_states[None], plans[:-1]], 0)
        h1 = plans

    vf_inputs = torch.cat([
        h0.view(-1, 720),
        h1.view(-1, 720),
    ], 1)
    values = vf(vf_inputs).detach()
    print('values: ', values)
    print('avg_value: %.2f, max_value: %.2f, min_value: %.2f'
          % (
              u.get_numpy(torch.mean(values)),
              u.get_numpy(torch.max(values)),
              u.get_numpy(torch.min(values))
          ))

    if batched:
        values = values.view(num_samples, num_steps)
    else:
        values = values.view(num_steps)

    return values


def preprocess(vqvae, h):
    h = h.view(
        -1,
        vqvae.embedding_dim,
        vqvae.root_len,
        vqvae.root_len)
    return h


def encode(vqvae, init_obs, goal_obs):
    init_state = vqvae.encode(init_obs[None, ...], flatten=False)[0]
    goal_state = vqvae.encode(goal_obs[None, ...], flatten=False)[0]
    return init_state, goal_state


def decode(vqvae, h):
    if isinstance(h, list):
        h = torch.stack(h, 1)

    outer_shape = list(h.shape)[:-3]
    h = h.view(
        -1,
        vqvae.embedding_dim,
        vqvae.root_len,
        vqvae.root_len)
    s = vqvae.decode(h)

    s_shape = outer_shape + list(s.shape[-3:])
    s = s.view(*s_shape)

    return s


def select(zs, plans, costs, to_list=False):
    # Rank the plans.
    min_costs, min_steps = torch.min(costs, 1)
    top_cost, top_ind = torch.min(min_costs, 0)

    top_zs = zs[top_ind, :]
    top_plan = plans[top_ind, :]

    # Optional: Prevent the random actions after achieving the goal.
    top_step = min_steps[top_ind]
    top_plan[top_step:] = top_plan[top_step:top_step + 1]

    if to_list:
        top_plan = list(torch.unbind(top_plan, 0))

    info = {
        'top_ind': top_ind,
        'top_step': top_step,
        'top_cost': top_cost,
    }

    return top_zs, top_plan, info


def nms(data, scores, num_elites, dist_thresh, stepwise_nms=False):
    num_samples = data.shape[0]
    assert num_samples >= num_elites
    if stepwise_nms:
        num_steps = data.shape[1]

    if scores is None:
        # top_indices = torch.zeros((num_samples,), dtype=torch.int64)
        top_indices = torch.arange(0, num_samples, dtype=torch.int64)
    else:
        _, top_indices = torch.topk(scores, num_samples)

    chosen_inds = torch.zeros((num_elites,), dtype=torch.int64)

    valids = torch.ones((num_samples, ), dtype=torch.float32).to(u.device)

    num_chosen = 0
    for i_top in range(num_samples):
        if num_chosen >= num_elites:
            break

        this_ind = top_indices[i_top]

        if valids[this_ind] == 0:
            continue

        chosen_inds[num_chosen] = this_ind
        num_chosen += 1

        diffs = data[this_ind][None, ...] - data

        if stepwise_nms:
            diffs = diffs.view(num_samples, num_steps, -1)
            dists = torch.norm(diffs, dim=-1)
        else:
            diffs = diffs.view(num_samples, -1)
            dists = torch.norm(diffs, dim=-1)
            valids = torch.where((dists >= dist_thresh).to(torch.uint8),
                                 valids,
                                 torch.zeros_like(valids))

    return chosen_inds

def _normalize_metric(metric:Tensor):
    assert isinstance(metric, Tensor)
    return (metric - metric.min()) / (metric.max() - metric.min())

class Planner(object):

    def __init__(
            self,
            model,
            predict_mode='affordance',
            cost_mode='l2_vf',
            encoding_type=None,
            debug=False,

            initial_collect_episodes=32, #32,
            buffer_size=0, 
            max_steps=4,
            **kwargs):

        self.encoding_type = encoding_type

        if hasattr(model, 'rnd'):
            self.rnd = model.rnd
            
        if hasattr(model, 'image_encoder'):
            self.obs_encoder = model.image_encoder
        else:
            self.obs_encoder = None

        if hasattr(model, 'affordance'):
            self.affordance = model.affordance
        else:
            assert False, 'no affordance'
            self.affordance = None

        if hasattr(model, 'classifier'):
            self.classifier = model.classifier
        else:
            self.classifier = None

        if hasattr(model, 'plan_vf') or hasattr(model, 'vf'):
            self._vf = model.vf
        else:
            self._vf = None

        if hasattr(model, 'qf1'):
            self._qf1 = model.qf1
        else:
            self._qf1 = None

        if hasattr(model, 'qf2'):
            self._qf2 = model.qf2
        else:
            self._qf2 = None
            
        if hasattr(model, 'policy'):
            self._policy = model.policy
        else:
            self._policy = None


        self._predict_mode = predict_mode
        self._cost_mode = cost_mode
        self._debug = debug

        self._max_steps = max_steps
        self._buffer_size = buffer_size
        self._initial_collect_episodes = initial_collect_episodes

        self.sub_planners = []

        self._buffer = None
        self._buffer_head = 0

        if self._buffer_size > 0:
            if self.affordance is None:
                representation_size = 8 
            else:
                try:
                    representation_size = self.affordance.representation_size
                except Exception:
                    representation_size = (
                        self.affordance.networks[0].representation_size)

            self._buffer = np.ones(
                (self._buffer_size,
                 self._max_steps,
                 representation_size),
                dtype=np.float32)

    @property
    def debug(self):
        return self._debug

    @property
    def vf(self):
        return self._vf

    @vf.setter
    def vf(self, value):
        self._vf = value

    @property
    def qf1(self):
        return self._qf1

    @qf1.setter
    def qf1(self, value):
        self._qf1 = value
        for sub_planner in self.sub_planners:
            sub_planner.qf1 = value

    @property
    def qf2(self):
        return self._qf2

    @qf2.setter
    def qf2(self, value):
        self._qf2 = value
        for sub_planner in self.sub_planners:
            sub_planner.qf2 = value

    @property
    def buffer_head(self):
        return self._buffer_head

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def initial_collect_episodes(self):
        return self._initial_collect_episodes

    @torch.no_grad()
    def _compute_rnd_penalty(self, plans, goal_states, replace_last=True):
        assert self.rnd is not None
        # plans: (num_samples, num_steps, 1)
        num_samples = plans.shape[0]
        num_steps = plans.shape[1]

        h1 = plans
        h1 = h1.view(-1, h1.shape[-1])  # num_samples * num_steps
        rnd_penalty, _  = self.rnd.get_reward(h1)  # h1 represents obs
        rnd_penalty = rnd_penalty.detach().view(num_samples, num_steps)[:, -1] 
        return  1 * rnd_penalty

    @torch.no_grad()
    def _generate_planned_trajectory(self, plans, init_state):
        batchsize, pred_length = plans.shape[0], plans.shape[1]
        pose_shape = 3
        planned_trajectory = np.zeros([batchsize, pred_length + 1, pose_shape])
        cur_pose = [0, 0, 0]
        planned_trajectory[:, 0] = torch.tensor(cur_pose).unsqueeze(0).repeat(batchsize, 1)
        init_state = init_state.to(self.device)
        j = 0
        for i in range(pred_length):
            h_t = plans[:, i, :]
            if i == 0:
                action_t = get_numpy(self._policy.act(None, init_state, h_t, deterministic=True))
            else:
                action_t = get_numpy(self._policy.act(None, pre_h, h_t, deterministic=True))
            rel_pose_i = action2pose(action_t, interval=2)
            pre_pose = copy.deepcopy(planned_trajectory[:, i, :])  #! use copy
            world_rel_pose_i = get_new_pose(pre_pose, rel_pose_i)  #! careful about the coordinate
            
            planned_trajectory[:, i + 1, :] = copy.deepcopy(world_rel_pose_i)
            pre_h = h_t

        delta_pose = torch.from_numpy(np.array(planned_trajectory[:, -1, :]) - np.array(cur_pose)).to(self.device)
        return planned_trajectory, delta_pose
        
        
    def _sample_z(self, num_samples, num_steps):

        if (np.random.rand() < 0.1 or
                self._buffer_head < self._initial_collect_episodes):
            self._frac_buffer = 0.0
        else:
            self._frac_buffer = 0.5

        num_buffer_samples = min(int(self._frac_buffer * num_samples),
                                 self._buffer_head)
        num_prior_samples = num_samples - num_buffer_samples

        z1 = self._sample_prior(num_prior_samples, num_steps)
        if num_buffer_samples > 0:
            print("num_buffer_samples:", num_buffer_samples)
            z2 = self._sample_buffer(num_buffer_samples, num_steps)
            z = torch.cat([z1, z2], 0)
        else:
            z = z1

        return z

    def _sample_prior(self, num_samples, num_steps):
        z = [self.affordance.sample_prior(num_samples, self.device)
             for t in range(num_steps)]
        z = np.stack(z, 1)
        z = u.from_numpy(z, self.device)
        return z

    def _sample_buffer(self, num_samples, num_steps):
        assert num_steps <= self._max_steps

        sampled_inds = np.random.choice(
            self._buffer_head,
            num_samples,
            replace=False)

        z = self._buffer[sampled_inds, :num_steps]
        z = u.from_numpy(z, self.device)
        return z

    def _add_to_buffer(self, z):
        num_steps = z.shape[0]
        self._buffer[self._buffer_head, :num_steps] = z
        self._buffer_head = (self._buffer_head + 1) % self._buffer_size
        return

    @torch.no_grad()
    def _predict(self, z=None, init_state=None, goal_state=None, 
                 rnn_afford=False, rnn_horizon=4, sample=True, num_steps=None):
        if self._predict_mode == 'affordance':
            assert self.affordance is not None
            if z is not None:
                num_steps = z.shape[1]
                bs = z.shape[0]
            else:
                assert num_steps is not None
                bs = init_state.shape[0]
            if rnn_afford:
                h_preds = [init_state[:, i] for i in range(init_state.shape[1])]
            else:
                h_preds = []
            h_t = init_state
            for t in range(num_steps):
                
                if rnn_afford:
                    assert hasattr(self.affordance, 'rnn')
                    length = min(len(h_preds), rnn_horizon)
                    batch_length = torch.tensor(length, dtype=torch.int64).repeat(bs)
                    
                    batch_feats = torch.stack(h_preds[-length:], dim=0).transpose(0, 1)
                    
                    h_t = self.affordance.rnn_features(batch_feats, batch_length)

                    if sample:
                        z_t = z[:, t]
                        h_pred = self.affordance.decode(z_t, cond=h_t).detach()
                    else:
                        (u_mu, u_logvar), u, h_pred = self.affordance(goal_state, batch_feats, batch_length)
                        
                else:
                    if sample:
                        z_t = z[:, t]
                        h_pred = self.affordance.decode(z_t, cond=h_t).detach()
                        
                    else:
                        (u_mu, u_logvar), u, h_pred = self.affordance(goal_state, h_t)
                    
                if self.encoding_type == 'vqvae':
                    _, h_pred = self.vqvae.vector_quantizer(h_pred)

                h_preds.append(h_pred.clone())
                h_t = h_pred

            h_preds = h_preds[init_state.shape[1]:] if rnn_afford else h_preds  # no initial_state
            plans = torch.stack(h_preds, 1)

        else:
            raise ValueError('Unrecognized prediction mode: %s'
                             % (self._predict_mode))

        return plans

    def _compute_costs(self, plans, init_states, goal_states, 
                       use_rnd=False, restrict_theta=False, zs=None):  # NOQA

        if self.encoding_type in ['vqvae']:
            start_dim = -3
        else:
            start_dim = -1

        if self._cost_mode == 'l2_vf':  #! default
            if len(init_states.shape) > 2:
                init_states = init_states[:, -1]
                    
            l2_dists_1 = l2_distance(plans[:, -1], goal_states, start_dim)

            values = compute_value(self.vf, plans, init_states, goal_states,
                                   replace_last=False)  # num_steps + 1(initial state)

            thresh = -10.0  # lower bound
            overage_all = torch.clamp(values - thresh, max=0.0)
            
            costs = (
                1. * l2_dists_1
                - 1. * overage_all.sum(1)  #*
            )

            z_costs = torch.mean(torch.norm(zs, dim=-1) ** 2, -1)
            costs += 0.1 * z_costs
            

            if hasattr(self, 'rnd'):
                rnd_penalty = self._compute_rnd_penalty(plans, goal_states)
                if use_rnd:
                    costs += rnd_penalty
            else:
                rnd_penalty = torch.zeros_like(costs)
                
            if restrict_theta:
                thred_theta = 0.25
                planned_traj, delta_pose = self._generate_planned_trajectory(plans, init_states)
                
                for j in range(len(delta_pose)):
                    delta_pose[j, 2] = torch.abs(_norm_heading(delta_pose[j, 2]))

                theta_cost = torch.clamp(delta_pose[:, 2] - thred_theta, max=0.0).squeeze()
                # tan = torch.atan2(delta_pose[:, 1], delta_pose[:, 0]).abs()
                # theta_cost = torch.clamp(tan - thred_theta, max=0.0).squeeze()
                
                costs -= 100 * theta_cost

            
            min_indx = torch.argmin(costs)
            print("min last value:", overage_all.sum(1)[min_indx].item(), "l2_dists:", l2_dists_1[min_indx].item(),"rnd penalty:", rnd_penalty[min_indx].item(), "z_costs:", z_costs[min_indx].item(), "theta cost:", theta_cost[min_indx].item()) 

        else:
            raise ValueError

        return costs, planned_traj


