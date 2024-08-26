import numpy as np
import pickle
from random import randint
from copy import deepcopy
import torch
import os
import copy
from torchvision import transforms as T

from dataset.dataset_utils import ImageGoalBatch, DataDict, _norm_heading
from utils.trainer_utils import transform_torch

    
class ImageGoalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dirs, device, config, 
                 train_val_divide=False, is_positive=True, repeat=False, is_train=True):
        
        self.dataset = deepcopy(DataDict)
        dataset_root = config['dataset_root']
        self.goal_range_max = config['Data']['goal_range_max']   # 29
        if dataset_dirs is None:
            dataset_dirs = sorted(os.listdir(dataset_root))
        if train_val_divide:
            file_name = 'train_idx_list.pkl' if is_train else 'eval_idx_list.pkl'
            divide_idx_path = os.path.join(dataset_root, '../', file_name)
            
            with open(divide_idx_path, 'rb') as f:
                divide_idx_list = pickle.load(f)
            dataset_dirs = np.array(dataset_dirs)[divide_idx_list]
            print("load dataset:", dataset_root, " len of dataset_dirs:", len(dataset_dirs), " is_train:", is_train)
        
        
        self.traj_range_dict = {}
        for i, data_dir in enumerate(dataset_dirs):
            with open(os.path.join(dataset_root, data_dir), 'rb') as f:
                data = pickle.load(f)
            max_range_indices = []
            lower_range_indices = []
            for idx in range(len(data["collision"])):
                max_range = min(self.goal_range_max, 
                                len(data["collision"]) - 1 - idx)
                max_range_indices.append(max_range)
                
                lower_range = min(self.goal_range_max, idx)
                lower_range_indices.append(lower_range)

            
            self.dataset['start_idx'].extend([len(self.dataset['collision'])] * 
                                             len(data['collision']))
            self.traj_range_dict[i] = [len(self.dataset['collision'])]
            for k, v in data.items():
                self.dataset[k].extend(v)
            
            self.dataset['max_range'].extend(max_range_indices)
            self.dataset['lower_range'].extend(lower_range_indices)
            self.dataset['traj_idx'].extend([i] * len(data['collision']))
            
            self.dataset['end_idx'].extend([len(self.dataset['collision'])] * 
                                           len(data['collision']))
            self.traj_range_dict[i].append(len(self.dataset['collision']))
            
        self.size = len(self.dataset['collision'])
        self.is_train = is_train
        self.repeat = repeat
        self.is_positive = is_positive
        self.repeat_num = config['Data']['dataset_repeat_time'] if self.repeat else 1
        self.norep_dataset = copy.deepcopy(self.dataset)
        for k, v in self.dataset.items():
            if len(v) != self.size and k not in ['terminals', 'actions_odom', 'frame_id', 'next_frame_id']:
                print(k, len(v), self.size)
                raise ValueError("inconsistent length of values")  
            if k in ['image_observations', 'next_image_observations']:
                self.dataset[k] = np.array(v, dtype=np.uint8)  #! image type should be uint8
            else:
                self.dataset[k] = np.array(v, dtype=np.float32)
            if self.repeat:
                self.dataset[k] = np.repeat(self.dataset[k], self.repeat_num, axis=0)

        # clip actions range
        self.dataset['actions'][:, 0] = self.dataset['actions'][:, 0].clip(min=-2, max=2) # linear range
        self.dataset['actions'][:, 1] = self.dataset['actions'][:, 1].clip(min=-1, max=1) # angular range
       
       
        # calculate reward    
        self.config = config
        self.collision = config['Data']['collision']
        self.success = config['Data']['success']
        self.lambda_dist = config['Data']['lambda_dist']
        self.lambda_angle = config['Data']['lambda_angle']
        self.dist_thred = config['Data']['dist_thred']
        self.angle_thred = config['Data']['angle_thred']
        self.step = config['Data']['goal_range_step']
        self.goal_range_min = config['Data']['goal_range_min']
        self.rewards_type = config["rewards_type"]
        self.curriculum = config["curriculum"]
        self.device = device
        self._epoch = 0
        self.no_aug = config["no_aug"]
        self.augment_rate = config['Data']['augment_rate']
        self.neg_sample = config['Data']['neg_sample']
        self.neg_fraction = config['Data']['neg_fraction']
        self.random_mask = config['random_mask']
        self.seg = config['seg']
        self.seg_category_num = config['Data']['seg_category_num']
        self.rnn_afford = config['Affordance']['rnn_afford']
        self.rnn_horizon = config['Affordance']['rnn_horizon']
    
    def __call__(self,):
        return self.dataset
    
    @property
    def obs_dim(self):
        return self.dataset['observations'].shape[1]
    
    @property
    def act_dim(self):
        return self.dataset['actions'].shape[1]
    
    def __len__(self):
        return len(self.dataset['observations'])
        
    
    @staticmethod
    def polar2goal(cur_pose, goal):
        goal_dist = np.linalg.norm(cur_pose[:2] - goal[:2])
        goal_angle = abs(_norm_heading(goal[2] - cur_pose[2]))
        return [goal_dist, goal_angle]
        
    def reward_relabel(self, cur_pose, next_pose, goal_pose, collision, reach_goal):
        """rewards normalization is unnecessary"""
        pre_goal_dist, pre_goal_angle = self.polar2goal(cur_pose, goal_pose)
        goal_dist, goal_angle = self.polar2goal(next_pose, goal_pose)
        if collision:
            return self.collision # -10
        elif reach_goal:
            return self.success if not self.rewards_type == "polar" else 10.0  # 0
        else:
            if self.rewards_type == "survival":
                return -1
            elif self.rewards_type == "polar":
                return (self.lambda_dist * (pre_goal_dist - goal_dist) + 
                     self.lambda_angle * (pre_goal_angle - goal_angle))
            else:
                raise ValueError(self.rewards_type)
    
    
    def set_epoch(self, epoch):
        self._epoch = epoch + 1
        
    
    def get_epoch(self):
        return self._epoch
        
    
    def __getitem__(self, idx):
        max_range = int(self.dataset["max_range"][idx])
        lower_range = int(self.dataset["lower_range"][idx])
        if self.is_train:
            ori_idx = idx // self.repeat_num
            
            if self.repeat and not self.is_positive and self.neg_sample:
                if ori_idx + max_range + 1 < self.dataset['end_idx'][idx]: 
                    idx2goal = randint(max_range + 1, self.dataset['end_idx'][idx] - ori_idx - 1)
                    goal = int(idx + self.repeat_num * idx2goal)
                else:
                    goal = -1  # vertically flip
                    idx2goal = max_range + 1
                
            elif idx % self.repeat_num < 2 and self.repeat:
                idx2goal = 0
                goal = idx
            else:
                idx2goal = randint(self.goal_range_min, max_range)
                goal = int(idx + self.repeat_num * idx2goal)
        else:
            idx2goal = max_range 
            goal = int(idx + self.repeat_num * idx2goal)
            
        #* pose is not necessary to use
        cur_pose = self.dataset['observations'][idx]
        next_pose = self.dataset['next_observations'][idx]
        goal_pose = self.dataset['next_observations'][goal]
        reach_goal = idx == goal     
        if self.rewards_type == "temporal":
            if reach_goal:
                reward = self.success
            else:
                reward = -(self.dataset['next_timesteps'][idx] - self.dataset['timesteps'][idx])
        else:
            reward = self.reward_relabel(cur_pose, 
                                        next_pose, 
                                        goal_pose,  
                                        self.dataset['collision'][idx],
                                        reach_goal)
        
        if self.rnn_afford: # or self.att_afford:
            ori_idx = idx // self.repeat_num
            stack_imgs = []
            for j in range(max(ori_idx - self.rnn_horizon + 1, int(self.dataset['start_idx'][idx])), ori_idx + 1):
                stack_imgs.append(transform_torch(self.dataset['image_observations'][j * self.repeat_num],
                                               (self.is_train and not self.no_aug), 
                                                self.augment_rate,
                                                self.random_mask))
            stack_lengths = torch.tensor(len(stack_imgs))
            # stack_imgs += [torch.zeros(*self.dataset['image_observations'][idx].shape)] * \
            #         (self.rnn_horizon - stack_length)  # padding
            stack_imgs = torch.stack(stack_imgs, dim=0)  # (rnn_horizon, img_shape)
        else:
            stack_imgs = None
            stack_lengths = None
        
        if goal == -1:
            image_goals = np.fliplr(self.dataset['image_observations'][idx]).copy()
        else:
            image_goals = self.dataset['next_image_observations'][goal]
            
        return ImageGoalBatch(
            raw_observations=self.dataset['image_observations'][idx], 
            observations=self.polar2goal(cur_pose, goal_pose),
            image_observations=transform_torch(self.dataset['image_observations'][idx], 
                                               (self.is_train and not self.no_aug), 
                                               self.augment_rate,
                                               self.random_mask),
            actions=self.dataset['actions'][idx],
            rewards=reward, 
            next_observations=self.polar2goal(next_pose, goal_pose),
            next_image_observations=transform_torch(self.dataset['next_image_observations'][idx], 
                                                    (self.is_train and not self.no_aug), 
                                                    self.augment_rate,
                                                    self.random_mask),
            next_actions=self.dataset['actions'][idx + 1 * self.repeat_num] \
                        if not int(reach_goal) and self.is_positive \
                        else np.zeros(2),
            image_goals=transform_torch(image_goals, 
                                        (self.is_train and not self.no_aug), 
                                        self.augment_rate,
                                        self.random_mask),  
            pose_to_goals=self.polar2goal(cur_pose, goal_pose),
            time_to_goals=min(idx2goal, max_range + 1),
            is_positive=int(idx2goal <= max_range),
            is_collision=self.dataset['collision'][idx],
            terminals=int(self.dataset['collision'][idx] or reach_goal),
            stack_imgs=stack_imgs,
            stack_lengths=stack_lengths,
        )
    
