import cv2
import torch
import numpy as np
from torchvision import transforms as T
import os
import pickle
import copy
from .dataset_utils import get_files_ending_with
from utils.trainer_utils import transform_torch


class Dataset(torch.utils.data.Dataset):
    '''
    This class cannot be modified once a training task is launched, 
    Since at each epoch the process will reload this class while other files will not be reloaded.
    Thus the process will break as the updated files are not reloaded simultaneously. 
    '''
    def __init__(self, dataset_dir, augment_rate=0.9, use_raw=False):
        self.data_path_list = []
        self.augment_rate = augment_rate
        self.use_raw = use_raw
        for per_dir in dataset_dir:
            raw_path_list = os.listdir(per_dir)
            raw_path_list = [os.path.join(per_dir, name) for name in raw_path_list]     
            self.data_path_list.extend(raw_path_list)
        print(f"length of dataset:{len(self.data_path_list)}")

    
    def __len__(self):
        return len(self.data_path_list)


    def __getitem__(self, idx):
        img = cv2.imread(self.data_path_list[idx]) 
        if self.use_raw:       
            return transform_torch(img, train=True, augment_rate=self.augment_rate), transform_torch(img, train=False)
        else:
            return transform_torch(img, train=False, augment_rate=self.augment_rate)


class PairDataset(torch.utils.data.Dataset):
    '''
    This class cannot be modified once a training task is launched, 
    Since at each epoch the process will reload this class while other files will not be reloaded.
    Thus the process will break as the updated files are not reloaded simultaneously. 
    '''
    def __init__(self, dataset_dirs, augment_rate):
        self.data_path_list = []
        self.augment_rate = augment_rate
        for per_dir in dataset_dirs:
            raw_path_list = []
            get_files_ending_with(per_dir, raw_path_list, suffix='.pkl', pair=False)
            self.data_path_list.extend(list(raw_path_list))
        print(f"length of dataset:{len(self.data_path_list)}")

            
    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        with open(self.data_path_list[idx], 'rb') as f:
            try:
                instance = pickle.load(f)
            except:
                raise ValueError(f"file {self.data_path_list[idx]} is broken")
        img_pair = instance['pair']
        label = instance['label']

        img = transform_torch(img_pair[0], augment_rate=self.augment_rate)
        img_goal = transform_torch(img_pair[1], augment_rate=self.augment_rate)       
        return img, img_goal, torch.tensor(label, dtype=torch.float32)
    


class SaDataset(torch.utils.data.Dataset):
    '''
    This class cannot be modified once a training task is launched, 
    Since at each epoch the process will reload this class while other files will not be reloaded.
    Thus the process will break as the updated files are not reloaded simultaneously. 
    '''
    def __init__(self, dataset_dirs, 
                 train=True, 
                 augment_rate=0.9, 
                 contrastive=False, 
                 affordance=False, 
                 rnn_afford=False,
                 rnn_horizon=4,
                 rnd=False):
        self.images_list = []
        self.next_images_list = []
        self.actions_list = []
        self.distance_list = []
        self.contrastive = contrastive
        self.affordance = affordance
        self.rnn_afford = rnn_afford
        self.rnn_horizon = rnn_horizon
        self.rnd = rnd
        self.augment_rate = augment_rate
        self.train = train
        self.start_idx = []
        self.end_idx = []
        self.traj_range_dict = {}
        for i, per_dir in enumerate(dataset_dirs):
            with open(per_dir, 'rb') as f: 
                dataset = pickle.load(f)
            self.start_idx.extend([len(self.images_list)] * len(dataset["image_observations"]))
            self.traj_range_dict[i] = [len(self.images_list)]
            self.images_list.extend(copy.deepcopy(dataset["image_observations"]))  #! copy is necessary
            self.next_images_list.extend(copy.deepcopy(dataset["next_image_observations"])) #! copy is necessary
            self.actions_list.extend(dataset["actions"]) 
            self.distance_list.extend(dataset["timesteps"])
            self.end_idx.extend([len(self.images_list)] * len(dataset["image_observations"]))
            self.traj_range_dict[i].append(len(self.images_list))
        print(f"length of dataset:{len(self.images_list)}")

            
    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        img = self.images_list[idx]
        next_img = self.next_images_list[idx]
        action = self.actions_list[idx]    
        dist = self.distance_list[idx]
        label = np.array(action + [dist])
        
        if self.affordance:
            if self.rnn_afford:
                stack_imgs = []
                for j in range(max(idx - self.rnn_horizon + 1, int(self.start_idx[idx])), idx + 1):
                    stack_imgs.append(transform_torch(self.images_list[j], 
                                                    self.is_train, 
                                                    self.augment_rate))
                stack_lengths = torch.tensor(len(stack_imgs))
                stack_imgs = torch.stack(stack_imgs, dim=0)  # (rnn_horizon, img_shape)
            else:
                stack_imgs = None
                stack_lengths = None
            return transform_torch(img, self.train, self.augment_rate), transform_torch(next_img, self.train, self.augment_rate), stack_imgs, stack_lengths
        elif self.contrastive:
            idx = np.random.randint(self.start_idx[idx], self.end_idx[idx] - 2)
            pos_idx = np.random.randint(idx + 1, self.end_idx[idx] - 1)  # [, )
            neg_idx = np.random.randint(pos_idx + 1, self.end_idx[idx])  # [, )
            pos_img = self.images_list[pos_idx]
            neg_img = self.images_list[neg_idx]
            return transform_torch(img, self.train, self.augment_rate), transform_torch(img, train=False), transform_torch(pos_img, self.train, self.augment_rate), transform_torch(neg_img, self.train, self.augment_rate)
        elif self.rnd:
            return transform_torch(img, self.train, self.augment_rate), torch.tensor(action, dtype=torch.float32)
        else:
            return transform_torch(img, self.train, self.augment_rate)