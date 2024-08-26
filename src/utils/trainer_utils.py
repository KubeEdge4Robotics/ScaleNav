import torch
import random
import os
import numpy as np
import time
import json
from yaml import full_load
import torch.nn as nn
import collections.abc
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence

from dataset.dataset_utils import Batch, ImageGoalBatch, _list_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def from_numpy(inputs, device):
    if isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs).to(device).float()
    else:
        return inputs.to(device).float()


def get_numpy(tensor):
    return tensor.cpu().detach().numpy()

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def is_main_device(device):
    return isinstance(device, torch.device) or device == 0


def _process_image(img, _device, _train=False):
    return transform_torch(img.astype(np.uint8), train=_train).unsqueeze(0).to(_device) # .unsqueeze(0)
    
def set_seed(seed, env=None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)

def plot_training_losses(writer, metric_params, _n_train_steps):
    # global writer
    for k, v in metric_params.items():
        writer.add_scalar(f"Train/{k}", v, _n_train_steps)
    # writer.add_scalar("Train/Rewards", rewards, _n_train_steps)
    # writer.add_scalar("Train/QF1_loss", qf1_loss, _n_train_steps)
    # writer.add_scalar("Train/QF2_loss", qf2_loss, _n_train_steps)
    # writer.add_scalar("Train/VF_loss", vf_loss, _n_train_steps)
    # writer.add_scalar("Train/Policy_loss", policy_loss, _n_train_steps)
        

def read_config(args):
    try:
        config_path = 'src/configs/' + str(args.config_dir)
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        config = full_load(content)
        config['env_name'] = args.env_name
        config['exp_time'] = time.asctime(time.localtime(time.time()))
        var_args = vars(args)
        config.update(var_args)
        if not args.test:
            if args.save_dir is None:
                args.save_dir = os.path.join('experiments', args.log_dir.split('/')[-1])
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(config, f)
    except:
        assert False, f'config [{args.config_dir}] does not exist.'
    return config

def convert_to_image_batch(batch):
    inputs = []
    for img in batch:
        inputs.append(img)
    inputs = torch.stack(inputs).type(torch.float32)
    return inputs

def convert_to_contrast_batch(batch):
    images, raw_imgs, pos_imgs, neg_imgs = [], [], [], []
    for (img, raw_img, pos_img, neg_img) in batch:
        images.append(img)
        raw_imgs.append(raw_img)
        pos_imgs.append(pos_img)
        neg_imgs.append(neg_img)
        
    images = torch.stack(images).type(torch.float32)
    raw_imgs = torch.stack(raw_imgs).type(torch.float32)
    pos_imgs = torch.stack(pos_imgs).type(torch.float32)
    neg_imgs = torch.stack(neg_imgs).type(torch.float32)
    return images, raw_imgs, pos_imgs, neg_imgs

def convert_to_afford_batch(batch):
    images, next_imgs, stack_imgs, stack_lengths = [], [], [], []
    for (img, next_img, stack_img, stack_length) in batch:
        images.append(img)
        next_imgs.append(next_img)
        if stack_img is not None:
            stack_imgs.append(stack_img)
            stack_lengths.append(stack_length)
    images = torch.stack(images).type(torch.float32)
    next_imgs = torch.stack(next_imgs).type(torch.float32)
    if stack_imgs:
        stack_imgs = torch.stack(stack_imgs).type(torch.float32)
        stack_lengths = torch.stack(stack_lengths).type(torch.int64)
    else:
        stack_imgs = stack_lengths = torch.tensor([0])
    return images, next_imgs

def convert_to_recon_batch(batch):
    images, next_imgs, labels = [], [], []
    for (img, next_img, label) in batch:
        images.append(img)
        next_imgs.append(next_img)
        labels.append(label)
    images = torch.stack(images).type(torch.float32)
    next_imgs = torch.stack(next_imgs).type(torch.float32)
    labels = torch.stack(labels).type(torch.float32)
    return images, next_imgs, labels


def convert_to_image_action_batch(batch):
    images, actions = [], []
    for img, action in batch:
        images.append(img)
        actions.append(action)
    images = torch.stack(images).type(torch.float32)
    actions = torch.stack(actions).type(torch.float32)
    # images = np.stack(images).astype(np.float32)
    # actions = np.stack(actions).astype(np.float32)
    return images, actions

def convert_to_image_dist_batch(batch):
    imgs, imgs_goal, labels = [], [], []
    for img, img_goal, label in batch:
        imgs.append(img)
        imgs_goal.append(img_goal)
        labels.append(label)
    imgs = torch.stack(imgs).type(torch.float32)
    imgs_goal = torch.stack(imgs_goal).type(torch.float32)
    labels = torch.stack(labels).type(torch.float32)
    return imgs, imgs_goal, labels

def convert_to_batch_tensor(batch):
    observations = []
    image_observations = []
    actions = []
    rewards = []
    next_observations = []
    next_image_observations = []
    next_actions = []
    image_goals = []
    pose_to_goals = []
    time_to_goals = []
    terminals = []
    is_positive = []
    is_collision = []
    raw_observations = []
    stack_imgs = []
    stack_lengths = []
    for sample in batch:
        observations.append(sample.observations)
        image_observations.append(sample.image_observations)
        actions.append(sample.actions)
        rewards.append(sample.rewards)
        next_observations.append(sample.next_observations)
        next_image_observations.append(sample.next_image_observations)
        next_actions.append(sample.next_actions)
        image_goals.append(sample.image_goals)
        pose_to_goals.append(sample.pose_to_goals)
        time_to_goals.append(sample.time_to_goals)
        terminals.append(sample.terminals)
        is_positive.append(sample.is_positive)
        is_collision.append(sample.is_collision)
        raw_observations.append(sample.raw_observations)
        if sample.stack_imgs is not None:
            stack_imgs.append(sample.stack_imgs)
            stack_lengths.append(sample.stack_lengths)
        
    return ImageGoalBatch(
            observations=_list_to_tensor(observations),
            image_observations=torch.stack(image_observations, dim=0), 
            actions=_list_to_tensor(actions),
            rewards=_list_to_tensor(rewards),
            next_observations=_list_to_tensor(next_observations), 
            next_image_observations=torch.stack(next_image_observations, dim=0),
            next_actions=_list_to_tensor(next_actions),
            image_goals=torch.stack(image_goals, dim=0),
            pose_to_goals=_list_to_tensor(pose_to_goals),
            time_to_goals=_list_to_tensor(time_to_goals),
            terminals=_list_to_tensor(terminals),
            is_positive=_list_to_tensor(is_positive, _dtype=torch.bool),
            is_collision=_list_to_tensor(is_collision, _dtype=torch.bool),
            raw_observations=_list_to_tensor(raw_observations),
            stack_imgs=pad_sequence(stack_imgs, batch_first=True, padding_value=0) \
                        if stack_imgs else torch.ones(1),
            stack_lengths=_list_to_tensor(stack_lengths, _dtype=torch.int64) \
                        if stack_lengths else torch.ones(1)
        )

def concat_batch(batch_pos, batch_neg):
    batch_pos_dict = batch_pos._asdict()
    batch_neg_dict = batch_neg._asdict()
    batch = {}
    for k, vp in batch_pos_dict.items():
        vn = batch_neg_dict[k]
        # print(k, vn.shape, vp.shape)
        batch[k] = torch.cat((vp, vn), dim=0)
    newbatch = ImageGoalBatch(**batch)
    # print(newbatch._fields)

    return newbatch

def _share_encoder(source, target):
    # Use critic conv layers in actor:
    # target.image_enc.parameters().data.copy_(source.image_enc.parameters())
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for k, v in source_dict.items():
        if 'image_enc' in k:
            target_dict[k] = v
    target.load_state_dict(target_dict)
    # for (t_name, target_param), (name, param) in zip(target.named_parameters(), source.named_parameters()):
    #     print(t_name, name)
    #     if 'image_enc' in name:
    #         target_param.data.copy_(
    #             param
    #         )


def transform_torch(img, train=True, augment_rate=0.9, random_mask=False):
    """
    img: (h, w, c)
    return: normalized torch.tensor(c, h, w)
    """
    assert img.dtype == np.uint8 or img.dtype == torch.uint8
    width = height = 256
    if train:
        transforms_stack = [
            T.ToPILImage(),
            T.Resize((width, height), antialias=True),  # T.Resize(256),
        ]
        if random.random() < augment_rate:
            transforms_stack.extend(_augmentations(width, height))
            
        # augmentations_stack.append(T.RandomErasing())  # does not support PIL Image.
        transforms_stack.extend([T.ToTensor(),  #! ToTensor() should be at the last two layer
                                 T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        if random_mask:
            transforms_stack.insert(-1, T.RandomErasing()) # does not support PIL Image.
        transform = T.Compose(transforms_stack)
    else:
        transforms_stack = [
            T.ToPILImage(),
            T.Resize((width, height), antialias=True),  # (256, 256)
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
            ]
        transform = T.Compose(transforms_stack)
    # out = transform(img)
    # return out - 0.5  # [-0.5, 0.5]
    return transform(img)


def _augmentations(width=256, height=256):
    RandomResizedCrop_params = dict(
        size=(width, height),
        scale=(0.9, 1.0),
        ratio=(0.9, 1.1),
    )
    ColorJitter_params = dict(
        brightness=(0.75, 1.25),
        contrast=(0.9, 1.1),
        saturation=(0.9, 1.1),
        hue=(-0.1, 0.1),
    )
    sharpness_factor = random.uniform(0.5, 1.5)
    augmentations_stack = [
        T.RandomResizedCrop(**RandomResizedCrop_params), 
        T.ColorJitter(**ColorJitter_params),
        T.RandomAdjustSharpness(sharpness_factor),
    ]
    
    # return T.Compose(augmentations_stack)
    return augmentations_stack


def numpy_to_torch(x, device=None):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    if device is not None:
        x = x.to(device=device)
    return x


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:  #* -1
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    start_time = time.time()
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    
    res = {k: v[indices] for k, v in dataset.items()}
    end_time = time.time()
    return Batch(**res)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon
        self.std = np.ones(shape, 'float32')

    def __repr__(self):
        return f'mean:{self.mean.shape}, var:{self.var.shape}'
    
    # @property
    # def get_std(self):
    #     return np.sqrt(self.var) if (self.std == 1).all() else self.std
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        n = self.count + batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / n
        new_var = M2 / n

        new_count = batch_count + self.count
        
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        self.std = np.sqrt(self.var)

def get_rnn_imgs(idx, eval_images_list, start_idx=0, rnn_horizon=4):
    stack_imgs = []
    for j in range(max(idx - rnn_horizon + 1, start_idx), idx + 1):
        stack_imgs.append(transform_torch(eval_images_list[j], train=False))
    
    stack_imgs = torch.stack(stack_imgs, dim=0).to(device)
    stack_lengths = torch.tensor(len(stack_imgs)).type(torch.int64)
    return stack_imgs, stack_lengths