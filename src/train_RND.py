
from arguments import get_args
import os
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['NUMEXPR_MAX_THREADS'] = "64"

import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset.img_dataset import SaDataset
from utils.trainer_utils import read_config, convert_to_image_action_batch, get_numpy, update
from models.RND import RND, RND_SA
from trainer.build_trainer import build_iql_trainer
writer = SummaryWriter(args.log_dir)
config = read_config(args)
epochs = 4000
batch_size = config["Train"]["batch_size"]


train_dataset_dirs = [
    ""
]

train_dataset = SaDataset(train_dataset_dirs, train=True, rnd=True)
eval_dataset_dirs = [
    "" 
]
eval_dataset = SaDataset(eval_dataset_dirs, train=False, rnd=True)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    collate_fn=convert_to_image_action_batch
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    drop_last=False,
    collate_fn=convert_to_image_action_batch
)
iql_pretrain_path = ''
pre_args_path = iql_pretrain_path.split('/')[:-1]
pre_args_path.append('args.json')
pre_args_path = '/'.join(pre_args_path)
with open(pre_args_path, 'r') as f:
    pre_args = json.load(f)
config = update(config, pre_args)

iql = build_iql_trainer(config, iql_pretrain_path).to(device)
iql.eval()
hidden_dim = config['RND']['hidden_dim']
encoder_dim = config['IQL']['encoder_dim']
RND = RND(hidden_dim, enc_dim=encoder_dim).to(device)
optimizer = torch.optim.AdamW(RND.predictor.parameters(),lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, epochs)

print('initialize mean and std')
for input_imgs, actions in tqdm(train_loader):
    input_imgs = input_imgs.to(device)
    actions = actions.to(device)
    encoding, (mu, logvar) = iql.image_encoder(input_imgs)
    RND._running_obs.update(get_numpy(mu))
RND.save_running_params()

for epoch in tqdm(range(epochs + 1)):
    running_rewards = []
    running_normalized_rewards = []
    iter_loader = tqdm(train_loader)
    for input_imgs, actions in iter_loader:
        input_imgs = input_imgs.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        encoding, (mu, logvar) = iql.image_encoder(input_imgs)
        input_imgs = mu.detach()
        rewards, normalized_reward = RND.get_reward(input_imgs)
        rewards.backward()
        optimizer.step()
        
        running_rewards.append(rewards.item())
        running_normalized_rewards.append(normalized_reward.item())
        iter_loader.set_description(f"rewards:{rewards.item():.2f}, normalized:{normalized_reward.item():.2f}")


    scheduler.step()
    avg_rewards = np.mean(running_rewards) 
    avg_normalized_rewards = np.mean(running_normalized_rewards)
    print(f"epoch:{epoch}, avg_rewards:{avg_rewards:.2f}, avg_normalized_rewards:{avg_normalized_rewards:.2f}")
    if (epoch + 1) % 100 == 0:
        writer.add_scalar("avg_rewards", avg_rewards, epoch)
        running_rewards = []
        running_normalized_rewards = []
        RND.predictor.eval()
        iter_eval = tqdm(eval_loader)
        with torch.no_grad():
            for input_imgs, actions in iter_eval:
                input_imgs = input_imgs.to(device)
                actions = actions.to(device)
                encoding, (mu, logvar) = iql.image_encoder(input_imgs)
                input_imgs = mu.detach()
                
                rewards, normalized_reward = RND.get_reward(input_imgs)
                running_rewards.append(rewards.item())
                running_normalized_rewards.append(normalized_reward.item())
                iter_loader.set_description(f"rewards:{rewards.item():.2f}, normalized:{normalized_reward.item():.2f}")
        avg_rewards = np.mean(running_rewards)
        avg_normalized_rewards = np.mean(running_normalized_rewards)
        print(f"evaluation: epoch:{epoch}, avg_rewards:{avg_rewards:.2f}, avg_normalized_rewards:{avg_normalized_rewards:.2f}")
        writer.add_scalar("eval_avg_rewards", avg_rewards, epoch)
        RND.predictor.train()
    if (epoch + 1)  % 1000 == 0:
        RND.save_running_params()
        print('save RND at epoch:', epoch)
        torch.save({'model_state': RND.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch},
                    os.path.join(args.save_dir, f'ckpt_{epoch}.pt'))

