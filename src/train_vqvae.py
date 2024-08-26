
from arguments import get_args
import os
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['NUMEXPR_MAX_THREADS'] = "64"

import sys
import cv2
import random
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset.img_dataset import Dataset, SaDataset
from utils.trainer_utils import read_config, convert_to_image_batch, convert_to_afford_batch, convert_to_contrast_batch
from models.encoders.vqvae import VqVae
writer = SummaryWriter(args.log_dir)
config = read_config(args)
epochs = config["Train"]["max_train_steps"]
batch_size = config["Train"]["batch_size"]

augment_rate = config['Data']['augment_rate']
train_dataset_dirs = [
    ""
]

print("num of dataset dirs:", len(train_dataset_dirs))
train_dataset = SaDataset(train_dataset_dirs, augment_rate=augment_rate, contrastive=args.contrastive)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    collate_fn=convert_to_contrast_batch if args.contrastive else convert_to_image_batch
)
epochs = 200
gradient_clip_value = 1.0
embedding_dim = config['VQVAE']['embedding_dim']
hidden_dim = config['VQVAE']['hidden_dim']
vqvae = VqVae(embedding_dim=embedding_dim, hidden_dim=hidden_dim, contrastive_loss=args.contrastive).to(device)
optimizer = optim.AdamW(vqvae.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, epochs)
for epoch in tqdm(range(epochs)): 
    running_loss = 0.0
    iter_loader = tqdm(train_loader)
    for batch in iter_loader:
        optimizer.zero_grad()
        if args.contrastive:
            input_imgs, raw_inputs, pos_imgs, neg_imgs = batch
            input_imgs = input_imgs.to(device)
            raw_inputs = raw_inputs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)
        else:
            input_imgs = batch
            input_imgs = input_imgs.to(device)
            raw_inputs = pos_imgs = neg_imgs = None
        
        loss, extra = vqvae.compute_loss(input_imgs, raw_inputs, pos_imgs, neg_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_vq = extra['loss_vq'].item()
        loss_recon = extra['loss_recon'].item()
        loss_contrast1 = extra['loss_contrast1'].item()
        loss_contrast2 = extra['loss_contrast2'].item()
        iter_loader.set_description(f"loss:{loss.item():.2f}|loss_vq:{loss_vq:.2f}|loss_recon:{loss_recon:.2f}|loss_contrast1:{loss_contrast1:.2f}|loss_contrast2:{loss_contrast2:.2f}")
        
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"epoch:{epoch}, avg_loss:{avg_loss:.2f}")
    
    if (epoch + 1)  % 50 == 0:
        torch.save({'model_state': vqvae.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch},
                    os.path.join(args.save_dir, f'ckpt_{epoch}.pt'))

