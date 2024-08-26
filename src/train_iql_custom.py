"""The training of offline rl does not need agent to interact with env warper."""
import os
from arguments import get_args
args = get_args()  # get args from command
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['NUMEXPR_MAX_THREADS'] = "64"
import sys
from tensorboardX import SummaryWriter
import time
import logging
from yaml import full_load
from tqdm import tqdm, trange
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
writer = SummaryWriter(args.log_dir)

import torch
# torch.autograd.set_detect_anomaly(True)
from utils.trainer_utils import set_seed, numpy_to_torch, return_range, sample_batch, convert_to_batch_tensor, plot_training_losses, get_numpy, read_config, concat_batch

from trainer.build_trainer import build_iql_trainer
from dataset.custom_dataset import ImageGoalDataset
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='[%d/%b/%Y %H:%M:%S]')
logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_policy_custom(eval_dataloader, iql, step, device, writer, train_set=False):
    iql.eval()  #*
    value_diff = []
    action_diff = []
    q_values = []
    td_errors = []
    afford_preds = []
    afford_klds = []
    collision_penalty = -config['Data']['collision']
    for batch in tqdm(eval_dataloader):
        obs = batch.observations.to(device)
        imgs = batch.image_observations.to(device)
        actions = batch.actions.to(device)
        time_to_goals = batch.time_to_goals.to(device)
        img_goals = batch.image_goals.to(device) 
        rewards = batch.rewards.to(device)
        is_collision = batch.is_collision.to(device)
        next_imgs = batch.next_image_observations.to(device)
        terminals = batch.terminals.to(device)
        time_to_goals = torch.where(is_collision, time_to_goals + collision_penalty, time_to_goals)
        
        imgs_emb, (imgs_mu, _) = iql.image_encoder(imgs)
        next_imgs_emb, (next_imgs_mu, _) = iql.image_encoder(next_imgs) 
        img_goals_emb, (img_goals_mu, _) = iql.image_encoder(img_goals)
        stack_lengths = batch.stack_lengths.to(device)
        stack_imgs = batch.stack_imgs.to(device)
        ## v function
        if hasattr(iql, 'plan_vf'):
            vf_pred = -iql.plan_vf(obs, imgs_emb, img_goals_emb)
        else:
            vf_pred = -iql.vf(obs, imgs_emb, img_goals_emb)  # edge_cost = -V 
        value_diff.append(get_numpy((vf_pred - time_to_goals).abs().mean()))  

        ## action
        action_pred = iql.policy.act(obs, imgs_emb, img_goals_emb, deterministic=True)
        action_diff.append(get_numpy((action_pred - actions).abs().mean(dim=0)))
        
        ## affordance
        if hasattr(iql, 'affordance'):
            if iql.affordance.use_rnn or iql.affordance.use_attention:
                bn, seq_len = stack_imgs.shape[0], stack_imgs.shape[1]
                stack_imgs = stack_imgs.reshape(bn * seq_len, *stack_imgs.shape[2:])
                _, (stack_imgs_mu, _) = iql.image_encoder(stack_imgs)
                stack_imgs_mu = stack_imgs_mu.reshape(bn, seq_len, -1)
                afford_pred, afford_kld = iql._compute_affordance_loss(
                    stack_imgs_mu, img_goals_mu, stack_lengths)
            else:
                afford_pred, afford_kld = iql._compute_affordance_loss(
                    imgs_mu, img_goals_mu, stack_lengths)
            afford_preds.append(get_numpy(afford_pred))
            afford_klds.append(get_numpy(afford_kld))
            
        ## metric 1: average q value
        q_pred = torch.min(iql.target_qf1(obs, actions, imgs_emb, img_goals_emb),
                            iql.target_qf2(obs, actions, imgs_emb, img_goals_emb)).detach()
        q_values.append(get_numpy(q_pred.mean()))
        
        
        ## metric 2: TD error
        target_next_vf_pred = iql.vf(None, next_imgs_mu, img_goals_mu).detach().clamp(max=iql.max_value)
        q_target = iql.reward_scale * rewards + \
                (1. - terminals) * iql.discount * target_next_vf_pred
        td_errors.append(get_numpy((q_target - q_pred).abs().mean()))
        
        print("vf_pred:", vf_pred, " q_pred:", -q_pred, " time_to_goals:", time_to_goals)
        

    avg_q_value = np.mean(q_values)  #* metric 1: average q value for detecting overfitting
    avg_td_errors = np.mean(td_errors)  #* metric2: td error for detecting underfitting
    avg_value_diff = np.mean(value_diff)
    avg_action_diff = np.mean(action_diff, axis=0)
    iql.train()  #*
    verbose = f"Evaluate|train_set:{train_set}, step:{step}, value_diff:{avg_value_diff:.2f}, linear_diff:{avg_action_diff[0]:.2f}, angular_diff:{avg_action_diff[1]:.2f}, avg_q_value:{avg_q_value:.2f}, avg_td_errors:{avg_td_errors:.2f}"
    prefix = "Train" if train_set else "Val"
    if len(afford_preds):
        avg_afford_pred = np.mean(afford_preds) * iql.affordance_pred_weight
        avg_afford_kld = np.mean(afford_klds) * iql.affordance_beta
        verbose += f"avg_afford_pred:{avg_afford_pred:.2f}, avg_afford_kld:{avg_afford_kld:.2f}"
        writer.add_scalar(f"{prefix}/avg_afford_pred", avg_afford_pred, step)
        writer.add_scalar(f"{prefix}/avg_afford_kld", avg_afford_kld, step)
    print(verbose)
    writer.add_scalar(f"{prefix}/value_diff", avg_value_diff, step)
    writer.add_scalar(f"{prefix}/linear_diff", avg_action_diff[0], step)
    writer.add_scalar(f"{prefix}/angular_diff", avg_action_diff[1], step)
    writer.add_scalar(f"{prefix}/avg_q_value", avg_q_value, step)
    writer.add_scalar(f"{prefix}/avg_td_errors", avg_td_errors, step)
    


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.num_workers + 4)
    print("device:", device)

    print("get offline dataset.")
    train_dataset_dirs = config['train_data_dirs'][args.env_name]
    eval_dataset_dirs = config['eval_data_dirs'][args.env_name]
    train_val_divide = config['Data']['train_val_divide']
    train_dataset = ImageGoalDataset(train_dataset_dirs, device, config, train_val_divide,
                                     repeat=True,
                                     is_train=True)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=config['Train']['batch_size']  // 2,
        collate_fn=convert_to_batch_tensor,
        num_workers=args.num_workers // 2, 
        drop_last=False
    )
    train_dataset_neg = ImageGoalDataset(train_dataset_dirs, device, config, train_val_divide,
                                        is_positive=False,
                                        repeat=True, 
                                        is_train=True)
    train_neg_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset_neg,
        shuffle=True,
        batch_size=config['Train']['batch_size'] // 2,
        collate_fn=convert_to_batch_tensor, 
        num_workers=args.num_workers // 2, 
        drop_last=False
    )
    
    eval_train_dataset = ImageGoalDataset(train_dataset_dirs, device, config, train_val_divide, 
                                          repeat=False, is_train=True)
    eval_train_dataset = torch.utils.data.DataLoader(
        dataset=eval_train_dataset,
        shuffle=False,  # Sequential
        batch_size=config['Eval']['batch_size'],
        collate_fn=convert_to_batch_tensor, 
        num_workers=2, 
        drop_last=False
    )
    eval_dataset = ImageGoalDataset(eval_dataset_dirs, device, config, train_val_divide, 
                                    repeat=False, is_train=False)
    eval_dataloader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        shuffle=False,  # Sequential
        batch_size=config['Eval']['batch_size'],
        collate_fn=convert_to_batch_tensor, 
        num_workers=2, 
        drop_last=False
    )
    env = None
    
    set_seed(config['seed'], env)
    iql = build_iql_trainer(config).to(device)
    total = sum(p.numel() for p in iql.parameters())
    print("total model params: %.2fM." % (total / 1e6))
    
    start_step = 0
    if args.checkpoint_path is not None:
        print(f"load checkpoint {args.checkpoint_path}.")
        checkpoint = torch.load(args.checkpoint_path)
        iql.load_state_dict(checkpoint['model_state'])
        start_step = checkpoint['step']
        # scheduler_warmup = None
    # else:
    #     warm_iter_num = int(config['Train']['lr_warmup'] * len(train_dataloader))
    #     def linear_warmup(iter_num): return iter_num / \
    #         warm_iter_num if iter_num <= warm_iter_num else 1
    #     scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
    #         iql.optimizer, lr_lambda=linear_warmup)
    

    print("start training.")
    print("len of train_dataloader:", len(train_dataloader))
    iter_epoches = tqdm(range(start_step, config['Train']['max_train_steps'] // len(train_dataloader)))
    n_train_steps = 0
    first_epoch = True
    warm_up_steps = config['Train']['warm_up_steps']
    image_encode = config['IQL']['image_encode']
    for epoch in iter_epoches:
        iql.train()
        train_dataloader.dataset.set_epoch(epoch)
        iter_bar = tqdm(train_dataloader)
        epoch_rewards = 0
        epoch_qf1_loss = 0
        epoch_qf2_loss = 0
        epoch_vf_loss = 0
        epoch_policy_loss = 0
        epoch_afford_loss = 0
        i = 0
        iter_neg = iter(train_neg_dataloader)
        for batch_pos in iter_bar:
            iql.train()
            et = time.time()
            try:  # ! do not use cycle, otherwise data leakage occurs
                batch_neg = next(iter_neg)
            except StopIteration:
                iter_neg = iter(train_neg_dataloader)
                batch_neg = next(iter_neg)
            batch = concat_batch(batch_pos, batch_neg)
            
            if first_epoch and args.warm_up:
                print(f"warming up, training for {warm_up_steps} steps.")
                for _ in trange(warm_up_steps):
                    rewards, qf1_loss, qf2_loss, vf_loss, policy_loss, extra = iql(batch, image_encode=image_encode, warm_up=True)
                first_epoch = False
                eval_policy_custom(eval_train_dataset, iql, n_train_steps, device, writer, train_set=True)
            else:
                rewards, qf1_loss, qf2_loss, vf_loss, policy_loss, extra = iql(batch, image_encode=image_encode)
            
            # if epoch == 0 and scheduler_warmup is not None:
            #     scheduler_warmup.step()
            
            ## verbose
            verbose_text = f"Epoch:{epoch}|steps:{n_train_steps}|rewards:{rewards.item():.2f}|qf1:{qf1_loss.item():.2f}|qf2:{qf2_loss.item():.2f}|vf:{vf_loss.item():.2f}|policy:{policy_loss.item():.2f}"
            for k, v in extra.items():
                if v is not None and k != 'afford':
                    verbose_text += f"|{k}:{v.item():.2f}"
            iter_bar.set_description(verbose_text)

            ## plot metrics
            epoch_rewards += rewards.item()
            epoch_qf1_loss += qf1_loss.item()
            epoch_qf2_loss += qf2_loss.item()
            epoch_vf_loss += vf_loss.item()
            epoch_policy_loss += policy_loss.item()
            if extra['afford'] is not None:
                epoch_afford_loss += extra['afford'].item()
                
            if (n_train_steps + 1) % config['Eval']['eval_period'] == 0:
                eval_policy_custom(eval_dataloader, iql, n_train_steps, device, writer)

            if (n_train_steps + 1) % config['Train']['save_period'] == 0:
                torch.save({'model_state': iql.state_dict(),
                            "steps": n_train_steps,
                            'epoch': epoch},
                            os.path.join(args.save_dir, f'ckpt_{n_train_steps}.pt'))
                
            if (n_train_steps + 1) % config['Train']['plot_period'] == 0:
                eval_policy_custom(eval_train_dataset, iql, n_train_steps, device, writer,train_set=True)
                metric_params = {
                    "QF1_loss": qf1_loss,
                    "QF2_loss": qf2_loss,
                    "VF_loss": vf_loss,
                    "Policy_loss": policy_loss,
                }
                plot_training_losses(writer, metric_params, n_train_steps)
            n_train_steps += 1
            i += 1
        iter_epoches.set_description(f"Epoch:{epoch}, cur_steps:{n_train_steps}, avg_rewards:{epoch_rewards / i:.2f}, avg_qf1:{epoch_qf1_loss / i:.2f}, avg_qf2:{epoch_qf2_loss / i:.2f}, avg_vf:{epoch_vf_loss / i:.2f}, avg_policy:{epoch_policy_loss / i:.2f}")
    
    torch.save({'model_state': iql.state_dict(),
                "steps": n_train_steps,
                'epoch': epoch},
                os.path.join(args.save_dir, f'ckpt_{n_train_steps}.pt'))
        


if __name__ == '__main__':
    config = read_config(args)
    main(args, config)
