from arguments import get_args
import os
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['NUMEXPR_MAX_THREADS'] = "64"

import rospy
import sys
import cv2
import random
import json
import pickle
import queue
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset.img_dataset import Dataset, SaDataset
from utils.trainer_utils import read_config, convert_to_image_batch, convert_to_image_action_batch, convert_to_afford_batch, transform_torch, get_numpy, update, _process_image

from trainer.build_trainer import build_iql_trainer
from dataset.custom_dataset import ImageGoalDataset
from dataset.dataset_utils import _norm_heading, get_rel_pose_change, get_new_pose, get_action, action2pose
from utils.ros_interface import ROSInterface
from planning.mppi_planner import MppiPlanner



class Deploy(object):
    value_thred = 10
    def __init__(self, config, args, model_path, eval_images_list, eval_poses_list):
        self.eval_images_list = eval_images_list
        self.eval_poses_list = eval_poses_list
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self._revise_mode = args.revise_mode
        self.no_ros = args.no_ros
        self.max_dist = config['Data']['goal_range_max'] + 1
        self.hist_states = []
        self.rnn_horizon = config['Affordance']['rnn_horizon']

        
        print('loading pretrained iql:', model_path)
        self.iql = build_iql_trainer(config, model_path).to(self.device)
        self.iql.eval()
        
        print('loading mppi planner')
        z_dim = config['Affordance']['z_dim']
        num_samples = config['Affordance']['num_samples']  # 1024
        num_iters = config['Affordance']['num_iters']  # 3
        self.planner = MppiPlanner(model=self.iql, num_samples=num_samples,
                        num_iters=num_iters, representation_size=z_dim, 
                        device=self.device, debug=True)
        

        if not self.no_ros:
            self.ros_api = ROSInterface(args)
            print('initializing ros interface.')
        else:
            print('test in no ros env.')
    
    def planning(self, 
                init_state, 
                goal_state,
                hist_states,
                cur_img_idx=None,
                goal_img_idx=None,
                num_steps = 4, 
                num_rollouts = 1, 
                restrict_theta = True,
                use_rnd = True,
                rnn_afford = True, 
                visualize = False,
                save_path = './utils/deploy_vis/'
                ):
        assert hasattr(self.planner, 'affordance')
        print('use rnd:', use_rnd, 'has rnd:', hasattr(self.planner, 'rnd'), 'use rnn:', rnn_afford)

        if visualize and goal_state is not None:
            assert cur_img_idx is not None and goal_img_idx is not None
            cur_pose = np.array(self.eval_poses_list[cur_img_idx])
            goal_pose = np.array(self.eval_poses_list[goal_img_idx])
            
            action_gt = get_action(cur_pose, goal_pose)
            trajectory_gt = np.array([pose for pose in self.eval_poses_list[cur_img_idx: goal_img_idx + 1]])
            rel_pose_gt = get_rel_pose_change(cur_pose, goal_pose)
            world_rel_pose_gt = get_new_pose(np.zeros(3), rel_pose_gt)
            world_rel_pose_gt2 = goal_pose - cur_pose
        else:
            cur_pose = np.array([0, 0, 0])
        
        stack_lengths = torch.tensor([0])
        initial_zs = None
        act = np.array([0.25, 0])
        rel_pos = action2pose(act, interval=2)
        for j in range(num_rollouts): 
        
            if rnn_afford:
                hist_states.append(init_state)  # T, B, D
                init_state = torch.stack(hist_states).transpose(1, 0)  # B, T, D

                stack_lengths = torch.tensor(init_state.shape[0])  
            init_state = init_state.squeeze(0)
                
            for use_rnd in [True]:
                plan, info = self.planner._plan(init_state, stack_lengths, goal_state, num_steps,
                                        initial_zs, use_rnd, restrict_theta, rnn_afford, input_info=None)
                initial_zs = info['initial_zs']
                # print("plan:", plan.shape) # [5, 128]
    
                if visualize:
                    optimal_trajectory, _ = self.planner._generate_planned_trajectory(plan.unsqueeze(0), init_state)  # B, T, D
                    optimal_trajectory = optimal_trajectory.squeeze(0)
                    
                    
                    optimal_trajectory = np.array(optimal_trajectory)
                    t = np.arange(0, len(optimal_trajectory), 1)
                    #text = f"rnd_{j}" if use_rnd else str(j) 
                    text = 'planning w/ RND' if use_rnd else 'planning w/o RND'
                    plt.plot(-optimal_trajectory[:, 1], optimal_trajectory[:, 0], marker='o', label=text)
                    
                    last_point = info['last_point']
                    color = 'bisque' if not use_rnd else 'cyan'
                    text = 'last point area w/ RND' if use_rnd else 'last point area w/o RND'
                    plt.scatter(-last_point[:, 1], last_point[:, 0], color=color, label=text)
    
                    
                    world_rel_pose = get_new_pose(cur_pose, rel_pos)
                    initial_heading = np.stack((cur_pose, world_rel_pose))
                    #plt.plot(-initial_heading[:, 1], initial_heading[:, 0], label='initial_heading', color='black', linewidth=2.0)
                    plt.arrow(0, 0, -world_rel_pose[1], world_rel_pose[0],
                            color='black', width=0.01, shape='full')
                            
                    
                    world_last_point = get_new_pose(optimal_trajectory[-1], rel_pos)
                    #last_heading = np.stack((optimal_trajectory[-1], world_last_point))        
                    plt.arrow(-optimal_trajectory[-1][1], optimal_trajectory[-1][0],
                           -(world_last_point[1] - optimal_trajectory[-1][1]), world_last_point[0] - optimal_trajectory[-1][0],
                            color='pink', width=0.01, shape='full')
                    
                    if goal_state is not None:
                      print("use_rnd:", use_rnd, j, optimal_trajectory[-1] - goal_pose) #world_rel_pose_gt2)
                      plt.plot(-trajectory_gt[:, 1], trajectory_gt[:, 0], marker='.', label='gt')
                      plt.scatter(-goal_pose[1], goal_pose[0], marker='*')  
            
            
            #goal_img = cv2.imread('./images/start.jpeg')
            #goal_img_emb = self.embed_obs(goal_img)
            #value = self.predict_value(init_state, goal_img_emb)
            #plt.text(0, 0, 'value=-' + str(int(value)))
            #plt.scatter(0, 0, color='red', label='start')
            #plt.legend()
            #plt.title('local coordinate')
            #os.makedirs(save_path, exist_ok=True)
            #plt.savefig(save_path + 'plan_1.jpeg')
            #plt.show()
            #plt.clf()
            #print("save fig")
        
        return plan
            
    
    def localize(self, cur_img_emb:Tensor, start_idx=0, span=10, verbose=True):
        localized_list = []
        for idx in range(
            max(0, start_idx-2), min(start_idx + span, len(self.eval_images_list))
        ):
            
            img_emb = self.embed_obs(self.eval_images_list[idx])
            value = self.predict_value(cur_img_emb, img_emb)
            
            localized_list.append((idx, value))
            if verbose:
                print('{},{}'.format(idx, value))  # timestep
                
        if len(localized_list) == 0:
            # * handle cases when localization fail
            print((" ======= Error! Fail to localize the image!")) 
            return None
        else:
            final_localized_idx, min_value = localized_list[
                np.argmin(np.array(localized_list)[:, 1])
            ]
            
            print('localized_idx:', final_localized_idx, 'min_value:', min_value)
            return final_localized_idx
   
    def backtracking(self, i, pre_localized_idx, next_interval):
        rospy.loginfo("launch backtracking")
        dist_list = []
        next_waypoint_idx = pre_localized_idx + next_interval
        goal_emb = self.embed_obs(self.eval_images_list[next_waypoint_idx])
        for (img, _) in self.ros_api._buffer:
            img_emb = self.embed_obs(img)
            dist = self.predict_value(img_emb, goal_emb)
            dist_list.append(dist)
        backtrack_step = np.argmin(dist_list) if np.min(dist_list) <= self.max_dist else 0
        (_, nearest_pose) = self.ros_api._buffer[backtrack_step]
        rospy.loginfo("find min dist in buffer:{}".format(np.min(dist_list)))
        cur_pose = self.ros_api.get_odom()
        local_rel_pose = get_rel_pose_change(cur_pose, nearest_pose)
        
        rospy.loginfo("backtrack to last {} step".format(len(dist_list) - 1 - backtrack_step))
        self.ros_api.reach_next_waypoint(i, local_rel_pose, backtrack=True)
        
        if np.min(dist_list) <= self.value_threshold:
            return next_waypoint_idx
        else:
            return pre_localized_idx

    @torch.no_grad()
    def model_infer(self, cur_img_emb:Tensor, img_goal_emb:Tensor):
        value = self.predict_value(cur_img_emb, img_goal_emb)
        action = self.predict_action(cur_img_emb, img_goal_emb)
        action_tensor = torch.from_numpy(action).to(self.device).unsqueeze(0)
        q_value = self.predict_q_value(cur_img_emb, img_goal_emb, action_tensor)

        return value, q_value, action
    
    def predict_action(self, cur_img_emb:Tensor, img_goal_emb:Tensor):
        action = self.iql.policy.act(None, cur_img_emb, img_goal_emb, deterministic=True)
        return get_numpy(action)[0]
    
    def predict_value(self, cur_img_emb:Tensor, img_goal_emb:Tensor):
        value = -self.iql.vf(None, cur_img_emb, img_goal_emb)
        value = value.clamp(max=self.max_dist)
        return get_numpy(value)[0]
    
    def predict_q_value(self, cur_img_emb:Tensor, img_goal_emb:Tensor, action:Tensor):
        q_value = torch.min(self.iql.target_qf1(None, action, cur_img_emb, img_goal_emb),
                            self.iql.target_qf2(None, action, cur_img_emb, img_goal_emb))
        return get_numpy(q_value)[0]
    
    def embed_obs(self, obs):
        obs_torch = _process_image(obs, _device=self.device)
        obs_emb, _ = self.iql.image_encoder(obs_torch)
        return obs_emb
    
    def get_real_time_obs_emb(self):
        cur_img = self.ros_api.get_real_time_img()
        return self.embed_obs(cur_img)
        

    @torch.no_grad()
    def navigate(self, cur_img=None, goal_img=None, goal_idx=None, next_interval=2):
        i = 0
        default_span = 10
        finish = False
        pre_localized_idx = localized_idx = 0  + len(self.eval_images_list) // 2 
        collision = False
        if self.no_ros: 
          if cur_img is None:
            cur_img = self.eval_images_list[2]
          cur_img_emb = self.embed_obs(cur_img)
        
        if goal_img is not None:
            goal_img_emb = self.embed_obs(goal_img)
            target_idx, min_value = self.localize(goal_img_emb, localized_idx + next_interval, 
                                       span=len(self.eval_images_list))
        else:
            target_idx = goal_idx
        while not rospy.is_shutdown():
            if not self.no_ros:
                cur_img_emb = self.get_real_time_obs_emb()
            
            ## localize:
            localized_idx = self.localize(cur_img_emb, localized_idx, 
                    span=len(self.eval_images_list) // 2 if i == 0 else default_span)
            
            cmd = input('press enter to launch relocalization, press g to do gloablly.')  # press 'g' to do global localization
            if  localized_idx is None or collision or cmd == '' or cmd == 'g':
                print('localization is lost, start to relocalize.')
                if self._revise_mode == 'backtrack':
                    relocalized_idx = self.backtracking(i, pre_localized_idx, next_interval)
                    cur_img_emb = self.get_real_time_obs_emb()
                elif self._revise_mode == 'affordance':
                    if i == 0 or cmd == 'g':
                        goal_img_emb = None
                    else:
                        goal_img_emb = self.embed_obs(self.eval_images_list[pre_localized_idx + next_interval])
                    while not rospy.is_shutdown():
                        plan = self.planning(cur_img_emb, goal_img_emb, self.hist_states[-min(self.rnn_horizon - 1, len(self.hist_states)):], rnn_afford=True, visualize=False)

                        # execute first step of plan: plan[0]
                        reloc_action = self.predict_action(cur_img_emb, plan[0])
                        print("reloc_action:", reloc_action)

                        if not self.no_ros:
                            while input('press enter to send action command') != '':
                                pass
                            self.ros_api.send_action(reloc_action)
                            rospy.sleep(1.0)
                            cur_img_emb = self.get_real_time_obs_emb()
                        
                        
                        start_idx = 0 if cmd == 'g' else pre_localized_idx
                        relocalized_idx = self.localize(cur_img_emb, start_idx, verbose=False,
                                span=len(self.eval_images_list) // 3 if goal_img_emb is None or cmd == 'g' else default_span)
                        if input('continue?') !='':
                            break
                else:
                    raise ValueError(self._revise_mode)
                localized_idx = relocalized_idx
                print('successfully relocalize to:', relocalized_idx)
            elif localized_idx == target_idx:
                print('reach goal!')
                finish = True

            pre_localized_idx = localized_idx
            next_waypoint_idx = localized_idx + next_interval
            goal_img_emb = self.embed_obs(self.eval_images_list[min(len(self.eval_images_list) - 1, next_waypoint_idx)])
            
            ## inference
            value, q_value, action = self.model_infer(cur_img_emb, goal_img_emb)
            print(f"next_waypoint_idx:{next_waypoint_idx}, value:{value}, action:{action}")

            ## execute action
            duration = 1
            if not self.no_ros:
                while input('press enter to send action command') != '':
                    pass
                _, _, pre_yaw = self.ros_api.get_odom()
                self.ros_api.send_action(action, duration)
                
                rospy.sleep(1.0)
            if finish or self.no_ros:
                break
            self.hist_states.append(cur_img_emb.clone())
            i += 1
            localized_idx = next_waypoint_idx
            
            
    def shut_down(self):
        if not self.no_ros:
            self.ros_api.shutdown()



if __name__ == '__main__':
    config = read_config(args)
    model_paths = [
        ""
    ]
    model_name = model_paths[0]
    model_path = "pretrain/" + model_name
    pre_args_path = model_path.replace('pt', 'json')
    with open(pre_args_path, 'r') as f:
        pre_args = json.load(f)
    config = update(config, pre_args)
    config['VQVAE']['pretrain_path'] = ''

    config['RND']['use_rnd_penalty'] = True
    config['RND']['pretrain_path'] = ''
    
    data_path = "" 
    
    eval_images_list = []
    eval_poses_list = []
    with open(os.path.join(data_path), 'rb') as f: 
        dataset = pickle.load(f)
    eval_images_list.extend(dataset["image_observations"])
    eval_poses_list.extend(dataset["observations"])
    
    deploy = Deploy(config, args, model_path, eval_images_list, eval_poses_list)
    
    #* visual localization
    # next_interval = 2
    # localized_idx = -next_interval
    # localized_idx = deploy.localize(cur_img, localized_idx, )
    
    #* navigate to goal specified by an index
    goal_idx = len(eval_images_list) - 1
    deploy.navigate(goal_idx=goal_idx)
    
    #* navigate to goal specified by an image
    # cur_img = cv2.imread('./images/1.jpeg')
    # deploy.navigate(cur_img = cur_img)
    
    deploy.shut_down() 
