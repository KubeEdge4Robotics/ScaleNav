
import math
import numpy as np
import collections
import os
from scipy.spatial.transform import Rotation as R
import cv2
from typing import List
from math import pi
import torch

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'terminals', 'next_observations'])

ImageBatch = collections.namedtuple('ImageBatch', [
    'observations', 'image_observations', 'actions', 'rewards', 'terminals',
    'next_observations', 'next_image_observations'
])

ImageGoalBatch = collections.namedtuple('ImageGoalBatch', [
    'observations', 'image_observations', 'actions', 'rewards', 'terminals',
    'next_observations', 'next_image_observations', 'next_actions', 'image_goals', 
    'pose_to_goals', 'time_to_goals', 'is_positive', 'raw_observations', 
    'stack_imgs', 'stack_lengths', 'is_collision'
])

DataDict = dict(
    image_observations=[],
    observations=[], 
    actions=[], 
    actions_odom=[],
    next_image_observations=[], 
    next_observations=[], 
    timesteps=[],
    next_timesteps=[],
    terminals=[], 
    collision=[], 
    step_idx=[],  
    traj_idx=[],
    start_idx = [],
    end_idx = [],
    max_range = [],
    lower_range = [],
    frame_id = [],
    next_frame_id = []
)


def _max_time_to_idx(timesteps_list, goal_range_max):
    """timesteps_list = current + goal_range_max * state"""
    
    cur_time = timesteps_list[0]
    for i, t in enumerate(timesteps_list[::-1]):
        if t - cur_time <= goal_range_max:  #! find max point before goal_range_max rather than max_time
            return len(timesteps_list) - i - 1, t - cur_time
            
            
def _list_to_tensor(sample_list, _dtype=torch.float32):
    return torch.tensor(np.array(sample_list), dtype=_dtype)
 

def _norm_heading(heading):
    if heading > pi:
        heading -= 2 * pi
    if heading <= -pi:
        heading += 2 * pi
    return heading 
    
def euc2polar(curr, goal, rot):
    """refer to ReViND"""
    
    curr = np.array(curr[:2])
    goal = np.array(goal[:2])

    total = np.array([curr, goal, goal])
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]).T
    currac = np.dot(R, total.T).T
    currac = currac - currac[0]

    [x, y] = currac[1][0], currac[1][1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([np.sin(phi), np.cos(phi), rho])


def get_action(pre_pos, pos, interval=2, noise_mu_sigma=(0, 0.005)):  # 0.007
    goal_dist = _get_dist([pos[0] - pre_pos[0], pos[1] - pre_pos[1]])
    goal_angle = _norm_heading(_norm_heading(pos[2]) - _norm_heading(pre_pos[2]))
    linear_sign = 1.0
    radius = abs((goal_dist / 2) / math.sin(goal_angle / 2))
    arc_length = abs(goal_angle * radius)
    
    linear_vel = arc_length / interval #+ np.random.normal(*noise_mu_sigma)
    linear_vel *= linear_sign

    angular_vel = abs(linear_vel) / radius # + np.random.normal(*noise_mu_sigma)  
    angular_sign = -1 if goal_angle < 0 else 1
    angular_vel *= angular_sign
    
    return [linear_vel, angular_vel]


def action2pose(action, interval=2):
    if len(action.shape) > 1:
        v_linear, v_angular = action[:, 0], action[:, 1]
    else:
        v_linear, v_angular = action[0], action[1]
    arc_length = v_linear * interval
    goal_angle = v_angular * interval 
    if len(action.shape) == 1 and goal_angle == 0:
        goal_dist = arc_length
    else:
        radius = np.abs(arc_length / goal_angle)
        goal_dist = np.abs(radius * np.sin(goal_angle / 2) * 2)
    dx, dy = goal_dist * np.cos(goal_angle / 2), goal_dist * np.sin(goal_angle / 2)
    
    if len(action.shape) > 1:
        return np.stack([dx, dy, goal_angle], axis=1)
    else:
        return np.array([dx, dy, goal_angle])  # x is positive to front, y is positive to left

def image_pose_matching(images_id_list, pose_pool_dict, thred=0.05):
    new_pose_dict = {}
    sorted_pose_list = list(sorted(pose_pool_dict.items()))  # sorted by key
    pre_j = -1
    valid_list = []
    for i in range(len(images_id_list)):
        timestep = float(images_id_list[i].replace(".jpeg", ""))
        for j, (k, v) in enumerate(sorted_pose_list[pre_j + 1:]):
            if abs(k - timestep) < thred:
                new_pose_dict[timestep] = v
                valid_list.append(True)
                pre_j = j
                break
            if k - timestep >= thred:
                valid_list.append(False)
                break
        if k - timestep < -thred:
            assert k == sorted_pose_list[-1][0]
            valid_list += [False] * (len(images_id_list) - len(valid_list))
            break
            
    assert len(valid_list) == len(images_id_list), f'{len(valid_list)}, {len(images_id_list)}'
    return new_pose_dict, valid_list


def get_actions_from_poses(poses_dict):
    actions = []
    sorted_poses_dict = dict(sorted(poses_dict.items(), key=lambda item: item[0]))
    sorted_poses = list(sorted_poses_dict.values())
    pre_pos = sorted_poses[0]
    for pos in sorted_poses[1:]:
        action = get_action(pre_pos, pos)
        actions.append(action)
        pre_pos = pos
    return actions


def image_resize(img, resize_shape=256):
    return cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)


def _resize_np_with_smaller_edge(img, size=256):
    ''' implement transform used in mobilenet-v2 for np.ndarray input.
    1) use cv2.resize to implement torchvision.transform.resize(256).
    3) transpose image shape from (h, w, c) to (c, h, w).
    '''
    assert isinstance(img, np.ndarray)
    # resize: set the smaller one between h and w to size=256 and scale another one.
    h, w = img.shape[:2]
    if h >= w:
        resize_shape = (size, int(size / w * h)) # (width, height)
    else:
        resize_shape = (int(size / h * w), size) # (width, height)
    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    return img


def _get_dist(delta_xy):
    return np.linalg.norm(delta_xy)

def get_rel_pose_change(pos1, pos2):
    """taken from chaplot's ANS: get relative pose in local coordinate"""
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = _get_dist([x1 - x2, y1 - y2])
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    """taken from chaplot's ANS: 
    pose is in world coordinate.
    rel_pose_change is in local coordinate.
    """
    if len(pose.shape) > 1:
        x, y, o = pose[:, 0], pose[:, 1], pose[:, 2]
        dx, dy, do = rel_pose_change[:, 0], rel_pose_change[:, 1], rel_pose_change[:, 2]
    else:
        x, y, o = pose
        dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(o) + dy * np.cos(o)
    global_dy = dx * np.cos(o) - dy * np.sin(o)
    x += global_dy
    y += global_dx
    o += do

    if len(pose.shape) > 1:
        for i in range(len(o)):
            o[i] = _norm_heading(o[i])
        return np.stack([x, y, o], axis=1)
    else:
        o = _norm_heading(o)
        return np.array([x, y, o])

def _query_data_by_idx(frame_id_j, images_dir, pose_pool_dict):
    img_j = _get_image_from_path(frame_id_j, images_dir)
    key_j = float(frame_id_j.replace('.jpeg', ''))
    try:
        raw_pose_j = np.array(pose_pool_dict[key_j])  
        
        pose_j, speed_j = raw_pose_j[:3], raw_pose_j[3:5]
        if len(raw_pose_j) == 6:
            scan_j = raw_pose_j[5]
        else:
            scan_j = None

    except:
        print("keyerror:", key_j)
        pose_j = None
        speed_j = None
        scan_j = None

    return img_j, pose_j, speed_j, scan_j

def get_files_ending_with(path:str, all_path:List, suffix:str, pair:bool):
    all_file_list = os.listdir(path) 
    for file in all_file_list:
        filepath = os.path.join(path, file)
        if os.path.isdir(filepath):  
            if  not pair and file == 'pair_data':
                continue
            get_files_ending_with(filepath, all_path, suffix, pair)
        elif os.path.isfile(filepath) and not filepath.endswith('list.pkl'):
            all_path.append(filepath)  
            
def _get_xy_yaw_from_pose(pose):
    x, y, qx, qy, qz, qw = pose
    # _, _, yaw = _quart_to_rpy(qx, qy, qz, qw)
    _, _, yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz')
    return np.array([x, y, yaw], dtype=np.float32)

def _get_image_from_path(frame_id, images_dir):
    image_path = os.path.join(images_dir, frame_id)
    name_list = os.listdir(images_dir)
    if not frame_id in name_list:
        return None
    else:
        image = cv2.imread(image_path)
        return image
        
        
def _rotate_coord(dx, dy, angle):
    """
    angle in radians.
    Notice the difference between coordinate rotation and vector rotation.
    This function rotates the coordinate.
    
    Notice the coordinate is unusual, where the front is positive x,
    and left is positive y, thus we should asign x := -x 
    """
    local_dx = -dx * math.cos(angle) + dy * math.sin(angle)
    local_dy = dx * math.sin(angle) + dy * math.cos(angle)
    new_x, new_y = local_dy, -local_dx
    return new_x, new_y  # asign x := -x