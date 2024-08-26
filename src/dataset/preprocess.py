import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import pickle
from tqdm import tqdm
from copy import deepcopy
from yaml import full_load

from dataset_utils import DataDict, _query_data_by_idx, get_action, image_resize, _norm_heading, image_pose_matching

rel_pose_thred = 1
rel_heading_thred = 0.5


def read_custom_data(raw_data_root,
                     data_name, 
                     trajectory,
                     original_rate=6, 
                     downsampled_rate=3, 
                     collision_thred=0.4,
                     use_downsample=False, 
                     ):

    images_dir = os.path.join(raw_data_root, "images", data_name)
    pose_dir = os.path.join(raw_data_root, "poses", data_name + ".pkl")
    ## read poses
    with open(pose_dir, "rb") as f:
        pose_pool_dict = pickle.load(f)

    ## read and sort images
    images_id_list = sorted(os.listdir(images_dir))
    
        
    if use_downsample:
        print(f"downsample: original_rate:{original_rate}, downsampled_rate:{downsampled_rate}")
        step = int(original_rate // downsampled_rate)
        images_id_list = images_id_list[::step]
    
    pose_pool_dict, valid_list = image_pose_matching(images_id_list, pose_pool_dict)
    pre_pose = None
    pre_i = 0
    print("data_name:", data_name, "length of images list:", len(images_id_list), " valid num:", sum(valid_list))
    count = 0
    scan_list = []
    for i, frame_id in tqdm(enumerate(images_id_list)):
        if valid_list[i]:
            img, pose, speed, scan = _query_data_by_idx(
                frame_id, images_dir, pose_pool_dict)
            if speed is None or len(speed) != 2:
                speed = np.zeros(2)
            if pre_pose is None: 
                pre_pose = pose
                pre_i = i
                continue
            elif  np.linalg.norm(pre_pose[:2] - pose[:2]) < rel_pose_thred and \
                abs(_norm_heading(pre_pose[2] - pose[2])) < rel_heading_thred: 
                pass
            else:
                trajectory["img"].append(image_resize(img, (256, 256)).astype(np.uint8))
                trajectory["pose"].append(pose.astype(np.float32))
                trajectory["speed"].append(speed.astype(np.float32))
                scan_list.append(scan)
                if scan <= collision_thred:
                    collision = True
                    print('colision', frame_id, scan)
                else:
                    collision = False
                count += collision
                trajectory["collision"].append(collision)
                trajectory["timestep"].append(i) 
                trajectory["frame_id"].append(float(frame_id.replace(".jpeg", "")))
                pre_pose = pose
                pre_i = i
            
    print("collision:", count, "min scan:", min(scan_list))
    return trajectory
    
    
def process(trajectory, save_file, config, original_rate=3, duration=2):
    data = deepcopy(DataDict)
    length = len(trajectory["img"])
    if length < 3:
        return
    intervals = []
    for j in range(min(length - 1, config["Data"]["max_episode_steps"])):  # drop last step
        data["image_observations"].append(trajectory["img"][j]) 
        data["observations"].append(trajectory["pose"][j])
        action_j = get_action(trajectory["pose"][j], trajectory["pose"][j + 1], duration) 
        data["actions_odom"].append(trajectory["speed"][j])
        is_collision = trajectory["collision"][j] 
        if is_collision:
            data["next_image_observations"].append(trajectory["img"][j])
            data["next_observations"].append(trajectory["pose"][j])
        else:
            data["next_image_observations"].append(trajectory["img"][j + 1])
            data["next_observations"].append(trajectory["pose"][j + 1])
        data["actions"].append(action_j)
        
        intervals.append((trajectory["timestep"][j + 1] - 
                            trajectory["timestep"][j]) / original_rate)
        
        data["collision"].append(is_collision)
        data["step_idx"].append(j)
        data["timesteps"].append(trajectory["timestep"][j])
        data["next_timesteps"].append(trajectory["timestep"][j + 1])
        if trajectory["frame_id"]:
            data["frame_id"].append(trajectory["frame_id"][j])
            data["next_frame_id"].append(trajectory["frame_id"][j + 1])
    print(np.min(data["actions"], axis=0), np.max(data["actions"], axis=0), np.mean(data["actions"], axis=0), np.std(data["actions"], axis=0))
    print("intervals:", np.min(intervals), np.max(intervals), np.mean(intervals), np.median(intervals))
    print("processed length:", len(data["image_observations"]))
    file_path = os.path.join(config["dataset_root"], save_file + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    

if __name__ == "__main__":
    """
    cd /src/dataset/
    python3 preprocess.py
    """
    config_path = "../configs/custom.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    config = full_load(content)
    raw_data_root = config["raw_data_root"]
    os.makedirs(config["dataset_root"], exist_ok=True)
    selected_data_list = config["selected_data"]
    if selected_data_list is None:
        selected_data_list = os.listdir(raw_data_root)
    for data_name in selected_data_list:
        trajectory = dict(
            img=[],
            pose=[],
            speed=[],
            timestep=[],
            frame_id=[],
            collision=[]
        )
        read_custom_data(raw_data_root, data_name, trajectory)
        process(trajectory, data_name, config)

    
    