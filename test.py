import argparse
from pathlib import Path
import time
import cv2
import hydra
import numpy as np
import torch
import torchvision
import yaml
import os
import json
import random
import pickle as pkl
import matplotlib.pyplot as plt
from robobuf import ReplayBuffer as RB
from data4robotics.transforms import get_transform_by_name
from mimicgen.envs.robosuite import three_piece_assembly
import mimicgen.utils.robomimic_utils as RoboMimicUtils
import robomimic.utils.file_utils as FileUtils

from scipy.spatial.transform import Rotation as R
from data_processing.common_transforms import rotm_to_rot6d,rot6d_to_rotm, cart2se3,vee_map,get_rel_command



class BaselinePolicy: 

    def __init__(self, agent_path, model_name): 

        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "exp_config.yaml"), "r") as f:
            config_yaml = f.read()
            exp_config = yaml.safe_load(config_yaml)
            self.cam_idx = exp_config['params']['task']['train_buffer']['cam_indexes']
        with open(Path(agent_path, "ac_norm.json"), "r") as f: 
            self.norm_config = json.load(f)
        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict['model'])
        self.agent = agent.eval().cuda()
        self.device = "cuda:0"
        self.transform = get_transform_by_name('preproc')

    
    def _proc_image(self, rgb_img, size=(256, 256)):
        cam_idx = 0 
        imgs = {}
        for img in rgb_img:
            rgb_tensor = torch.from_numpy(img).to(self.device)
            rgb_tensor = rgb_tensor.float().permute((2, 0, 1)) / 255
            # rgb_tensor = torchvision.transforms.Resize(size, antialias=True)(rgb_tensor)
            rgb_tensor = rgb_tensor.unsqueeze(0)
            imgs[f"cam{cam_idx}"] = rgb_tensor.unsqueeze(0)
            cam_idx+=1
        return imgs
    
    def __proc_state(self, state): 
        state = state.cpu().numpy().T.reshape((state.shape[1],))
        xyz_state = state[:3] # Extract the XYZ position
        quat = state[3:7] 
         # Extract the roll, pitch, yaw angles
        gripper_state = state[7:]
        rotm = R.from_quat(quat).as_matrix()
        rot6d_state = rotm_to_rot6d(rotm)
        processed_state = np.concatenate((xyz_state, rot6d_state, gripper_state), axis=0)
        processed_state = torch.from_numpy(processed_state).to(self.device)
        mean = torch.from_numpy(np.array(self.norm_config["state_norm"]["loc"])).float().to(self.device)
        std = torch.from_numpy(np.array(self.norm_config["state_norm"]["scale"])).float().to(self.device)
        state_norm = (processed_state-mean)/std
        return state_norm 
    
    def denorm_action(self, raw_action): 
        mean = torch.from_numpy(np.array(self.norm_config["action_norm"]["loc"])).float().to(self.device)
        std = torch.from_numpy(np.array(self.norm_config["action_norm"]["scale"])).float().to(self.device)
        action = (raw_action * std) + mean
        return action 
    
    def __proc_action(self, raw_action): 
        xyz = raw_action[:3].reshape((-1,))
        rot6d = raw_action[3:9]
        rotm = rot6d_to_rotm(rot6d)
        rotvec = R.from_matrix(rotm).as_rotvec().reshape((-1,))
        gripper =  raw_action[-1]
        proc_action = np.concatenate((xyz, rotvec, np.array([gripper])))
        return proc_action
    
    def forward(self, img, obs):
        img_proc = self._proc_image(img)
        state = torch.from_numpy(obs)[None].float().cuda()
        state = self.__proc_state(state).float()
        with torch.no_grad(): 
            ac = self.agent.get_actions(img_proc, state.unsqueeze(0))
        ac = self.denorm_action(ac)
        ac = ac.squeeze(0).cpu().numpy().astype(np.float32)
        proc_ac = []
        for i in range(ac.shape[0]): 
            proc_action = self.__proc_action(ac[i])
            proc_ac.append(proc_action)
        return proc_ac


def create_base_env(env_name, raw_data_file, 
                    cameras=["agentview", "robot0_eye_in_hand"],
                    img_width=256, 
                    img_height=256): 

    if env_name == "three_piece_assembly": 
        env_metadata = FileUtils.get_env_metadata_from_dataset(raw_data_file)
        env_metadata['env_kwargs']["render_camera"] = None
        env_metadata['env_kwargs']['controller_configs']["input_min"] = -5
        env_metadata['env_kwargs']['controller_configs']['input_max'] = 5
        env_metadata['env_kwargs']['controller_configs']['input_type'] = "absolute"
        env_metadata['env_kwargs']['controller_configs']['input_ref_frame'] = "base"
        env_metadata['env_kwargs']['controller_configs']["output_min"] = [-5, -5, -5, -5, -5, -5]
        env_metadata['env_kwargs']['controller_configs']['output_max'] = [5, 5, 5,5, 5, 5]
        env_metadata['env_kwargs']['controller_configs']["control_delta"] = False 
        env_metadata['env_kwargs']['controller_configs']['uncouple_pos_ori'] = False
        env = RoboMimicUtils.create_env(env_metadata, 
                                    camera_names= cameras, 
                                    camera_width= img_width, 
                                    camera_height= img_height, 
                                    render=True)
        return env

def extract_policy_obs(obs, img_keys= ["agentview_image", "robot0_eye_in_hand_image"], state_keys=["robot0_eef_pos",
                                  "robot0_eef_quat",
                                  "robot0_gripper_qpos" 
                                    ]): 
    imgs = []
    for camera in img_keys: 
        # img_mod = cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR)
        img_mod = obs[camera]
        imgs.append(img_mod)
    state = []
    
    for state_var in state_keys: 
        state.append(obs[state_var])
    print(f"Current state: {state}")
    return np.array(imgs), np.concatenate(state)

if __name__ == "__main__": 
    import imageio
    video_dir = "/home/horowitz3/latent-diffusion_project/ldp/inference/sim_videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"video_{0}.mp4"
    video_writer = imageio.get_writer(os.path.join(video_dir, video_path), fps=20)
    env = create_base_env("three_piece_assembly", "/home/horowitz3/latent-diffusion_project/datasets/threading/demo_src_threading_task_D0/demo.hdf5")
    # agent = BaselinePolicy("/home/horowitz3/latent-diffusion_project/ldp/bc_finetune/dit_threading_test_04/wandb_None_mimicgen_3piece_assembly_resnet_gn_2025-11-22_05-04-08", 
    #                        "dit_threading_test_04.ckpt")
    agent = BaselinePolicy("/home/horowitz3/dit-policy/bc_finetune/test/wandb_None_mimicgen_threading_resnet_gn_nopool_2025-11-24_23-28-07", 
                           "test.ckpt")
    obs = env.reset()
    curr_obs = obs
    timesteps = 1000
    action_buffer = []
    chunk = 30
    for i in range(timesteps): 
        # print(f"Action shape: {init_action.shape}")
        
        imgs, state = extract_policy_obs(obs)
        # print(f"Current state: {state}")
        # if len(action_buffer) == 0:
        #     acs = agent.forward(imgs, state)
        #     action_buffer.extend(acs)
        #     ac = action_buffer[0]
        # else: 
        #     ac = action_buffer.pop(0)
        # print(f"Action : {ac}")
        # acs = agent.forward(imgs, state)
        # action_buffer.extend(acs)
        # ac = action_buffer[0]
        acs = agent.forward(imgs, state)
        # action_buffer.extend(acs)
        # ac = acs[0]
        # if ac[-1] > 0.5: 
        #     ac[-1] = 1
        # elif ac[-1] < 0.0: 
        #     ac[-1] = -1
        for ac in acs:
            obs, reward, done, info = env.step(ac)
            curr_obs = obs  #take action in the environment
            # print(curr_obs, )
            env.render()
        # frame = env.env.sim.render(
        # camera_name="agentview",
        # height=256,
        # width=256
        # )
        # frame = np.flipud(frame)
        # video_writer.append_data(frame)
        # cv2.imshow("Agentview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # if done: 
        #     break
    
    video_writer.close()
    cv2.destroyAllWindows()
