import cv2 
import h5py
import numpy as np 
import os 
import json
import time 
import pickle as pkl
import yaml
import copy
import torch 
from tqdm import tqdm
import torch.nn as nn 
from mlp_ae import MLP_VQVAE, MLPVAE
from cnn_ae import CNN_VAE
from scipy.spatial.transform import Rotation as R
from common_transforms import rotm_to_rot6d,cart2se3,vee_map,get_rel_command

# class DataProcessorLatentAction:

#     def __init__(self, config):
#         with open(config, 'r') as f:
#             self.config = yaml.safe_load(f)
#         self.data = {}
#         self.data_paths = self.config["data_paths"]
#         self.cameras = self.config["cameras"]
#         self.load_autoencoder()
#         self.action_chunk = self.config["autoencoder"]["ac_dim"][0] 
#         self.state_vars = self.config["state"]

#     def load_autoencoder(self): 
#         type = self.config["autoencoder"]["type"]
#         model_params = self.config["autoencoder"]
#         ckpt_path = model_params["ckpt_path"]
#         if type == "MLP_VAE": 
#             obs_dim = model_params["obs_dim"]
#             ac_dim = model_params["ac_dim"]
#             latent_dim = model_params["latent_dim"]
#             hidden = model_params["hidden"]
#             self.model = MLPVAE(obs_dim, ac_dim, latent_dim, hidden)
#         elif type == "MLP_VQVAE": 
#             obs_dim = model_params["obs_dim"]
#             ac_dim = model_params["ac_dim"]
#             latent_dim = model_params["latent_dim"]
#             hidden = model_params["hidden"]
#             self.model = MLP_VQVAE(obs_dim, ac_dim, latent_dim, hidden)
#         elif type == "CNN_VAE":
#             obs_dim = model_params["obs_dim"]
#             ac_dim = model_params["ac_dim"]
#             latent_dim = model_params["latent_dim"]
#             channels = model_params["channels"]
#             dec_in_padding = model_params["dec_in_padding"]
#             dec_out_padding =  model_params["dec_out_padding"]
#             self.model = CNN_VAE(ac_dim[0], obs_dim[0], ac_dim[-1], channels,dec_in_padding, dec_out_padding, latent_dim)
#         state_dict = torch.load(ckpt_path)
#         self.model.load_state_dict(state_dict["model_state_dict"])
#         self.model.eval().cuda()
#         self.enc = self.model.enc
#         self.dec = self.model.dec
#         if type == "CNN_VQVAE" or type == "MLP_VQVAE": 
#             self.quantizer = self.model.quantizer()
        

#     def preprocess_img(self, img, convert_color=True, size=(256,256)): 
#         img = cv2.resize(img, size)
#         if convert_color:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         _, img_encoded = cv2.imencode('.jpg', img)
#         return img_encoded
    
#     def preprocess_action(self, action): 
#         xyz_state = action[:3]
#         rpy_state = action[3:]
#         rotm = R.from_euler('xyz', rpy_state, degrees=True).as_matrix()
#         rot6d_state = rotm_to_rot6d(rotm)
#         return np.concatentate((xyz_state, rot6d_state, action[-1]), axis=0)

#     def preprocess_state(self, state): 
#         xyz_state = state[:3] # Extract the XYZ position
#         quat = state[3:7] 
#         quat = np.array([quat[1], quat[2], quat[3], quat[0]])  
#          # Extract the roll, pitch, yaw angles
#         # gripper_state = state[7:]
#         rotm = R.from_quat(quat).as_matrix()
#         rot6d_state = rotm_to_rot6d(rotm)
#         processed_state = np.concatenate((xyz_state, rot6d_state), axis=0)
#         return processed_state
    
#     def proc_dataset(self): 
        
#         save_dir = self.config["save_dir"]
#         os.makedirs(save_dir, exist_ok=True)

#         norm_dict = {}
#         dataset = []
#         states = []
#         actions = []
#         start_time = time.time()

#         for i in tqdm(range(len(self.data_paths))): 
#             path = self.data_paths[i]
#             with open(path, "rb") as f: 
#                 ep = pkl.load(f)
#                 ep_states, ep_actions, ep_data = self.proc_ep(ep)
#                 dataset.append(ep_data)
#                 states.extend(ep_states)
#                 actions.extend(ep_actions)
        
#         if self.config["normalize_state"]: 
#             norm_dict["state_norm"] = self._max_min_norm(copy.deepcopy(states))
#             # state_norm = norm_dict["state_norm"]
#             # print(f"Normalizing state: {state_norm}")
#             for ep_idx, ep_data in enumerate(dataset): 
#                 for t in range(len(ep_data)): 
#                     obs, act, rew = ep_data[t]
#                     # print(f'Current State: {obs["state"]}, mean: {np.array(norm_dict["state_norm"]["loc"])}')
#                     obs["state"] = (obs["state"] - np.array(norm_dict["state_norm"]["loc"])) / np.array(norm_dict["state_norm"]["scale"])
                    
#                     dataset[ep_idx][t] = (obs, act, rew) 
#         else: 
#             norm_dict["state_norm"] = {"loc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0]}
#         if self.config["normalize_action"]:
#             norm_dict["action_norm"] = self._max_min_norm(actions)
#         else: 
#             norm_dict["action_norm"] = {"loc": 0.0, "scale": 1.0}
#         with open(os.path.join(save_dir, "ac_norm.json"), "w") as f:
#             json.dump(norm_dict, f)
#         with open(os.path.join(save_dir, 'buf.pkl'), "wb") as f: 
#             pkl.dump(dataset, f)
#         print(f"Processing time {time.time() - start_time}") 
#         print(f"Saved dataset at path: {save_dir}")
    
#     def process_episode(self, ep): 
#         ep_data = []
#         ep_states = []
#         ep_actions = []
#         obs = ep["obs"]
#         actions = ep["actions"]
#         n = actions.shape[0]
#         for t in range(n): 
#             counter = 0
#             obs_t = obs[t]
#             data_obs = {}
#             for i, cam in enumerate(self.cameras): 
#                 img = obs_t[cam]
#                 data_obs[f"enc_cam_{i}"] = self.proc_img(img, convert_color=True)
#             full_state = []
#             for var in self.state_vars:
#                 full_state.append(obs_t[var])
            
#             full_state = np.concatenate(full_state, axis=0)
#             proc_state = self.preprocess_state(full_state)
#             reward = float(0)
#             ac_chunk = self.get_ac_chunk(t, n, full_state, actions)
#             obs_t = torch.from_numpy(proc_state).unsqueeze(0)
#             traj = torch.from_numpy(ac_chunk).unsqueeze(0)
#             latent_action = self.encode_ac_chunk(obs_t, traj)
#             ep_states.append(proc_state)
#             ep_actions.append(latent_action)
#             ep_data.append((proc_state, latent_action, 0.0))
#         return ep_data, ep_states, ep_actions 
    
#     def encode_ac_chunk(self, obs_t, traj): 
#         model_type = self.config["autoencoder"]["type"]
#         if model_type == "MLP_VAE" or model_type == "CNN_VAE": 
#             return self.enc(obs_t, traj)
#         elif model_type == "MLP_VQVAE" or model_type == "CNN_VQVAE": 
#             z_e = self.enc(obs_t, traj)
#             z_q, indices = self.quantizer(z_e)
#             return z_q
        
#     def get_ac_chunk(self, t, max_t, curr_state, actions): 
#         ac_chunk = []
#         for i in range(self.action_chunk): 
#             idx = np.max(np.array([t+i, max_t-1]))
#             if self.config["mimicgen"]: 
#                 action = self.mimicgen2real_ac(actions[idx], curr_state)
#                 ac_chunk.append(action)
#             else: 
#                 ac_chunk.append(actions[idx])
#             return np.array(ac_chunk)
    
#     def verify_encoding(self, proc_state, latent_action, ac_chunk): 
#         self.decoded_chunk = self.dec(proc_state, latent_action)
#         model_params = self.config["autoencoder"]
#         model_type = model_params["model_type"]
#         if model_type == "MLP_VAE": 
#             self.decoded_chunk = self.decoded_chunk.squeeze(0).cpu().numpy()
#             obs_dim = model_params["obs_dim"]
#             ac_dim = model_params["ac_dim"]
#             obs_dim_flat = obs_dim[0] * obs_dim[1]
#             ac_dim_flat = ac_dim[0] * ac_dim[1]
#             pred_ac_chunk = self.decoded_chunk[obs_dim_flat:]
#             print(f"Error between {np.mean(np.abs(pred_ac_chunk - ac_chunk))}")
#         # elif model_type == "CNN_VAE": 

#         print(f"Error between ac_chunk ")
#     def mimicgen2real_ac(self, action, state): 
#         pose_t = np.eye(4)
#         pose_t[:3, :3] = R.from_quat(state[3:7]).as_matrix()
#         pose_t[:3, 3] = state[:3].reshape((-1,))
#         cur_rotm = pose_t[:3, :3]
#         cur_xyz = pose_t[:3, 3]
        
#         xyz = action[:3]
#         rpy = action[3:6]
#         offset_mat = np.array([[0, -1 , 0],
#                             [ 1 , 0, 0],
#                             [ 0,   0  ,1]])
#         target_xyz = cur_xyz + xyz * 0.05 
#         target_rotm = R.from_rotvec(rpy * 0.05).as_matrix() @ offset_mat @ cur_rotm

#         target_rot6d = rotm_to_rot6d(target_rotm)
#         return np.concatenate((target_xyz, target_rot6d, [action[-1]]), axis=0)
#     def _gaussian_norm(self, all_acs):
#         all_acs_arr = np.array(all_acs)
#         mean = np.mean(all_acs_arr, axis=0)
#         std =  np.std(all_acs_arr, axis=0)
#         if not std.all(): # handle situation w/ all 0 actions
#             std[std == 0] = 1e-17

#         for a in all_acs:
#             a -= mean
#             a /= std

#         return dict(loc=mean.tolist(), scale=std.tolist())
    
#     def _max_min_norm(self, all_acs):
#         print('Using max min norm')
#         all_acs_arr = np.array(all_acs)
#         max_ac = np.max(all_acs_arr, axis=0)
#         min_ac = np.min(all_acs_arr, axis=0)

#         mid = (max_ac + min_ac) / 2
#         delta = (max_ac - min_ac) / 2

#         for a in all_acs:
#             a -= mid
#             a /= delta
#         return dict(loc=mid.tolist(), scale=delta.tolist())

class DataProcessorLatentAction: 

    def __init__(self, config_path): 

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.src_dir = self.config["src_dir"]
        self.ep_paths = os.listdir(self.src_dir)
        self.data_paths = []
        for ep_path in self.ep_paths: 
            self.data_paths.append(os.path.join(self.src_dir, ep_path))

        self.cameras = self.config["cameras"]
        self.state_vars = self.config["state"]
        self.load_autoencoder()
        self.action_chunk = self.config["autoencoder"]["ac_dim"][0] 

    
    def load_autoencoder(self): 
        type = self.config["autoencoder"]["type"]
        model_params = self.config["autoencoder"]
        ckpt_path = model_params["ckpt_path"]
        if type == "MLP_VAE": 
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            hidden = model_params["hidden"]
            self.model = MLPVAE(obs_dim, ac_dim, latent_dim, hidden)
        elif type == "MLP_VQVAE": 
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            hidden = model_params["hidden"]
            self.model = MLP_VQVAE(obs_dim, ac_dim, latent_dim, hidden)
        elif type == "CNN_VAE":
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            channels = model_params["channels"]
            dec_in_padding = model_params["dec_in_padding"]
            dec_out_padding =  model_params["dec_out_padding"]
            self.model = CNN_VAE(ac_dim[0], obs_dim[0], ac_dim[-1], channels,dec_in_padding, dec_out_padding, latent_dim)
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.model.eval().cuda()
        self.enc = self.model.enc
        self.dec = self.model.dec
        if type == "CNN_VQVAE" or type == "MLP_VQVAE": 
            self.quantizer = self.model.quantizer()
        with open(model_params["norm_path"], "r") as f: 
            self.norm = json.load(f)
        print(self.norm)

    def proc_state(self, state): 
        
        trans = state[:3]
        ori = state[3:7]
        ori_rot6d = rotm_to_rot6d(R.from_quat(ori).as_matrix())
        proc_state = np.concatenate([trans, ori_rot6d])
        return proc_state 
    
    def proc_img(self, img, size=[256, 256], convert_color=True ): 
        
        img = cv2.resize(img, size)

        if convert_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', img)
        return img_encoded
    
    def proc_action(self, action): 
        
        trans = action[:3]
        ori = action[3:6]
        gripper = action[-1]
        ori_rot6d = rotm_to_rot6d(R.from_rotvec(ori).as_matrix())
        proc_action = np.concatenate([trans, ori_rot6d, [gripper]])
        return proc_action 
    
    def proc_ep(self, ep): 

        ep_data = []
        ep_states = []
        ep_actions = []
        ep_len = len(ep["observations"])
        obs = ep["observations"]
        acs = ep["actions"]
        for t in range(ep_len): 
            obs_t = obs[t]
            data_obs = {}
            for i, cam in enumerate(self.cameras): 
                img = obs_t[cam]
                data_obs[f"enc_cam_{i}"] = self.proc_img(img, convert_color=True)
            state = []
            for var in self.state_vars:
                state.append(obs_t[var])
            state = np.concatenate(state)
            proc_state = self.proc_state(state)
            # print(f"Default state: {obs_t['robot0_eef_pos']} and processed_state: {proc_state}")
            data_obs["state"] = proc_state
            
            ac_chunk = self.get_ac_chunk(t, ep_len, state, acs)
            obs_t = torch.from_numpy(proc_state).unsqueeze(0)
            traj = torch.from_numpy(ac_chunk).unsqueeze(0)
            latent_action = self.encode_ac_chunk(obs_t, traj)
            pred_ac_chunk = self.verify_encoding(obs_t, traj, latent_action, ac_chunk)
            # latent_action=latent_action.detach().cpu().numpy()
            latent_action = latent_action.T.reshape((-1,))
            data_obs["pred_ac_chunk"] = pred_ac_chunk
            ep_states.append(proc_state)
            ep_actions.append(latent_action)
            ep_data.append((data_obs, pred_ac_chunk[0], 0.0))
        return ep_states, ep_actions, ep_data
    
    def proc_dataset(self): 
        
        save_dir = self.config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        norm_dict = {}
        dataset = []
        states = []
        actions = []
        start_time = time.time()

        for i in tqdm(range(len(self.data_paths))): 
            path = self.data_paths[i]
            with open(path, "rb") as f: 
                ep = pkl.load(f)
                ep_states, ep_actions, ep_data = self.proc_ep(ep)
                dataset.append(ep_data)
                states.extend(ep_states)
                actions.extend(ep_actions)
        
        if self.config["normalize_state"]: 
            norm_dict["state_norm"] = self._max_min_norm(copy.deepcopy(states))
            # state_norm = norm_dict["state_norm"]
            # print(f"Normalizing state: {state_norm}")
            for ep_idx, ep_data in enumerate(dataset): 
                for t in range(len(ep_data)): 
                    obs, act, rew = ep_data[t]
                    # print(f'Current State: {obs["state"]}, mean: {np.array(norm_dict["state_norm"]["loc"])}')
                    obs["state"] = (obs["state"] - np.array(norm_dict["state_norm"]["loc"])) / np.array(norm_dict["state_norm"]["scale"])
                    
                    dataset[ep_idx][t] = (obs, act, rew) 
        else: 
            norm_dict["state_norm"] = {"loc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0]}
        if self.config["normalize_action"]:
            norm_dict["action_norm"] = self._max_min_norm(actions)
        else:
            norm_dict["action_norm"] = {"loc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0, 1.0]}
        with open(os.path.join(save_dir, "ac_norm.json"), "w") as f:
            json.dump(norm_dict, f)
        with open(os.path.join(save_dir, 'buf.pkl'), "wb") as f: 
            pkl.dump(dataset, f)
        print(f"Processing time {time.time() - start_time}") 
        print(f"Saved dataset at path: {save_dir}")

    def encode_ac_chunk(self, obs_t, traj): 
        model_type = self.config["autoencoder"]["type"]
        obs_t = (obs_t - torch.from_numpy(np.array(self.norm["state_norm"]["loc"])))/torch.from_numpy(np.array(self.norm["state_norm"]["scale"]))
        traj = (traj - torch.from_numpy(np.array(self.norm["action_norm"]["loc"])))/torch.from_numpy(np.array(self.norm["action_norm"]["scale"]))
        obs_t = obs_t.float().to("cuda")
        traj = traj.float().to("cuda")
        if model_type == "MLP_VAE" or model_type == "CNN_VAE":
            if model_type == "MLP_VAE": 
                traj = traj.flatten(start_dim=1, end_dim=2)
            return self.enc(obs_t, traj)
        elif model_type == "MLP_VQVAE" or model_type == "CNN_VQVAE": 
            z_e = self.enc(obs_t, traj)
            z_q, indices = self.quantizer(z_e)
            return z_q
        
    def get_ac_chunk(self, t, max_t, curr_state, actions): 
        ac_chunk = []
        for i in range(self.action_chunk): 
            idx = np.min(np.array([t+i, max_t-1]))
            if self.config["mimicgen"]: 
                action = self.mimicgen2real_ac(actions[idx], curr_state)
                ac_chunk.append(action)
            else: 
                proc_ac = self.proc_action(actions[idx])
                ac_chunk.append(proc_ac)
            
        return np.array(ac_chunk)
    
    def verify_encoding(self, proc_state,traj, latent_action, ac_chunk): 
        proc_state = (proc_state - torch.from_numpy(np.array(self.norm["state_norm"]["loc"])))/torch.from_numpy(np.array(self.norm["state_norm"]["scale"]))
        proc_state = proc_state.to("cuda").float()
        z_sample, mu, sigma = self.model.reparametrize(latent_action)
        print(mu.shape)
        x_hat = self.dec(proc_state, mu)
        # self.decoded_chunk = self.dec(proc_state, latent_action)
        model_params = self.config["autoencoder"]
        model_type = model_params["type"]
        ac_mean = np.array(self.norm["action_norm"]["loc"])
        ac_std = np.array(self.norm["action_norm"]["scale"])
        if model_type == "MLP_VAE": 
            x_hat = x_hat.detach().cpu().squeeze(0)
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            obs_dim_flat = obs_dim[0] * obs_dim[1]
            ac_dim_flat = ac_dim[0] * ac_dim[1]
            pred_ac_chunk = x_hat[obs_dim_flat:].reshape(ac_dim)
            pred_ac_chunk = pred_ac_chunk * ac_std + ac_mean
            # print(f"Pred ac chunk shape: {pred_ac_chunk.shape}")
            # print(f"Error between {pred_ac_chunk - ac_chunk}")
        return pred_ac_chunk.numpy()
            
    def _gaussian_norm_state(self, states):
        all_acs_arr = np.array(states)
        # print(f"State shape:{all_acs_arr.shape}")
        mean = np.mean(all_acs_arr, axis=0)
        std =  np.std(all_acs_arr, axis=0)
        if not std.all(): # handle situation w/ all 0 actions
            std[std == 0] = 1e-17
        return dict(loc=mean.tolist(), scale=std.tolist())
    
    def _max_min_norm_state(self, all_acs):
        # print('Using max min norm')
        all_acs_arr = np.array(all_acs)
        max_ac = np.max(all_acs_arr, axis=0)
        min_ac = np.min(all_acs_arr, axis=0)

        mid = (max_ac + min_ac) / 2
        delta = (max_ac - min_ac) / 2

        for a in all_acs:
            a -= mid
            a /= delta
        return dict(loc=mid.tolist(), scale=delta.tolist())
    
    def _gaussian_norm(self, all_acs):
        all_acs_arr = np.array(all_acs)
        mean = np.mean(all_acs_arr, axis=0)
        std =  np.std(all_acs_arr, axis=0)


        for a in all_acs:
            a -= mean
            a /= std

        return dict(loc=mean.tolist(), scale=std.tolist())
    
    def _max_min_norm(self, all_acs):
        print('Using max min norm')
        all_acs_arr = np.array(all_acs)
        max_ac = np.max(all_acs_arr, axis=0)
        min_ac = np.min(all_acs_arr, axis=0)

        mid = (max_ac + min_ac) / 2
        delta = (max_ac - min_ac) / 2

        for a in all_acs:
            a -= mid
            a /= delta
        return dict(loc=mid.tolist(), scale=delta.tolist())
if __name__ == "__main__": 

    dp = DataProcessorLatentAction("latent_action_base.yaml")
    dp.proc_dataset()