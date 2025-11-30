import cv2 
import h5py
import numpy as np 
import os 
import json
import time 
import pickle as pkl
import yaml
import copy
from scipy.spatial.transform import Rotation as R
from common_transforms import rotm_to_rot6d,cart2se3,vee_map,get_rel_command

class DataProcessorStateAction:

    def __init__(self, config):
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data = {}
        self.data_paths = self.config["data_paths"]
        self.cameras = self.config["cameras"]
        print(self.cameras)
        self.state_vars = self.config["state"]

    def preprocess_img(self, img, convert_color=True, size=(256,256)): 
        img = cv2.resize(img, size)
        if convert_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', img)
        return img_encoded
    
    def preprocess_action(self, action): 
        xyz_state = action[:3]
        rpy_state = action[3:]
        rotm = R.from_euler('xyz', rpy_state, degrees=True).as_matrix()
        rot6d_state = rotm_to_rot6d(rotm)
        return np.concatentate((xyz_state, rot6d_state, action[-1]), axis=0)

    def preprocess_state(self, state): 
        xyz_state = state[:3] # Extract the XYZ position
        quat = state[3:7] 
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])  
         # Extract the roll, pitch, yaw angles
        gripper_state = state[7:]
        rotm = R.from_quat(quat).as_matrix()
        rot6d_state = rotm_to_rot6d(rotm)
        processed_state = np.concatenate((xyz_state, rot6d_state, gripper_state), axis=0)
        return processed_state
    
    def process_dataset(self): 
        normalize_state = self.config["normalize_state"]
        save_path = self.config["save_path"]
        os.makedirs(save_path, exist_ok=True)
        norm_dict = {}
        dataset =[]
        states = []
        actions = []
        start_time = time.time()
        for path in self.data_paths:
            print(f"Processing file: {path}")
            with h5py.File(path, "r") as f: 
                data = f["data"]
                counter = 0
                for demo in data.keys():
                    ep_data, ep_states, ep_actions = self.process_episode(data[demo])
                    print(f"Processed episode: {counter + 1}")
                    dataset.append(ep_data)
                    states.extend(ep_states)
                    actions.extend(ep_actions)

                    counter+=1

                print(f"Number of demos: {counter}")
        if normalize_state:
            norm_dict["state_norm"] = self._max_min_norm(copy.deepcopy(states))
            state_norm = norm_dict["state_norm"]
            print(f"Normalizing state: {state_norm}")
            for ep_idx, ep_data in enumerate(dataset): 
                for t in range(len(ep_data)): 
                    state, act = ep_data[t]
                    state = (state - norm_dict["state_norm"]["loc"]) / norm_dict["state_norm"]["scale"]
                    
                    dataset[ep_idx][t] = (state, act) 
        print(f"Pre action_normalization")
        actions_pre_norm = np.array(copy.deepcopy(actions))
        norm_dict["action_norm"] = self._max_min_norm(actions)
        print(f"Differential between pre norm and post-norm: {np.sum(np.abs(actions_pre_norm-actions))}")
        denorm_actions = np.array(actions) * np.array(norm_dict["action_norm"]["scale"]) + np.array(norm_dict["action_norm"]["loc"])
        print(f"Denorm actions: {np.sum(np.abs(actions_pre_norm-denorm_actions))}")
        with open(os.path.join(save_path, "ac_norm.json"), "w") as f:
            json.dump(norm_dict, f)
        with open(os.path.join(save_path, 'buf.pkl'), "wb") as f: 
            pkl.dump(dataset, f)
        print(f"Processing time {time.time() - start_time}") 
        print(f"Saved dataset at path: {save_path}")
    
    def process_episode(self, ep): 
        ep_data = []
        ep_states = []
        ep_actions = []
        obs = ep["obs"]
        actions = ep["actions"]
        n = actions.shape[0]
        for t in range(n): 
            counter = 0
            full_state = []
            for var in self.state_vars:
                full_state.append(obs[var][t])
            
            full_state = np.concatenate(full_state, axis=0)
            proc_state = self.preprocess_state(full_state)
            reward = float(0)
            ep_data.append((proc_state, actions[t]))
            ep_states.append(proc_state)
            ep_actions.append(actions[t])
        return ep_data, ep_states, ep_actions 
    

    def _gaussian_norm(self, all_acs):
        all_acs_arr = np.array(all_acs)
        mean = np.mean(all_acs_arr, axis=0)
        std =  np.std(all_acs_arr, axis=0)
        if not std.all(): # handle situation w/ all 0 actions
            std[std == 0] = 1e-17

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

    dp = DataProcessorStateAction("/home/horowitz3/dit-policy/latent_actions/latent_base.yaml")
    dp.process_dataset()