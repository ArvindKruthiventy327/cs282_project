import os 
import tqdm
import random
import pickle as pkl
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np 

class IterableWrapper(IterableDataset):
    def __init__(self, wrapped_dataset, max_count=float("inf")):
        self.wrapped = wrapped_dataset
        self.ctr, self.max_count = 0, max_count

    def __iter__(self):
        self.ctr = 0
        return self

    def __next__(self):
        if self.ctr > self.max_count:
            raise StopIteration

        self.ctr += 1
        idx = int(np.random.choice(len(self.wrapped)))
        return self.wrapped[idx]


class LatentActionBuffer(Dataset): 

    def __init__(self, 
                 buffer_path,
                 n_test, 
                 n_val,
                 mode="train", 
                 ac_chunk=1, 
                 obs_dim=11, 
                 ac_dim=7): 
        
        assert mode in ("train", "val", "test"), "Mode must be train/test"
        buf = self.load_buffer(buffer_path)
        assert len(buf) > n_test, "Dataset is too small"
        BUF_SHUFFLE_RNG = 3904767649
        rng = random.Random(BUF_SHUFFLE_RNG)

        # get and shuffle list of buf indices
        idx = list(range(len(buf)))
        rng.shuffle(idx)

        if mode == "train": 
            index_list = idx[n_test + n_val:]
        elif mode == "val": 
            index_list = idx[n_test: n_test + n_val]
        elif mode == "test": 
            index_list = idx[:n_test]
        print(f"Mode: {mode}, Index list length: {len(index_list)}")
        self.s_a = []
        for i in tqdm.tqdm(index_list): 
            # print(f"Current demo:{i}")
            info_t = buf[i]
            # print(len(info_t))
            for j in range(len(info_t)):
                # print(f"Current iteration: {j}")
                if len(info_t) - j < ac_chunk: 
                    actions_traj =  [np.array(item[1]) for item in info_t[j:]]
                    # print(F"Padding: {(ac_chunk - (len(info_t) - j))}, Padding array: {len(list(actions_traj[-1])*(ac_chunk - (len(info_t) - j)))}")
                    for k in range(ac_chunk - (len(info_t) - j)): 
                        actions_traj.append(actions_traj[-1])
                    # print(f"actions trajectory: {len(actions_traj)}")
                    actions_traj = torch.Tensor(np.array(actions_traj))
                    state_t = torch.Tensor(info_t[j][0])
                else:
                    actions_traj = [np.array(item[1]) for item in info_t[j:j+ac_chunk]]
                    # print(f"Action chunk size: {len(actions_traj)}")
                    actions_traj =  torch.Tensor(np.array(actions_traj))
                    # print(f"Action chunk size: {actions_traj.shape}")

                    state_t = torch.Tensor(info_t[j][0])
                self.s_a.append((state_t, actions_traj))


    def load_buffer(self, buffer_path): 
        with open(buffer_path, "rb") as f: 
            buffer = pkl.load(f)
        return buffer
    
    def __len__(self): 
        return len(self.s_a)
    
    def __getitem__(self, idx): 
        state_t, actions_chunk = self.s_a[idx]
        return state_t, actions_chunk

if __name__ == "__main__": 
    import pickle as pkl 
    from scipy.spatial.transform import Rotation as R
    import matplotlib.pyplot as plt
    from common_transforms import rotm_to_rot6d,cart2se3,vee_map,get_rel_command

    src_dir = "/extra_storage/equicontact_stacking_fixed_pose_sa/buf.pkl"
    n_test = 0
    n_val = 0
    ac_chunk = 30
    obs_dim = [11, 1]
    ac_dim = [30, 10]
    batch_size = 50
    action_dataset = LatentActionBuffer(src_dir, 
                                                0,
                                                0, 
                                                mode="test",
                                                obs_dim=obs_dim, 
                                                ac_chunk = ac_chunk, 
                                                ac_dim = ac_dim)
    test_loader = DataLoader(action_dataset, 
                                  batch_size = 1, 
                                  shuffle = False, 
                                  num_workers=1) 
    
    print(len(test_loader))
    src_demo_path = "/extra_storage/data_collection/datasets/stack_fixed_pose/demo_193.pkl"
    with open(src_demo_path, "rb") as f: 
        demo = pkl.load(f)
    # print(demo["observations"][0])
    def proc_action(action): 
        
        trans = action[:3]
        ori = action[3:6]
        gripper = action[-1]
        ori_rot6d = rotm_to_rot6d(R.from_rotvec(ori).as_matrix())
        proc_action = np.concatenate([trans, ori_rot6d, [gripper]])
        return proc_action 
    actions = []
    for i in demo["actions"][0:30]: 
        proc_ac = proc_action(i)
        actions.append(proc_ac)
    actions = np.array(actions)
    # print(demo["actions"][0:30])
    counter = 0
    for batch in test_loader: 
        if counter >= 1: 
            break
        print(torch.from_numpy(actions) - batch[1])
        counter+=1