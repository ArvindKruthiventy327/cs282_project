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
        print(len(buf))
        assert len(buf) > n_test, "Dataset is too small"
        BUF_SHUFFLE_RNG = 3904767649
        rng = random.Random(BUF_SHUFFLE_RNG)

        # get and shuffle list of buf indices
        idx = list(range(len(buf)))
        print(idx)
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
