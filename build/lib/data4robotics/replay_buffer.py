# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle as pkl
import random
import shutil
import hashlib
import tempfile #
import numpy as np
import torch
import tqdm
from robobuf import ReplayBuffer as RB
from tensorflow.io import gfile
from torch.utils.data import Dataset, IterableDataset

# cache loading from the buffer list to half memory overhead
buf_cache = dict()
BUF_SHUFFLE_RNG = 3904767649


# helper functions
_img_to_tensor = (
    lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
)
_to_tensor = lambda x: torch.from_numpy(x).float()


def _cached_load(path):
    global buf_cache

    if path in buf_cache:
        return buf_cache[path]

    with gfile.GFile(path, "rb") as f:
        buf = RB.load_traj_list(pkl.load(f))
    buf_cache[path] = buf
    return buf


def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    while len(imgs) < past_frames + 1:
        imgs.append(t.obs.image(cam_idx)[None])

        if t.prev is not None:
            t = t.prev
    return np.concatenate(imgs, axis=0)


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


# class RobobufReplayBuffer(Dataset):
#     def __init__(
#         self,
#         buffer_path,
#         transform=None,
#         n_test_trans=500,
#         mode="train",
#         ac_chunk=1,
#         cam_indexes=[0],
#         goal_indexes=[],
#         goal_geom_prob=0.01,
#         past_frames=0,
#         ac_dim=7,
#     ):
#         assert mode in ("train", "test"), "Mode must be train/test"
#         buf = _cached_load(buffer_path)
#         print(f"Length of buffer: {len(buf)}")
#         assert len(buf) > n_test_trans, "Not enough transitions!"

#         norm_file = os.path.join(os.path.dirname(buffer_path), "ac_norm.json")
#         if os.path.exists(norm_file):
#             shutil.copyfile(norm_file, "./ac_norm.json")

#         # shuffle the list with the fixed seed
#         rng = random.Random(BUF_SHUFFLE_RNG)

#         # get and shuffle list of buf indices
#         index_list = list(range(len(buf)))
#         rng.shuffle(index_list)

#         # split data according to mode
#         index_list = (
#             index_list[n_test_trans:] if mode == "train" else index_list[:n_test_trans]
#         )

#         self.transform = transform
#         self.s_a_mask = []

#         self.cam_indexes = cam_indexes = list(cam_indexes)
#         self.past_frames = past_frames
#         print(f"Building {mode} buffer with cam_indexes={cam_indexes}")

#         self.goal_geom_prob = goal_geom_prob
#         self.goal_indexes = set(goal_indexes)
#         assert all([g in self.cam_indexes for g in self.goal_indexes])

#         for idx in tqdm.tqdm(index_list):
#             t = buf[idx]

#             loop_t, chunked_actions, loss_mask = t, [], []
#             for _ in range(ac_chunk):
#                 chunked_actions.append(loop_t.action[None])
#                 loss_mask.append(1.0)

#                 if loop_t.next is None:
#                     break
#                 loop_t = loop_t.next

#             if len(chunked_actions) < ac_chunk:
#                 for _ in range(ac_chunk - len(chunked_actions)):
#                     chunked_actions.append(chunked_actions[-1])
#                     loss_mask.append(0.0)

#             a_t = np.concatenate(chunked_actions, 0).astype(np.float32)
#             assert ac_dim == a_t.shape[-1]

#             loss_mask = np.array(loss_mask, dtype=np.float32)
#             self.s_a_mask.append((t, a_t, loss_mask, loop_t))

#     def __len__(self):
#         return len(self.s_a_mask)

#     def __getitem__(self, idx):
#         step, a_t, loss_mask, goal = self.s_a_mask[idx]

#         if self.goal_indexes:
#             while np.random.uniform() > self.goal_geom_prob and goal.next is not None:
#                 goal = goal.next

#         i_t, o_t = dict(), step.obs.state
#         for idx, cam_idx in enumerate(self.cam_indexes):
#             i_c = _get_imgs(step, cam_idx, self.past_frames)
#             if self.goal_indexes:
#                 g_c = (
#                     _get_imgs(goal, cam_idx, 0)
#                     if cam_idx in self.goal_indexes
#                     else np.zeros_like(i_c[:1])
#                 )
#                 i_c = np.concatenate((g_c, i_c), axis=0)

#             i_c = _img_to_tensor(i_c)
#             if self.transform is not None:
#                 i_c = self.transform(i_c)

#             i_t[f"cam{idx}"] = i_c

#         o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
#         loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))
#         assert (
#             loss_mask.shape[0] == a_t.shape[0]
#         ), "a_t and mask shape must be ac_chunk!"
#         return (i_t, o_t), a_t, loss_mask


class RobobufReplayBuffer(Dataset):
    def __init__(
        self,
        buffer_path,
        transform=None,
        n_test_trans=500,
        mode="train",
        ac_chunk=1,
        cam_indexes=[0],
        goal_indexes=[],
        goal_geom_prob=0.01,
        past_frames=0,
        ac_dim=7,
        cache_to_disk=True,  # <--- NEW ARGUMENT
        cache_dir=None,      # <--- NEW ARGUMENT
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        
        # --- 1. Setup Caching ---
        self.cache_to_disk = cache_to_disk
        self.item_filepaths = []
        if self.cache_to_disk:
            if cache_dir is None:
                # Create a unique cache dir based on buffer_path hash
                path_hash = hashlib.md5(buffer_path.encode()).hexdigest()
                self.cache_dir = os.path.join(tempfile.gettempdir(), f"dit_policy_cache_{path_hash}", mode)
            else:
                self.cache_dir = os.path.join(cache_dir, mode)
            
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Using on-disk cache at: {self.cache_dir}")
        
        
        # --- 2. Load Buffer (Same as before) ---
        buf = _cached_load(buffer_path)
        print(len(buf))
        assert len(buf) > n_test_trans, "Not enough transitions!"

        norm_file = os.path.join(os.path.dirname(buffer_path), "ac_norm.json")
        if os.path.exists(norm_file):
            shutil.copyfile(norm_file, "./ac_norm.json")

        rng = random.Random(BUF_SHUFFLE_RNG)
        index_list = list(range(len(buf)))
        rng.shuffle(index_list)
        index_list = (
            index_list[n_test_trans:] if mode == "train" else index_list[:n_test_trans]
        )

        self.transform = transform
        self.s_a_mask = [] # Original list (only used if cache_to_disk=False)

        self.cam_indexes = cam_indexes = list(cam_indexes)
        self.past_frames = past_frames
        print(f"Building {mode} buffer with cam_indexes={cam_indexes}")

        self.goal_geom_prob = goal_geom_prob
        self.goal_indexes = set(goal_indexes)
        assert all([g in self.cam_indexes for g in self.goal_indexes])
        
        # --- 3. Modified Pre-processing Loop ---
        
        print("Processing dataset... (This will take a while, but only once!)")
        for i, idx in enumerate(tqdm.tqdm(index_list)):
            
            if self.cache_to_disk:
                # If caching, we store the *filename*
                item_path = os.path.join(self.cache_dir, f"{i:09d}.pt")
                self.item_filepaths.append(item_path)
                
                # If file exists, skip creation
                if os.path.exists(item_path):
                    continue
            
            # --- Start: Logic from original __init__ + __getitem__ ---
            t = buf[idx]

            # 1. Action Chunking (from original __init__)
            loop_t, chunked_actions, loss_mask = t, [], []
            for _ in range(ac_chunk):
                chunked_actions.append(loop_t.action[None])
                loss_mask.append(1.0)
                if loop_t.next is None:
                    break
                loop_t = loop_t.next

            if len(chunked_actions) < ac_chunk:
                for _ in range(ac_chunk - len(chunked_actions)):
                    chunked_actions.append(chunked_actions[-1])
                    loss_mask.append(0.0)

            a_t_np = np.concatenate(chunked_actions, 0).astype(np.float32)
            # print(a_t_np.shape[-1])
            assert ac_dim == a_t_np.shape[-1]
            loss_mask_np = np.array(loss_mask, dtype=np.float32)
            
            # --- IF NOT CACHING: Store original data and stop ---
            if not self.cache_to_disk:
                self.s_a_mask.append((t, a_t_np, loss_mask_np, loop_t))
                continue

            # --- IF CACHING: Do all the __getitem__ work NOW ---
            
            # 2. Goal Sampling (from original __getitem__)
            goal = loop_t
            if self.goal_indexes:
                # Note: This is stochastic, so we pre-sample it once.
                while np.random.uniform() > self.goal_geom_prob and goal.next is not None:
                    goal = goal.next
            
            # 3. Image Getting (from original __getitem__)
            i_t, o_t = dict(), t.obs.state
            for idx_cam, cam_idx in enumerate(self.cam_indexes):
                i_c = _get_imgs(t, cam_idx, self.past_frames)
                if self.goal_indexes:
                    g_c = (
                        _get_imgs(goal, cam_idx, 0)
                        if cam_idx in self.goal_indexes
                        else np.zeros_like(i_c[:1])
                    )
                    i_c = np.concatenate((g_c, i_c), axis=0)

                i_c = _img_to_tensor(i_c)
                if self.transform is not None:
                    i_c = self.transform(i_c)
                i_t[f"cam{idx_cam}"] = i_c

            # 4. Tensor Conversion (from original __getitem__)
            o_t_tensor, a_t_tensor = _to_tensor(o_t), _to_tensor(a_t_np)
            loss_mask_tensor = _to_tensor(loss_mask_np)[:, None].repeat((1, a_t_tensor.shape[-1]))
            
            item_data = ((i_t, o_t_tensor), a_t_tensor, loss_mask_tensor)
            
            # 5. Save the final pre-processed item to disk
            torch.save(item_data, item_path)
            # --- End: Caching Logic ---

        if self.cache_to_disk:
            print(f"Finished caching {len(self.item_filepaths)} items.")
        else:
            print(f"Finished loading {len(self.s_a_mask)} items into memory (no cache).")


    def __len__(self):
        if self.cache_to_disk:
            return len(self.item_filepaths)
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        # --- NEW: Fast loading from cache ---
        if self.cache_to_disk:
            filepath = self.item_filepaths[idx]
            return torch.load(filepath)

        # --- ELSE: Original __getitem__ logic ---
        step, a_t_np, loss_mask_np, goal = self.s_a_mask[idx]

        if self.goal_indexes:
            while np.random.uniform() > self.goal_geom_prob and goal.next is not None:
                goal = goal.next

        i_t, o_t = dict(), step.obs.state
        for idx_cam, cam_idx in enumerate(self.cam_indexes):
            i_c = _get_imgs(step, cam_idx, self.past_frames)
            if self.goal_indexes:
                g_c = (
                    _get_imgs(goal, cam_idx, 0)
                    if cam_idx in self.goal_indexes
                    else np.zeros_like(i_c[:1])
                )
                i_c = np.concatenate((g_c, i_c), axis=0)

            i_c = _img_to_tensor(i_c)
            if self.transform is not None:
                i_c = self.transform(i_c)

            i_t[f"cam{idx_cam}"] = i_c

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t_np)
        loss_mask = _to_tensor(loss_mask_np)[:, None].repeat((1, a_t.shape[-1]))
        assert (
            loss_mask.shape[0] == a_t.shape[0]
        ), "a_t and mask shape must be ac_chunk!"
        return (i_t, o_t), a_t, loss_mask
