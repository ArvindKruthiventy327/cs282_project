import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb 
import os
from collections import deque

class AE_Trainer(): 

    def __init__(self, model, loss_fn, device_id, optim_params, train_save_dir, model_type="VAE"): 

        self.model = model 
        self.device_id = device_id
        self.optim_params = optim_params
        self.train_save_dir = train_save_dir
        os.makedirs(self.train_save_dir, exist_ok = True)
        print(self.optim_params)
        self.optim = optim.Adam(model.parameters(), lr = self.optim_params["lr"])
        self.loss_fn = loss_fn
        self.log_losses = deque()
        self.losses = deque()
        self.val_losses = deque()
        self.running_mean = 0.0

        # self.lr_scheduler = optim.lr_scheduler
    
    def train_step(self, batch_cond_input, batch_input, val_loader, global_step ): 
        true_output = torch.cat([batch_cond_input, batch_input], dim=1)
        output = self.model(batch_cond_input, batch_input)
        loss = self.loss_fn(output, true_output)
        self.losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        self.running_mean += loss.item()
        if global_step % self.optim_params["log_freq"] == 0: 
            self.losses.append(self.running_mean / self.optim_params["log_freq"])
            self.running_mean = 0.0
            print(f"Logged loss: {self.losses[-1]}")
            # if self.model_type == "VQVAE":
            #     self.model.quantizer.refresh_codebook = True 
                

        if global_step % self.optim_params["save_freq"] == 0: 
            # Save both the model state_dict and the optimizer state_dict
            checkpoint = {
                'epoch': global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
            }
            val_losses = self.validation(val_loader)
            torch.save(checkpoint, os.path.join(self.train_save_dir, 'checkpoint.pth'))

    def validation(self, val_loader): 
        running_loss = 0.0
        batch_size = 0
        for batch in val_loader:
            batch_size = batch[0].shape[0]
            batch_cond_input = batch[0].to("cuda")
            batch_input = batch[1].flatten(start_dim=1, end_dim=2).to("cuda")
            true_output = torch.cat([batch_cond_input, batch_input], dim=1)
            output = self.model(batch_cond_input, batch_input)
            loss = self.loss_fn(output, true_output)
            running_loss += loss.item()
        self.val_losses.append(running_loss / (len(val_loader)))
        return running_loss / (len(val_loader))
    
    def train_loop(self, trainloader,val_loader):
        self.model.cuda()
        iter = 0
        for epoch in range(self.optim_params["epochs"]): 

            self.model.train()
            for batch in trainloader: 

                states = batch[0]
                actions = batch[1].flatten(start_dim=1, end_dim=2) 

                states = states.to("cuda")
                actions = actions.to("cuda")
                self.optim.zero_grad()

                self.train_step(states, actions, val_loader,iter)
                
                iter+= 1
        os.makedirs(self.train_save_dir, exist_ok = True)
        losses_path = os.path.join(self.train_save_dir, "losses.npy")
        losses = np.array(self.losses)
        np.save(losses_path, losses)
        log_losses_path = os.path.join(self.train_save_dir, "log_losses.npy")
        log_losses = np.array(self.log_losses)
        np.save(log_losses_path, log_losses)
        val_losses_path = os.path.join(self.train_save_dir, "val_losses.npy")
        val_losses = np.array(self.val_losses)
        np.save(val_losses_path, val_losses)
        



    


    
    