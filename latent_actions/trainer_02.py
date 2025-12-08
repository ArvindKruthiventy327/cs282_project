import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import wandb 
import os
from collections import deque
from functools import partial 

class AE_Trainer(): 

    def __init__(self, model, loss_fn, device_id, optim_params, train_save_dir, model_type="VAE", dist_reg = 0.8): 
        self.model = model 
        self.optim_params = optim_params
        self.train_save_dir = train_save_dir
        self.model_type = model_type
        print(f"Model type trainer: {model_type}")
        # Handle device setup
        if isinstance(device_id, int):
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device(device_id if torch.cuda.is_available() else "cpu")
            
        self.model.to(self.device)

        os.makedirs(self.train_save_dir, exist_ok = True)
        print(self.optim_params)
        
        self.optim = optim.Adam(model.parameters(), lr = self.optim_params["lr"])
        if model_type == "VAE":
            self.loss_fn = partial(loss_fn, beta=dist_reg)
        elif model_type == "VQVAE": 
            self.loss_fn = partial(loss_fn, commitment_weight=dist_reg )
        
        # Metrics
        self.log_losses = deque()
        self.losses = deque()
        self.val_losses = deque()
        self.running_loss_accum = 0.0

    def get_next_batch(self, dataloader, iterator):
        """Helper to get next batch or reset iterator if dataset is exhausted"""
        try:
            batch = next(iterator)
        except StopIteration:
            # Dataset exhausted, reset iterator
            iterator = iter(dataloader)
            batch = next(iterator)
        return batch, iterator

    def train_step(self, batch_cond_input, batch_input): 
        """
        Performs exactly one gradient update step.
        Returns the scalar loss for logging.
        """
        # Forward pass
        # Note: Ensure your model accepts inputs in this specific order
        output = self.model(batch_cond_input, batch_input)
        
        # Concat inputs for reconstruction target as per your original logic
        if self.arch == "mlp":
            true_output = torch.cat([batch_cond_input, batch_input], dim=1)
        elif self.arch == "conv": 
            true_output = batch_input
        loss = self.loss_fn(output, true_output)

        # Backward pass
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        
        return loss.item()

    def validation(self, val_loader): 
        self.model.eval() # Switch to eval mode
        running_loss = 0.0
        
        with torch.no_grad(): # No gradients needed for validation
            for batch in val_loader:
                batch_cond_input = batch[0].to(self.device)
                if self.arch=="mlp":
                    batch_input = batch[1].flatten(start_dim=1, end_dim=2).to(self.device)
                    true_output = torch.cat([batch_cond_input, batch_input], dim=1)
                elif self.arch=="conv": 
                    batch_input = batch[1].to(self.device)
                    true_output = batch_input
                # true_output = torch.cat([batch_cond_input, batch_input], dim=1)
                output = self.model(batch_cond_input, batch_input)
                loss = self.loss_fn(output, true_output)
                running_loss += loss.item()
        
        avg_val_loss = running_loss / len(val_loader)
        self.val_losses.append(avg_val_loss)
        
        self.model.train() # Switch back to train mode
        return avg_val_loss
    
    def train_loop(self, trainloader, val_loader, steps_per_epoch, arch="mlp",codebook_clear=True):
        """
        steps_per_epoch: How many batches to process before calling it an 'epoch'
        """
        total_steps = self.optim_params["epochs"] * steps_per_epoch 
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.optim_params["lr"],
            total_steps=total_steps,
            pct_start=0.1, # Warmup for first 10% of training
            anneal_strategy='cos'
        )
        self.model.train()
        
        # Create an iterator from the dataloader
        train_iter = iter(trainloader)
        global_step = 0

        print(f"Starting training for {self.optim_params['epochs']} epochs, {steps_per_epoch} steps per epoch.")

        for epoch in range(self.optim_params["epochs"]): 
            
            epoch_loss = 0.0

            # --- Inner Loop runs for fixed steps, not dataset length ---
            for i in range(steps_per_epoch):
                
                # 1. Get Data (Handle infinite loop)
                batch, train_iter = self.get_next_batch(trainloader, train_iter)
                
                states = batch[0].to(self.device)
                if arch=="mlp":
                    actions = batch[1].flatten(start_dim=1, end_dim=2).to(self.device)
                elif arch=="conv": 
                    actions = batch[1].to(self.device)
                self.arch = arch
                # 2. Train Step
                loss_val = self.train_step(states, actions)
                
                # 3. Update Metrics
                self.losses.append(loss_val)
                self.running_loss_accum += loss_val
                epoch_loss += loss_val
                global_step += 1

                # 4. Optional: Codebook expiration (VQVAE specific)
                if codebook_clear: 
                    # Ensure your model has these attributes if using this flag
                    if hasattr(self.model, 'quantizer'):
                        self.model.quantizer.expire_codes(self.model.z_e) 

                # 5. Logging (Step-based)
                if global_step % self.optim_params["log_freq"] == 0: 
                    avg_running_loss = self.running_loss_accum / self.optim_params["log_freq"]
                    self.log_losses.append(avg_running_loss)
                    self.running_loss_accum = 0.0
                    print(f"Step {global_step} | Logged Loss: {avg_running_loss:.5f}")

                # 6. Saving & Validation (Step-based)
                if global_step % self.optim_params["save_freq"] == 0: 
                    val_loss = self.validation(val_loader)
                    print(f"Step {global_step} | Val Loss: {val_loss:.5f} | Saving checkpoint...")
                    
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(self.train_save_dir, 'checkpoint.pth'))

            # --- End of Epoch ---
            print(f"End of Epoch {epoch} | Avg Epoch Loss: {epoch_loss / steps_per_epoch:.5f}")

        # Final Save
        self.save_logs()
        print("Training Finished.")

    def save_logs(self):
        """Helper to save numpy arrays"""
        os.makedirs(self.train_save_dir, exist_ok = True)
        np.save(os.path.join(self.train_save_dir, "losses.npy"), np.array(self.losses))
        np.save(os.path.join(self.train_save_dir, "log_losses.npy"), np.array(self.log_losses))
        np.save(os.path.join(self.train_save_dir, "val_losses.npy"), np.array(self.val_losses))
    


