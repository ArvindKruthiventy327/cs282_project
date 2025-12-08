import numpy as np
import torch
import yaml
import torch.nn as nn 

from trainer_02 import AE_Trainer
# from mlp_ae import MLPAutoEncoder, MLPVAE, MLP_VQVAE, ae_loss, vae_loss, vqvae_loss
from cnn_ae import CNN_VAE, CNN_VQVAE, ae_loss, vae_loss, vqvae_loss
from torch.utils.data import DataLoader
from latent_dataset import LatentActionBuffer, IterableWrapper

def get_test_accuracy(model, test_dataset, model_type="VAE"): 
    
    test_loader = DataLoader(test_dataset, 
                                  batch_size = 5, 
                                  shuffle = True, 
                                  num_workers=10) 
    model.eval().cuda()
    criterion = nn.MSELoss()
    test_losses = []
    counter = 0
    with torch.no_grad(): 
        for batch in test_loader: 
            states = batch[0]
            actions = batch[1]

            states = states.to("cuda")
            actions = actions.to("cuda")
            gt = actions
            if model_type == "VAE":
                pred, mu,logvar = model(states, actions)
            elif model_type == "VQVAE": 
                pred, z_q, z_e, indices = model(states, actions)
            # print(pred)
            loss = criterion(pred, gt)
            test_losses.append(loss.item())
        losses = np.array(test_losses)
        avg_loss = losses.mean()
        std_loss = losses.std()

        print(f"Evaluation complete. Average Reconstruction Loss (MSE): {avg_loss:.4f} (Std Dev: {std_loss:.4f})")
        test_stats = {"avg_loss": avg_loss, "std_loss": std_loss}
        return test_stats
    
def start_trainer(cfg_file): 

    with open(cfg_file) as stream: 
        cfg = yaml.safe_load(stream)
        model_params = cfg["model"]
        model_type = cfg["model"]["type"]

        if model_type == "VAE": 
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            channels = model_params["channels"]
            dec_in_padding = model_params["dec_in_padding"]
            dec_out_padding =  model_params["dec_out_padding"]
            model = CNN_VAE(ac_dim[0], obs_dim[0], ac_dim[-1], channels,dec_in_padding, dec_out_padding, latent_dim)
            loss_fn = vae_loss
                
        elif model_type == "VQVAE":
            n_embeddings = cfg["model"]["n_embeddings"] 
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            channels = model_params["channels"]
            dec_in_padding = model_params["dec_in_padding"]
            dec_out_padding =  model_params["dec_out_padding"]
            model = CNN_VQVAE(ac_dim[0], obs_dim[0], ac_dim[-1], channels,dec_in_padding, dec_out_padding, latent_dim, n_embeddings=n_embeddings)
            loss_fn = vqvae_loss

        raw_data = cfg["dataloader"]["raw_data"]
        n_test = cfg["dataloader"]["n_test"]
        n_val = cfg["dataloader"]["n_val"]
        ac_chunk = cfg["dataloader"]["ac_chunk"]
        obs_dim = cfg["dataloader"]["obs_dim"]
        ac_dim = cfg["dataloader"]["ac_dim"]
        action_dataset = LatentActionBuffer(raw_data, 
                                            n_test,
                                            n_val, 
                                            obs_dim=obs_dim, 
                                            ac_chunk = ac_chunk, 
                                            ac_dim = ac_dim)
        val_action_dataset = LatentActionBuffer(raw_data, 
                                            n_test,
                                            n_val,
                                            mode="val", 
                                            obs_dim=obs_dim, 
                                            ac_chunk = ac_chunk, 
                                            ac_dim = ac_dim)
        test_action_dataset = LatentActionBuffer(raw_data, 
                                            n_test,
                                            n_val,
                                            mode="test", 
                                            obs_dim=obs_dim, 
                                            ac_chunk = ac_chunk, 
                                            ac_dim = ac_dim)
        train_loader = DataLoader(action_dataset, 
                                  batch_size = cfg["dataloader"]["batch_size"], 
                                  shuffle = True, 
                                  num_workers=10)
        val_loader = DataLoader(val_action_dataset, 
                                  batch_size = cfg["dataloader"]["batch_size"], 
                                  shuffle = True, 
                                  num_workers=10) 
        if model_type == "VQVAE":
            trainer = AE_Trainer(model,loss_fn, "cuda", cfg["optim"], cfg["train_dir"],1000)
        else: 
            trainer = AE_Trainer(model,loss_fn, "cuda", cfg["optim"], cfg["train_dir"],1000)
        return model, train_loader, val_loader, test_action_dataset, trainer

if __name__ == "__main__": 

    from argparse import ArgumentParser
    import json
    import os  
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", help="Config file path to instantiate model, dataloader, trainer")
    
    args = parser.parse_args()
    model, train_loader, val_loader, test_action_dataset,trainer =  start_trainer(args.cfg_path)
    trainer.train_loop(train_loader, val_loader, 1000, arch="conv")
    with open(args.cfg_path) as stream: 
        cfg = yaml.safe_load(stream)
    test_stats = get_test_accuracy(trainer.model, test_action_dataset,cfg["model"]["type"])
    print(test_stats)
    
    with open(os.path.join(cfg["train_dir"], "test_stats.json"),"w") as f :
        json.dump(test_stats, f)