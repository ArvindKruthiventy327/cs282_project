import torch
import yaml
import torch.nn as nn 

from trainer import AE_Trainer
from mlp_ae import MLPAutoEncoder, MLPVAE, MLP_VQVAE, ae_loss, vae_loss, vqvae_loss
from torch.utils.data import DataLoader
from latent_dataset import LatentActionBuffer, IterableWrapper


def start_trainer(cfg_file): 

    with open(cfg_file) as stream: 
        cfg = yaml.safe_load(stream)
        model_type = cfg["model"]["type"]
        obs_dim = cfg["model"]["obs_dim"]
        ac_dim = cfg["model"]["ac_dim"]
        latent_dim = cfg["model"]["latent_dim"]
        hidden = cfg["model"]["hidden"]
        if model_type == "VAE": 
            model = MLPVAE(obs_dim, ac_dim, latent_dim, hidden)
            loss_fn = vae_loss
        elif model_type == "AE": 
            model = MLPAutoEncoder(obs_dim, ac_dim, latent_dim, hidden)
            loss_fn = ae_loss
        
        elif model_type == "VQVAE":
            # n_embeddings = cfg["model"]["n_embeddings"] 
            
            model = MLP_VQVAE(obs_dim, ac_dim, latent_dim, hidden)
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
        train_loader = DataLoader(action_dataset, 
                                  batch_size = cfg["dataloader"]["batch_size"], 
                                  shuffle = True, 
                                  num_workers=10)
        val_loader = DataLoader(val_action_dataset, 
                                  batch_size = cfg["dataloader"]["batch_size"], 
                                  shuffle = True, 
                                  num_workers=10) 
        trainer = AE_Trainer(model,loss_fn, "cuda", cfg["optim"], cfg["train_dir"] )
        return model, train_loader, val_loader, trainer

if __name__ == "__main__": 

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--cfg_path", help="Config file path to instantiate model, dataloader, trainer")
    
    args = parser.parse_args()
    model, train_loader, val_loader, trainer =  start_trainer(args.cfg_path)
    trainer.train_loop(train_loader, val_loader)
