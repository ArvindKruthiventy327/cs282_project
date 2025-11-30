import torch
import torch.nn as nn 
import torch.nn.functional as F 
import copy 

class Encoder(nn.Module): 

    def __init__(self, obs_dim, ac_dim, latent_dim, hidden): 

        super().__init__()
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        flattened_in_dim = (self.obs_dim[0] * self.obs_dim[1]) + (self.ac_dim[0] * self.ac_dim[1])
        layer_dims = copy.deepcopy(hidden)
        layer_dims.reverse()
        layer_dims.append(flattened_in_dim)
        layer_dims.reverse()
        layer_dims.append(latent_dim)
        model = []
        for i in range(len(layer_dims) - 1):
            if i > 0: 
                model.append(nn.LayerNorm(layer_dims[i]))
            model.append(nn.Linear(layer_dims[i], 
                                        layer_dims[i+1]))
            model.append(nn.GELU())
        self.model = nn.ModuleList(model)
    def forward(self, obs_t, traj): 

        x = torch.cat([obs_t, traj], dim=1) 
        for layer in self.model: 
            x = layer(x)
        return x
    
class Decoder(nn.Module): 

    def __init__(self, latent_dim, output_dim, hidden): 
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim 
        layer_dims = copy.deepcopy(hidden)
        layer_dims.reverse()
        layer_dims.append(latent_dim)
        layer_dims.reverse()
        layer_dims.append(output_dim)
        model = []
        # print("decoder_layers", layer_dims)
        for i in range(len(layer_dims) - 1):
            if i > 0: 
                model.append(nn.LayerNorm(layer_dims[i]))
            model.append(nn.Linear(layer_dims[i], 
                                        layer_dims[i+1]))
            model.append(nn.GELU())
        model.pop(-1)
        self.model = nn.ModuleList(model)
    def forward(self, latent): 

        x = latent
        for layer in self.model: 
            x = layer(x)
        return x
    
class MLPAutoEncoder(nn.Module): 

    def __init__(self, obs_dim, ac_dim, latent_dim, hidden, dec_hidden=None): 
        super().__init__()
        self.input_dim = obs_dim[0] * obs_dim[1] + ac_dim[0] * ac_dim[1]
        self.latent_dim = latent_dim
        self.enc_layers = copy.deepcopy(hidden) 
        if dec_hidden is None: 
            self.dec_layers = hidden[::-1]
        else: 
            self.dec_layers = dec_hidden 
        # print("decoder_layers", self.dec_layers)
        self.enc = Encoder(obs_dim, ac_dim, latent_dim, self.enc_layers)
        self.dec = Decoder(latent_dim + obs_dim[0] * obs_dim[1], self.input_dim, self.dec_layers)

    def forward(self, obs_t, traj): 
        z = self.enc(obs_t, traj)
        z_hat = torch.cat([obs_t, z], axis=1)
        x_hat = self.dec(z_hat)
        return x_hat 
    
class MLPVAE(MLPAutoEncoder): 

    def __init__(self,obs_dim, ac_dim, latent_dim, hidden, dec_hidden=None): 
        super().__init__(obs_dim, ac_dim,latent_dim, hidden, dec_hidden)
        self.mu_proj = nn.Linear(latent_dim, latent_dim)
        self.sigma_proj = nn.Linear(latent_dim, latent_dim)
    
    def reparametrize(self, z): 
        eps = torch.randn_like(z).to(z.device)
        mu = self.mu_proj(z)
        sigma = self.sigma_proj(z)
        z_sample = mu + torch.exp(0.5 * sigma) * eps
        return z_sample, mu, sigma 
    
    def forward(self, obs_t, traj): 
        z = self.enc(obs_t, traj)
        z_sample, mu, sigma = self.reparametrize(z)
        x_hat = self.dec(torch.cat([obs_t, z_sample], dim=1))
        # x_hat = torch.sigmoid(x_hat)
        return x_hat, mu, sigma
            

# --------------------------------------------------------
#      Vector Quantizer with EMA (same as CNN version)
# --------------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", embed.clone())

    def forward(self, z_e):  # z_e: (B, T, D)
        flat = z_e

        # squared Euclidean distance
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (B*T, n_embeddings)

        indices = torch.argmin(dist, dim=1)  # (B*T,)
        z_q = F.embedding(indices, self.embedding)  # (B*T, D)
        # EMA updates
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices.view(-1), self.n_embeddings).float()

                # update ema count
                new_count = encodings.sum(0)
                self.ema_count.mul_(self.decay).add_(new_count, alpha=1 - self.decay)

                # normalize
                total = self.ema_count.sum()
                self.ema_count = (self.ema_count + self.eps) / \
                                 (total + self.n_embeddings * self.eps) * total

                # update embedding
                new_weight = encodings.t() @ flat
                self.ema_weight.mul_(self.decay).add_(new_weight, alpha=1 - self.decay)

                # final normalized embedding
                self.embedding.copy_(self.ema_weight / self.ema_count.unsqueeze(1))

        # loss
        # straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices

    def expire_codes(self, z_e):
        # z_e: (Batch, Latent_Dim) - The actual encoder outputs from the current batch
        if self.training:
            with torch.no_grad():
                # 1. Calculate usage of each code
                # We use the 'ema_count' buffer you already have
                # If a code's usage is below a threshold (e.g., 1.0 or 0.1), it is "dead"
                expired_indices = (self.ema_count < 1.0).nonzero(as_tuple=True)[0]
                
                n_expired = expired_indices.size(0)
                if n_expired > 0:
                    # 2. Randomly sample real encoder outputs from the current batch
                    # This ensures the new code is definitely inside the data distribution
                    random_indices = torch.randint(0, z_e.size(0), (n_expired,)).to(z_e.device)
                    new_codes = z_e[random_indices].detach()
                    
                    # 3. Reset the dead embeddings to these new active points
                    self.embedding[expired_indices] = new_codes
                    
                    # 4. Reset the EMA counters for these codes so they have a fresh start
                    self.ema_weight[expired_indices] = new_codes
                    self.ema_count[expired_indices] = 1.0

class MLP_VQVAE(MLPAutoEncoder):
    def __init__(
        self,
        obs_dim, 
        ac_dim, 
        latent_dim, 
        hidden,
        n_embeddings=512,
    ):
        super().__init__(obs_dim, ac_dim, latent_dim, hidden)

        self.quantizer = VectorQuantizerEMA(
            n_embeddings=n_embeddings,
            embedding_dim=latent_dim
        )

        

    def forward(self, obs_t, traj):  # x: (B, T, action_dim)
        z_e = self.enc(obs_t, traj)
        z_q, indices = self.quantizer(z_e)
        out = self.dec(torch.cat([obs_t, z_q], dim=1))
        return out, z_q, z_e, indices
    
def ae_loss(x, x_hat): 
    loss_fn = nn.MSELoss(reduction="mean")
    loss = loss_fn(x_hat, x)
    return loss

def vae_loss(input, x):
    x_hat, mu, logvar = input
    recon_loss_fn = nn.MSELoss(reduction="mean")
    recon_loss = recon_loss_fn(x_hat, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def vqvae_loss(input, x): 
    x_hat, z_q, z_e, indices = input
    codebook_loss = F.mse_loss(z_e.detach(), z_q)
    commitment_loss = F.mse_loss(z_e, z_q.detach())
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    return recon_loss + 0.25*commitment_loss
