import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module): 

    def __init__(self, channels_in, channels_out): 
        super().__init__()
        self.gamma = nn.Linear(channels_in, channels_out)
        self.beta = nn.Linear(channels_in, channels_out)

    def forward(self, cond, x): 
        gamma = self.gamma(cond).unsqueeze(2)
        beta = self.beta(cond).unsqueeze(2)
        out = (x * gamma) + beta 
        return out

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: (Batch, Channels, Length)
        
        # 1. Permute to (Batch, Length, Channels)
        x = x.transpose(1, 2)
        
        # 2. Apply LayerNorm
        x = self.norm(x)
        
        # 3. Permute back to (Batch, Channels, Length)
        x = x.transpose(1, 2)
        
        return x
    
class ConvEncoder(nn.Module):
    
    def __init__(self, seq_len=60, state_dim = 9, action_dim=10, channels=[64,128,256], latent_dim=64, film=True):
        super().__init__()

        layers = []
        in_ch = action_dim
        length = seq_len
        for i, ch in enumerate(channels):
            layers.append(LayerNorm1d(in_ch))
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            if i == 0: 
                layers.append(FiLM(state_dim, ch))
            layers.append(nn.GELU())

            in_ch = ch
            length = (length + 1) // 2
        layers.append(nn.AdaptiveAvgPool1d(output_size=1))
        self.model = nn.ModuleList(layers)
        self.final_len = length
    
    def forward(self, obs_t, traj):
        
        h = traj.permute(0,2,1)    # → (B,10,seq_len)
        # print(f"Obs dimension: {obs_t.shape} and traj dimension: {h.shape}") 
        for l, layer in enumerate(self.model): 
            if l == 2: 
                h = layer(obs_t, h)
                continue
            h = layer(h)
        h = h.flatten(1)
        return h 

class ConvDecoder(nn.Module):
    def __init__(self, seq_len=60, state_dim = 9, action_dim=10, channels=[256,128,64], in_padding=[1, 1, 1], out_padding=[1,1,1], latent_dim=64):
        super().__init__()

        # self.init_len = seq_len // (2 ** len(channels))
        # self.fc = nn.Linear(latent_dim, channels[0] * self.init_len)
        
        layers = []
        self.upscale = (nn.Upsample(scale_factor=4, mode='linear', align_corners=False))
        for i in range(len(channels)-1):
            layers.append(LayerNorm1d(channels[i]))
            layers.append(nn.ConvTranspose1d(
                channels[i], channels[i+1],
                kernel_size=3, stride=2, padding=in_padding[i], output_padding=out_padding[i]
            ))
            if i == 0: 
                layers.append(FiLM(state_dim, channels[i+1]))
            layers.append(nn.GELU())

        layers.append(nn.ConvTranspose1d(
            channels[-1], action_dim,
            kernel_size=3, stride=2, padding=1, output_padding=1
        ))

        # self.model = nn.Sequential(*layers)
        self.model = nn.ModuleList(layers)
    # def forward(self, z):
    #     h = self.fc(z)
    #     h = h.view(z.size(0), -1, self.init_len)
    #     x = self.model(h)      # (B, action_dim, seq_len)
    #     return x.permute(0,2,1)
    
    def forward(self, obs_t, z): 
        # h = self.fc(z)
        h = z.unsqueeze(-1)
        h = self.upscale(h)
        for l, layer in enumerate(self.model): 
            if l == 2: 
                h = layer(obs_t, h)
                continue
            h = layer(h)
        h = h.transpose(2,1)
        return h 
    
class CNN_VAE(nn.Module): 

    def __init__(self, seq_len=60, state_dim = 9, action_dim=10, 
                       channels=[256,128,64], in_padding=[1, 1, 1], 
                       out_padding=[1,1,1], latent_dim=64): 

        super().__init__()
        self.enc = ConvEncoder(
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            channels=channels,
            latent_dim=latent_dim
        )

        self.dec = ConvDecoder(
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            channels=channels[::-1],
            latent_dim=latent_dim, 
            in_padding=in_padding, 
            out_padding=out_padding
        )

        self.mu_proj = nn.Linear(latent_dim, latent_dim)
        self.sigma_proj = nn.Linear(latent_dim, latent_dim)

    def reparametrize(self, z): 
        eps = torch.randn_like(z).to(z.device)
        mu = self.mu_proj(z)
        sigma = self.sigma_proj(z)
        sigma = torch.clamp(sigma, min=-20, max=3)
        z_sample = mu + torch.exp(0.5 * sigma) * eps
        return z_sample, mu, sigma 
    
    def forward(self, obs_t, traj): 
        z = self.enc(obs_t, traj)
        z_sample, mu, sigma = self.reparametrize(z)
        x_hat = self.dec(obs_t, z_sample)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat, mu, sigma

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

class CNN_VQVAE(nn.Module): 

    def __init__(self, seq_len=60, state_dim = 9, action_dim=10, 
                       channels=[256,128,64], in_padding=[1, 1, 1], 
                       out_padding=[1,1,1], latent_dim=64, n_embeddings=256): 
        super().__init__()
        self.enc = ConvEncoder(
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            channels=channels,
            latent_dim=latent_dim
        )

        self.dec = ConvDecoder(
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            channels=channels[::-1],
            latent_dim=latent_dim, 
            in_padding=in_padding, 
            out_padding=out_padding
        )
        self.quantizer = VectorQuantizerEMA(
            n_embeddings=n_embeddings,
            embedding_dim=latent_dim
        )
        self.z_e = None
    
    def forward(self, obs_t, traj):  # x: (B, T, action_dim)
        z_e = self.enc(obs_t, traj)
        self.z_e = z_e.clone().detach()
        z_q, indices = self.quantizer(z_e)
        out = self.dec(obs_t, z_q)
        return out, z_q, z_e, indices
    
def ae_loss(x, x_hat): 
    loss_fn = nn.MSELoss(reduction="mean")
    loss = loss_fn(x_hat, x)

    return loss

def vae_loss(input, x, beta=0.8):
    x_hat, mu, logvar = input
    recon_loss_fn = nn.MSELoss(reduction="mean")
    recon_loss = recon_loss_fn(x_hat, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def vqvae_loss(input, x, commitment_weight=0.25): 
    x_hat, z_q, z_e, indices = input
    # codebook_loss = F.mse_loss(z_e.detach(), z_q)
    commitment_loss = F.mse_loss(z_e, z_q.detach())
    recon_loss = F.mse_loss(x_hat, x, reduction="mean")
    return recon_loss + commitment_weight*commitment_loss