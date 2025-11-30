import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    
    def __init__(self, seq_len=60, action_dim=10, channels=[64,128,256], latent_dim=64):
        super().__init__()

        layers = []
        in_ch = action_dim
        length = seq_len

        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch
            length = (length + 1) // 2
            
        self.conv = nn.Sequential(*layers)
        self.final_len = length

        self.fc_mu = nn.Linear(channels[-1] * length, latent_dim)
        self.fc_logvar = nn.Linear(channels[-1] * length, latent_dim)

    def forward(self, x):       # x: (B, seq_len, 10)
        x = x.permute(0,2,1)    # → (B,10,seq_len)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)



class ConvDecoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, channels=[256,128,64], latent_dim=64):
        super().__init__()

        self.init_len = seq_len // (2 ** len(channels))
        self.fc = nn.Linear(latent_dim, channels[0] * self.init_len)

        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.ConvTranspose1d(
                channels[i], channels[i+1],
                kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose1d(
            channels[-1], action_dim,
            kernel_size=3, stride=2, padding=1, output_padding=1
        ))

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), -1, self.init_len)
        x = self.deconv(h)      # (B, action_dim, seq_len)
        return x.permute(0,2,1)


class VAE(nn.Module):
    def __init__(self, seq_len=60, action_dim=10,
                 enc_channels=[64,128,256],
                 dec_channels=[256,128,64],
                 latent_dim=64):
        super().__init__()

        self.encoder = ConvEncoder(
            seq_len=seq_len,
            action_dim=action_dim,
            channels=enc_channels,
            latent_dim=latent_dim
        )

        self.decoder = ConvDecoder(
            seq_len=seq_len,
            action_dim=action_dim,
            channels=dec_channels,
            latent_dim=latent_dim
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        out = self.decoder(z)
        return out, mu, logvar

    
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl
    
# --------------------------------------------------------
#      Vector Quantization module (EMA version)
# --------------------------------------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings=512, embedding_dim=256, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", embed.clone())

    def forward(self, z_e):   # z_e: (B, C, T)
        B, C, T = z_e.shape

        # (B*T, C)
        flat = z_e.permute(0, 2, 1).reshape(-1, C)

        # compute distances
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (B*T, n_embeddings)

        # nearest embedding index
        indices = torch.argmin(dist, dim=1)  # (B*T,)
        z_q = F.embedding(indices, self.embedding)  # (B*T, C)

        # reshape back
        z_q = z_q.view(B, T, C).permute(0, 2, 1)  # (B, C, T)
        indices = indices.view(B, T)

        # EMA updates (no grad)
        if self.training:
            with torch.no_grad():
                # one-hot: (B*T, n_embed)
                encodings = F.one_hot(indices.view(-1), self.n_embeddings).float()

                # count update
                new_count = encodings.sum(0)
                self.ema_count.mul_(self.decay).add_(new_count, alpha=1 - self.decay)
                total = self.ema_count.sum()
                self.ema_count = (self.ema_count + self.eps) / (total + self.n_embeddings * self.eps) * total

                # weight update
                new_weight = encodings.t() @ flat
                self.ema_weight.mul_(self.decay).add_(new_weight, alpha=1 - self.decay)

                # final normalized embedding
                self.embedding.copy_(self.ema_weight / self.ema_count.unsqueeze(1))

        # losses
        # z_e → z_q with straight-through
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())

        # straight-through
        z_q = z_e + (z_q - z_e).detach()

        return z_q, indices, codebook_loss, commitment_loss



# --------------------------------------------------------
#      Encoder (Conv1D, same as your VAE encoder)
# --------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, channels=[64,128,256], hidden_dim=256):
        super().__init__()

        layers = []
        in_ch = action_dim
        length = seq_len

        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch
            length = (length + 1) // 2

        self.conv = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self.out_len = length

        # project to latent embedding dimension
        self.proj = nn.Conv1d(channels[-1], hidden_dim, kernel_size=1)

    def forward(self, x):  # x: (B, seq_len, action_dim)
        x = x.permute(0, 2, 1)
        h = self.conv(x)
        return self.proj(h)   # (B, hidden_dim, T')


# --------------------------------------------------------
#      Decoder (ConvTranspose1D, same as VAE decoder)
# --------------------------------------------------------
class ConvDecoder(nn.Module):
    def __init__(self, seq_len=60, action_dim=10, channels=[256,128,64], hidden_dim=256):
        super().__init__()

        # compute starting length
        self.init_len = seq_len // (2 ** len(channels))
        self.input_channels = hidden_dim

        layers = []
        in_ch = hidden_dim
        for ch in channels:
            layers.append(nn.ConvTranspose1d(
                in_ch, ch,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            layers.append(nn.ReLU())
            in_ch = ch

        # final projection
        layers.append(nn.ConvTranspose1d(
            in_ch, action_dim,
            kernel_size=3, stride=2, padding=1, output_padding=1
        ))

        self.deconv = nn.Sequential(*layers)

    def forward(self, z_q):
        return self.deconv(z_q).permute(0, 2, 1)  # (B, seq_len, action_dim)



# --------------------------------------------------------
#                     VQ-VAE
# --------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(self, seq_len=60, action_dim=10,
                 enc_channels=[64,128,256],
                 dec_channels=[256,128,64],
                 hidden_dim=256,
                 n_embeddings=512):
        super().__init__()

        self.encoder = ConvEncoder(
            seq_len=seq_len,
            action_dim=action_dim,
            channels=enc_channels,
            hidden_dim=hidden_dim
        )

        self.quantizer = VectorQuantizerEMA(
            n_embeddings=n_embeddings,
            embedding_dim=hidden_dim
        )

        self.decoder = ConvDecoder(
            seq_len=seq_len,
            action_dim=action_dim,
            channels=dec_channels,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        z_e = self.encoder(x)  # (B, C, T')
        z_q, indices, codebook_loss, commitment_loss = self.quantizer(z_e)
        out = self.decoder(z_q)
        return out, codebook_loss, commitment_loss, indices

   
def vqvae_loss(recon_x, x, codebook_loss, commitment_loss, beta=0.25):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    loss = recon_loss + codebook_loss + beta * commitment_loss
    return loss, recon_loss, codebook_loss, commitment_loss