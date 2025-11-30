import math
import torch 
import torch.nn as nn 
import torch.nn.functional as f 
from rope import RotaryPositionEmbedding2D

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class DropOut(nn.Module): 
    def __init__(self, dropout=0.0, training=False):
        super().__init__()
        self.dropout = dropout
        self.training = training
    def forward(self, x): 
        if self.dropout == 0.0 or not self.training: 
            return x 
        else: 
            keep = 1 - self.dropout
            B = x.shape[0]
            drop_mask= torch.empty((B,) + (1,) * (len(x.shape) - 1)).bernoulli(keep)
            if keep > 0.0:
                drop_mask.div_(keep)
            return x * drop_mask
    
class MultiHeadAttn(nn.Module): 

    def __init__(self, num_heads, emb_dim, qkv_bias = False, dropout=0.0, proj_dropout=0.0, rope=None): 
        super().__init__()
        self.num_heads = num_heads 
        self.emb_dim = emb_dim 
        self.c_q = nn.Linear(emb_dim, emb_dim)
        self.c_k = nn.Linear(emb_dim, emb_dim)
        self.c_v = nn.Linear(emb_dim, emb_dim)
        self.c_out_proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(proj_dropout)
        self.rope = rope

    def forward(self, query, key, value, positions, mask = None): 
        B, T, C = query.shape
        q = self.c_q(query)
        k = self.c_k(key)
        v = self.c_v(value)
        q = q.view((B, T, self.num_heads, C//self.num_heads)).transpose(1, 2)
        k = k.view((B, T, self.num_heads, C//self.num_heads)).transpose(1, 2)
        v = v.view((B, T, self.num_heads, C//self.num_heads)).transpose(1, 2)
        if self.rope is None: 
            pass
        else:
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        attn_1 = (q @ k.transpose(-2, -1)) * (1.0/torch.sqrt(torch.tensor(k.shape[-1])))
        if mask is not None: 
            attn_1 = attn_1.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        softmax = nn.Softmax(dim=-1)
        attn_drop = self.attn_dropout(softmax(attn_1))
        attn_out = (attn_drop@v).transpose(1,2).contiguous().view((B, T, C))
        attn_out = self.c_out_proj(attn_out)
        out = self.out_dropout(attn_out)
        return out 

class FFN(nn.Module): 

    def __init__(self, in_dim, hidden_dim, dropout=[0.0, 0.0]): 
        super().__init__()
        self.in_dim = int(in_dim) 
        self.hidden_dim = int(hidden_dim)
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            nn.Linear(self.hidden_dim, self.in_dim),
            nn.Dropout(dropout[0])
        )

    def forward(self, x): 
        return self.net(x)
    

class TransformerEncoderLayer(nn.Module): 

    def __init__(self, emb_dim, num_heads, ffn_dim=None, qkv_bias=False, res_dropout = 0.0, dropout=0.0, proj_dropout=0.0, rope=False): 

        super().__init__()
        if ffn_dim is None: 
            ffn_dim = 4 * emb_dim
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        if rope:
            self.rope = RotaryPositionEmbedding2D()
        else: 
            self.rope = None
        self.self_attn = MultiHeadAttn(num_heads, emb_dim, qkv_bias, dropout, proj_dropout, rope=self.rope)
        self.ffn = FFN(emb_dim, ffn_dim)
        self.res_drop = DropOut(res_dropout, training=False)
        self.training = False
        
    def forward(self, x, positions, training=False): 
        self.res_drop.training = training
        x = self.ln1(x)
        x = x + self.res_drop(self.self_attn(x, x, x, positions))
        x = x + self.res_drop(self.ffn(self.ln2(x)))
        return x
    
class TransformerDecoderLayer(nn.Module): 

    def __init__(self, emb_dim, num_heads, ffn_dim=None, qkv_bias=False, res_dropout = 0.0, dropout=0.0, proj_dropout=0.0, rope=False): 
        super().__init__()
        if ffn_dim is None: 
            ffn_dim = 4 * emb_dim
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        if rope:
            self.rope = RotaryPositionEmbedding2D()
        else: 
            self.rope = None
        self.self_attn = MultiHeadAttn(num_heads, emb_dim, qkv_bias, dropout, proj_dropout, rope=self.rope)
        self.cross_attn = MultiHeadAttn(num_heads, emb_dim, qkv_bias, dropout, proj_dropout, rope=self.rope)
        self.ffn = FFN(emb_dim, ffn_dim)
        self.res_drop = DropOut(res_dropout, training=False)
        self.training = False

    def forward(self, x, positions, training=False): 
        self.res_drop.training = training 
        x0 = self.ln1(x)


        