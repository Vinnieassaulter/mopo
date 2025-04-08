import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

class VQVAE(nn.Module):
    """VQ-VAE for marker point cloud sequences"""
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, 
                 num_embeddings=512, num_markers=53):
        super().__init__()
        
        self.encoder = PointCloudEncoder(input_dim, hidden_dim, latent_dim, num_markers)
        self.decoder = PointCloudDecoder(latent_dim, hidden_dim, input_dim, num_markers)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        
    def encode(self, x):
        z = self.encoder(x)
        z_q, vq_loss, perplexity, encodings = self.vq(z)
        return z_q, vq_loss, perplexity, encodings
    
    def decode(self, z_q):
        return self.decoder(z_q)
    
    def forward(self, x):
        z_q, vq_loss, perplexity, _ = self.encode(x)
        recon_x = self.decode(z_q)
        return recon_x, vq_loss, perplexity

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs from B x ... x C to B x ... x C
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) 
        + torch.sum(self.embedding.weight**2, dim=1) 
        - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1])

class PointCloudTransformer(nn.Module):
    """Transformer block for processing point cloud data with permutation invariance"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PointCloudEncoder(nn.Module):
    """Encoder for marker point cloud sequences"""
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, num_markers=20):
        super().__init__()
        
        # Initial processing of each point independently
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Transformer for inter-point relationships
        self.transformer = PointCloudTransformer(hidden_dim)
        
        # Temporal processing (1D convs)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Downsample to latent dimension
        self.final_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, num_markers, 3)
        batch_size, seq_len, num_markers, _ = x.shape
        
        # Process each point independently
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.point_mlp(x)  # (b*t, n, hidden_dim)
        
        # Apply transformer to capture point interactions
        x = self.transformer(x)
        
        # Average over points (permutation invariant aggregation)
        x = x.mean(dim=1)  # (b*t, hidden_dim)
        x = rearrange(x, '(b t) d -> b d t', b=batch_size)
        
        # Process temporally
        x = self.temporal_conv(x)  # (b, hidden_dim, t)
        x = rearrange(x, 'b d t -> b t d')
        
        # Project to latent dimension
        x = self.final_proj(x)
        return x

class PointCloudDecoder(nn.Module):
    """Decoder for marker point cloud sequences"""
    def __init__(self, latent_dim=32, hidden_dim=64, output_dim=3, num_markers=20):
        super().__init__()
        self.num_markers = num_markers
        
        # Initial expansion
        self.init_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Point generator
        self.point_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        # z shape: (batch, seq_len, latent_dim)
        batch_size, seq_len, _ = z.shape
        
        # Initial projection
        x = self.init_proj(z)  # (b, t, hidden_dim)
        x = rearrange(x, 'b t d -> b d t')
        
        # Temporal processing
        x = self.temporal_conv(x)  # (b, hidden_dim, t)
        x = rearrange(x, 'b d t -> (b t) d')
        
        # Generate points for each frame
        # First create a learned "seed" for each point
        seeds = torch.randn(batch_size * seq_len, self.num_markers, x.shape[-1], 
                          device=x.device) * 0.02
        x = x.unsqueeze(1) + seeds  # (b*t, n, hidden_dim)
        
        # Final point generation
        x = self.point_generator(x)  # (b*t, n, 3)
        x = rearrange(x, '(b t) n d -> b t n d', b=batch_size)
        return x