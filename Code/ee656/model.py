import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(SpatialTemporalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "Dimension must be divisible by num_heads"
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mode='spatial'):
        batch_size, seq_len, num_patches, dim = x.shape
        q = self.query(x).view(batch_size, seq_len, num_patches, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, num_patches, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, num_patches, self.num_heads, self.head_dim)

        if mode == 'spatial':
            q = q.permute(0, 3, 1, 2, 4)  # (batch, heads, seq_len, patches, head_dim)
            k = k.permute(0, 3, 1, 2, 4)
            v = v.permute(0, 3, 1, 2, 4)
            attn_scores = torch.einsum('bhtpd,bhtqd->bhtpq', q, k) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.einsum('bhtpq,bhtqd->bhtpd', attn_weights, v)
            out = out.permute(0, 2, 3, 1, 4).reshape(batch_size, seq_len, num_patches, dim)
        else:  # temporal
            q = q.permute(0, 3, 2, 1, 4)  # (batch, heads, patches, seq_len, head_dim)
            k = k.permute(0, 3, 2, 1, 4)
            v = v.permute(0, 3, 2, 1, 4)
            attn_scores = torch.einsum('bhptd,bhqtd->bhptq', q, k) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.einsum('bhptq,bhqtd->bhptd', attn_weights, v)
            out = out.permute(0, 3, 2, 1, 4).reshape(batch_size, seq_len, num_patches, dim)
        
        return out, attn_weights

class SelfSubtractMechanism(nn.Module):
    def __init__(self, dim):
        super(SelfSubtractMechanism, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)  # Original input used for values

    def forward(self, x, original_x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(original_x)  # Use original input for values
        return q, k, v

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.spatial_attn = SpatialTemporalSelfAttention(dim, num_heads)
        self.temporal_attn = SpatialTemporalSelfAttention(dim, num_heads)
        self.self_subtract = SelfSubtractMechanism(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, original_x):
        # Spatial self-attention
        x = self.norm1(x)
        spatial_out, spatial_attn = self.spatial_attn(x, mode='spatial')
        x = x + self.dropout(spatial_out)
        
        # Temporal self-attention with self-subtract
        x = self.norm2(x)
        q, k, v = self.self_subtract(x, original_x)
        temporal_out, temporal_attn = self.temporal_attn(x, mode='temporal')
        x = x + self.dropout(temporal_out)
        
        # Feed-forward network
        x = self.norm3(x)
        x = x + self.dropout(self.ffn(x))
        
        return x, spatial_attn, temporal_attn

class ISTVT(nn.Module):
    def __init__(self, img_size=112, patch_size=16, seq_len=2, dim=768, num_heads=12, ff_dim=3072, num_blocks=12, num_classes=2):
        super(ISTVT, self).__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim
        
        # Feature extractor (e.g., CNN backbone)
        self.feature_extractor = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim) for _ in range(num_blocks)
        ])
        
        # Prediction head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        
        # Feature extraction
        x = x.view(-1, c, h, w)
        x = self.feature_extractor(x)  # (batch*seq_len, dim, h', w')
        x = x.view(batch_size, seq_len, self.dim, -1).permute(0, 1, 3, 2)  # (batch, seq_len, num_patches, dim)
        
        # Add classification token
        cls_token = self.cls_token.expand(batch_size, seq_len, 1, -1)
        x = torch.cat([cls_token, x], dim=2)  # (batch, seq_len, num_patches+1, dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        original_x = x.clone()  # For self-subtract mechanism
        
        # Transformer blocks
        spatial_attns, temporal_attns = [], []
        for block in self.blocks:
            x, spatial_attn, temporal_attn = block(x, original_x)
            spatial_attns.append(spatial_attn)
            temporal_attns.append(temporal_attn)
        
        # Extract classification token for prediction
        cls_output = x[:, 0, 0, :]  # (batch, dim)
        logits = self.mlp_head(cls_output)
        
        return logits, spatial_attns, temporal_attns