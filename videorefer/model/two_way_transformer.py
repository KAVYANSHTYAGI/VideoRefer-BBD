"""
Simplified TwoWayTransformer for MaskDecoder.
Based on SAM's architecture but standalone.
"""

import torch
from torch import nn
import math


class TwoWayTransformer(nn.Module):
    """
    Two-way transformer for bidirectional attention between image and prompts.
    """
    
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        activation: nn.Module = nn.ReLU,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        image_embedding: torch.Tensor,  # [b, c, h, w]
        image_pe: torch.Tensor,  # [b, c, h, w]
        point_embedding: torch.Tensor,  # [b, n, c]
    ) -> tuple:
        """
        Args:
            image_embedding: Image features [b, c, h, w]
            image_pe: Positional encodings [b, c, h, w]
            point_embedding: Prompt tokens [b, n, c]
        
        Returns:
            updated_point_embedding: [b, n, c]
            updated_image_embedding: [b, c, h, w]
        """
        # Flatten image embedding
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # [b, h*w, c]
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # [b, h*w, c]
        
        # Prepare queries (prompts) and keys (image)
        queries = point_embedding
        keys = image_embedding
        
        # Apply transformer layers
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        
        # Final attention from tokens to image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        
        return queries, keys.permute(0, 2, 1).view(bs, c, h, w)


class TwoWayAttentionBlock(nn.Module):
    """Single two-way attention block."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: nn.Module = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        
        self.skip_first_layer_pe = skip_first_layer_pe
    
    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor,
        query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> tuple:
        # Self attention
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)
        
        # Cross attention: tokens → image
        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm2(queries)
        
        # MLP
        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)
        
        # Cross attention: image → tokens
        q = queries + query_pe
        k = keys + key_pe
        keys = keys + self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.norm4(keys)
        
        return queries, keys


class Attention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Simplified attention (can use torch.nn.functional.scaled_dot_product_attention)
        B, N, C = q.shape
        qkv = self.qkv(q).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLPBlock(nn.Module):
    """MLP block with activation."""
    
    def __init__(self, embedding_dim: int, mlp_dim: int, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = activation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
