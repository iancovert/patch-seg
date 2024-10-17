import timm

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Transformer(nn.Module):
    """Small output head transformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_size: int,
        num_tokens: int,
        num_prefix_tokens: int,
        mode: str,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        assert mode in ("keep_prefix", "keep_suffix", "mean")
        self.mode = mode

        # Positional embeddings.
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

        # Input normalization.
        self.norm_pre = nn.LayerNorm(embed_dim)

        # Transformer blocks (parameters set to match ViT-B architecture).
        self.blocks = nn.Sequential(
            *[
                timm.models.vision_transformer.Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=False,
                    qk_norm=False,
                    init_values=0.01,
                    proj_drop=0,
                    attn_drop=0,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    mlp_layer=timm.layers.Mlp,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization.
        self.norm = nn.LayerNorm(embed_dim)

        # Output head.
        self.head = nn.Linear(embed_dim, output_size)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply positional embeddings and transformer blocks.
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)

        # Extract output tokens.
        if self.mode == "keep_prefix":
            x = x[:, : self.num_prefix_tokens].contiguous()
            if x.shape[1] > 1:
                x = x.mean(dim=1, keepdim=True)
        elif self.mode == "keep_suffix":
            x = x[:, self.num_prefix_tokens :].contiguous()
        elif self.mode == "mean":
            x = torch.mean(x, dim=1, keepdim=True)

        return x


class TransformerIntervention(nn.Module):
    """Small output head transformer that makes predictions via latent tokens."""

    def __init__(
        self,
        num_latent_tokens: int,
        embed_dim: int,
        num_heads: int,
        output_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_latent_tokens = num_latent_tokens

        # Latent embeddings.
        self.latent_embed = nn.Parameter(torch.randn(num_latent_tokens, embed_dim) * 0.02)

        # Input normalization.
        self.norm_pre = nn.LayerNorm(embed_dim)

        # Transformer blocks (parameters set to match ViT-B architecture).
        self.blocks = nn.Sequential(
            *[
                timm.models.vision_transformer.Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=False,
                    qk_norm=False,
                    init_values=0.01,
                    proj_drop=0,
                    attn_drop=0,
                    drop_path=0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    mlp_layer=timm.layers.Mlp,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization.
        self.norm = nn.LayerNorm(embed_dim)

        # Output head.
        self.head = nn.Linear(embed_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, self.latent_embed.expand(x.shape[0], -1, -1)], dim=1)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)

        return x[:, -self.num_latent_tokens :].contiguous()
