import timm
import torch
import logging
from typing import Any, Callable
from patchseg.custom_backbones import CUSTOM_BACKBONES
from timm.models.vision_transformer import VisionTransformer
from timm.models.beit import Beit
from timm.models.eva import Eva

# Groups of model configs.
IN1K_BACKBONES = {
    "in1k-vit-t": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "in1k-vit-s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "in1k-vit-b": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "in1k-vit-l": "vit_large_patch16_224.augreg_in21k_ft_in1k",
}

CLIP_BACKBONES = {
    "clip-vit-b": "vit_base_patch16_clip_quickgelu_224.openai",
    "clip-vit-l": "vit_large_patch14_clip_quickgelu_224.openai",
    "clip-vit-l-336px": "vit_large_patch14_clip_quickgelu_336.openai",
}

SIGLIP_VISION_BACKBONES = {
    "siglip-vit-b": "vit_base_patch16_siglip_224",
    "siglip-vit-b-256px": "vit_base_patch16_siglip_256",
    "siglip-vit-b-384px": "vit_base_patch16_siglip_384",
    "siglip-vit-so400m": "vit_so400m_patch14_siglip_224",
    "siglip-vit-so400m-384px": "vit_so400m_patch14_siglip_384",
}

MAE_BACKBONES = {
    "mae-vit-b": "vit_base_patch16_224.mae",
    "mae-vit-l": "vit_large_patch16_224.mae",
    "mae-vit-h": "vit_huge_patch14_224.mae",
}

OPENCLIP_BACKBONES = {
    "openclip-vit-b": "vit_base_patch16_clip_224.laion2b",
    "openclip-vit-l": "vit_large_patch14_clip_224.laion2b",
    "openclip-vit-h": "vit_huge_patch14_clip_224.laion2b",
}

DFN_BACKBONES = {
    "dfn-vit-b": "vit_base_patch16_clip_224.dfn2b",
    "dfn-vit-l": "vit_large_patch14_clip_quickgelu_224.dfn2b",
    "dfn-vit-h": "vit_huge_patch14_clip_224.dfn5b",
    "dfn-vit-h-378px": "vit_huge_patch14_clip_378.dfn5b",
}

IJEPA_BACKBONES = {
    "ijepa-vit-h": "vit_huge_patch14_gap_224.in22k_ijepa",
}

DINO_BACKBONES = {
    "dino-vit-b": "vit_base_patch16_224.dino",
}

MOCO_BACKBONES = {
    "moco-vit-b": "vit_base_patch16_mocov3_224",
}

BEIT_BACKBONES = {
    "beit-vit-b": "beit_base_patch16_224.in22k_ft_in22k_in1k",
    "beitv2-vit-b": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
}

EVA_BACKBONES = {
    "eva02-vit-b": "eva02_base_patch16_clip_224.merged2b",
    "eva02-vit-l": "eva02_large_patch14_clip_224.merged2b",
}

DBOT_BACKBONES = {
    "dbot-clip-vit-b": "vit_base_patch16_dbot_clip_224",
    "dbot-random-vit-b": "vit_base_patch16_dbot_random_224",
}

# Merge into single dict.
TIMM_BACKBONES = {
    **IN1K_BACKBONES,
    **CLIP_BACKBONES,
    **SIGLIP_VISION_BACKBONES,
    **MAE_BACKBONES,
    **OPENCLIP_BACKBONES,
    **DFN_BACKBONES,
    **IJEPA_BACKBONES,
    **DINO_BACKBONES,
    **MOCO_BACKBONES,
    **BEIT_BACKBONES,
    **EVA_BACKBONES,
    **DBOT_BACKBONES,
}


def load_backbone(backbone_name: str) -> torch.nn.Module:
    """Load backbone by config name."""
    if backbone_name in TIMM_BACKBONES:
        return timm.create_model(
            model_name=TIMM_BACKBONES[backbone_name],
            pretrained=True,
            num_classes=0,
        )

    elif backbone_name in CUSTOM_BACKBONES:
        return CUSTOM_BACKBONES[backbone_name]()

    else:
        raise ValueError(f"Unknown model name: {backbone_name}")


def get_backbone_config(backbone: torch.nn.Module) -> dict:
    """Get configuration options from backbone model."""
    if isinstance(backbone, (VisionTransformer, Beit, Eva)):
        data_cfg = timm.data.resolve_data_config(backbone.pretrained_cfg)
        config = {
            "image_mean": data_cfg["mean"],
            "image_std": data_cfg["std"],
            "image_width": data_cfg["input_size"][-1],
            "patch_size": backbone.patch_embed.patch_size[0],
            "num_patches": backbone.patch_embed.num_patches,
            "num_prefix_tokens": backbone.num_prefix_tokens,
            "embed_dim": backbone.embed_dim,
            "num_heads": backbone.blocks[0].attn.num_heads,
        }

    else:
        raise ValueError(f"Unknown backbone type: {type(backbone)}")

    return config


def keep_prefix(fn: Callable[[Any], torch.Tensor], num_prefix_tokens: int) -> Callable[[Any], Any]:
    """Keep prefix tokens only, and calculate mean in case of multiple tokens."""

    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = fn(*args, **kwargs)
        return result[:, :num_prefix_tokens].mean(dim=1, keepdim=True)

    return wrapper


def keep_suffix(fn: Callable[[Any], torch.Tensor], num_prefix_tokens: int) -> Callable[[Any], Any]:
    """Keep suffix tokens only."""

    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = fn(*args, **kwargs)
        return result[:, num_prefix_tokens:]

    return wrapper


def keep_mean(fn: Callable[[Any], torch.Tensor]) -> Callable[[Any], Any]:
    """Keep mean of tokens only."""

    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = fn(*args, **kwargs)
        return torch.mean(result, dim=1, keepdim=True)

    return wrapper


def modify_backbone(model: torch.nn.Module, mode: str, skip_blocks: int = 0) -> torch.nn.Module:
    """
    Modify forward function to extract intermediate representations.

    Args:
        model: vision transformer model.
        mode: one of "keep_prefix", "keep_suffix", "keep_all" or "mean".
        skip_blocks: number of transformer blocks to skip.
    """
    # Prune layers.
    if skip_blocks > 0:
        num_blocks = len(model.blocks)
        take_indices = model.prune_intermediate_layers(indices=[len(model.blocks) - 1 - skip_blocks])
        logging.info(f"Pruning backbone to {take_indices[0] + 1}/{num_blocks} blocks")

    # Replace forward function.
    if mode == "keep_prefix":
        model.forward = keep_prefix(model.forward_features, num_prefix_tokens=model.num_prefix_tokens)

    elif mode == "keep_suffix":
        model.forward = keep_suffix(model.forward_features, num_prefix_tokens=model.num_prefix_tokens)

    elif mode == "keep_all":
        model.forward = model.forward_features

    elif mode == "mean":
        model.forward = keep_mean(model.forward_features)

    else:
        raise ValueError(f"Unknown backbone mode: {mode}")

    return model
