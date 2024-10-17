import os
import timm
import torch
from functools import partial
from typing import Optional
# from locality_alignment import load_checkpoint_auto


def load_dinov2_backbone(arch_name: str, img_size: Optional[int]) -> torch.nn.Module:
    """Load DINOv2 backbone and override image size."""
    backbone = timm.create_model(arch_name, pretrained=True, num_classes=0, img_size=img_size)
    if img_size is not None:
        backbone.default_cfg["input_size"] = (3, img_size, img_size)
    return backbone


# def load_aligned_backbone(
#     checkpoint_name: str,
#     arch_name: str,
#     img_size: Optional[int] = None,
#     model_kwargs: dict = {},
# ) -> torch.nn.Module:
#     # Create model.
#     backbone = timm.create_model(arch_name, pretrained=False, num_classes=0, img_size=img_size, **model_kwargs)
#     if img_size is not None:
#         backbone.default_cfg["input_size"] = (3, img_size, img_size)

#     # Load checkpoint.
#     non_matching_keys = load_checkpoint_auto(backbone, checkpoint_name, strict=False)

#     # Remove output head.
#     backbone.head = torch.nn.Identity()
#     return backbone


DINOV2_BACKBONES = {
    # # Without registers https://arxiv.org/abs/2304.07193
    # "dinov2-vit-b": partial(
    #     load_dinov2_backbone,
    #     arch_name="vit_base_patch14_dinov2.lvd142m",
    #     img_size=224,
    # ),
    # "dinov2-vit-l": partial(
    #     load_dinov2_backbone,
    #     arch_name="vit_large_patch14_dinov2.lvd142m",
    #     img_size=224,
    # ),
    # With registers https://arxiv.org/abs/2309.16588
    "dinov2-vit-b": partial(
        load_dinov2_backbone,
        arch_name="vit_base_patch14_reg4_dinov2.lvd142m",
        img_size=224,
    ),
    "dinov2-vit-l": partial(
        load_dinov2_backbone,
        arch_name="vit_large_patch14_reg4_dinov2.lvd142m",
        img_size=224,
    ),
}


# LOCALITY_ALIGNED_BACKBONES = {
#     # Main locality alignment runs.
#     "in1k-vit-b-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "in1k-vit-b-maskembed"),
#         arch_name="svit_base_patch16_224",
#     ),
#     "in1k-vit-l-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "in1k-vit-l-maskembed"),
#         arch_name="svit_large_patch16_224",
#     ),
#     "clip-vit-b-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "clip-vit-b-maskembed"),
#         arch_name="svit_base_patch16_clip_quickgelu_224",
#     ),
#     "clip-vit-l-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "clip-vit-l-maskembed"),
#         arch_name="svit_large_patch14_clip_quickgelu_224",
#     ),
#     "clip-vit-l-336px-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "clip-vit-l-336px-maskembed"),
#         arch_name="svit_large_patch14_clip_quickgelu_336",
#     ),
#     "siglip-vit-b-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "siglip-vit-b-maskembed"),
#         arch_name="svit_base_patch16_siglip_224",
#     ),
#     "siglip-vit-so400m-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "siglip-vit-so400m-maskembed"),
#         arch_name="svit_so400m_patch14_siglip_224",
#     ),
#     "siglip-vit-so400m-384px-aligned": partial(
#         load_aligned_backbone,
#         checkpoint_name=os.path.join("vision_checkpoints", "siglip-vit-so400m-384px-maskembed"),
#         arch_name="svit_so400m_patch14_siglip_384",
#     ),
# }


CUSTOM_BACKBONES = {
    # **LOCALITY_ALIGNED_BACKBONES,
    **DINOV2_BACKBONES,
}
