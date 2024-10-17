import torch.nn as nn
from patchseg.model import MLP, Transformer, TransformerIntervention
from patchseg.vision_backbones import get_backbone_config, load_backbone, modify_backbone


def prepare_backbone(backbone_name: str, output_head: str, label_reduction: str, skip_blocks: int) -> nn.Module:
    # Load backbone.
    backbone = load_backbone(backbone_name)

    # Get backbone configuration.
    backbone_config = get_backbone_config(backbone)

    # Determine intermediate feature representation mode.
    if output_head in ("linear", "mlp"):
        if backbone_config["num_prefix_tokens"] > 0:
            mode = "keep_prefix" if label_reduction == "global-union" else "keep_suffix"
        else:
            mode = "mean" if label_reduction == "global-union" else "keep_suffix"
    else:
        if backbone_config["num_prefix_tokens"] > 0:
            mode = "keep_prefix" if output_head == "transformer-cls" else "keep_all"
        else:
            mode = "mean" if output_head == "transformer-cls" else "keep_all"

    modify_backbone(backbone, mode, skip_blocks)

    return backbone, backbone_config


def prepare_output_head(
    output_head: str,
    backbone_config: dict,
    label_reduction: str,
    hidden_size: int,
    num_layers: int,
    num_classes: int,
) -> nn.Module:
    if output_head == "linear":
        return nn.Linear(backbone_config["embed_dim"], num_classes)

    elif output_head == "mlp":
        return MLP(
            input_size=backbone_config["embed_dim"],
            hidden_size=hidden_size,
            output_size=num_classes,
        )

    elif output_head == "transformer":
        # Determine transformer output mode.
        if label_reduction == "global-union":
            if backbone_config["num_prefix_tokens"] > 0:
                mode = "keep_prefix"
            else:
                mode = "mean"
        else:
            mode = "keep_suffix"

        return Transformer(
            embed_dim=backbone_config["embed_dim"],
            num_heads=backbone_config["num_heads"],
            output_size=num_classes,
            num_tokens=backbone_config["num_patches"] + backbone_config["num_prefix_tokens"],
            num_prefix_tokens=backbone_config["num_prefix_tokens"],
            mode=mode,
            num_layers=num_layers,
        )

    elif output_head in ("transformer-cls", "transformer-separate"):
        assert label_reduction == "union"  # Interventions are meant to be applied with local labels.
        return TransformerIntervention(
            num_latent_tokens=backbone_config["num_patches"],
            embed_dim=backbone_config["embed_dim"],
            num_heads=backbone_config["num_heads"],
            output_size=num_classes,
            num_layers=num_layers,
        )

    else:
        raise ValueError(f"Unknown output head type: {output_head}")
