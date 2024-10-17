import os
import random
import logging
import argparse
import numpy as np
from typing import ContextManager, Tuple
from functools import partial
from contextlib import suppress

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from timm.scheduler import CosineLRScheduler

from patchseg.dataset import CocoPatchDataset, Ade20kPatchDataset
from patchseg.loss import create_metrics
from patchseg.model import MLP, Transformer, TransformerIntervention
from patchseg.vision_backbones import get_backbone_config, load_backbone, modify_backbone


# Set up argument parser.
# fmt: off
parser = argparse.ArgumentParser()

# Vision backbone arguments.
parser.add_argument("--backbone", type=str, default="in1k-vit-b")
parser.add_argument("--skip-blocks", type=int, default=0)
parser.add_argument("--full-finetune", action="store_true")

# Output head arguments.
parser.add_argument("--output-head", type=str, default="transformer", choices=["linear", "mlp", "transformer", "transformer-cls", "transformer-separate"])
parser.add_argument("--hidden-size", type=int, default=1024)
parser.add_argument("--num-layers", type=int, default=2)

# Task arguments.
parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "ade20k"])
parser.add_argument("--label-reduction", type=str, default="union", choices=["union", "global-union", "majority"])

# Optimization arguments.
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--min-lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-epochs", type=int, default=5)
parser.add_argument("--warmup-steps", type=int, default=500)
parser.add_argument("--weight-decay", type=float, default=0.01)

# Systems arguments.
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--num-workers", type=int, default=24)

# Logging and saving.
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--no-save", action="store_true")
# fmt: on


# Configure logger.
logging.basicConfig(format="[Patch-Seg] %(message)s", level=logging.INFO)


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


def prepare_loaders(
    dataset_name: str,
    image_width: int,
    image_mean: Tuple[float],
    image_std: Tuple[float],
    patch_size: int,
    label_reduction: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    # Get dataset class.
    if dataset_name == "coco":
        dataset_class = CocoPatchDataset
    elif dataset_name == "ade20k":
        dataset_class = Ade20kPatchDataset

    # Prepare train data.
    train_dataset = dataset_class(
        image_width=image_width,
        patch_size=patch_size,
        split="train",
        image_mean=image_mean,
        image_std=image_std,
        label_reduction=label_reduction,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
        drop_last=True,
    )

    # Prepare val data.
    val_dataset = dataset_class(
        image_width=image_width,
        patch_size=patch_size,
        split="val",
        image_mean=image_mean,
        image_std=image_std,
        label_reduction=label_reduction,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=False,
        drop_last=False,
    )

    # Initialize dataloader workers.
    next(iter(train_loader))
    next(iter(val_loader))

    return train_loader, val_loader, train_dataset.num_classes


def evaluate(
    backbone: nn.Module,
    output_head: nn.Module,
    loader: DataLoader,
    loss_criterion: nn.Module,
    metrics_helper: nn.Module,
    amp_autocast: ContextManager,
) -> None:
    # Setup.
    backbone.eval()
    output_head.eval()
    metrics_helper.reset()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for image, annotation in loader:
            image = image.cuda()
            annotation = annotation.cuda()
            with amp_autocast():
                features = backbone(image)
                logits = output_head(features)
                loss = loss_criterion(logits, annotation)
                metrics_helper(logits, annotation)
            total_loss += loss.item() * len(image)
            total_samples += len(image)

    # Log results.
    loss = total_loss / total_samples
    accuracy = metrics_helper.compute_accuracy()
    macro_accuracy = metrics_helper.compute_macro_accuracy()
    macro_recall = metrics_helper.compute_macro_recall()
    macro_precision = metrics_helper.compute_macro_precision()
    macro_f1 = metrics_helper.compute_macro_f1()
    logging.info(
        f"Validation loss {loss:.5f}, "
        f"accuracy {accuracy:.5f}, "
        f"macro accuracy {macro_accuracy:.5f}, "
        f"macro recall {macro_recall:.5f}, "
        f"macro precision {macro_precision:.5f}, "
        f"macro f1 {macro_f1:.5f}"
    )

    # Return all metrics.
    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
    }


def save_results(metrics_dict: dict[str, float], args: argparse.Namespace):
    # Determine filename.
    filename = f"results/outputs-{args.dataset}.csv"
    os.makedirs("results", exist_ok=True)

    # Write header if file doesn't exist.
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(
                "dataset, label_reduction, output_head, backbone, "
                "skip_blocks, full_finetune, num_epochs, "
                "loss, accuracy, macro_accuracy, macro_recall, macro_precision, macro_f1\n"
            )

    # Add line for this model.
    with open(filename, "a") as f:
        f.write(
            f"{args.dataset}, {args.label_reduction}, {args.output_head}, {args.backbone}, "
            f"{args.skip_blocks}, {args.full_finetune}, {args.num_epochs}, "
            f"{metrics_dict['loss']}, {metrics_dict['accuracy']}, {metrics_dict['macro_accuracy']}, "
            f"{metrics_dict['macro_recall']}, {metrics_dict['macro_precision']}, "
            f"{metrics_dict['macro_f1']}\n"
        )


if __name__ == "__main__":
    # Parse arguments.
    args = parser.parse_args()

    # Seed everything.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Prepare feature extractor.
    backbone, backbone_config = prepare_backbone(
        args.backbone,
        output_head=args.output_head,
        label_reduction=args.label_reduction,
        skip_blocks=args.skip_blocks,
    )

    # Extract relevant attributes.
    image_mean = backbone_config["image_mean"]
    image_std = backbone_config["image_std"]
    image_width = backbone_config["image_width"]
    patch_size = backbone_config["patch_size"]
    embed_dim = backbone_config["embed_dim"]

    # Prepare data loaders.
    train_loader, val_loader, num_classes = prepare_loaders(
        args.dataset,
        image_width,
        image_mean,
        image_std,
        patch_size,
        args.label_reduction,
        args.batch_size,
        args.num_workers,
    )

    # Prepare output head.
    output_head = prepare_output_head(
        args.output_head,
        backbone_config,
        label_reduction=args.label_reduction,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
    )
    logging.info(
        f"Training with {args.dataset} and {args.label_reduction} label reduction ({num_classes} classes), "
        f"{args.output_head} output head (full finetune = {args.full_finetune})"
    )

    # Set up loss function and optimizer.
    loss_criterion, metrics_helper = create_metrics(num_classes, args.label_reduction)
    backbone_context = torch.no_grad if not args.full_finetune else suppress
    params = (
        list(backbone.parameters()) + list(output_head.parameters())
        if args.full_finetune
        else list(output_head.parameters())
    )
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=args.num_epochs * updates_per_epoch,
        lr_min=args.min_lr,
        t_in_epochs=False,
        warmup_t=args.warmup_steps,
    )

    # Model setup.
    backbone.cuda()
    output_head.cuda()

    # Set up AMP.
    dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_dict[args.dtype]
    if dtype == torch.bfloat16:
        logging.info("Using bfloat16 AMP")
        amp_autocast = partial(torch.autocast, device_type="cuda", dtype=dtype)
    elif dtype == torch.float32:
        logging.info("Using float32, no AMP")
        amp_autocast = suppress
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # For storing best results.
    best_metrics = None
    best_recall = float("-inf")

    # Training loop.
    num_updates = 0
    for epoch in range(args.num_epochs):
        backbone.train()
        output_head.train()
        for i, (image, annotation) in enumerate(train_loader):
            image = image.cuda()
            annotation = annotation.cuda()
            with amp_autocast():
                with backbone_context():
                    features = backbone(image)
                logits = output_head(features)
                loss = loss_criterion(logits, annotation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_updates += 1
            scheduler.step_update(num_updates=num_updates, metric=None)
            if i % args.log_every == 0:
                logging.info(f"Epoch {epoch}, " f"iter {i}, " f"loss {loss.item():.5f}")

        # Validation.
        metrics_dict = evaluate(
            backbone,
            output_head,
            val_loader,
            loss_criterion,
            metrics_helper,
            amp_autocast,
        )
        if metrics_dict["macro_recall"] > best_recall:
            # Save best metrics.
            best_recall = metrics_dict["macro_recall"]
            best_metrics = metrics_dict

    # Shutdown dataloader workers.
    train_loader._iterator._shutdown_workers()
    val_loader._iterator._shutdown_workers()

    if not args.no_save:
        # Write results to file.
        save_results(best_metrics, args)
