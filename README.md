# Patch-Seg

[![arXiv](https://img.shields.io/badge/arXiv-2410.11087-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2410.11087)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Timm](https://img.shields.io/badge/TIMM-1.0.8-black.svg?style=for-the-badge&logo=huggingface)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/github/license/iancovert/patch-seg?style=for-the-badge)](LICENSE)

This is a repository for Patch-Seg, a probing benchmark to test local feature extraction abilities of pre-trained vision transformers (ViTs). It was developed alongside a post-training technique for ViTs called *locality alignment*, and both are described in [this paper](https://arxiv.org/abs/2410.11087). To test locality-aligned models with PatchSeg, you'll also want to use the [locality-alignment](https://github.com/iancovert/locality-alignment) repository.

The primary usage of Patch-Seg is to cast semantic segmentation as a patch-level multi-label classification task, where each patch either contains or does not contain each class in the dataset (e.g., person, dog, etc). We train for this task using a frozen ViT backbone and a learnable output head, typically a transformer. The package also supports other options for the output head (MLP, linear) and other label reduction strategies (majority labels, union of labels in image).

# Installation and usage

First, clone the repository and install it in your conda environment:

```bash
git clone https://github.com/iancovert/patch-seg.git
cd patch-seg
pip install -e .
```

Utilities for training are in the `patchseg` package, and the main entrypoint is the training script at `scripts/train.py`.

**Data.** Before training, you need to set up the training dataset. PatchSeg currently supports MSCOCO and ADE20k - you can download MSCOCO using the shell script at `scripts/download_coco.sh`, and ADE20k from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/). Datasets should be placed in the root directory of the repository, e.g., `coco` or `ade20k`.

**Training.** To train a model, run the following command from the root directory:

```bash
python scripts/train.py --backbone clip-vit-b --label-reduction union --output-head transformer
```

There are several configuration options for the training script, including:

- `--backbone`: short name for the backbone model. All available options are registered in `patchseg/vision_backbones.py` and `patchseg/custom_backbones.py`
- `--label-reduction`: set to `"union"` by default. Can also be set to `"majority"` for patch-level multi-class classification, or `"global-union"` for image-level multi-label classification
- `--output-head`: set to `"transformer"` by default, can also be set to `"mlp"` or `"linear"`
- `--dataset`: set to `"coco"` by default, can also be set to `"ade20k"`

Training runs are fast (minutes), so progress is logged to the console and the final metrics are saved in the `results` directory. We observed minor variability in the results across different hardware, so for consistency we performed all training runs on a single Nvidia H100 GPU.

**Registering new backbones.** For models available on [timm](https://github.com/huggingface/pytorch-image-models), you can simply add the model to `patchseg/vision_backbones.py`. For other models with custom loading logic or local checkpoints, you can register them in `patchseg/custom_backbones.py` (we provide an example there).


### Citation

If you find our code useful in your work, please cite [our paper](https://arxiv.org/abs/2410.11087):

```bibtex
@article{covert2024locality,
  title = {Locality Alignment Improves Vision-Language Models},
  author = {Covert, Ian and Sun, Tony and Zou, James and Hashimoto, Tatsunori},
  year = {2024},
  journal = {arXiv preprint arXiv:2410.11087},
}
```
