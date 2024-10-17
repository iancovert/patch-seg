import torch
from PIL import Image
from typing import Any
from torchvision.transforms.v2 import Transform


class LargestCenterCrop(Transform):
    """Take largest center crop from image."""

    def __init__(self) -> None:
        super().__init__()

    def _transform(self, image: Image, params: dict[str, Any]) -> Image:
        min_dim = min(image.width, image.height)
        x = (image.width - min_dim) // 2
        y = (image.height - min_dim) // 2
        return image.crop((x, y, x + min_dim, y + min_dim))


class ReduceAnnotation(Transform):
    """Reduce segmentation mask labels to patch level."""

    def __init__(self, patch_size: int, num_classes: int, label_reduction: str) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.label_reduction = label_reduction

    def _transform(self, annotation: Image, params: dict[str, Any]) -> torch.Tensor:
        # Unfold annotation into patches.
        patches = annotation[0].unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        patches = patches.flatten(2)
        # patches.shape = (num_patches, num_patches, patch_size * patch_size)

        if self.label_reduction == "majority":
            # Most common label for each patch.
            labels = patches.mode(dim=-1).values

        elif self.label_reduction == "union":
            # Binary indicator for each class in each patch.
            labels = torch.zeros(patches.shape[0], patches.shape[1], self.num_classes)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    # Set all unique classes to present.
                    unique_classes = torch.unique(patches[i, j])
                    labels[i, j, unique_classes] = 1

                    # Override "unlabeled" class, only allowed if it's the unique class.
                    if len(unique_classes) > 1:
                        labels[i, j, 0] = 0

        elif self.label_reduction == "global-union":
            # Binary indicator for each class in entire image.
            labels = torch.zeros(self.num_classes)
            unique_classes = torch.unique(patches.flatten())
            labels[unique_classes] = 1

        else:
            raise ValueError(f"Invalid label_reduction method: {self.label_reduction}")

        return labels


class CocoLabelAssignment(Transform):
    """Compress segmentation mask labels to contiguous integers."""

    def __init__(self, class_set: str = "all") -> None:
        super().__init__()
        assert class_set in ["all", "things"]
        self.class_set = class_set

    def _transform(self, annotation: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        # Increment classes and wrap around 256.
        annotation = (annotation + 1) % 256

        if self.class_set == "things":
            # Collape "stuff" classes into "unlabeled" class.
            annotation[annotation > 91] = 0

        return annotation
