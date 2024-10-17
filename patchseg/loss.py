import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from abc import ABC, abstractmethod


def create_metrics(num_classes: int, label_reduction: str) -> Tuple[nn.Module, nn.Module]:
    if label_reduction == "majority":
        return MajorityLoss(), MajorityMetricsHelper(num_classes)

    elif label_reduction == "union":
        return UnionLoss(), UnionMetricsHelper(num_classes)

    elif label_reduction == "global-union":
        return GlobalUnionLoss(), GlobalUnionMetricsHelper(num_classes)


class MajorityLoss(nn.Module):
    """Standard cross-entropy loss for training with majority labels."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, num_patches * num_patches, num_classes)
        # targets.shape = (batch, num_patches, num_patches)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        return F.cross_entropy(logits, targets)


class UnionLoss(nn.Module):
    """Binary cross-entropy loss for training with union labels."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, num_patches * num_patches, num_classes)
        # targets.shape = (batch, num_patches, num_patches, num_classes)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1))
        return F.binary_cross_entropy_with_logits(logits, targets)


class GlobalUnionLoss(nn.Module):
    """Binary cross-entropy loss for training with global union labels."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, 1, num_classes)
        # targets.shape = (batch, num_classes)
        logits = logits[:, 0]
        return F.binary_cross_entropy_with_logits(logits, targets)


class ConfusionMatrix(nn.Module):
    """Confusion matrix that be can incrementally updated."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Update confusion matrix with new predictions and targets."""
        if self.num_classes == 2:
            # Binary case.
            self.confusion_matrix[0, 0] += torch.sum((targets == 0) & (predictions == 0))
            self.confusion_matrix[0, 1] += torch.sum((targets == 0) & (predictions == 1))
            self.confusion_matrix[1, 0] += torch.sum((targets == 1) & (predictions == 0))
            self.confusion_matrix[1, 1] += torch.sum((targets == 1) & (predictions == 1))
        else:
            # Multiclass case.
            for target, prediction in zip(targets, predictions):
                self.confusion_matrix[target, prediction] += 1

        return self.confusion_matrix


class MajorityMetricsHelper(nn.Module):
    """Metrics helper for training with patch-level majority labels."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cmat = ConfusionMatrix(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, num_patches * num_patches, num_classes)
        # targets.shape = (batch, num_patches, num_patches)
        logits, targets = logits.cpu(), targets.cpu()
        logits = logits.view(-1, logits.size(-1))
        probs = logits.softmax(dim=-1)
        targets = targets.view(-1)
        predictions = probs.argmax(dim=-1)
        return self.cmat(predictions, targets)

    def reset(self) -> None:
        self.cmat.reset()

    def compute_class_accuracy(self) -> torch.Tensor:
        cmat = self.cmat.confusion_matrix
        true_positives = cmat.diag()
        true_negatives = cmat.sum() - cmat.sum(dim=0) - cmat.sum(dim=1) + true_positives
        total_instances = cmat.sum()
        class_accuracy = (true_positives + true_negatives) / total_instances
        return class_accuracy

    def compute_class_recall(self) -> torch.Tensor:
        cmat = self.cmat.confusion_matrix
        class_recall = cmat.diag() / cmat.sum(dim=1)
        return class_recall

    def compute_class_precision(self) -> torch.Tensor:
        cmat = self.cmat.confusion_matrix
        class_precision = cmat.diag() / cmat.sum(dim=0)
        return class_precision

    def compute_class_f1(self) -> torch.Tensor:
        precision = self.compute_class_precision()
        recall = self.compute_class_recall()
        class_f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return class_f1

    def compute_macro_accuracy(self) -> torch.Tensor:
        valid_classes = self.cmat.confusion_matrix.sum(dim=1) > 0
        return self.compute_class_accuracy()[valid_classes].mean()

    def compute_macro_recall(self) -> torch.Tensor:
        valid_classes = self.cmat.confusion_matrix.sum(dim=1) > 0
        return self.compute_class_recall()[valid_classes].mean()

    def compute_macro_precision(self) -> torch.Tensor:
        valid_classes = self.cmat.confusion_matrix.sum(dim=1) > 0
        return self.compute_class_precision()[valid_classes].mean()

    def compute_macro_f1(self) -> torch.Tensor:
        valid_classes = self.cmat.confusion_matrix.sum(dim=1) > 0
        return self.compute_class_f1()[valid_classes].mean()

    def compute_micro_accuracy(self) -> torch.Tensor:
        cmat = self.cmat.confusion_matrix
        return cmat.diag().sum() / cmat.sum()

    def compute_accuracy(self) -> torch.Tensor:
        return self.compute_micro_accuracy()


class MultilabelMetricsHelper(ABC, nn.Module):
    """ABC for metrics when training with multilabel classification."""

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def reset(self) -> None:
        for cmat in self.cmats:
            cmat.reset()

    def compute_class_accuracy(self) -> torch.Tensor:
        class_accuracies = []
        for cmat in self.cmats:
            cmat = cmat.confusion_matrix
            accuracy = cmat.diag().sum() / cmat.sum()
            class_accuracies.append(accuracy)
        return torch.tensor(class_accuracies)

    def compute_class_recall(self) -> torch.Tensor:
        class_recalls = []
        for cmat in self.cmats:
            cmat = cmat.confusion_matrix
            class_recall = cmat.diag() / cmat.sum(dim=1)
            # Only include for positive instances of the class.
            class_recalls.append(class_recall[1])
        return torch.tensor(class_recalls)

    def compute_class_precision(self) -> torch.Tensor:
        class_precisions = []
        for cmat in self.cmats:
            cmat = cmat.confusion_matrix
            class_precision = cmat.diag() / cmat.sum(dim=0)
            # Only include for positive instances of the class.
            class_precisions.append(class_precision[1])
        return torch.tensor(class_precisions)

    def compute_class_f1(self) -> torch.Tensor:
        precision = self.compute_class_precision()
        recall = self.compute_class_recall()
        class_f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return class_f1

    def compute_macro_accuracy(self) -> torch.Tensor:
        valid_classes = torch.tensor([cmat.confusion_matrix.sum(dim=1)[1] > 0 for cmat in self.cmats])
        return self.compute_class_accuracy()[valid_classes].mean()

    def compute_macro_recall(self) -> torch.Tensor:
        valid_classes = torch.tensor([cmat.confusion_matrix.sum(dim=1)[1] > 0 for cmat in self.cmats])
        return self.compute_class_recall()[valid_classes].mean()

    def compute_macro_precision(self) -> torch.Tensor:
        valid_classes = torch.tensor([cmat.confusion_matrix.sum(dim=1)[1] > 0 for cmat in self.cmats])
        return self.compute_class_precision()[valid_classes].mean()

    def compute_macro_f1(self) -> torch.Tensor:
        valid_classes = torch.tensor([cmat.confusion_matrix.sum(dim=1)[1] > 0 for cmat in self.cmats])
        return self.compute_class_f1()[valid_classes].mean()

    def compute_micro_accuracy(self) -> torch.Tensor:
        numerator = sum([cmat.confusion_matrix.diag().sum() for cmat in self.cmats])
        denominator = sum([cmat.confusion_matrix.sum() for cmat in self.cmats])
        return numerator / denominator

    def compute_accuracy(self) -> torch.Tensor:
        return self.compute_micro_accuracy()


class GlobalUnionMetricsHelper(MultilabelMetricsHelper):
    """Confusion matrix for training with image-level global union labels."""

    def __init__(self, num_classes: int, detection_threshold: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.detection_threshold = detection_threshold
        self.cmats = [ConfusionMatrix(2) for _ in range(num_classes)]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, 1, num_classes)
        # targets.shape = (batch, num_classes)
        logits, targets = logits.cpu(), targets.cpu()
        probs = logits[:, 0].sigmoid()
        predictions = (probs > self.detection_threshold).long()
        for i in range(self.num_classes):
            self.cmats[i](predictions[:, i], targets[:, i])


class UnionMetricsHelper(MultilabelMetricsHelper):
    """Confusion matrix for training with patch-level union labels."""

    def __init__(self, num_classes: int, detection_threshold: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.detection_threshold = detection_threshold
        self.cmats = [ConfusionMatrix(2) for _ in range(num_classes)]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, num_patches * num_patches, num_classes)
        # targets.shape = (batch, num_patches, num_patches, num_classes)
        logits, targets = logits.cpu(), targets.cpu()
        probs = logits.view(-1, logits.size(-1)).sigmoid()
        targets = targets.view(-1, targets.size(-1))
        predictions = (probs > self.detection_threshold).long()
        for i in range(self.num_classes):
            self.cmats[i](predictions[:, i], targets[:, i])
        return self.cmats


class SizeUnionMetricsHelper(MultilabelMetricsHelper):
    """Confusion matrix for evaluating with union labels, stratified by size."""

    def __init__(self, num_buckets: int, detection_threshold: float = 0.5) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.detection_threshold = detection_threshold
        self.cmats = [ConfusionMatrix(2) for _ in range(num_buckets)]
        self.buckets = torch.linspace(0, 1, steps=self.num_buckets + 1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch, num_patches * num_patches, num_classes)
        # targets.shape = (batch, num_patches, num_patches, num_classes)
        logits, targets = logits.cpu(), targets.cpu()
        batch_size, num_patches, num_classes = logits.shape
        targets = targets.view(batch_size, num_patches, num_classes)
        probs = logits.sigmoid()
        predictions = (probs > self.detection_threshold).long()

        for i in range(batch_size):
            class_patch_pct = targets[i].sum(dim=0) / num_patches
            bins = torch.bucketize(class_patch_pct, self.buckets[1:-1])
            for j in range(self.num_buckets):
                self.cmats[j](predictions[i, :, bins == j].flatten(), targets[i, :, bins == j].flatten())

        return self.cmats
