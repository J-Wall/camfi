"""Defines transformations used for data augmentation during automatic annotation model
training. Depends on camfi.datamodel.autoannotation."""

from math import inf, sqrt
import random
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field
import torch

from camfi.datamodel.autoannotation import ImageTransform, Target


class Compose(ImageTransform):
    transforms: Sequence[ImageTransform]

    def apply_to_tensor_dict(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        for transform in self.transforms:
            image, target = transform.apply_to_tensor_dict(image, target)
        return image, target


class RandomHorizontalFlip(ImageTransform):
    prob: float = Field(..., ge=0.0, le=1.0)

    def apply_to_tensor_dict(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            try:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
            except KeyError:
                pass
            try:
                target["masks"] = target["masks"].flip(-1)
            except KeyError:
                pass

        return image, target
