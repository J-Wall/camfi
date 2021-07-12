from math import inf, sqrt
import random
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from torch import Tensor

from camfi.data import ImageTransform, Target


class Compose(ImageTransform):
    transforms: Sequence[ImageTransform]

    def apply_to_tensor_dict(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        for transform in self.transforms:
            image, target = transform.apply_to_tensor_dict(image, target)
        return image, target


class RandomHorizontalFlip(ImageTransform):
    prob: float = Field(..., ge=0.0, le=1.0)

    def apply_to_tensor_dict(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
