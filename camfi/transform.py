"""Defines transformations used for data augmentation during automatic annotation model
training. Depends on camfi.datamodel.autoannotation."""

from math import inf, sqrt
import random
from typing import Iterable, Optional, Sequence, Union

from pydantic import BaseModel
import torch

from camfi.datamodel.autoannotation import ImageTransform, Target
from camfi.util import Field


class Compose(ImageTransform):
    """Composition of transformations.

    Parameters
    ----------
    transforms: Sequence[ImageTransform]
        Sequence of transorms to compose.
    """

    transforms: Sequence[ImageTransform]

    def apply_to_tensor_dict(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Applies each transormation in self.transforms to an image and target dict.

        Parameters
        ----------
        image : torch.Tensor
            Input image to first transformation.
        target : dict[str, torch.Tensor]
            Target tensor dict of annotations on image.

        Returns
        -------
        transformed_image : torch.Tensor
            Output image of last transformation.
        transformed_target : dict[str, torch.Tensor]
            Output target dict of last transformation.
        """
        for transform in self.transforms:
            image, target = transform.apply_to_tensor_dict(image, target)
        return image, target


class RandomHorizontalFlip(ImageTransform):
    """Image transform which applies a horizontal flip to an image with a fixed
    probability.

    Parameters
    ----------
    prob : float
        Probability of flipping image.
    """

    prob: float = Field(..., ge=0.0, le=1.0)

    def apply_to_tensor_dict(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Applies each random horizontal flip transormation to an image and target
        dict.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor.
        target : dict[str, torch.Tensor]
            Input target tensor dict.

        Returns
        -------
        transformed_image : torch.Tensor
            Output image of transformation.
        transformed_target : dict[str, torch.Tensor]
            Output target dict of transformation.
        """
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
