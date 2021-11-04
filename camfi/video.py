"""Provides camfi with functionality to work with short video clips.
"""

import itertools
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, PositiveInt
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor
from torchvision.io import read_video

from camfi import annotator
from camfi.datamodel.autoannotation import CamfiDataset, Prediction
from camfi.datamodel.via import ViaRegion


class FrameAnnotator(annotator.Annotator):
    """Subclasses annotator.Annotator to remove reliance on a still image dataset
    (i.e. does not require a via project, and can operate on individual frames).

    **Note:** The implementation of FrameAnnotator is pretty heavily dependent on
    dynamic typing, which feels a bit hacky, but reduces the amount of repeated code
    here and/or refactoring of ``annotator``. Hence the code is littered with
    ``# type: ignore[...]`` comments. Technical debt ftw?
    """

    dataset: Optional[CamfiDataset] = None  # type: ignore[assignment]

    def get_prediction(self, img: Tensor) -> Prediction:  # type: ignore[override]
        """Gets instance segmentation prediction from already loaded image.

        Parameters
        ----------
        img : Tensor
            Image frame to perform prediction on. Shape should be (3, height, width).

        Returns
        -------
        prediction : Prediction
        """
        with torch.no_grad():
            try:
                prediction = self.model([img.to(self.device)])[0]
            except RuntimeError:
                if self.backup_model:
                    prediction = self.backup_model([img.to(self.backup_device)])[0]
                    self.backup_model_used += 1
                else:
                    raise

        return Prediction.from_tensor_dict(prediction)

    def annotate_img(self, img: Tensor) -> list[ViaRegion]:  # type: ignore[override]
        """See `camfi.annotator.Annotator.annotate_img`. This is the equivalent method
        but operates on image tensor instead of index.

        Parameters
        ----------
        img : Tensor
            Image frame to perform prediction on. Shape should be (3, height, width).

        Returns
        -------
        regions : list[ViaRegion]
            list of annotations for image.
        """
        return super().annotate_img(img)  # type: ignore[arg-type]


def all_channels_equal(frames: np.ndarray) -> bool:
    """Returns true if all channels (colours) are equal for every pixel."""
    return all(
        (frames[..., i] == frames[..., i + 1]).all()
        for i in range(frames.shape[-1] - 1)
    )


def temporal_filter(
    frames: np.ndarray,
    width: int = 2,
    use_max: Callable[[np.ndarray], bool] = all_channels_equal,
) -> np.ndarray:
    """Merges consecutive frames using a 1D minimum or maximum filter across the frames,
    depending on the output of ``use_max``.

    Parameters
    ----------
    frames : np.ndarray
        With shape [n_frames, height, width, channels]
    width : int
        Number of frames to include in maximum/minimum filter for each step.
    use_max : Callable[[np.ndarray], bool]
        Function which determines which method to use.
    """
    filter_fn = maximum_filter1d if use_max(frames) else minimum_filter1d
    return filter_fn(frames, width, axis=0, mode="nearest")


class RegionStringMember(BaseModel):

    region: ViaRegion
    frame_index: int
    colour: int = -1


class VideoAnnotator(BaseModel):
    """Implements video clip annotation procedure."""

    frame_annotator: FrameAnnotator
    temporal_filter_width: PositiveInt = 2
    use_max: Callable[[np.ndarray], bool] = all_channels_equal
    max_matching_dist: float = np.inf
    metadata: Optional[dict] = None

    def prep_video(self, filepath: Path) -> np.ndarray:
        v, _, self.metadata = read_video(filepath)
        v = temporal_filter(
            v.numpy(), width=self.temporal_filter_width, use_max=self.use_max
        )
        v = np.moveaxis(v, -1, 1).astype("f4") / 255  # [frame, channel, height, width]
        return torch.tensor(v)

    def annotate_frames(self, video: Tensor) -> list[list[ViaRegion]]:
        regions = []
        for i in range(video.shape[0] - 1):
            r = self.frame_annotator.annotate_img(video[i])
            regions.append(r)

        return regions

    def matching_distances(self, regions: list[list[ViaRegion]]) -> list[np.ndarray]:
        distances = []
        for regions0, regions1 in zip(regions, regions[1:]):
            ds = np.empty((len(regions0), len(regions1)))
            for (i, r0), (j, r1) in itertools.product(
                enumerate(regions0), enumerate(regions1)
            ):
                ds[i, j] = r0.matching_distance(r1)

            distances.append(ds)

        return distances

    def colour_regions(
        self, regions: list[list[ViaRegion]], distances: list[np.ndarray]
    ) -> dict[int, list[RegionStringMember]]:
        # Encapsualate ViaRegion instances in RegionStringMember instances
        region_string_members = []
        for i, region_list in enumerate(regions):
            r = []
            for region in filter(x: x.shape_attributes.name == "polyline", region_list):
                r.append(RegionStringMember(region=region, frame_index=i))
            region_string_members.append(r)

        # Colour them
        colour = 0
        out: dict[int, list[RegionStringMember]] = {}
        # Initialise colours if there are regions in the first frame
        for region in region_string_members[0]:
            if region.colour == -1:
                region.colour = colour
                out[region.colour] = [region]
                colour += 1

        # Iteratively colour regions in each frame
        for ds, regions0, regions1 in zip(
            distances, region_string_members, region_string_members[1:]
        ):
            # Propagate colours
            rows, cols = linear_sum_assignment(ds)
            for r, c in zip(rows, cols):
                if ds[r, c] <= self.max_matching_dist:
                    regions1[c].colour = regions0[r].colour
                    out[regions0[r].colour].append(regions1[c])

            # Initalise new strings
            for region in regions1:
                if region.colour == -1:
                    region.colour = colour
                    out[region.colour] = [region]
                    colour += 1

        return out
