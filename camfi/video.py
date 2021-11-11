"""Provides camfi with functionality to work with short video clips.
"""

from argparse import ArgumentParser
import itertools
from pathlib import Path
from typing import Callable, Optional, Union

import imageio
import numpy as np
from pydantic import BaseModel, PositiveInt, NonNegativeFloat
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import linear_sum_assignment
from strictyaml import load
import torch
from torch import Tensor

from camfi import annotator
from camfi.projectconfig import InferenceConfig
from camfi.datamodel.autoannotation import CamfiDataset, Prediction
from camfi.datamodel.via import ViaRegion
from camfi.datamodel.geometry import CircleShapeAttributes, PolylineShapeAttributes
from camfi.util import endpoint_methods


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

    def convert_to_circle(
        self,
        polyline: PolylineShapeAttributes,
        img_shape: tuple[PositiveInt, PositiveInt],
    ) -> Union[PolylineShapeAttributes, CircleShapeAttributes]:
        return polyline

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


def return_true(frames: np.ndarray) -> bool:
    return True


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

    @property
    def all_points_x(self) -> list[NonNegativeFloat]:
        if self.region.shape_attributes.name == "polyline":
            return self.region.shape_attributes.all_points_x
        else:
            return [self.region.shape_attributes.cx]

    @property
    def all_points_y(self) -> list[NonNegativeFloat]:
        if self.region.shape_attributes.name == "polyline":
            return self.region.shape_attributes.all_points_y
        else:
            return [self.region.shape_attributes.cy]


def get_all_points_x(regions: list[RegionStringMember]) -> list[NonNegativeFloat]:
    return [x for r in regions for x in r.all_points_x]


def get_all_points_y(regions: list[RegionStringMember]) -> list[NonNegativeFloat]:
    return [y for r in regions for y in r.all_points_y]


class RegionString(BaseModel):
    regions: list[RegionStringMember]


class ColouredRegions(BaseModel):
    region_strings: dict[int, RegionString]
    video_file: Optional[str] = None


class VideoAnnotator(BaseModel):
    """Implements video clip annotation procedure."""

    frame_annotator: FrameAnnotator
    temporal_filter_width: PositiveInt = 2
    use_max: Callable[[np.ndarray], bool] = all_channels_equal
    max_matching_dist: float = np.inf
    min_string_length: PositiveInt = 3
    metadata: Optional[dict] = None

    def prep_video(self, filepath: Path) -> torch.Tensor:
        reader = imageio.get_reader(filepath)
        self.metadata = reader.get_meta_data()
        v = np.stack([img for img in reader])

        v = temporal_filter(v, width=self.temporal_filter_width, use_max=self.use_max)
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
            for region in region_list:
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
            # Mask out above-threshold distances
            ds = ds.copy()
            ds[ds > self.max_matching_dist] = self.max_matching_dist + 1.0

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

    def filter_region_strings(
        self, coloured_regions: dict[int, list[RegionStringMember]]
    ) -> dict[int, list[RegionStringMember]]:
        return {
            k: v
            for (k, v) in coloured_regions.items()
            if len(v) >= self.min_string_length
        }


def reorient_regions(regions: list[RegionStringMember]) -> None:
    """Takes a list of RegionStringMembers, and reorients them so that they are all
    pointing forwards in time. Operates inplace.

    Parameters
    ----------
    regions : list[RegionStringMember]
        List of regions to reorient. It is assumed that they are in chronological order.
    """
    centres = [r.region.shape_attributes.centre_of_mass() for r in regions]

    for region, source, target in zip(
        regions, [None] + centres[:-1], centres[1:] + [None]
    ):
        if region.region.shape_attributes.name == "polyline":
            region.region.shape_attributes.reorient(source=source, target=target)


def parse_args():
    parser = ArgumentParser(
        prog="camfi-video",
        description=None,
        epilog=None,
    )
    parser.add_argument(
        "-c",
        "--annotator-config",
        required=True,
        help=("" ""),
    )
    parser.add_argument(
        "-t",
        "--temporal-filter-width",
        type=int,
        default=2,
        help=("" ""),
    )
    parser.add_argument(
        "-m",
        "--max-filter-always",
        action="store_true",
        help=("" ""),
    )
    parser.add_argument(
        "-d",
        "--max-matching-dist",
        type=float,
        default=np.inf,
        help=("" ""),
    )
    parser.add_argument(
        "-l",
        "--min-string-length",
        type=int,
        default=3,
        help=("" ""),
    )
    parser.add_argument(
        "-i",
        "--infile",
        nargs="+",
        help=("" ""),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.annotator_config, "r") as f:
        annotator_config = InferenceConfig.parse_obj(load(f.read()).data)

    frame_annotator = FrameAnnotator(
        model=annotator_config.model,
        device=annotator_config.device,
        backup_device=annotator_config.backup_device,
        split_angle=annotator_config.split_angle,
        poly_order=annotator_config.poly_order,
        endpoint_method=endpoint_methods[annotator_config.endpoint_method],
        endpoint_extra_args=annotator_config.endpoint_extra_args,
        score_thresh=annotator_config.score_thresh,
        overlap_thresh=annotator_config.overlap_thresh,
        edge_thresh=annotator_config.edge_thresh,
    )

    video_annotator = VideoAnnotator(
        frame_annotator=frame_annotator,
        temporal_filter_width=args.temporal_filter_width,
        use_max=return_true if args.max_filter_always else all_channels_equal,
        max_matching_dist=args.max_matching_dist,
        min_string_length=args.min_string_length,
    )

    for infile in args.infile:
        print(f"Processing {infile}")

        video = video_annotator.prep_video(args.infile)

        regions = video_annotator.annotate_frames(video)
        distances = video_annotator.matching_distances(regions)

        coloured_regions = video_annotator.colour_regions(regions, distances)
        for region_string in coloured_regions.values():
            if len(region_string) > 1:
                reorient_regions(region_string)

        filtered_regions = video_annotator.filter_region_strings(coloured_regions)

        out_str = ColouredRegions(
            region_strings={
                k: RegionString(regions=v) for (k, v) in filtered_regions.items()
            },
            video_file=args.infile,
        ).json(
            exclude_none=True,
        )
        with open(infile + ".annotated.json", "w") as f:
            f.write(out_str)


if __name__ == "__main__":
    main()
