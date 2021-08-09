"""Defines data structures used during automatic annotation model training and
inference. Depends on camfi.util, camfi.datamodel.geometry, and camfi.datamodel.via."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import inf
from pathlib import Path
from typing import Optional, Union

from numpy import array
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    root_validator,
    validator,
)
from skimage import draw
import torch
from torch.utils.data import Dataset

from camfi.util import dilate_idx, Field
from .geometry import (
    BoundingBox,
    PointShapeAttributes,
    CircleShapeAttributes,
    PolylineShapeAttributes,
)
from .via import ViaMetadata, ViaProject


class MaskMaker(BaseModel):
    """Defines methods to generate target instance segmentation masks for training a
    camfi annotation model.

    Parameters
    ----------
    shape : tuple[PositiveInt, PositiveInt]
        Shape of image (height, width). This will also be the shape of the instance
        segmentation masks.
    mask_dilate : Optional[PositiveInt] = None
        sets the amount of morphological dilation to apply to segmentation skeletons
        to produce instance segmentation masks.
    """

    shape: tuple[PositiveInt, PositiveInt] = Field(
        ..., description="Shape of images (height, width) in pixels."
    )
    mask_dilate: Optional[PositiveInt] = Field(
        None, description="Morphological dilation to apply to segmentation skeletons."
    )

    class Config:
        schema_extra = {
            "description": "Contains settings for segmentation mask generation."
        }

    def get_point_mask(self, point: PointShapeAttributes) -> torch.Tensor:
        """Produces a mask Tensor for a given point. Raises a ValueError if point lies
        outside self.shape.

        Parameters
        ----------
        point : PointShapeAttributes
            Geometry from which to generate instance segmentation mask.

        Returns
        -------
        mask : torch.Tensor
            Instance segmentation mask.
        """
        cx = array([int(point.cx)])
        cy = array([int(point.cy)])

        if cx >= self.shape[1] or cy >= self.shape[0]:
            raise ValueError(
                f"point ({cy}, {cx}) lies outside image with shape {self.shape}"
            )

        mask = torch.zeros(self.shape)
        if self.mask_dilate is None:
            rr, cc = cy, cx
        else:
            rr, cc = dilate_idx(cy, cx, self.mask_dilate, img_shape=self.shape)

        mask[rr, cc] = 1

        return mask

    def get_circle_mask(self, circle: CircleShapeAttributes) -> torch.Tensor:
        """Produces a mask Tensor for a given circle. Raises a ValueError if centre lies
        outside self.shape.

        Parameters
        ----------
        circle : CirlceShapeAttributes
            Geometry from which to generate instance segmentation mask.

        Returns
        -------
        mask : torch.Tensor
            Instance segmentation mask.
        """
        return self.get_point_mask(circle.as_point())

    def get_polyline_mask(self, polyline: PolylineShapeAttributes) -> torch.Tensor:
        """Produces a mask Tensor for a given polyline. Raises a ValueError if any point
        lies outside self.shape.

        Parameters
        ----------
        polyline : PolylineShapeAttributes
            Geometry from which to generate instance segmentation mask.

        Returns
        -------
        mask : torch.Tensor
            Instance segmentation mask.
        """
        x = polyline.all_points_x
        y = polyline.all_points_y
        mask = torch.zeros(self.shape)

        for i in range(len(x) - 1):
            if x[i] >= self.shape[1] or y[i] >= self.shape[0]:
                raise ValueError(
                    f"point ({x[i]}, {y[i]}) lies outside image with shape {self.shape}"
                )

            rr, cc = draw.line(int(y[i]), int(x[i]), int(y[i + 1]), int(x[i + 1]))
            if self.mask_dilate is not None:
                rr, cc = dilate_idx(rr, cc, self.mask_dilate, img_shape=self.shape)

            mask[rr, cc] = 1

        return mask

    def get_mask(
        self,
        shape_attributes: Union[
            PointShapeAttributes, CircleShapeAttributes, PolylineShapeAttributes
        ],
    ) -> torch.Tensor:
        """Produces a mask Tensor for a given geometry. Raises a ValueError if any point
        lies outside self.shape. Calls one of self.get_point_mask,
        self.get_circle_mask, or self.get_polyline_mask, depending on the type of
        shape_attributes.

        Parameters
        ----------
        shape_attributes : Union[
            PointShapeAttributes, CircleShapeAttributes, PolylineShapeAttributes
        ]
            Geometry from which to generate instance segmentation mask.

        Returns
        -------
        mask : torch.Tensor
            Instance segmentation mask.
        """
        if isinstance(shape_attributes, PointShapeAttributes):
            return self.get_point_mask(shape_attributes)
        elif isinstance(shape_attributes, CircleShapeAttributes):
            return self.get_circle_mask(shape_attributes)
        else:
            return self.get_polyline_mask(shape_attributes)

    def get_masks(self, metadata: ViaMetadata) -> list[torch.Tensor]:
        """Calls self.get_mask on all regions in metadata.

        Parameters
        ----------
        metadata : ViaMetadata
            Annotation data for an image.

        Returns
        -------
        masks : list[torch.Tensor]
            list of instance segmentation masks. One for each item in
            metadata.regions.
        """
        return [self.get_mask(region.shape_attributes) for region in metadata.regions]


class TargetPredictionABC(BaseModel, ABC):
    """Abstract base class for defining Targets (the input to the instance segmentation
    model training) and Predictions (the ouput of instance segmentation inference).

    Parameters
    ----------
    boxes : list[BoundingBox]
        list of bounding boxes of annotated objects in an image.
    labels : list[PositiveInt]
        list of integer class labels of annotated objects in an image. Zero is reserved
        for the background.
    masks : list[torch.Tensor]
        list of instance segmentation masks of annotated objects in an image.
    """

    boxes: list[BoundingBox]
    labels: list[PositiveInt]
    masks: list[torch.Tensor]

    class Config:
        """Pydantic configuration for TargetPredictionABC."""

        arbitrary_types_allowed = True

    @root_validator
    def all_fields_have_same_length(cls, values):
        """Pydantic validation method, called when instantiating subclasses of
        TargetPredictionABC. Ensures that all fields have the same length."""
        try:
            length = len(values["boxes"])
            if not all(len(values[k]) == length for k in ["labels", "masks"]):
                raise ValueError("Fields must have same length")
        except KeyError:
            raise ValueError("Invalid parameters given to Target")

        return values

    @validator("masks")
    def all_masks_same_shape(cls, v):
        """Pydantic validation method, called when instantiating subclasses of
        TargetPredictionABC. Ensures that all masks have the same shape."""
        if len(v) <= 1:
            return v

        shape = v[0].shape
        for mask in v[1:]:
            if mask.shape != shape:
                raise ValueError("All masks must have same shape")

        return v

    def __len__(self):
        return len(self.labels)

    @abstractmethod
    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Send data to a dict of Tensors.

        Returns
        -------
        tensor_dict : dict[str, torch.Tensor]
            Compatible with pytorch instance segmentation model.
        """

    @classmethod
    @abstractmethod
    def from_tensor_dict(
        cls, tensor_dict: dict[str, torch.Tensor]
    ) -> TargetPredictionABC:
        """Load Target or Prediction from tensor_dict.

        Parameters
        -------
        tensor_dict : dict[str, torch.Tensor]
            Compatible with pytorch instance segmentation model.

        Returns
        -------
        target_or_prediction : TargetPredictionABC
            Sub-classes of TargetPredictionABC should define this to return an instance
            of themselves.
        """


class Target(TargetPredictionABC):
    """Sub-class of TargetPredictionABC, defining the input to camfi model training.

    Parameters
    ----------
    boxes : list[BoundingBox]
        list of bounding boxes of annotated objects in an image.
    labels : list[PositiveInt]
        list of integer class labels of annotated objects in an image. Zero is reserved
        for the background.
    masks : list[torch.Tensor]
        list of instance segmentation masks of annotated objects in an image.
    image_id : NonNegativeInt
        Unique index of image to which Target instance relates.
    """

    image_id: NonNegativeInt

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Send data to a dict of Tensors.

        Returns
        -------
        tensor_dict : dict[str, torch.Tensor]
            Compatible with pytorch instance segmentation model.
        """
        return dict(
            boxes=torch.tensor([[b.x0, b.y0, b.x1, b.y1] for b in self.boxes]),
            labels=torch.tensor(self.labels),
            image_id=torch.tensor([self.image_id]),
            masks=torch.stack(self.masks) if len(self.masks) > 0 else torch.tensor([]),
        )

    @classmethod
    def from_tensor_dict(cls, tensor_dict: dict[str, torch.Tensor]) -> Target:
        """Load Target from tensor_dict.

        Parameters
        -------
        tensor_dict : dict[str, torch.Tensor]
            Compatible with pytorch instance segmentation model.

        Returns
        -------
        target : Target
            For use by camfi.
        """
        return Target(
            boxes=[
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                for x0, y0, x1, y1 in tensor_dict["boxes"]
            ],
            labels=[int(v) for v in tensor_dict["labels"]],
            masks=list(tensor_dict["masks"]),
            image_id=int(tensor_dict["image_id"]),
        )

    @classmethod
    def empty(cls, image_id: NonNegativeInt = 0) -> Target:
        """Initialises a Target with and image_id but no other data.

        Parameters
        ----------
        image_id : NonNegativeInt
            Unique image index to initialise Target with.

        Returns
        -------
        target : Target
            Target instance with no data (except image_id).

        Examples
        --------
        >>> Target.empty()
        Target(boxes=[], labels=[], masks=[], image_id=0)
        """
        return Target.construct(boxes=[], labels=[], image_id=image_id, masks=[])


class Prediction(TargetPredictionABC):
    """Sub-class of TargetPredictionABC, defining the output of camfi model inference.

    Parameters
    ----------
    boxes : list[BoundingBox]
        list of bounding boxes of annotated objects in an image.
    labels : list[PositiveInt]
        list of integer class labels of annotated objects in an image. Zero is reserved
        for the background.
    masks : list[torch.Tensor]
        list of instance segmentation masks of annotated objects in an image.
    scores : list[NonNegativeFloat]
        list of annotation scores (from 0.0 to 1.0)
    """

    scores: list[NonNegativeFloat]

    @root_validator
    def all_fields_have_same_length(cls, values):
        """Pydantic validation method, called when instantiating Prediction.
        Ensures that all fields have the same length."""
        length = len(values["boxes"])
        if not all(len(values[k]) == length for k in ["labels", "masks", "scores"]):
            raise ValueError("Fields must have same length")

        return values

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Send data to a dict of Tensors.

        Returns
        -------
        tensor_dict : dict[str, torch.Tensor]
            Compatible with pytorch instance segmentation model.
        """
        return dict(
            boxes=torch.tensor([[b.x0, b.y0, b.x1, b.y1] for b in self.boxes]),
            labels=torch.tensor(self.labels),
            scores=torch.tensor(self.scores),
            masks=torch.stack(self.masks) if len(self.masks) > 0 else torch.tensor([]),
        )

    @classmethod
    def from_tensor_dict(cls, tensor_dict: dict[str, torch.Tensor]) -> Prediction:
        """Load Prediction from tensor_dict.

        Parameters
        -------
        tensor_dict : dict[str, torch.Tensor]
            Compatible with pytorch instance segmentation model.

        Returns
        -------
        prediction : Prediction
            For use by camfi.
        """
        return Prediction(
            boxes=[
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                for x0, y0, x1, y1 in tensor_dict["boxes"]
            ],
            labels=[int(v) for v in tensor_dict["labels"]],
            masks=list(tensor_dict["masks"]),
            scores=[int(v) for v in tensor_dict["scores"]],
        )

    @classmethod
    def empty(cls) -> Prediction:
        """Initialises an Prediction with no data.

        Returns
        -------
        prediction : Prediction
            Prediction instance with no data.

        Examples
        --------
        >>> Prediction.empty()
        Prediction(boxes=[], labels=[], masks=[], scores=[])
        """
        return Prediction.construct(boxes=[], labels=[], masks=[], scores=[])

    def filter_by_score(self, score_thresh: float) -> Prediction:
        """Returns a Prediction instance with items with scores below score_thresh
        removed.

        Parameters
        ----------
        score_thresh: float
            Score threshold. Annotations with score below score_thresh are not
            included in the returned Prediction object.

        Returns
        -------
        prediction : Prediction
            Prediction object with below-score-threshold annotations removed.
        """
        filtered_prediction = Prediction.empty()
        for i in range(len(self)):
            if self.scores[i] >= score_thresh:
                filtered_prediction.boxes.append(self.boxes[i])
                filtered_prediction.labels.append(self.labels[i])
                filtered_prediction.masks.append(self.masks[i])
                filtered_prediction.scores.append(self.scores[i])

        return filtered_prediction

    def get_subset_from_index(self, subset: list[NonNegativeInt]) -> Prediction:
        """Returns a Prediction instance from self with items indexed by the elements
        of subset.

        Parameters
        ----------
        subset : list[NonNegativeInt]
            list of indices to include in subset.

        Returns
        -------
        prediction : Prediction
            Prediction object only including specified subset of annotations.

        Examples
        --------
        >>> prediction = Prediction(
        ...     boxes=[
        ...         BoundingBox(x0=0, y0=0, x1=1, y1=1),
        ...         BoundingBox(x0=1, y0=1, x1=2, y1=2),
        ...     ],
        ...     labels=[1, 2],
        ...     masks=[torch.zeros(2, 2), torch.zeros(2, 2)],
        ...     scores=[0.0, 1.0],
        ... )
        >>> subset_prediction = prediction.get_subset_from_index([0])
        >>> subset_prediction.boxes == prediction.boxes[0:1]
        True
        >>> subset_prediction.labels == prediction.labels[0:1]
        True
        >>> len(subset_prediction.masks) == 1
        True
        >>> subset_prediction.scores == prediction.scores[0:1]
        True
        """
        filtered_prediction = Prediction.empty()
        for i in subset:
            filtered_prediction.boxes.append(self.boxes[i])
            filtered_prediction.labels.append(self.labels[i])
            filtered_prediction.masks.append(self.masks[i])
            filtered_prediction.scores.append(self.scores[i])

        return filtered_prediction


class ImageTransform(BaseModel, ABC):
    """Abstract base class for transforms on images with segmentation data.
    Instances of subclasses of ImageTransform are callables which when called, apply
    a transformation to an image Tensor and an accosiated target_dict.
    """

    def __call__(
        self, image: torch.Tensor, target: Target
    ) -> tuple[torch.Tensor, Target]:
        image, target_dict = self.apply_to_tensor_dict(image, target.to_tensor_dict())
        return image, Target.from_tensor_dict(target_dict)

    @abstractmethod
    def apply_to_tensor_dict(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Subclasses of ImageTransform should implement this to return a transformed
        image and target dict.

        Parameters
        ----------
        image : torch.Tensor
            Image tensor on which annotations are made.
        target : dict[str, torch.Tensor]
            Specifies annotations on image. Output of Target.to_tensor_dict().

        Returns
        -------
        transformed_image : torch.Tensor
            Transformed image.
        transformed_target : dict[str, torch.Tensor]
            Specifies transformed annotations on image. Can be used as input to
            Target.from_tensor_dict().
        """


class CamfiDataset(BaseModel, Dataset):
    """Defines a camfi dataset, compatible with torch.utils.data.DataLoader, for
    loading image (and annotation) data for camfi model training and inference.

    When indexed with an int, a CamfiDataset object returns an image Tensor and a Target
    object, which contains the manual annotations for that image. If using in inference
    mode, the Target will not contain any data (except the image index).

    Parameters
    ----------
    root: Path
        Path to directory containing images (in subdirectories). Filenames specified in
        via_project should be relative paths from this root directory.
    via_project: ViaProject
        Via project data, containing all annotation data, and all images to be trained
        on (or for inference to run on).
    crop: Optional[BoundingBox]
        If specified, images will be cropped to this bounding box after loading them
        from file. This can be useful when working with images taken by trail cameras,
        as some models burn a information bar onto the bottom of the image, which
        ideally should be removed before training or inference on that image.
    inference_mode: bool
        Flag to indicate whether dataset is being used for inference or training. In
        training mode, data augmentation etc. will be performed on the dataset.
    mask_maker: Optional[MaskMaker]
        Instance of MaskMaker which produces instance segementation masks for model
        training. Only set if inference_mode = False.
    transform: Optional[ImageTransform]
        Image transformations to apply for data augmentation. Only set
        if inference_mode = False.
    min_annotations: int
        Only train on images which have at least this many annotations.
        Only set if inference_mode = False.
    max_annotations: float = inf
        Only train on images which have at most this many annotations. Can be useful
        if running into memory errors on the GPU, since memory consumption depends on
        the number of annotated objects in the image.
        Only set if inference_mode = False.
    box_margin: PositiveInt = 10
        Margin to add to bounding boxes of object annotations, for model training.
        Only set if inference_mode = False.
    exclude: set[Path]
        Optionally specify a set of image files to exclude for training or inference.
        Elements in exclude must match the filenames (ie. relative file paths) of
        images to exclude.
    keys: list[str]
        list of image keys. This field is automatically generated upon instantiation,
        and should not be set.
    """

    root: Path
    via_project: ViaProject
    crop: Optional[BoundingBox] = None

    inference_mode: bool = False

    # Only set if inference_mode = False
    mask_maker: Optional[MaskMaker] = None
    transform: Optional[ImageTransform] = None
    min_annotations: Optional[int] = None
    max_annotations: Optional[int] = None
    box_margin: PositiveInt = 10

    # Optionally exclude some files
    exclude: set[Path] = None  # type: ignore[assignment]

    # Automatically generated. No need to set.
    keys: list[str] = None  # type: ignore[assignment]

    @validator("exclude", pre=True, always=True)
    def default_exclude(cls, v):
        """Pydantic validation method, called when instantiating CamfiDataset.
        sets default for exclude."""
        if v is None:
            return set()
        return v

    @validator("transform")
    def only_set_transform_if_not_inference_mode(cls, v, values):
        """Pydantic validation method, called when instantiating CamfiDataset.
        Ensures that transform is only set if in training mode."""
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is None, "Only set if inference_mode=False"
        return v

    @validator("min_annotations")
    def only_set_min_annotations_if_not_inference_mode(cls, v, values):
        """Pydantic validation method, called when instantiating CamfiDataset.
        Ensures that min_annotations is only set if in training mode."""
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is None, "Only set if inference_mode=False"
        return v

    @validator("max_annotations")
    def only_set_max_annotations_if_not_inference_mode(cls, v, values):
        """Pydantic validation method, called when instantiating CamfiDataset.
        Ensures that max_annotations is only set if in training mode."""
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is None, "Only set if inference_mode=False"
        return v

    @validator("box_margin")
    def only_set_box_margin_if_not_inference_mode(cls, v, values):
        """Pydantic validation method, called when instantiating CamfiDataset.
        Ensures that box_margin is only set if in training mode."""
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v == 10, "Only set if inference_mode=False"
        return v

    @validator("mask_maker", always=True)
    def set_iff_not_inference_mode(cls, v, values):
        """Pydantic validation method, called when instantiating CamfiDataset.
        Ensures that mask_maker is set iff in training mode."""
        if "inference_mode" in values and values["inference_mode"] is True:
            assert v is None, "Only set if inference_mode=False"
        else:
            assert isinstance(
                v, MaskMaker
            ), "mask_maker must be set if inference_mode=False"
        return v

    @validator("keys", pre=True, always=True)
    def generate_filtered_keys(cls, v, values):
        """Pydantic validation method, called when instantiating CamfiDataset.
        Generates list of image keys."""
        min_annotations = values.get("min_annotations", 0)
        if min_annotations is None:
            min_annotations = 0
        max_annotations = values.get("max_annotations", inf)
        if max_annotations is None:
            max_annotations = inf
        return list(
            dict(
                filter(
                    lambda e: e[1].filename not in values["exclude"]
                    and min_annotations <= len(e[1].regions) <= max_annotations,
                    values["via_project"].via_img_metadata.items(),
                )
            )
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Target]:
        metadata = self.via_project.via_img_metadata[self.keys[idx]]
        image = metadata.read_image(root=self.root)

        if self.crop is not None:
            image = self.crop.crop_image(image)

        if self.inference_mode:
            target = Target.empty(image_id=idx)
        else:  # Training mode
            boxes = metadata.get_bounding_boxes()
            for box in boxes:
                box.add_margin(self.box_margin)
            target = Target(
                boxes=boxes,
                labels=metadata.get_labels(),
                image_id=idx,
                masks=self.mask_maker.get_masks(metadata),  # type: ignore[union-attr]
            )

            if self.transform is not None:
                image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.keys)

    def metadata(self, idx: int) -> ViaMetadata:
        """Returns the ViaMetadata object of the image at idx.

        Parameters
        ----------
        idx : int
            Index of image in dataset.

        Returns
        -------
        metadata : ViaMetadata
            Annotation data for the image.
        """
        return self.via_project.via_img_metadata[self.keys[idx]]
