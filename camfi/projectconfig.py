"""Provides camfi project config parser.
"""

from __future__ import annotations

from datetime import date, timezone
from functools import cached_property
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence

import pandas as pd
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    root_validator,
    validator,
)
from strictyaml import as_document, load

from camfi.annotator import (
    Annotator,
    AnnotationValidationResult,
    train_model,
    validate_annotations,
)
from camfi.datamodel.autoannotation import CamfiDataset, MaskMaker
from camfi.datamodel.geometry import BoundingBox
from camfi.datamodel.locationtime import LocationTimeCollector
from camfi.datamodel.via import ViaProject
from camfi.datamodel.weather import LocationWeatherStationCollector
from camfi.models import model_urls
from camfi.util import Timezone, endpoint_methods
from camfi.transform import RandomHorizontalFlip
from camfi.wingbeat import WingbeatExtractorConfig, extract_all_wingbeats


class ParameterUnspecifiedError(Exception):
    """Base exception called when a parameter which needs to be specified is not."""


class ViaProjectUnspecifiedError(ParameterUnspecifiedError):
    """Raised by CamfiConfig.load_via_project."""


class PlaceUnspecifiedError(ParameterUnspecifiedError):
    """Raised when CamfiConfig.place is requested, but it is unspecified."""


class WingbeatExtractorConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised when a WingbeatExtractorConfig is needed, but was not supplied."""


class CameraConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised when a method requies a camera config but none was supplied."""


class AnnotatorConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised if AnnotatorConfig is required but is not specified."""


class TrainingConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised if TrainingConfig is required but is not specified."""


class MaskMakerUnspecifiedError(ParameterUnspecifiedError):
    """Raised if MaskMaker is required but is not specified."""


class InferenceConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised if InferenceConfig is required but is not specified."""


class ValidationConfigUnspecifiedError(ParameterUnspecifiedError):
    """Raised if ValidationConfig is required but is not specified."""


class CameraConfig(BaseModel):
    """Camera hardware-related configuration."""

    camera_time_to_actual_time_ratio: Optional[float] = Field(
        None, description="Used for correcting timestamps from an inaccurate clock."
    )
    line_rate: Optional[PositiveFloat] = Field(
        None, description="Rolling-shutter line rate of camera (lines per second)."
    )


class TrainingConfig(BaseModel):
    """Contains settings for camfi annotator model training.

    Parameters
    ----------
    mask_maker : Optional[MaskMaker]
        Instance of MaskMaker which produces instance segementation masks for model
        training.
    min_annotations: int
        Only train on images which have at least this many annotations.
    max_annotations : float
        Only train on images which have at most this many annotations. Can be useful
        if running into memory errors on the GPU, since memory consumption depends on
        the number of annotated objects in the image.
    box_margin : PositiveInt
        Margin to add to bounding boxes of object annotations, for model training.
    test_set_file : Optional[FilePath]
        Alternative to setting test_set directly, load it from a file.
    test_set : List[Path]
        List of images to exclude from training.
    load_pretrained_model : Optional[Union[Path, str]]
        Path or url to model parameters file. If set, will load the pretrained
        parameters. By default, will start with a model pre-trained on the Microsoft
        COCO dataset.
    device : str
        E.g. "cpu" or "cuda". Training is typically much faster on a GPU. Use "cuda" for
        Nvidia GPUs.
    batch_size : int
        Number of images to load at once.
    num_workers : int
        Number of worker processes for data loader to spawn.
    num_epochs : int
        Number of epochs to train.
    outdir : DirectoryPath
        Path to directory where to save model(s).
    model_name : Optional[str]
        Identifier to include in model save file. By default the current date in
        YYYYmmdd format.
    save_intermediate : bool
        If True, model is saved after each epoch, not just after all epoch are complete.
        This is recommended, especially if training on a service which could terminate
        unpredicatbly.
    """

    mask_maker: Optional[MaskMaker] = None
    min_annotations: Optional[int] = Field(
        None,
        description="Only train on images which have at least this many annotations.",
    )
    max_annotations: Optional[int] = Field(
        None,
        description="Only train on images which have at most this many annotations.",
    )
    box_margin: PositiveInt = Field(
        10, description="Margin to add to bounding boxes of object annotations."
    )
    test_set_file: Optional[FilePath] = Field(
        None,
        description="Contains filepaths (one per line). Do not set if test_set is set.",
    )
    test_set: List[Path] = Field(
        [],
        description="List of images to exclude from training.",
    )
    load_pretrained_model: Optional[Union[Path, str]] = Field(
        None, description="Path or url to model parameters file to initialise training."
    )
    device: str = Field(
        "cpu", description="Device to run training on (e.g. cuda or cpu)."
    )
    batch_size: int = Field(5, description="Number of images to load at once.")
    num_workers: int = Field(
        2, description="Number of worker processes for data loader to spawn."
    )
    num_epochs: int = Field(10, description="Number of epochs to train.")
    outdir: DirectoryPath = Field(
        Path(), description="Path to directory where to save model(s)."
    )
    model_name: Optional[str] = Field(
        None, description="Identifier to include in model save file. Default YYYYmmdd."
    )
    save_intermediate: bool = Field(
        False, description="If True, model is saved after each epoch."
    )

    class Config:
        schema_extra = {
            "description": "Contains settings for camfi annotator model training."
        }

    @validator("test_set", always=True)
    def load_test_set(cls, v, values):
        test_set_file = values.get("test_set_file", None)
        if test_set_file is not None:
            if len(v) > 0:
                raise ValueError("test_set should not be set if test_set_file is set.")
            with open(test_set_file, "r") as f:
                v = [Path(l.strip()) for l in f]
        return v


class InferenceConfig(BaseModel):
    """Contains settings for camfi annotator inference.

    Parameters
    ----------
    output_path : Optional[Path]
        If set, automatically generated annotations will be saved to file.
    model : Union[Path, str]
        Path to .pth file specifying model parameters, model name defined in
        camfi.models.model_urls, or url to model to download from the internet.
    device : str
        Specifies device to run inference on. E.g. set to "cuda" to use an Nvidia GPU.
    backup_device : Optional[str]
        Specifies device to run inference on when a runtime error occurs while using
        device. Probably only makes sense to set this to "cpu" if device="cuda". This
        option enables the annotator to leverage a GPU with limited memory capacity
        without crashing if a difficult image is encountered.
    split_angle : PositiveFloat
        Approximate maximum angle between polyline segments in degrees. Note that this
        will immediately be converted to radians upon instantiation of Annotator.
    poly_order : PositiveInt
        Order of polynomial used for fitting motion blur paths.
    endpoint_method : str
        Method to find endpoints of motion blurs. One of camfi.utils.endpoint_methods.
    endpoint_extra_args : List[Any]
        Extra arguments to pass to endpoint method function.
    score_thresh : float
        Score threshold between 0.0 and 1.0 for automatic annotations to be kept.
    overlap_thresh : float
        Minimum proportion of overlap (weighted intersection over minimum) between two
        instance segmentation masks to infer that one of the masks should be discarded.
    edge_thresh : NonNegativeInt
        Minimum distance an annotation has to be from the edge of the image before it is
        converted from a polyline annotation to a circle annotation.
    """

    output_path: Optional[Path] = Field(
        None,
        description="If set, automatically generated annotations will be saved to file.",
    )
    model: Union[Path, str] = Field(
        "release",
        description="Path or url to model .pth file or one of camfi.models.model_urls.",
        examples=list(model_urls),
    )
    device: str = Field(
        "cpu",
        description="Specifies device to run inference on.",
        examples=["cpu", "cuda"],
    )
    backup_device: Optional[str] = Field(
        None,
        description="Used for images which fail on device due to memory constraints.",
    )
    split_angle: PositiveFloat = Field(
        15.0,
        description="Approximate maximum angle between polyline segments in degrees.",
    )
    poly_order: PositiveInt = Field(
        2, description="Order of polynomial used for fitting motion blur paths."
    )
    endpoint_method: str = Field(
        "truncate",
        description="Method to find endpoints of motion blurs.",
        examples=list(endpoint_methods),
    )
    endpoint_extra_args: List[Any] = Field(
        [10], description="Extra arguments to pass to endpoint method function."
    )
    score_thresh: float = Field(
        0.4,
        description="Score threshold between 0.0 and 1.0 for automatic annotations to be kept.",
    )
    overlap_thresh: float = Field(
        0.4,
        description="Minimum weighted IoM for non-maximum suppression of detections.",
    )
    edge_thresh: NonNegativeInt = Field(
        20,
        description="Polyline annotations this close to edge of image are converted to circles.",
    )

    class Config:
        schema_extra = {
            "description": "Contains settings for camfi annotator inference."
        }

    @validator("endpoint_method")
    def check_endpoint_method(cls, v):
        assert (
            v in endpoint_methods
        ), "endpoint_method must be one of {list(endpoint_methods)}. Got {v}."


class ValidationConfig(BaseModel):
    """Contains settings for camfi annotator validation.

    Parameters
    ----------
    autoannotated_via_project_file : Optional[Path]
        Path to file containing VIA Project with annotations to validate.
    iou_thresh : float
        Threshold of intersection-over-union to match annotations.
    """

    autoannotated_via_project_file: Optional[Path] = Field(
        None,
        description="Path to file containing VIA Project with annotations to validate.",
    )
    iou_thresh: float = Field(
        0.5, description="Threshold of intersection-over-union to match annotations."
    )
    image_sets: List[str] = Field(
        ["all"],
        description="Image sets to perform training on.",
        regex="^(all|test|train)$",
    )
    output_dir: Optional[DirectoryPath] = Field(
        None,
        description="If set, results of validation will be saved.",
    )
    output_stem: str = Field("validation", description="Stem of output files.")

    class Config:
        schema_extra = {
            "description": "Contains settings for camfi annotator validation."
        }


class AnnotatorConfig(BaseModel):
    """Contains settings for automatic annotation training and inference.

    Parameters
    ----------
    crop : Optional[BoundingBox]
        If specified, images will be cropped to this bounding box after loading them
        from file. This can be useful when working with images taken by trail cameras,
        as some models burn a information bar onto the bottom of the image, which
        ideally should be removed before training or inference on that image.
    training : Optional[TrainingConfig]
        Contains settings for camfi annotator model training.
    inference : Optional[InferenceConfig]
        Contains settings for camfi annotator inference.
    """

    crop: Optional[BoundingBox] = None
    training: Optional[TrainingConfig] = None
    inference: Optional[InferenceConfig] = None
    validation: Optional[ValidationConfig] = None

    class Config:
        schema_extra = {
            "description": "Settings for automatic annotation training and inference."
        }


class CamfiConfig(BaseModel):
    """Defines structure of Camfi's config.json files, and provides methods for loading
    data from various sources.
    """

    root: DirectoryPath = Field(
        Path(), description="Directory containing all images for the project."
    )
    disable_progress_bar: Optional[bool] = Field(
        None, description="Disables progress bars. By default, disable on non-TTY."
    )
    via_project_file: Optional[FilePath] = Field(
        None, description="Path to file containing VIA project."
    )
    day_zero: Optional[date] = Field(
        None, description="Used as reference date for plotting etc."
    )
    output_tz: Timezone = Field(
        description="Sets a global timezone used for various analyses.",
        default_factory=lambda: Timezone("Z"),
    )
    default_output: Optional[Path] = Field(
        None,
        description=(
            "Path to write output. Outputs defined in "
            "wingbeat_extractor and annotator "
            "take precedence over this."
        ),
    )
    camera: Optional[CameraConfig] = None
    time: Optional[LocationTimeCollector] = None
    place: Optional[LocationWeatherStationCollector] = None
    wingbeat_extraction: Optional[WingbeatExtractorConfig] = None
    annotator: Optional[AnnotatorConfig] = None

    @property
    def timestamp_zero(self) -> Optional[pd.Timestamp]:
        if self.day_zero is None:
            return None
        return pd.to_datetime(self.day_zero).tz_localize(self.output_tz._timezone)

    @cached_property
    def via_project(self) -> ViaProject:
        """Loads ViaProject from file. Raises ViaProjectUnspecifiedError if
        self.via_project is None.

        Returns
        -------
        via_project : ViaProject
            VIA project loaded from self.via_project file.
        """
        if self.via_project_file is None:
            raise ViaProjectUnspecifiedError
        return ViaProject.parse_file(self.via_project_file)

    class Config:
        json_encoders = {Timezone: str}
        keep_untouched = (cached_property,)
        schema_extra = {"description": "Camfi configuration."}

    @root_validator
    def check_all_loctions_defined(cls, values):
        if values.get("time") is None:
            return values
        default_place = LocationWeatherStationCollector(
            locations=[], weather_stations=[], location_weather_station_mapping={}
        )
        specified_locations = set(
            location.name for location in values.get("place", default_place).locations
        )
        for camera_placement in values["time"].camera_placements.values():
            if camera_placement.location not in specified_locations:
                raise ValidationError(
                    f"location {camera_placement.location} unspecified in place field."
                )

        return values

    @classmethod
    def parse_yaml(cls, document: str) -> CamfiConfig:
        """Parses yaml document and returns a CamfiConfig instance.

        Parameters
        ----------
        document : str
            StrictYAML document string.

        Returns
        -------
        config : CamfiConfig
            CamfiConfig instance with settings defined in document.
        """
        return cls.parse_obj(load(document).data)

    @classmethod
    def parse_yaml_file(cls, document_path: Path) -> CamfiConfig:
        """Parses yaml document read from file and returns a CamfiConfig instance.

        Parameters
        ----------
        document_path : Path
            Path to file containing yaml document.

        Returns
        -------
        config : CamfiConfig
            CamfiConfig instance with settings defined in document.
        """
        with open(document_path, "r") as f:
            document = f.read()

        return cls.parse_yaml(document)

    def yaml(self, **kwargs) -> str:
        """Serialises self to yaml string.

        Parameters
        ----------
        **kwargs
            Passed to self.json(). E.g. exclude_unset. Note that by default,
            exclude_none=True to avoid YAMLSerializationError. Also, exclude_unset=True
            by default, as this makes more sense for CamfiConfig. This differs from
            the defaults of CamfiConfig.json(). Other defaults are the same.

        Returns
        -------
        document : str
            String containing yaml document.
        """
        kwargs = kwargs.copy()
        kwargs.setdefault("exclude_unset", True)
        if not kwargs.setdefault("exclude_none", True):
            raise NotImplementedError(
                "exclude_none=False is not implemented for yaml. "
                "Consider using json, or leave exclude_none=True."
            )
        config_dict = json.loads(self.json(**kwargs))
        if len(config_dict) == 0:
            return ""
        return as_document(config_dict).as_yaml()

    def get_image_dataframe(self) -> pd.DataFrame:
        """Calls self.via_project.to_image_dataframe(tz=self.output_tz), returning the
        result."""
        if self.place is None:
            raise PlaceUnspecifiedError
        return self.via_project.to_image_dataframe(tz=self.output_tz._timezone)

    def get_weather_dataframe(self) -> pd.DataFrame:
        """Calls self.place.get_weather_dataframe(), returning the result."""
        if self.place is None:
            raise PlaceUnspecifiedError
        return self.place.get_weather_dataframe()

    def get_sun_time_dataframe(
        self, days: Union[str, Dict[str, Sequence[date]]]
    ) -> pd.DataFrame:
        """Calls self.place.get_sun_time_dataframe"""
        if self.place is None:
            raise PlaceUnspecifiedError
        if days == "images":
            image_df = self.get_image_dataframe()
            days = {}
            for location in self.place.locations:
                days[location.name] = image_df[image_df["location"] == location.name][
                    "datetime_corrected"
                ].dt.date.unique()

        elif days == "weather":
            weather_df = self.get_weather_dataframe()
            days = {}
            for location in self.place.locations:
                weather_station: str = self.place.location_weather_station_mapping[
                    location.name
                ]
                days[location.name] = weather_df.loc[weather_station].index.date

        if isinstance(days, dict):
            return self.place.get_sun_time_dataframe(days)

        raise TypeError(
            "days must be one of ['images' | 'weather' | Dict[str, Sequence[date]]]. "
            f"Got {days} of type {type(days)}."
        )

    def get_merged_dataframe(self) -> pd.DataFrame:
        if self.place is None:
            raise PlaceUnspecifiedError
        image_df = self.get_image_dataframe()
        image_df["date"] = pd.to_datetime(image_df["datetime_corrected"].dt.date)

        days = {}
        for location in self.place.locations:
            days[location.name] = image_df[image_df["location"] == location.name][
                "date"
            ].dt.date.unique()

        weather_sun_df = self.place.get_weather_sun_dataframe(days=days)

        merged_df = pd.merge(
            image_df, weather_sun_df, how="left", on=["location", "date"], sort=True
        )
        merged_df.set_index(["location", "date"], inplace=True)

        return merged_df

    def load_all_exif_metadata(self) -> None:
        """Calls self.via_project.load_all_exif_metadata with appropriate arguments,
        set by config. Operates in place.
        """
        time_ratio = None
        if self.camera is not None:
            time_ratio = self.camera.camera_time_to_actual_time_ratio

        location_dict = None
        datetime_correctors = None
        if self.time is not None:
            location_dict = self.time.get_location_dict()
            datetime_correctors = self.time.get_correctors(
                camera_time_to_actual_time_ratio=time_ratio
            )

        self.via_project.load_all_exif_metadata(
            root=self.root,
            location_dict=location_dict,
            datetime_correctors=datetime_correctors,
            disable_progress_bar=self.disable_progress_bar,
        )

    def extract_all_wingbeats(self) -> None:
        """Calls extract_all_wingbeats on self.via_project with parameters taken from
        configuration."""
        if self.wingbeat_extraction is None:
            raise WingbeatExtractorConfigUnspecifiedError
        if self.camera is None or self.camera.line_rate is None:
            raise CameraConfigUnspecifiedError
        extract_all_wingbeats(
            self.via_project,
            root=self.root,
            line_rate=self.camera.line_rate,
            disable_progress_bar=self.disable_progress_bar,
            **self.wingbeat_extraction.dict(),
        )

    @property
    def training_dataset(self) -> CamfiDataset:
        """Gets CamfiDataset suitable for training camfi automatic annotation model."""
        if self.annotator is None:
            raise AnnotatorConfigUnspecifiedError

        if self.annotator.training is None:
            raise TrainingConfigUnspecifiedError

        if self.annotator.training.mask_maker is not None:
            mask_maker = self.annotator.training.mask_maker
        elif self.annotator.crop is not None:
            mask_maker = MaskMaker(shape=self.annotator.crop.shape)
        else:
            raise MaskMakerUnspecifiedError(
                "Need to set annotator.training.mask_maker and/or annotator.crop"
            )

        return CamfiDataset(
            root=self.root,
            via_project=self.via_project,
            crop=self.annotator.crop,
            inference_mode=False,
            mask_maker=mask_maker,
            transform=RandomHorizontalFlip(prob=0.5),
            min_annotations=self.annotator.training.min_annotations,
            max_annotations=self.annotator.training.max_annotations,
            box_margin=self.annotator.training.box_margin,
            exclude=set(self.annotator.training.test_set),
        )

    def train_model(self) -> None:
        """Calls camfi.annotator.train_model with appropriate arguments from self."""
        if self.annotator is None:
            raise AnnotatorConfigUnspecifiedError

        if self.annotator.training is None:
            raise TrainingConfigUnspecifiedError

        return train_model(
            self.training_dataset,
            load_pretrained_model=self.annotator.training.load_pretrained_model,
            device=self.annotator.training.device,
            batch_size=self.annotator.training.batch_size,
            num_workers=self.annotator.training.num_workers,
            num_epochs=self.annotator.training.num_epochs,
            outdir=self.annotator.training.outdir,
            model_name=self.annotator.training.model_name,
            save_intermediate=self.annotator.training.save_intermediate,
        )

    @property
    def inference_dataset(self) -> CamfiDataset:
        """Gets CamfiDataset suitable for automatic annotaton."""
        if self.annotator is None:
            raise AnnotatorConfigUnspecifiedError

        return CamfiDataset(
            root=self.root,
            via_project=self.via_project,
            crop=self.annotator.crop,
            inference_mode=True,
        )

    def annotate(self) -> ViaProject:
        """Performs automatic annotation on all the images in project. Saves project to
        a file if self.annotator.inference.output_path is set.

        Returns
        -------
        project : ViaProject
            With automatic annotations made.
        """
        if self.annotator is None:
            raise AnnotatorConfigUnspecifiedError

        if self.annotator.inference is None:
            raise InferenceConfigUnspecifiedError

        annotator = Annotator(
            dataset=self.inference_dataset,
            model=self.annotator.inference.model,
            device=self.annotator.inference.device,
            backup_device=self.annotator.inference.backup_device,
            split_angle=self.annotator.inference.split_angle,
            poly_order=self.annotator.inference.poly_order,
            endpoint_method=endpoint_methods[self.annotator.inference.endpoint_method],
            endpoint_extra_args=self.annotator.inference.endpoint_extra_args,
            score_thresh=self.annotator.inference.score_thresh,
            overlap_thresh=self.annotator.inference.overlap_thresh,
            edge_thresh=self.annotator.inference.edge_thresh,
        )
        annotated_project = annotator.annotate(
            disable_progress_bar=self.disable_progress_bar
        )

        # Optionally save ouptut to file
        output_path = (
            self.annotator.inference.output_path
            if self.annotator.inference.output_path
            else self.default_output
        )
        if output_path:
            with open(output_path, "w") as f:
                f.write(annotated_project.json(indent=2))

        return annotated_project

    def get_autoannotated_via_project(self):
        """Loads automatic annotations from file.

        Returns
        -------
        project : ViaProject
            ViaProject with automatically aquired annotations.
        """
        if self.annotator is None:
            raise AnnotatorConfigUnspecifiedError

        if (
            self.annotator.validation is not None
            and self.annotator.validation.autoannotated_via_project_file is not None
        ):
            return ViaProject.parse_file(
                self.annotator.validation.autoannotated_via_project_file
            )

        if (
            self.annotator.inference is None
            or self.annotator.inference.output_path is None
        ):
            raise ValidationConfigUnspecifiedError(
                "Could not determine which file to validate."
            )

        return ViaProject.parse_file(self.annotator.inference.output_path)

    def validate_annotations(self) -> List[AnnotationValidationResult]:
        """Validates automatically aquired annotations against ground-truth annotations.

        Returns
        -------
        validation_results : List[AnnotationValidationResult]
            Results from validation.
        """
        if self.annotator is None:
            raise AnnotatorConfigUnspecifiedError

        if self.annotator.validation is None:
            raise ValidationConfigUnspecifiedError

        subset_functions = {}
        for image_set in self.annotator.validation.image_sets:
            if image_set == "all":
                subset_functions["all"] = lambda x: True
            elif image_set == "train":
                if self.annotator.training is None:
                    exclude_set = set()
                else:
                    exclude_set = set(self.annotator.training.test_set)
                subset_functions["train"] = lambda x: x.filename not in exclude_set
            elif image_set == "test":
                if self.annotator.training is None:
                    include_set = set()
                else:
                    include_set = set(self.annotator.training.test_set)
                subset_functions["test"] = lambda x: x.filename in include_set
            else:
                raise ValueError(f"Expected one of all|train|test. Got {image_set}.")

        validation_results = validate_annotations(
            auto_annotations=self.get_autoannotated_via_project(),
            ground_truth=self.via_project,
            iou_thresh=self.annotator.validation.iou_thresh,
            subset_functions=subset_functions,
            disable_progress_bar=self.disable_progress_bar,
        )

        # Optionally save to file before returning
        output_dir = (
            self.annotator.validation.output_dir
            if self.annotator.validation.output_dir
            else self.default_output
        )
        if output_dir:
            for image_set, validation_result in zip(
                self.annotator.validation.image_sets, validation_results
            ):
                output_file = (
                    output_dir
                    / f"{self.annotator.validation.output_stem}.{image_set}.json"
                )
                with open(output_file, "w") as f:
                    f.write(validation_result.json(indent=2))

        return validation_results

    def write_project(self) -> None:
        """Writes self.via_project as json to self.default_output (if set) or stdout."""
        if self.default_output:
            with open(self.default_output, "w") as f:
                f.write(self.via_project.json(indent=2, exclude_unset=True))

        else:
            print(self.via_project.json(indent=2, exclude_unset=True))
