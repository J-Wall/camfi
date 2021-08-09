"""Defines data structures relating to VGG Image Annotator. Depends on
camfi.datamodel.geometry."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import exif
import pandas as pd
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    validator,
)
import torch
import torchvision.io
from tqdm import tqdm

from .geometry import (
    BoundingBox,
    PointShapeAttributes,
    CircleShapeAttributes,
    PolylineShapeAttributes,
)
from .via_region_attributes import ViaRegionAttributes
from .region_filter_config import RegionFilterConfig
from camfi.util import DatetimeCorrector, Field


class ViaFileAttributes(BaseModel):
    """Contains file-level metadata for a single photograph.

    Parameters
    ----------
    datetime_corrected : Optional[datetime]
        Time image was taken (after taking the error of the camera's clock into
        account).
    datetime_original : Optional[datetime]
        Time image was taken (according to camera).
    exposure_time : Optional[PositiveFloat]
        The exposure time of the image in seconds, as reported by the camera.
    location : Optional[str]
        The location the image was taken.
    pixel_x_dimension : Optional[PositiveInt]
        The width of the image in pixels.
    pixel_y_dimension : Optional[PositiveInt]
        The height of the image in pixels.
    """

    datetime_corrected: Optional[datetime]
    datetime_original: Optional[datetime]
    exposure_time: Optional[PositiveFloat]
    location: Optional[str]
    pixel_x_dimension: Optional[PositiveInt]
    pixel_y_dimension: Optional[PositiveInt]

    @validator("datetime_corrected", "datetime_original", pre=True)
    def valid_datetime_str(cls, v):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return datetime.strptime(v, "%Y:%m:%d %H:%M:%S")


class ViaRegion(BaseModel):
    """Combines region metadata with a geometry to define a complete annotation of a
    single flying insect motion blur.

    Parameters
    ----------
    region_attributes : ViaRegionAttributes
        Metadata of annotation.
    shape_attributes : Union[
        PolylineShapeAttributes, CircleShapeAttributes, PointShapeAttributes
    ]
        Geometry of annotation.
    """

    region_attributes: ViaRegionAttributes
    shape_attributes: Union[
        PolylineShapeAttributes, CircleShapeAttributes, PointShapeAttributes
    ]

    @validator("shape_attributes")
    def only_polylines_get_wingbeat_data(cls, v, values):
        if "region_attributes" in values and v.name != "polyline":
            for field in [
                values["region_attributes"].best_peak,
                values["region_attributes"].blur_length,
                values["region_attributes"].snr,
                values["region_attributes"].wb_freq_up,
                values["region_attributes"].wb_freq_down,
                values["region_attributes"].et_up,
                values["region_attributes"].et_dn,
            ]:
                assert field is None, "Wingbeat data is invalid for non-polylines"
        return v

    def get_bounding_box(self) -> BoundingBox:
        """Calls .get_bounding_box on self.shape_attributes to get the bounding box
        of the annotation.

        Returns
        -------
        box : BoundingBox
            Bounding box of annotation.
        """
        return self.shape_attributes.get_bounding_box()

    def in_box(self, box: BoundingBox) -> bool:
        """Returns True if all points in region are within bounding box.

        Parameters
        ----------
        box: BoundingBox
            Box to test against.

        Examples
        --------
        >>> polyline = PolylineShapeAttributes(
        ...     all_points_x=[1, 3],
        ...     all_points_y=[15, 13],
        ... )
        >>> region = ViaRegion(
        ...     region_attributes=ViaRegionAttributes(),
        ...     shape_attributes=polyline,
        ... )
        >>> region.in_box(BoundingBox(x0=1, y0=13, x1=4, y1=16))
        True
        >>> region.in_box(BoundingBox(x0=2, y0=13, x1=4, y1=16))
        False
        >>> region.in_box(BoundingBox(x0=1, y0=14, x1=4, y1=16))
        False
        >>> region.in_box(BoundingBox(x0=1, y0=13, x1=3, y1=16))
        False
        >>> region.in_box(BoundingBox(x0=1, y0=13, x1=4, y1=15))
        False
        """
        return self.shape_attributes.in_box(box)

    def passes_filter(self, filters: RegionFilterConfig) -> bool:
        """Determine whether self passes filters.

        Parameters
        ----------
        filters : RegionFilterConfig
            Filters to check.

        Returns
        -------
        passes : bool
           True if all filters are passed.
        """
        for field in filters.__fields__.keys():
            attribute = getattr(self.region_attributes, field)
            filt = getattr(filters, field)
            if filt.exclude_none and attribute is None:
                return False
            if attribute > filt.le or attribute < filt.ge:
                return False

        return True


class ViaMetadata(BaseModel):
    """Combines file-level image metadata with a list of annotations contained within
    the image.

    Parameters
    ----------
    file_attributes : ViaFileAttributes
        File-level image metadata.
    filename : Path
        Relative path to image file.
    regions : list[ViaRegion]
        list of flying insect annotations.
    size : int = -1
        Not used. (Included for compatability with VIA).
    """

    file_attributes: ViaFileAttributes
    filename: Path
    regions: list[ViaRegion]
    size: int = -1

    def get_bounding_boxes(self) -> list[BoundingBox]:
        """Calls .get_bounding_box on each region in self.regions.

        Returns
        -------
        boxes : list[BoundingBox]
            list of bounding boxes, with one BoundingBox per item in self.regions.
        """
        return [region.get_bounding_box() for region in self.regions]

    def get_labels(self) -> list[PositiveInt]:
        """Gets a list full of 1's with same length as self.regions.

        Returns
        -------
        labels : list[int]
            [1, 1, 1, ...]
        """
        return [1 for _ in range(len(self.regions))]

    def load_exif_metadata(
        self,
        root: Path = Path(),
        location: Optional[str] = None,
        datetime_corrector: Optional[DatetimeCorrector] = None,
    ) -> None:
        """Extract EXIF metadata from an image file and put it in
        self.file_attributes.

        **Note: this will overwrite all contents in self.file_attributes.**

        EXIF tags loaded:
          - datetime_original: datetime
          - exposure_time: PositiveFloat
          - pixel_x_dimension: PositiveInt
          - pixel_y_dimension: PositiveInt

        Extra tags:
          - datetime_corrected: datetime
                if datetime_corrector is set, this is calculated by
                calling datetime_corrector(datetime_original).
          - location: str
                set if location is set

        Parameters
        ----------
        root : Path
            Root directory from which the relative path in self.filename is resolved.
            Defaults to current working directory. If a str is passed it will be coerced
            to a Path.
        location : Optional[str]
            Option to also apply a location
        datetime_corrector : Optional[DatetimeCorrector]
            If set, then will be used to calculate datetime_corrected

        Returns
        -------
        None (operates in place)

        Examples
        --------
        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata.load_exif_metadata()
        >>> metadata.file_attributes.datetime_original
        datetime.datetime(2019, 11, 14, 20, 30, 29)
        >>> print(round(metadata.file_attributes.exposure_time, 6))
        0.111111
        >>> metadata.file_attributes.pixel_y_dimension
        3456
        >>> metadata.file_attributes.pixel_x_dimension
        4608

        Optionally specify root directory. Here we are loading the same file, but using
        a root parameter. Note that root may also be a relative path (as in this case).
        Absolute paths are also acceptable.

        >>> metadata_with_root = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata_with_root.load_exif_metadata(root="camfi/test")
        >>> metadata_with_root.file_attributes == metadata.file_attributes
        True

        If location is set, this will be reflected

        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata.load_exif_metadata(location="cabramurra")
        >>> metadata.file_attributes.location
        'cabramurra'

        If a time correction needs to be made (for example if the camera's clock is
        known to have been incorrectly set), then we can correct the datetime by
        supplying a function to the datetime_corrector parameter.

        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> metadata.load_exif_metadata(
        ...     datetime_corrector=lambda dt: dt - timedelta(days=30)
        ... )
        >>> metadata.file_attributes.datetime_original
        datetime.datetime(2019, 11, 14, 20, 30, 29)
        >>> metadata.file_attributes.datetime_corrected
        datetime.datetime(2019, 10, 15, 20, 30, 29)
        """
        with open(Path(root) / self.filename, "rb") as image_file:
            image = exif.Image(image_file)

        self.file_attributes = ViaFileAttributes(
            datetime_original=image.datetime_original,
            exposure_time=image.exposure_time,
            pixel_x_dimension=image.pixel_x_dimension,
            pixel_y_dimension=image.pixel_y_dimension,
            location=location,
        )

        if (
            datetime_corrector is not None
            and self.file_attributes.datetime_original is not None
        ):
            self.file_attributes.datetime_corrected = datetime_corrector(
                self.file_attributes.datetime_original
            )

    def read_image(self, root: Path = Path()) -> torch.Tensor:
        """Read an image from a file.

        Parameters
        ----------
        root: Path
            Root directory from which the relative path in self.filename is resolved.
            Defaults to current working directory. If a str is passed it will be coerced
            to a Path.

        Returns
        -------
        torch.Tensor[colour, height (y), width (x)]
            Image as RGB float32 tensor.

        Examples
        --------
        >>> metadata = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="camfi/test/data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> image = metadata.read_image()
        >>> image.shape == (3, 3456, 4608)
        True
        >>> image.dtype
        torch.float32

        Optionally specify root directory. Here we are loading the same file, but using
        a root parameter. Note that root may also be a relative path (as in this case).
        Absolute paths are also acceptable.

        >>> metadata_with_root = ViaMetadata(
        ...     file_attributes=ViaFileAttributes(),
        ...     filename="data/DSCF0010.JPG",
        ...     regions=[],
        ... )
        >>> image_with_root = metadata_with_root.read_image(root="camfi/test")
        >>> image_with_root.allclose(image)
        True
        """
        image = torchvision.io.read_image(
            str(Path(root) / self.filename), mode=torchvision.io.image.ImageReadMode.RGB
        )
        return image / 255  # Converts from uint8 to float32

    def filter_regions(self, region_filters: RegionFilterConfig) -> None:
        """Filters regions in-place.

        Parameters
        ----------
        region_filters : RegionFilterConfig
            Filters to apply.
        """
        self.regions = list(
            filter(lambda x: x.passes_filter(region_filters), self.regions)
        )


class ViaProject(BaseModel):
    """Defines the structure of a VIA project file. Can be used for loading and saving
    VIA project data.

    Parameters
    ----------
    via_attributes : dict
        Unused by camfi. (Included for compatability with VIA).
    via_img_metadata : dict[str, ViaMetadata]
        dict of {str: ViaMetadata} pairs. Keys can be arbitrary strings, however they
        usually bare some resemblance to the .filename attribute of the ViaMetadata
        instance.
    via_settings : dict
        Unused by camfi. (Included for compatability with VIA).
    """

    via_attributes: dict
    via_img_metadata: dict[str, ViaMetadata]
    via_settings: dict

    class Config:
        """Sets pydantic.BaseModel configuration of ViaProject."""

        fields = {
            "via_attributes": "_via_attributes",
            "via_img_metadata": "_via_img_metadata",
            "via_settings": "_via_settings",
        }

    def load_all_exif_metadata(
        self,
        root: Path = Path(),
        location_dict: Optional[Mapping[Path, Optional[str]]] = None,
        datetime_correctors: Optional[
            Mapping[Path, Optional[DatetimeCorrector]]
        ] = None,
        disable_progress_bar: Optional[bool] = True,
    ) -> None:
        """Calls the .load_exif_metadata method on all ViaMetadata instances in
        self.via_img_metadata, extracting the EXIF metadata from each image file.

        Parameters
        ----------
        root : Path
            Root directory from which the relative path in self.filename is resolved.
            Defaults to current working directory. If a str is passed it will be coerced
            to a Path.
        location_dict : Optional[Mapping[Path, Optional[str]]]
            A mapping from filenames (i.e. relative paths to images under root) to
            location strings, which are passed to ViaMetadata.load_exif_metadata.
            Typically, an instance of camfi.util.SubDirdict should be used.
        datetime_correctors : Optional[Mapping[Path, Optional[DatetimeCorrector]]]
            A mapping from filenames (i.e. relative paths to images under root) to
            DatetimeCorrector instances, which are passed to
            ViaMetadata.load_exif_metadata
            Typically, an instance of camfi.util.SubDirdict should be used.
        disable_progress_bar : Optional[bool]
            If True (default), progress bar is disabled.
            If set to None, disable on non-TTY.

        Returns
        -------
        None
            Operates in place.

        Examples
        --------
        >>> with open("camfi/test/data/sample_project_images_included.json") as f:
        ...     project = ViaProject.parse_raw(f.read())

        The file which has been loaded contains no metadata

        >>> for meta in project.via_img_metadata.values():
        ...     print(meta.filename, str(meta.file_attributes.datetime_original))
        DSCF0010.JPG None
        DSCF0011.JPG None

        After load_all_exif_metadata is called, project does contain image metadata

        >>> project.load_all_exif_metadata(root=Path("camfi/test/data"))
        >>> for meta in project.via_img_metadata.values():
        ...     print(meta.filename, str(meta.file_attributes.datetime_original))
        DSCF0010.JPG 2019-11-14 20:30:29
        DSCF0011.JPG 2019-11-14 20:40:32

        If location_dict and/or datetime_correctors are set, the metadata will
        include location and/or datetime_corrected, respectively. Normally, these
        would be set as instances of camfi.util.SubDirdict, but for brevity we use
        a regular dict for each of them here.

        >>> project.load_all_exif_metadata(
        ...     root=Path("camfi/test/data"),
        ...     location_dict={
        ...         Path("DSCF0010.JPG"): "loc0", Path("DSCF0011.JPG"): "loc1"
        ...     },
        ...     datetime_correctors={
        ...         Path("DSCF0010.JPG"): lambda dt: dt + timedelta(hours=1),
        ...         Path("DSCF0011.JPG"): lambda dt: dt - timedelta(hours=1),
        ...     },
        ... )
        >>> for meta in project.via_img_metadata.values():
        ...     print(
        ...         meta.filename,
        ...         meta.file_attributes.location,
        ...         str(meta.file_attributes.datetime_corrected),
        ...     )
        DSCF0010.JPG loc0 2019-11-14 21:30:29
        DSCF0011.JPG loc1 2019-11-14 19:40:32
        """
        if location_dict is None:
            location_dict = defaultdict(lambda: None)
        if datetime_correctors is None:
            datetime_correctors = defaultdict(lambda: None)

        for metadata in tqdm(
            self.via_img_metadata.values(),
            disable=disable_progress_bar,
            desc="Loading EXIF metadata",
            unit="img",
            dynamic_ncols=True,
            ascii=True,
            color="green",
        ):
            metadata.load_exif_metadata(
                root=root,
                location=location_dict[metadata.filename],
                datetime_corrector=datetime_correctors[metadata.filename],
            )

    def to_region_dataframe(self) -> pd.DataFrame:
        """Returns a Pandas DataFrame with one row per region (annotation).

        Returns
        -------
        regions : pd.DataFrame
            DataFrame with a row for every annotation in self.via_img_metadata. Contains
            a column for every field in ViaFileAttributes and ViaRegionAttributes, as
            well as img_key, filename, and name columns.
        """
        rows: list[dict[str, Any]] = []
        for img_key, metadata in self.via_img_metadata.items():
            for region in metadata.regions:
                row = {
                    "img_key": img_key,
                    "filename": metadata.filename,
                    "name": region.shape_attributes.name,
                }
                row.update(dict(metadata.file_attributes))
                row.update(dict(region.region_attributes))
                rows.append(row)

        return pd.DataFrame(rows)

    def to_image_dataframe(self, tz: Union[None, str, tzinfo] = None) -> pd.DataFrame:
        """Returns a Pandas DataFrame with one row per image.

        Parameters
        ----------
        tz: Optional[tzinfo]
           If set, all datetime_corrected values will be converted to the specified
           timezone.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with one row per image. Contains columns for each field in
            ViaFileAttributes, as well as img_key, filename, and n_annotations.
            Note: values in datetime_corrected (but not datetime_original) will be
            converted to pandas.Timestamp, with timezone conversion/localization applied
            if applicable.
        """
        rows: list[dict[str, Any]] = []
        for img_key, metadata in self.via_img_metadata.items():
            row = {
                "img_key": img_key,
                "filename": metadata.filename,
                "n_annotations": len(metadata.regions),
            }
            row.update(dict(metadata.file_attributes))

            # Convert datetime to pandas.Timestamp before putting in the DataFrame
            row["datetime_corrected"] = pd.to_datetime(row["datetime_corrected"])
            assert isinstance(row["datetime_corrected"], pd.Timestamp)

            if row["datetime_corrected"] is not None and tz is not None:
                # Fix timezone
                if row["datetime_corrected"].tzinfo is not None:
                    row["datetime_corrected"] = row["datetime_corrected"].astimezone(tz)
                else:
                    row["datetime_corrected"] = row["datetime_corrected"].tz_localize(
                        tz
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    def filtered_copy(
        self, function: Callable[[ViaMetadata], bool], deep: bool = False
    ) -> ViaProject:
        """Filters images in self.via_img_metadata, returning a new ViaProject instance.

        Parameters
        ----------
        function : Callable[[ViaMetadata], bool]
            Called on each value in self.via_img_metadata to determine if it should be
            included in output.
        deep : bool
            If True, make a deep copy.

        Returns
        -------
        project : ViaProject
            Copy of self with via_img_metadata filtered.
        """
        if deep:
            filtered_img_metadata = {
                key: value.copy(deep=True)
                for key, value in self.via_img_metadata.items()
                if function(value)
            }
        else:
            filtered_img_metadata = {
                key: value
                for key, value in self.via_img_metadata.items()
                if function(value)
            }
        return self.copy(update={"via_img_metadata": filtered_img_metadata}, deep=deep)

    def filter_inplace(self, function: Callable[[ViaMetadata], bool]) -> None:
        """Filters images in self.via_img_metadata in-place.

        Parameters
        ----------
        function : Callable[[ViaMetadata], bool]
            Called on each value in self.via_img_metadata to determine if it should be
            included in output.
        """
        _new_img_metadata = {
            key: value
            for key, value in self.via_img_metadata.items()
            if function(value)
        }
        self.via_img_metadata = _new_img_metadata
