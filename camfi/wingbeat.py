"""Implements wingbeat frequency measurement from annotated images of flying insects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property, total_ordering
from math import sqrt
from pathlib import Path
from typing import Optional, Union

from bces import bces
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    validator,
)
import sklearn.mixture
import torch
from tqdm import tqdm

from camfi.datamodel.geometry import PolylineShapeAttributes
from camfi.datamodel.via import ViaRegionAttributes, ViaMetadata, ViaProject
from camfi.util import DatetimeCorrector, Field


def autocorrelation(roi: torch.Tensor, max_pixel_period: PositiveInt) -> torch.Tensor:
    """Calculates the autocorrelation along axis 1 of roi. Will run entirely on the
    device specified by roi.device, and is optimised for running on the GPU.

    Parameters
    ----------
    roi : torch.Tensor
        Tensor of shape (width, blur_length).
    max_pixel_period : Optional[PositiveInt]
        Maximum period to consider (choosing a smaller number will increase the speed of
        execution if running on cpu, and decrease memory consumption regardless of
        which device it's running on). If None, calculated as roi.shape[1] // 2.

    Returns
    -------
    mean_autocorrelation : torch.Tensor
        Tensor of shape (max_pixel_period,). Contains the autocorrelation with of the
        roi along axis 1, with integer step-sizes.

    Examples
    --------
    >>> from torch import arange, cos, sin, stack
    >>> from math import pi
    >>> theta = arange(0., 4. * pi, pi / 4)
    >>> autocorrelation(stack([sin(theta), cos(theta)]), len(theta) // 2)
    tensor([ 0.4375,  0.2500,  0.0000, -0.2500, -0.3750, -0.2500,  0.1250,  0.3750])
    """
    mean_diff = roi - roi.mean(axis=0)  # type: ignore[call-overload]
    std = roi.std(axis=0)  # type: ignore[call-overload]

    index = torch.arange(roi.shape[1] - max_pixel_period, device=roi.device)
    step, origin = torch.meshgrid(index[: max_pixel_period + 1], index)
    step = step + origin

    autocovariance = (mean_diff[:, origin] * mean_diff[:, step]).mean(axis=0)

    denominator = std[origin] * std[step]

    return (autocovariance / denominator).nan_to_num().mean(axis=1)


def find_best_peak(
    values: torch.Tensor,
) -> tuple[Optional[PositiveInt], Optional[float]]:
    """Takes a Tensor of values (with 1 dimension), and finds the index of the best peak
    If peak finding fails, (None, None) is returned.

    Parameters
    ----------
    values : torch.Tensor
        1-D tensor of values to find peaks in.

    Returns
    -------
    best_peak : Optional[PositiveInt]
        Index of best peak.
    snr : Optional[float]
        Score of peak.

    Examples
    --------
    >>> t = torch.zeros(100)
    >>> t[0] = 1.  # The first peak is always ignored
    >>> t[50] = 1.
    >>> find_best_peak(t)
    (50, inf)

    >>> from torch import cos, linspace
    >>> from math import pi
    >>> t = (cos(linspace(0., 8. * pi, 32)) + 1) * linspace(0.5, 0., 32)
    >>> t[4] = max(t[3], t[5]) + 0.01  # small local peak is not picked up
    >>> t[2] = t[0] + 0.01  # The highest peak is not necessarily the best peak
    >>> best_peak, score = find_best_peak(t)
    >>> best_peak
    8
    >>> 0 < score < 10
    True

    If no peak is found, then (None, None) is returned
    >>> find_best_peak(torch.zeros(10))
    (None, None)
    """
    peaks = ((values > values.roll(-1)) & (values > values.roll(1))).nonzero(
        as_tuple=True
    )[0][1:]
    sorted_peaks = peaks[values[peaks].argsort(descending=True)]

    best_peak: Optional[PositiveInt] = None
    snr: Optional[float] = None
    snrs: list[float] = []

    for peak in sorted_peaks:
        peak_idx = int(peak)
        # Find snr
        trough_values = torch.cat(
            [
                values[peak_idx // 3 : (peak_idx * 3) // 4],
                values[(peak_idx * 5) // 3 : (peak_idx * 7) // 4],
            ]
        )
        snr = float((values[peak] - trough_values.mean()) / trough_values.std())
        if len(snrs) > 0:
            if snrs[-1] > snr:
                snr = snrs[-1]
                break  # Previous peak was the best peak
        snrs.append(snr)
        best_peak = int(peak)

    return best_peak, snr


class WingbeatSuppFigPlotter(ABC, BaseModel):
    """Defines an interface for producing suplementary figures from the wingbeat
    extraction procedure, used by WingbeatExtractor. This abstract base class is
    agnostic to which plotting library is used. Subclasses will naturally have to use
    a particular plotting library, such as matplotlib.

    Concrete subclasses must implement the .__call__ method.

    Parameters
    ----------
    root: Path
        Root directory to put supplementary figures in.
    image_filename: Path
        Relative path to image file. `self.get_filepath` uses this to compute the path
        where the supplementary figure will be written (note that if annotation_idx is
        set, then the filename will be modified to include the annotation index - useful
        for plotting multple supplementary figures relating to annotations from one
        image).
    suffix: str
        File suffix for output plots. Defaults to ".png", but could be ".pdf", ".html",
        or ".eps", etc. depending on the subclass of WingbeatSuppFigPlotter.
    annotation_idx: NonNegativeInt
        Used to infer the correct filename. Defaults to 0 (which is almost always what
        you would want when initialising a subclass of WingbeatSuppFigPlotter).

    Examples
    --------
    >>> class DummyWingbeatSuppFigPlotter(WingbeatSuppFigPlotter):
    ...     def __call__(
    ...         self,
    ...         region_attributes: ViaRegionAttributes,
    ...         region_of_interest: torch.Tensor,
    ...         mean_autocorrelation: torch.Tensor,
    ...     ) -> None:
    ...         self.get_filepath()
    ...         return None
    >>> supplementary_figure_plotter = DummyWingbeatSuppFigPlotter(
    ...     root="foo", image_filename="bar/baz.jpg"
    ... )
    >>> supplementary_figure_plotter.get_filepath() == Path("foo/bar/baz_0.png")
    True

    The annotation index has now been incremented by 1.

    >>> supplementary_figure_plotter.annotation_idx
    1
    >>> supplementary_figure_plotter.get_filepath() == Path("foo/bar/baz_1.png")
    True
    """

    root: Path
    image_filename: Path
    suffix: str = ".png"
    annotation_idx: NonNegativeInt = 0

    def get_filepath(self):
        """Computes the filepath for the supplementary figure and increments
        `self.annotation_idx` by 1.

        Returns
        -------
        filepath : Path
            Full path to supplementary figure file.
        """
        name = f"{self.image_filename.stem}_{self.annotation_idx}{self.suffix}"
        self.annotation_idx += 1
        return self.root.joinpath(self.image_filename.parent, name)

    @abstractmethod
    def __call__(
        self,
        region_attributes: ViaRegionAttributes,
        region_of_interest: torch.Tensor,
        mean_autocorrelation: torch.Tensor,
    ) -> None:
        """Implementations produce a supplementary figure of a wingbeat extraction
        process, perhaps writing this to a file. Should call `self.get_filepath()` once
        only to get the path to the file where the figure should be written.

        Parameters
        ----------
        region_attributes : ViaRegionAttributes
            With fields calculated (e.g. by WingbeatExtractor.process_blur)
        region_of_interest : torch.Tensor
            Greyscale image Tensor displaying region of interest
        mean_autocorrelation : torch.Tensor
            1-d Tensor with values containing autocorrrelation along axis 1 of
            `region_of_interest`
        """


class WingbeatExtractorConfig(BaseModel):
    """Contains configurable parameters for WingbeatExtractor. Provides a creation
    methods for WingbeatExtractor. See WingbeatExtractor for futher documentation.
    """

    device: str = Field(
        "cpu",
        description="Using GPU ('cuda') can give 4x speedups for certain operations.",
    )
    backup_device: Optional[str] = Field(
        None,
        description="Used when computation fails on main device due to memory limit.",
    )
    scan_distance: PositiveInt = Field(
        50,
        description="Max. distance from polyline used for wingbeat extraction.",
    )
    max_pixel_period: Optional[PositiveInt] = Field(
        None, description="Max. pixel period to check during wingbeat extraction."
    )
    force_load_exif_metadata: bool = Field(
        False,
        description="Forces EXIF metadata to be read from image instead of ViaProject.",
    )

    class Config:
        schema_extra = {
            "description": "Contains configurable parameters for WingbeatExtractor."
        }

    @validator("device", "backup_device")
    def check_device_available(cls, v):
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "No NVIDIA driver on your system. Install one or set device to 'cpu'."
            )
        try:
            torch.tensor([], device=v)
        except RuntimeError as e:
            raise ValueError(*e.args())

        return v

    def get_wingbeat_extractor(self, **kwargs):
        """Instantiates WingbeatExtractor.

        Parameters
        ----------
        **kwargs
            Passed to WingbeatExtractor

        Returns
        -------
        wingbeat_extractor : WingbeatExtractor
            WingbeatExtractor with fields taken from self and kwargs.
        """
        return WingbeatExtractor(**self.dict(), **kwargs)


class WingbeatExtractor(WingbeatExtractorConfig):
    """Class for measuring wingbeat frequencies of annotated flying insects in an image.
    A new instance of WingbeatExtractor should be used for each distinct image file.

    Parameters
    ----------
    metadata: ViaMetadata
        Containing annotations of flying insects, as well as file-level image metadata.
        If file-level metadata is missing (specifically, exposure_time), this will be
        read from the image file.
    root: Path
        Path to root directory containing all image directories.
    line_rate: PositiveFloat
        Rolling shutter line rate of the camera used to take the photo, in lines per
        second. This must be measured separately. See
        https://camfi.readthedocs.io/en/latest/usage/notebooks/camera_calibration.html
        for a guide on measuring the rolling shutter line rate.
    device: str
        Some steps can run on the GPU, which can give speedups of over 4x. To enable,
        set e.g. device="cuda".
    backup_device: Optional[str]
        If a step raises a RuntimeError, it can be re-attempted on an alternative
        device. It is recommended to set to "cpu" if running on a GPU with limited
        memory.
    scan_distance: PositiveInt
        Optional parameter used in WingbeatExtractor.process_blur. This defines the
        maximum perpendicular distance from the polyline annotation of pixels included
        in the rotated, cropped, and straightened region of interest images.
    max_pixel_period: Optional[PositiveInt]
        Optional parameter used in WingbeatExtractor.process_blur. By default,
        autocorrelation is calculated up to half the length of each motion blur. For
        speed of execution or to reduce memory footprint, a maximum value can be set.
    force_load_exif_metadata: bool
        If True, EXIF metadata will be read from the image file, regardless of whether
        exposure_time is already set in metadata.file_attributes. By default, EXIF
        metadata will only be read if it is missing from metadata.file_attributes.
    location: Optional[str]
        Sets location string when loading EXIF metadata, placed in
        metadata.file_attributes.location. Has no effect if EXIF metadata is not loaded.
        Recommended to use in conjunction with force_load_exif_metadata.
    datetime_corrector: Optional[DatetimeCorrector]
        If provided, will be called while loading EXIF metadata to obtain a corrected
        timestamp, which is placed in metadata.file_attributes.datetime_corrected. Has
        no effect if EXIF metadata is not loaded.
        Recommended to use in conjunction with force_load_exif_metadata.
    supplementary_figure_plotter: Optional[WingbeatSuppFigPlotter]
        If set, will be called to plot supplementary figures during self.process_blur.
        A custom implementation of WingbeatSuppFigPlotter may be used, or one from
        camfi.plotting.
    """

    metadata: ViaMetadata
    root: Path
    line_rate: PositiveFloat
    location: Optional[str] = None
    datetime_corrector: Optional[DatetimeCorrector] = None
    supplementary_figure_plotter: Optional[WingbeatSuppFigPlotter] = None

    # image and exposure_time may require expensive IO operations, so should only happen
    # once each, if at all. They should also be treated as immutable for the life of the
    # WingbeatExtractor instance. Hence, the property and cache decorators.
    @cached_property
    def image(self) -> torch.Tensor:
        """Loads image from file and converts it to a greyscale tensor. Output is cached
        (so image is only loaded once for the life of the WingbeatExtractor instance).

        Returns
        -------
        image : torch.Tensor
            Image tensor with shape [height, width].
        """
        return self.metadata.read_image(root=self.root).to(self.device).mean(axis=-3)  # type: ignore[call-overload]

    @cached_property
    def exposure_time(self) -> PositiveFloat:
        """Gets exposure time either from self.metadata.file_attributes, or from the
        EXIF metadata of the image file. Caches output so file is only read once for
        life of WingbeatExtractor instance.

        Returns
        -------
        expoosure_time : PositiveFloat
            Exposure time of photograph in seconds.
        """
        if (
            self.force_load_exif_metadata
            or self.metadata.file_attributes.exposure_time is None
        ):
            self.metadata.load_exif_metadata(
                root=self.root,
                location=self.location,
                datetime_corrector=self.datetime_corrector,
            )
        assert isinstance(self.metadata.file_attributes.exposure_time, float)
        return self.metadata.file_attributes.exposure_time

    class Config:
        keep_untouched = (cached_property,)

    # In order to use the cache decorator, we need to define __eq__ and __hash__
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self) // 16

    def process_blur(
        self, polyline: PolylineShapeAttributes, score: Optional[float] = None
    ) -> ViaRegionAttributes:
        """Performs the camfi algorithm to takes a measurement of wingbeat frequency
        from a flying insect motion blur which has been annotated with a polyline.

        Parameters
        ----------
        polyline: PolylineShapeAttributes
            Polyline annotation following the path of the flying insect's motion blur.
        score: Optional[float]
            Score parameter to be passed to ViaRegionAttributes constructor (should set
            if processing an annotation which was generated automatically, so that the
            score is reflected in the output).

        Returns
        -------
        region_attributes : ViaRegionAttributes
            With all fields set (including score iff a value was given).
        """
        # Load region of interest. Mypy complains about self.image not being the right
        try:
            roi = polyline.extract_region_of_interest(self.image, self.scan_distance)
        except RuntimeError:
            if self.backup_device is None:
                raise
            roi = polyline.extract_region_of_interest(
                self.image.to(self.backup_device), self.scan_distance
            ).to(self.device)

        # Infer max_pixel_period if not set
        max_pixel_period = roi.shape[1] // 2
        if self.max_pixel_period is not None:
            max_pixel_period = min(max_pixel_period, self.max_pixel_period)

        # Calculate autocorrelation. This step can run on the GPU
        try:
            mean_autocorrelation = autocorrelation(roi, max_pixel_period).cpu()
        except RuntimeError:
            if self.backup_device is None:
                raise
            mean_autocorrelation = autocorrelation(
                roi.to(self.backup_device), max_pixel_period
            ).cpu()

        # Find wingbeat peak
        best_peak, snr = find_best_peak(mean_autocorrelation)

        if best_peak is None:  # Failed to find peak
            region_attributes = ViaRegionAttributes(
                score=score, blur_length=polyline.length()
            )

        else:
            # Calculate wingbeat frequency from peak.
            # Note that due to the ambiguity in the direction of the moth's flight,
            # as well as rolling shutter, there are two possible frequency estimates,
            # with the lower frequency corresponding to an upward direction and the
            # higher frequency corresponding to a downward direction of flight with
            # respect to the camera's orientation.
            y_diff = polyline.y_diff()
            corrected_exposure_time = (
                (self.exposure_time + torch.tensor([y_diff, -y_diff]) / self.line_rate)
                .sort()
                .values
            )
            period = [
                torch.arange(1, max_pixel_period + 1) * float(et) / roi.shape[1]
                for et in corrected_exposure_time
            ]
            wb_freq_up, wb_freq_down = [1 / period[i][best_peak] for i in (0, 1)]

            region_attributes = ViaRegionAttributes(
                score=score,
                best_peak=best_peak,
                blur_length=polyline.length(),
                snr=snr,
                wb_freq_up=wb_freq_up,
                wb_freq_down=wb_freq_down,
                et_up=float(corrected_exposure_time[0]),
                et_dn=float(corrected_exposure_time[1]),
            )

        # Plot supplementary figure
        if self.supplementary_figure_plotter is not None:
            self.supplementary_figure_plotter(
                region_attributes, roi, mean_autocorrelation
            )

        return region_attributes

    def extract_wingbeats(self) -> int:
        """Calls self.process_blur on the shape_attributes of all polyline regions in
        self.metadata, replacing the region_attributes of those regions with ones
        containing wingbeat data.

        Operates in place.

        Returns
        -------
        polylines_processed : int
            Number of polyline annotations processed.
        """
        polylines_processed = 0
        for region in filter(
            lambda r: r.shape_attributes.name == "polyline", self.metadata.regions
        ):
            # Get parameters for self.process_blur
            polyline = region.shape_attributes
            assert isinstance(polyline, PolylineShapeAttributes)
            score = region.region_attributes.score

            # Update region attributes with wingbeat data
            region.region_attributes = self.process_blur(polyline, score=score)
            polylines_processed += 1

        return polylines_processed


def extract_all_wingbeats(
    via_project: ViaProject, disable_progress_bar: Optional[bool] = True, **kwargs
) -> None:
    """Extracts wingbeat data from all images in via_project, inserting that data into
    via_project (operates in place).

    Parameters
    ----------
    via_project : ViaProject
        Contains metadata for each image to process.
    disable_progress_bar : Optional[bool]
        If True (default), progress bar is disabled.
        If set to None, disable on non-TTY.
    **kwargs
        Passed to WingbeatExtractor constructor (once for each image).
    """
    postfix = {"polylines_processed": 0}
    pb = tqdm(
        via_project.via_img_metadata.items(),
        disable=disable_progress_bar,
        desc="Extracting wingbeats",
        unit="img",
        dynamic_ncols=True,
        ascii=True,
        postfix=postfix,
    )
    for img_key, metadata in pb:
        wingbeat_extractor = WingbeatExtractor(metadata=metadata, **kwargs)
        postfix["polylines_processed"] += wingbeat_extractor.extract_wingbeats()
        via_project.via_img_metadata[img_key] = wingbeat_extractor.metadata
        pb.set_postfix(postfix, refresh=False)


@total_ordering
class BcesResult(BaseModel):
    """Stores parameters of one BCES linear regression.

    estimates : list[tuple[float, float, float, float, float]]
        list of (gradient, y_intercept, gradient_stderr, y_intercept_stderr, cov_xy)
        tuples of estimates and standard errors. Has length n_classes.
    err : np.ndarray
        Array of weighted errors for each measurement from each regression line. Has
        shape (n_classes, len(x)).

    Parameters
    ----------
    gradient : float
        Gradient Estimate.
    y_intercept : float
        Intercept estimate.
    gradient_stderr : float
        Standard error of the gradient esitmate.
    y_intercept_stderr : float
        Standard error of the y_intercept estimate.
    cov_xy : float
        Covariance estimate.
    """

    gradient: float
    y_intercept: float
    gradient_stderr: float
    y_intercept_stderr: float
    cov_xy: float

    def __lt__(self, other: BcesResult) -> bool:
        return (
            self.gradient,
            self.y_intercept,
            self.gradient_stderr,
            self.y_intercept_stderr,
            self.cov_xy,
        ) < (
            other.gradient,
            other.y_intercept,
            other.gradient_stderr,
            other.y_intercept_stderr,
            other.cov_xy,
        )


class BcesEM(BaseModel):
    """Implements an expectation-maximisation algorithm for fitting multiple BCES
    linear regresssions to a dataset.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Dependent variable values. Should have same shape as x.
    n_classes : int
        Number of classes (i.e. number of regressions in multiple regression).
    xerr : Union[float, np.ndarray]
        Error of independent variable measurments. Should have same shape as x. If float
        is given, it will be converted to an array.
    yerr : Union[float, np.ndarray]
        Error of dependent variable measurements. Should have same shape as x. If float
        is given, it will be converted to an array.
    cov : Union[float, np.ndarray]
        Covariance of independent and dependent variable measurements. Should have same
        shape as x. If float is given, it will be converted to an array.
    class_mask: Union[None, int, np.random.Generator, np.ndarray]
        Array of integers defining classes of measurements. Accepts values
        from the set {0, 1, ..., n_classes - 1}. Should have same shape as x.
        Alternatively, give a seed for a random number generator (or the Generator
        itself), and class_mask will be generated.
    prob_class : np.ndarray
        Marginal probabilities of each class. Should have shape (n_classes,), and sum
        to 1.
    """

    x: np.ndarray
    y: np.ndarray
    n_classes: PositiveInt
    xerr: Union[float, np.ndarray] = 0.0
    yerr: Union[float, np.ndarray] = 0.0
    cov: Union[float, np.ndarray] = 0.0
    class_mask: Union[None, int, np.random.Generator, np.ndarray] = None
    prob_class: np.ndarray = None  # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True

    @validator("x", pre=True)
    def x_is_1d(cls, v):
        assert len(v.shape) == 1
        return v

    @validator("y", "xerr", "yerr", "cov", pre=True, always=True)
    def values_same_length_as_x(cls, v, values):
        if isinstance(v, float):
            v = np.ones_like(values["x"]) * v
        if v.shape != values["x"].shape:
            raise ValidationError(
                f"Data must have same shape. {values['x'].shape}, {v.shape}."
            )
        return v

    @validator("class_mask", pre=True, always=True)
    def check_class_mask(cls, v, values):
        if v is None or isinstance(v, int):
            rng = np.random.default_rng(v)
            v = rng.integers(0, values["n_classes"], len(values["x"]), "u1")
        elif isinstance(v, np.random.Generator):
            v = v.integers(0, values["n_classes"], len(values["x"]), "u1")

        if v.shape != values["x"].shape:
            raise ValidationError(
                f"Data must have same shape. {values['x'].shape}, {v.shape}."
            )

        assert (
            v.min() >= 0
        ), f"class_mask must not contain negative values. Got {v.min()}."
        assert (
            v.max() < values["n_classes"]
        ), f"class_mask must not contain values greter than n_classes. Got {v.max()}."

        return v

    @validator("prob_class", pre=True, always=True)
    def check_prob_class(cls, v, values):
        if v is None:
            v = np.array(
                [np.mean(values["class_mask"] == i) for i in range(values["n_classes"])]
            )
        assert isinstance(v, np.ndarray), f"Expected np.ndarray, got {type(v)}."
        assert (
            len(v) == values["n_classes"]
        ), f"Expected {values['n_classes']} probabilities, got {len(v)}."
        assert v.min() >= 0.0, f"Probabilities cannot be negative. Got {v.min()}."
        assert v.max() <= 1.0, f"Probabilities cannot be greater than 1. got {v.max()}."
        assert (
            abs(1.0 - v.sum()) <= 1e-6
        ), f"Probabilities should sum to 1, but sum to {v.sum()}."
        return v

    @classmethod
    def from_region_dataframe(
        cls, regions: pd.DataFrame, n_classes: int, seed: Optional[int] = None
    ) -> BcesEM:
        """Initialises a BcesEM object from a regions dataframe.

        Parameters
        ----------
        regions : pd.DataFrame
            DataFrame containing wingbeat-extracted polylines. SNR threshold should
            already have been applied. Must contain columns "best_peak", "et_up",
            "et_dn", and "blur_length".
        n_classes : int
            Number of target classes.
        seed : Optional[int]
            Sets the seed for initialisation of class_mask.

        Returns
        -------
        bces_em : BcesEM
            Model to fit by calling bces_em.fit().
        """
        x = (
            np.array(
                regions["best_peak"] * regions["et_up"]
                + regions["best_peak"] * regions["et_up"]
            )
            / 2
        )
        xerr = (
            np.abs(
                np.array(
                    regions["best_peak"] * regions["et_up"]
                    - regions["best_peak"] * regions["et_up"]
                )
            )
            / 2
        )
        y = np.array(regions["blur_length"])
        yerr = np.zeros_like(y)
        cov = np.zeros_like(y)

        return BcesEM(
            x=x,
            y=y,
            n_classes=n_classes,
            xerr=xerr,
            yerr=yerr,
            cov=cov,
            class_mask=seed,
        )

    def fit_bces(self):
        """Fits BCES linear regressions to the data, subdivided into self.n_classes
        classes by self.class_mask.

        Returns
        -------
        estimates : list[BcesResult]
            list of self.n_classes BcesResult instances.
        err : np.ndarray
            Array of weighted errors for each measurement from each regression line. Has
            shape (n_classes, len(x)).
        """
        assert isinstance(self.class_mask, np.ndarray)
        estimates: list[BcesResult] = []
        err = np.zeros((self.n_classes, len(self.x)))
        for class_id in range(self.n_classes):
            mask = self.class_mask == class_id
            gradient, y_intercept, gradient_err, y_intercept_err, cov_xy = (
                e[0]
                for e in bces.bces(
                    self.x[mask],
                    self.xerr[mask],
                    self.y[mask],
                    self.yerr[mask],
                    self.cov[mask],
                )
            )
            estimates.append(
                BcesResult(
                    gradient=gradient,
                    y_intercept=y_intercept,
                    gradient_stderr=gradient_err,
                    y_intercept_stderr=y_intercept_err,
                    cov_xy=cov_xy,
                )
            )

            err[class_id, :] = (
                y_intercept + gradient * self.x - self.y
            ) ** 2 / self.prob_class[class_id]

        return estimates, err

    def fit(self, max_iterations: int = 100):
        """Performs expectation-maximisation to fit data to self.n_classes BCES linear
        regression lines. Modifies self.class_mask and self.prob_class.

        Parameters
        ----------
        max_iterations : int
            Maximum number of EM-iterations.

        Returns
        -------
        estimates : list[BcesResult]
            list of self.n_classes BcesResult instances.
        """
        assert isinstance(self.class_mask, np.ndarray)
        for i in range(max_iterations):
            # Fit BCES regressions based on existing class data
            estimates, err = self.fit_bces()

            # Update self.class_mask
            class_mask = np.argmin(err, axis=0)

            # Check if converged
            if (class_mask == self.class_mask).all():
                break

            # Reinitialise for next iteration
            self.class_mask[:] = class_mask
            self.prob_class[:] = np.array(
                [np.mean(self.class_mask == i) for i in range(self.n_classes)]
            )

        return estimates


@total_ordering
class WeightedGaussian(BaseModel):
    mean: float
    std: float
    weight: float = 1.0

    def __lt__(self, other: WeightedGaussian):
        return (self.mean, self.std, self.weight) < (
            other.mean,
            other.std,
            other.weight,
        )


class GMM(BaseModel):
    x: np.ndarray
    n_classes: PositiveInt
    seed: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("x")
    def x_is_1d(cls, v):
        assert len(v.shape) == 1, "Data must be 1-D. Got array with shape {v.shape}."
        return v

    @classmethod
    def log10_from_region_dataframe(
        cls, regions: pd.DataFrame, n_classes: int, seed: Optional[int] = None
    ) -> GMM:
        """Initialises a GMM object from a regions dataframe, taking the log10 values
        of preliminary wingbeat frequency.

        Parameters
        ----------
        regions : pd.DataFrame
            DataFrame containing wingbeat-extracted polylines. SNR threshold should
            already have been applied. Must contain columns "wb_freq_down" and
            "wb_freq_up".
        n_classes : int
            Number of target classes.
        seed : Optional[int]
            Sets the seed for the EM algorithm.

        Returns
        -------
        gmm : GMM
            Model to fit by calling gmm.fit().
        """
        return GMM(
            x=np.log10(
                np.concatenate([regions["wb_freq_down"], regions["wb_freq_up"]])
            ),
            n_classes=n_classes,
            seed=seed,
        )

    def fit(self) -> list[WeightedGaussian]:
        gmm = sklearn.mixture.GaussianMixture(
            n_components=self.n_classes, random_state=self.seed
        )
        gmm.fit(self.x.reshape(-1, 1))
        components: list[WeightedGaussian] = []
        for i in range(self.n_classes):
            components.append(
                WeightedGaussian(
                    mean=gmm.means_[i][0],
                    std=sqrt(gmm.covariances_[i][0]),
                    weight=gmm.weights_[i],
                )
            )

        return components
