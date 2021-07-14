from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import arange, cat, float32, meshgrid, tensor, Tensor, zeros

from camfi.data import (
    PolylineShapeAttributes,
    ViaRegionAttributes,
    ViaRegion,
    ViaMetadata,
)

from camfi.util import cache


def autocorrelation(roi: Tensor, max_pixel_period: PositiveInt) -> Tensor:
    """Calculate the autocorrelation along axis 1 of roi. Will run entirely on the
    device specified by roi.device, and is optimised for running on the GPU.

    Parameters
    ----------
    roi : Tensor
        tensor of shape (width, blur_length)
    max_pixel_period : Optional[PositiveInt]
        maximum period to consider (choosing a smaller number will increase the speed of
        execution if running on cpu, and decrease memory consumption regardless of
        which device it's running on). If None, calculated as roi.shape[1] // 2

    Returns
    -------
    Tensor
        of shape (max_pixel_period,). Contains the the autocorrelation with integer
        step-sizes.

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

    index = arange(roi.shape[1] - max_pixel_period, device=roi.device)
    step, origin = meshgrid(index, index)
    step = step + origin

    autocovariance = (mean_diff[:, origin] * mean_diff[:, step]).mean(axis=0)
    denominator = std[origin] * std[step]

    return (autocovariance / denominator).nan_to_num().mean(axis=1)


def find_best_peak(values: Tensor) -> Tuple[Optional[PositiveInt], Optional[float]]:
    """Takes a Tensor of values (with 1 dimension), and finds the index of the best peak
    If peak finding fails, (None, None) is returned.

    Parameters
    ----------
    values : Tensor

    Returns
    -------
    best_peak : Optional[PositiveInt]
        Index of best peak
    snr : Optional[float]
        Score of peak

    Examples
    --------
    >>> t = zeros(100)
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
    >>> find_best_peak(zeros(10))
    (None, None)
    """
    peaks = ((values > values.roll(-1)) & (values > values.roll(1))).nonzero(
        as_tuple=True
    )[0][1:]
    sorted_peaks = peaks[values[peaks].argsort(descending=True)]

    best_peak: Optional[PositiveInt] = None
    snr: Optional[float] = None
    snrs: List[float] = []

    for peak in sorted_peaks:
        # Find snr
        trough_values = cat(
            [
                values[peak // 3 : (peak * 3) // 4],
                values[(peak * 5) // 3 : (peak * 7) // 4],
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


class WingbeatExtractor(BaseModel):
    metadata: ViaMetadata
    root: Path
    line_rate: PositiveFloat

    # Optional parameters to process_blur
    scan_distance: PositiveInt = 50
    max_pixel_period: Optional[PositiveInt] = None

    # Optional extra parameters for when getting exif metadata
    force_load_exif_metadata: bool = False
    location: Optional[str] = None
    datetime_corrector: Optional[Callable[[datetime], datetime]] = None

    # image and exposure_time may require expensive IO operations, so should only happen
    # once each, if at all. They should also be treated as immutable for the life of the
    # WingbeatExtractor instance. Hence, the property and cache decorators.
    @property  # type: ignore[misc]
    @cache
    def image(self) -> Tensor:
        """Loads image from file and converts it to a greyscale tensor"""
        return self.metadata.read_image(root=self.root).mean(axis=-3)  # type: ignore[call-overload]

    @property  # type: ignore[misc]
    @cache
    def exposure_time(self) -> PositiveFloat:
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

    def process_blur(
        self, polyline: PolylineShapeAttributes, score: Optional[float] = None
    ) -> ViaRegionAttributes:
        """Performs the camfi algorithm to takes a measurement of wingbeat frequency
        from a flying insect motion blur which has been annotated with a polyline.

        Parameters
        ----------
        polyline: PolylineShapeAttributes
            Polyline annotation following the path of the flying insect's motion blur
        score: Optional[float]
            score parameter to be passed to ViaRegionAttributes constructor (should set
            if processing an annotation which was generated automatically, so that the
            score is reflected in the output).

        Returns
        -------
        ViaRegionAttributes
            with all fields set (including score iff a value was given).
        """
        # Load region of interest. Mypy complains about self.image not being the right
        roi = polyline.extract_region_of_interest(self.image, self.scan_distance)

        # Infer max_pixel_period if not set
        max_pixel_period = roi.shape[1] // 2
        if self.max_pixel_period is not None:
            max_pixel_period = min(max_pixel_period, self.max_pixel_period)

        # Calculate autocorrelation
        mean_autocorrelation = autocorrelation(roi, max_pixel_period)

        # Find wingbeat peak
        best_peak, snr = find_best_peak(mean_autocorrelation)

        # Calculate wingbeat frequency from peak.
        # Note that due to the ambiguity in the direction of the moth's flight,
        # as well as rolling shutter, there are two possible frequency estimates,
        # with the lower frequency corresponding to an upward direction and the
        # higher frequency corresponding to a downward direction of flight with
        # respect to the camera's orientation.
        y_diff = polyline.y_diff()
        corrected_exposure_time = (
            self.exposure_time + tensor([y_diff, -y_diff]) / self.line_rate
        ).sort()
        period = [
            arange(1, max_pixel_period) * et / roi.shape[1]
            for et in corrected_exposure_time
        ]
        wb_freq_up, wb_freq_down = [1 / period[i][best_peak] for i in (0, 1)]

        return ViaRegionAttributes(
            score=score,
            blur_length=roi.shape[1],
            snr=snr,
            wb_freq_up=wb_freq_up,
            wb_freq_down=wb_freq_down,
            et_up=corrected_exposure_time[0],
            et_dn=corrected_exposure_time[1],
        )
