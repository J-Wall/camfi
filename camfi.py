#!/usr/bin/env python

from datetime import datetime as dt
import itertools
import json
from math import inf
from multiprocessing import Pool
import os
import sys
from zipfile import ZipFile

import exif
import fire
import imageio
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform
from tqdm import tqdm


def extract_rois(regions, image_path, scan_distance):
    """
    Extracts regions of interest (ROIs) from an image using polyline annotations

    Parameters
    ----------

    regions : dict
        contains information fopr each annotation

    image_path : str
        path to annotated image

    scan_distance : int
        half-width of rois for motion blurs

    Returns
    -------

    capture_time : datetime
        image capture from image metadata

    exposure_time : float
       exposure time of image in seconds

    blurs : list of tuples [(roi, y_diff), ...]
        roi: rotated and cropped regions of interest
        y_diff: number of rows the blur spans
    """
    # Load image and get exposure time from metadata
    reader = imageio.get_reader(image_path)
    img = reader.get_data(0)
    reader.close()
    capture_time = dt.strptime(
        img.meta["EXIF_MAIN"]["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
    )
    exposure_meta = img.meta["EXIF_MAIN"]["ExposureTime"]
    if isinstance(exposure_meta, tuple):
        exposure_time = exposure_meta[0] / exposure_meta[1]
    else:
        exposure_time = float(img.meta["EXIF_MAIN"]["ExposureTime"])

    #
    xs, ys = [], []
    for a in range(len(regions)):
        if regions[a]["shape_attributes"]["name"] == "polyline":
            x = regions[a]["shape_attributes"]["all_points_x"]
            y = regions[a]["shape_attributes"]["all_points_y"]
            xs.append(x)
            ys.append(y)

    blurs = []
    for x, y in zip(xs, ys):
        sections = []
        tot_length = 0
        for i in range(len(x) - 1):
            # Calculate angle of section
            try:
                perp_grad = (x[i] - x[i + 1]) / (y[i + 1] - y[i])
            except ZeroDivisionError:
                perp_grad = np.inf
            rotation = np.arctan2(y[i + 1] - y[i], x[i + 1] - x[i])

            # Calculate upper corner of ROI
            if rotation > 0:
                trans_x = x[i] + scan_distance / np.sqrt(perp_grad ** 2 + 1)
                if perp_grad == np.inf:
                    trans_y = y[i] + scan_distance
                else:
                    trans_y = y[i] + scan_distance * perp_grad / np.sqrt(
                        perp_grad ** 2 + 1
                    )
            else:
                trans_x = x[i] - scan_distance / np.sqrt(perp_grad ** 2 + 1)
                if perp_grad == np.inf:
                    trans_y = y[i] - scan_distance
                else:
                    trans_y = y[i] - scan_distance * perp_grad / np.sqrt(
                        perp_grad ** 2 + 1
                    )

            # Rotate and translate image
            transform_matrix = transform.EuclideanTransform(
                rotation=rotation, translation=(trans_x, trans_y)
            )
            warped_img = transform.warp(img, transform_matrix)

            # Crop rotated image to ROI
            section_length = int(
                round(np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2))
            )
            cropped_img = warped_img[: 2 * scan_distance, :section_length, :]

            sections.append(cropped_img)
            tot_length += section_length

        # Join sections to form complete ROI
        joined_img = np.hstack(sections)
        blurs.append((joined_img, abs(y[-1] - y[0])))

    return capture_time, exposure_time, blurs


def process_blur(roi, exposure_time, y_diff, line_rate, max_dist=None):
    """
    Takes a straigtened region of interest image and a y_diff value (from extract_rois)
    to measure the wingbeat frequency of the moth in the ROI.

    Parameters
    ----------

    roi : array
        rotated and cropped regions of interest

    exposure_time : float
        exposure time of image in seconds

    y_diff : int
        number of rows the blur spans

    line_rate : float or int
        the line rate of the rolling shutter

    max_dist : int
        maximum number of columns to calculate autocorrelation over. Defaults to a
        half of the length of the image

    Returns
    -------

    spec_density : array
        y-values on spectrogram

    snr : float
        signal to noise ratio

    up_freq : float
        Wingbeat frequency estimate, assuming upward motion

    down_freq : float
        Wingbeat frequency estimate, assuming downward motion
    """
    if max_dist is None:
        max_dist = roi.shape[1] // 2

    spectral_density = np.zeros((max_dist, roi.shape[1] - max_dist), dtype=np.float64)
    gr_roi = roi.mean(axis=2)  # Operate on greyscale image
    # Calculate autocorrelation
    for step in range(max_dist):
        spectral_density[step, ...] = np.mean(
            (gr_roi[:, :-max_dist] - np.mean(gr_roi[:, :-max_dist], axis=0))
            * (
                gr_roi[:, step : step - max_dist]
                - np.mean(gr_roi[:, step : step - max_dist], axis=0)
            ),
            axis=0,
        ) / (
            np.std(gr_roi[:, :-max_dist], axis=0)
            * np.std(gr_roi[:, step : step - max_dist], axis=0)
        )

    # Find wingbeat peak
    total_spectral_density = spectral_density.mean(axis=1)[1:]
    peak_idxs = np.where(
        (total_spectral_density > np.roll(total_spectral_density, -1))
        & (total_spectral_density > np.roll(total_spectral_density, 1))
    )[0][1:]
    sorted_peak_idxs = peak_idxs[np.argsort(total_spectral_density[peak_idxs])][::-1]
    try:
        snrs = []
        best_peak = sorted_peak_idxs[0]  # Temporary value
        for peak_idx in sorted_peak_idxs:
            # Find snr
            trough_values = np.concatenate(
                [
                    total_spectral_density[peak_idx // 3 : (peak_idx * 3) // 4],
                    total_spectral_density[(peak_idx * 5) // 3 : (peak_idx * 7) // 4],
                ]
            )
            snr = (total_spectral_density[peak_idx] - np.mean(trough_values)) / np.std(
                trough_values
            )
            if len(snrs) > 0:
                if snrs[-1] > snr:
                    snr = snrs[-1]
                    break  # Previous peak was the best peak
            snrs.append(snr)
            best_peak = peak_idx

    except (ValueError, IndexError):
        best_peak = -1
        first_trough_idx = -1
        snr = np.nan

    # Calculate wingbeat frequency from peak.
    # Note that due to the ambiguity in the direction of the moth's flight,
    # as well as rolling shutter, there are two possible frequency estimates,
    # with the lower frequency corresponding to an upward direction and the
    # higher frequency corresponding to a downward direction of flight with
    # respect to the camera's orientation.
    corrected_exposure_time = np.sort(
        exposure_time + np.array([y_diff, -y_diff]) / line_rate
    )
    period = [
        np.arange(1, max_dist) * et / roi.shape[1] for et in corrected_exposure_time
    ]
    up_freq, down_freq = [1 / period[i][best_peak] for i in (0, 1)]

    return (
        corrected_exposure_time[0],
        corrected_exposure_time[1],
        period[0],
        period[1],
        total_spectral_density,
        snr,
        up_freq,
        down_freq,
        best_peak,
    )


def make_supplementary_figure(
    file_path, annotation_idx, roi, spectral_density, best_peak, snr
):
    """
    Saves supplementary figure for the wingbeat measurement for a particular annotation.

    Parameters
    ----------

    file_path : str
        Path supplementary figure file (will be overwritten if it already exists)

    annotation_idx : int
        Index of annotation (within the image). Used in plot title.

    roi : array
        rotated and cropped regions of interest

    spectral_density : array
        y-values on spectrogram

    best_peak : int
        period of wingbeat in pixels

    snr : float
        signal-to-noise ratio of autocorrelation at best_peak
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(
        211, title=f"Linearised view of insect motion blur (moth {annotation_idx})"
    )
    ax1.imshow(roi)
    ax1.axvline(best_peak, c="r")

    ax2 = fig.add_subplot(
        212,
        title=f"Autocorrelation along motion blur (SNR: {snr:.2f})",
        ylabel="Correlation",
        xlabel="Distance (pixel columns)",
    )
    ax2.plot(spectral_density)
    ax2.axvline(best_peak, c="r")
    ax2.fill_between(
        [best_peak // 4, (best_peak * 3) // 4], 0, 1, color="k", alpha=0.25, zorder=0
    )
    ax2.fill_between(
        [(best_peak * 5) // 4, (best_peak * 7) // 4],
        0,
        1,
        color="k",
        alpha=0.25,
        zorder=0,
    )
    try:
        ax2.set_ylim(spectral_density.min() - 0.01, spectral_density[best_peak] + 0.01)
    except ValueError:
        pass

    try:
        fig.savefig(file_path)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path)

    plt.close(fig)

    return None


def process_annotations(args):
    """
    Passed to worker processes by `AnnotationUtils.extract_wingbeats`. Calls
    `extract_rois`, then `process_blur` on each region. Optionally calls
    `make_supplementary_fig`.

    Parameters
    ----------

    args : tuple

    Returns
    -------

    results : list
    """
    image_metadata, scan_distance, line_rate, max_dist, supplementary_fig = args
    results = []
    regions = image_metadata["regions"]
    if len(regions) > 0:  # Check there are annotations on the image
        image_path = image_metadata["filename"]

        capture_time, exposure_time, blurs = extract_rois(
            regions, image_path, scan_distance
        )
        annotation_idx = 0
        for roi, y_diff in blurs:
            (
                et_up,
                et_dn,
                period_up,
                period_dn,
                spec_dens,
                snr,
                wb_freq_up,
                wb_freq_dn,
                best_peak,
            ) = process_blur(roi, exposure_time, y_diff, line_rate, max_dist)

            if snr is not np.nan:
                results.append(
                    (
                        image_path,
                        capture_time,
                        annotation_idx,
                        best_peak,
                        roi.shape[1],
                        snr,
                        wb_freq_up,
                        wb_freq_dn,
                        et_up,
                        et_dn,
                        ",".join([str(pu) for pu in period_up]),
                        ",".join([str(pd) for pd in period_dn]),
                        ",".join([str(sd) for sd in spec_dens]),
                    )
                )

                if supplementary_fig is not None:
                    supp_fig_path = os.path.join(
                        supplementary_fig,
                        os.path.splitext(image_metadata["filename"])[0]
                        + f"_{annotation_idx}_wingbeat.png",
                    )
                    make_supplementary_figure(
                        supp_fig_path, annotation_idx, roi, spec_dens, best_peak, snr
                    )

                annotation_idx += 1

    return results


def get_metadata(image_file, exif_tags):
    """
    Parameters
    ----------

    image_file : str
        Path to image file

    exif_tags : list
        Metadata tags to include

    Returns
    -------

    out : dict
        Contains the metadata of the image, with all keys and values coerced to str.
    """
    with open(image_file, "rb") as img_file:
        img = exif.Image(img_file)

    out = {}
    for exif_key in exif_tags:
        out[exif_key] = str(img[exif_key])

    return out


def _get_metadata_with_key(img_tup):
    return img_tup[0], get_metadata(img_tup[1], img_tup[2])


class AnnotationUtils(object):
    """
    Provides utilities for working with camfi projects

    Parameters
    ----------

    processes : int
        number of child processes to spawn

    i: str
        path to input VIA project json file. Defaults to sys.stdin

    o: str
        path to output file. Defaults to sys.stdout
    """

    def __init__(self, processes=1, i=None, o=None):
        self.processes = processes
        self.i = i
        self.o = o

    def _output(self, out_str, mode="w"):
        """
        Used by methods on `AnnotationUtils` (rather than print) to output to file or
        sys.stdout.
        """
        if self.o is None:
            print(out_str)
        else:
            with open(self.o, mode) as f:
                print(out_str, file=f)

    def _load(self):
        """
        Used by methods on `AnnotationUtils` to load json file (or sys.stdin)

        Returns
        -------

        annotations : dict
        """
        if self.i is None:
            annotations = json.load(sys.stdin)
        else:
            with open(self.i, "r") as jf:
                annotations = json.load(jf)
        return annotations

    def add_metadata(self, *exif_tags):
        """
        Adds image (EXIF) metadata to VIA project by reading image files. Optionally
        spawns multiple processes (reading the images is usually I/O bound and can take
        some time).
        """
        pool = Pool(self.processes)

        annotation_project = self._load()
        file_attributes = set()

        img_paths = [
            (img_key, annotation_project["_via_img_metadata"][img_key]["filename"], exif_tags)
            for img_key in annotation_project["_via_img_metadata"].keys()
        ]

        for img_key, metadata_dict in tqdm(
            pool.imap_unordered(_get_metadata_with_key, img_paths, 10),
            total=len(img_paths),
            desc="Adding metadata",
            unit="img",
            smoothing=0.0,
        ):
            annotation_project["_via_img_metadata"][img_key]["file_attributes"].update(
                metadata_dict
            )
            file_attributes.update(metadata_dict)

        pool.close()
        pool.join()

        for file_attribute in file_attributes:
            annotation_project["_via_attributes"]["file"][str(file_attribute)] = dict(
                type="text", description="", default_value=""
            )

        self._output(
            json.dumps(annotation_project, separators=(",", ":"), sort_keys=True)
        )

    def remove_unannotated(self):
        """
        Removes image metadata from VIA project file for images which have no
        annotations.
        """
        annotation_project = self._load()
        new_metadata = {}

        for key, img_data in annotation_project["_via_img_metadata"].items():
            if len(img_data["regions"]) > 0:
                new_metadata[key] = img_data

        annotation_project["_via_img_metadata"] = new_metadata

        self._output(
            json.dumps(annotation_project, separators=(",", ":"), sort_keys=True)
        )

    def merge_annotations(self, *annotation_files):
        """
        Takes a list of VIA project files and merges them into one. Ignores --i in
        favour of *annotation_files.

        Parameters
        ---------

        *annotation_files
            list of VIA project json files to merge. Project and VIA settings are taken
            from the first file.
        """
        with open(annotation_files[0], "r") as jf:
            annotations = json.load(jf)

        for f in annotation_files[1:]:
            with open(f, "r") as jf:
                next_anns = json.load(jf)
            annotations["_via_img_metadata"].update(next_anns["_via_img_metadata"])

        self._output(json.dumps(annotations, separators=(",", ":"), sort_keys=True))

    def zip_images(self, **kwargs):
        """
        Makes a zip archive of all the images in the provided VIA project file.
        If --i is set, then the annotation file itself will be included in the zip file.

        Parameters
        ----------

        **kwargs
            Passed to zipfile.ZipFile
        """
        assert self.o is not None, "Must specify output zipfile using --o=[file]"

        annotations = self._load()

        with ZipFile(self.o, mode="w", **kwargs) as outzip:
            if self.i is not None:
                outzip.write(self.i)
            for img_data in tqdm(
                annotations["_via_img_metadata"].values(),
                desc="Zipping images",
                unit="img",
                smoothing=0.1,
            ):
                outzip.write(img_data["filename"])

    def filter(self, by, minimum=-inf, maximum=inf, mode="warn"):
        """
        Filters VIA annotations by enforcing a minimum and/or maximum value for a
        numerical region attribute (eg. "score" which is defined during automatic
        automatic annotation)

        Parameters
        ----------

        by : str
            The region_attributes key to filter annotations by.

        minimum : float
            The minimum value of the region attribute to pass the filter

        maximum : float
            The maximum value of the region attribute to pass the filter

        mode : str
            One of {"pass", "fail", "raise", "warn"}. Defines how annotations missing
            the `by` region attribute are handled.
                "pass": These annotations pass the filter
                "fail": These annotations are removed
                "raise": A KeyError is raised if an annotation is missing the attribute
                "warn": Like "pass" but a warning is printed to sys.stderr
        """

        def _raise(ex):
            raise ex

        def _warn(ex):
            print(
                f"Warning: Missing region_attribute '{by}' for {img_data['filename']}",
                file=sys.stderr,
            )
            return True

        mode_fn = {
            "pass": lambda x: True,
            "fail": lambda x: False,
            "raise": _raise,
            "warn": _warn,
        }[mode]

        def _annotation_passes(region):
            try:
                return minimum < float(region["region_attributes"][by]) < maximum
            except KeyError as ex:
                return mode_fn(ex)

        annotation_project = self._load()

        for img_data in annotation_project["_via_img_metadata"].values():
            img_data["regions"] = list(filter(_annotation_passes, img_data["regions"]))

        self._output(
            json.dumps(annotation_project, separators=(",", ":"), sort_keys=True)
        )

    def extract_wingbeats(
        self,
        line_rate=inf,
        scan_distance=100,
        max_dist=None,
        supplementary_figures=None,
    ):
        """
        Uses the camfi algorithm to measure the wingbeat frequency of annotated flying
        insect motion blurs in still images.

        Parameters
        ---------

        line_rate : int
            The line rate of the rolling shutter

        scan_distance : int
            Half width of analysis windows (half width of blurs)

        max_dist : int
            Maximum number of columns to calculate autocorrelation over. Defaults to a
            half of the length of the image

        processes : int
            Number of worker processes to spawn. Default 1

        supplementary_figures : str
            Directory in which to put supplementary figures (optional)
        """
        # Print header
        self._output(
            "\t".join(
                [
                    "image_path",
                    "capture_time",
                    "annotation_idx",
                    "best_peak",
                    "blur_length",
                    "snr",
                    "wb_freq_up",
                    "wb_freq_dn",
                    "et_up",
                    "et_dn",
                    "period_up",
                    "period_dn",
                    "spec_dens",
                ]
            )
        )

        annotations = self._load()

        pool = Pool(self.processes)

        tot_annotations = 0
        for results in (
            pb := tqdm(
                pool.imap(
                    process_annotations,
                    zip(
                        annotations["_via_img_metadata"].values(),
                        itertools.repeat(scan_distance),
                        itertools.repeat(line_rate),
                        itertools.repeat(max_dist),
                        itertools.repeat(supplementary_figures),
                    ),
                    5,
                ),
                desc="Processing annotations",
                total=len(annotations["_via_img_metadata"]),
                unit="img",
                smoothing=0.0,
            )
        ):
            for result in results:
                if result[2] is not np.nan:  # Check if snr is not np.nan
                    self._output("\t".join(str(val) for val in result), mode="a")
                    tot_annotations += 1
            pb.set_postfix(refresh=False, tot_annotations=tot_annotations)

        pool.close()
        pool.join()


def main():
    fire.Fire(AnnotationUtils)


if __name__ == "__main__":
    main()
