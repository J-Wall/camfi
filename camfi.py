#!/usr/bin/env python

from datetime import datetime as dt
import itertools
import json
from multiprocessing import Pool
import os

import fire
import imageio
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform


def extract_rois(regions, image_path, scan_distance):
    """
    Args:
        regions         dict containing annotation information
        image_path      path to annotated image
        scan_distance   int half-width of rois for motion blurs
    Returns:
        capture_time    datetime of image capture from image metadata
        exposure_time   float of exposure time of image in seconds
        blurs           list of tuples (roi, y_diff)
            roi             rotated and cropped regions of interest
            y_diff          number of rows the blur spans
    """
    # Load image and get exposure time from metadata
    reader = imageio.get_reader(image_path)
    img = reader.get_data(0)
    reader.close()
    capture_time = dt.strptime(
        img.meta["EXIF_MAIN"]["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S"
    )
    exposure_tup = img.meta["EXIF_MAIN"]["ExposureTime"]
    exposure_time = exposure_tup[0] / exposure_tup[1]

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
    Args:
        roi             rotated and cropped regions of interest
        exposure_time   float of exposure time of image in seconds
        y_diff          number of rows the blur spans
        line_rate       the line rate of the rolling shutter
        max_dist        maximum number of columns to calculate autocorrelation
                        over. Defaults to a half of the length of the image
    Returns:
        spec_density    Array of y-values on spectrogram
        snr             float, signal to noise ratio
        up_freq         Wingbeat frequency estimate, assuming upward motion
        down_freq       Wingbeat frequency estimate, assuming downward motion
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


def main(
    annotation_file,
    line_rate=np.inf,
    scan_distance=100,
    max_dist=None,
    processes=1,
    supplementary_figures=None,
):
    """
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
    print(
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

    with open(annotation_file) as f:
        annotations = json.load(f)

    pool = Pool(processes)

    for results in pool.imap(
        process_annotations,
        zip(
            annotations["_via_img_metadata"].values(),
            itertools.repeat(scan_distance),
            itertools.repeat(line_rate),
            itertools.repeat(max_dist),
            itertools.repeat(supplementary_figures),
        ),
        10,
    ):
        for result in results:
            if result[2] is not np.nan:  # Check if snr is not np.nan
                print("\t".join(str(val) for val in result))


if __name__ == "__main__":
    fire.Fire(main)
