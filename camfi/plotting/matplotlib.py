from typing import List, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from torch import Tensor

from camfi.datamodel.via import ViaRegionAttributes
from camfi.wingbeat import WingbeatSuppFigPlotter


class MatplotlibWingbeatSuppFigPlotter(WingbeatSuppFigPlotter):
    """Implementation of WingbeatSuppFigPlotter using matplotlib."""

    def __call__(
        self,
        region_attributes: ViaRegionAttributes,
        region_of_interest: Tensor,
        mean_autocorrelation: Tensor,
    ) -> None:
        """Plot a supplementary figure of wingbeat extraction, saving the image to a
        file.

        Parameters
        ----------
        region_attributes : ViaRegionAttributes
            With fields calculated (e.g. by WingbeatExtractor.process_blur).
        region_of_interest : Tensor
            Greyscale image Tensor displaying region of interest.
        mean_autocorrelation : Tensor
            1-d Tensor with values containing autocorrrelation along axis 1 of
            region_of_interest.
        """

        fig = plt.figure()

        ax1 = fig.add_subplot(
            211,
            title=f"Linearised view of insect motion blur (moth {self.annotation_idx})",
        )
        ax1.imshow(region_of_interest.numpy())

        if region_attributes.snr is not None:
            ax2_title = (
                f"Autocorrelation along motion blur (SNR: {region_attributes.snr:.2f})"
            )
        else:
            ax2_title = f"Autocorrelation along motion blur (No peak found)"

        ax2 = fig.add_subplot(
            212,
            title=ax2_title,
            ylabel="Correlation",
            xlabel="Distance (pixel columns)",
        )
        ax2.plot(mean_autocorrelation.numpy())

        if region_attributes.best_peak is not None:
            best_peak = region_attributes.best_peak
            ax1.axvline(best_peak, c="r")
            ax2.axvline(best_peak, c="r")
            ax2.axvspan(
                best_peak // 4, (best_peak * 3) // 4, color="k", alpha=0.25, zorder=0
            )
            ax2.axvspan(
                (best_peak * 5) // 4,
                (best_peak * 7) // 4,
                color="k",
                alpha=0.25,
                zorder=0,
            )

            try:
                ymin = float(mean_autocorrelation.min())
                ymax = float(mean_autocorrelation[best_peak])
                yrange = ymin - ymax
                ymin -= yrange * 0.05
                ymax += yrange * 0.05
                ax2.set_ylim(ymin, ymax)
            except ValueError:
                pass

        filepath = self.get_filepath()
        try:
            fig.savefig(str(filepath))
        except FileNotFoundError:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath)

        plt.close(fig)


def plot_herror_bars(
    axes: plt.Axes, lower: np.ndarray, upper: np.ndarray, y: np.ndarray, **kwargs
) -> List[plt.Line2D]:
    """Plots horizontal error bars on a set of axes.

    Parameters
    ----------
    axes : plt.Axes
        Axes to plot error bars on.
    lower : np.ndarray
        Lower x-values of error bars.
    upper : np.ndarray
        Upper x-values of error bars. Should have same shape as lower.
    y : np.ndarray
        y-values. Should have same shape as upper and lower.
    **kwargs : dict
        Passed to plt.plot.
    """
    assert (
        lower.shape == upper.shape == y.shape
    ), f"Shapes do not match. Got {lower.shape}, {upper.shape}, {y.shape}."
    return axes.plot(
        np.stack((lower, upper)), np.broadcast_to(y, (2,) + y.shape), **kwargs
    )


class MatplotlibWingbeatFrequencyPlotter(BaseModel):
    polyline_regions: pd.DataFrame
    snr_thresh: float = 4.0
    figsize: Tuple[float, float] = (7.5, 5.2)
    left_border: float = 0.1
    bottom_border: float = 0.1
    snr_vs_pwf_ax_width: float = 0.4
    snr_vs_pwf_ax_height: float = 0.4
    hist_height: float = 0.2
    l_vs_pdt_spacing: float = -0.12
    snr_vs_pwf_alpha: float = 1.0
    snr_vs_pwf_abovethresh_c: str = "k"
    snr_vs_pwf_belowthresh_c: str = "grey"
    snr_thresh_line_c: str = "r"
    errorbar_lw: float = 1
    l_vs_pdt_alpha: float = 0.5

    def _init_figure(self) -> None:
        """Initialises Figure and all Axes into self.fig."""
        self.fig = plt.figure(figsize=self.figsize)
        self._init_snr_vs_pwf_ax()
        self._init_histx_ax()
        self._init_histy_ax()
        self._init_l_vs_pdt_ax()

    def _init_snr_vs_pwf_ax(self) -> None:
        """Initialises Axes into self.snr_vs_pwf_ax."""
        self.snr_vs_pwf_ax = self.fig.add_axes(
            [
                self.left_border,
                self.bottom_border,
                self.snr_vs_pwf_ax_width,
                self.snr_vs_pwf_ax_height,
            ],
            xlabel="Preliminary wingbeat frequency (Hz)",
            ylabel="SNR",
            xscale="log",
        )

    def _init_histx_ax(self) -> None:
        """Initialises Axes into self.histx_ax."""
        self.histx_ax = self.fig.add_axes(
            [
                self.left_border,
                self.bottom_border + self.snr_vs_pwf_ax_height,
                self.snr_vs_pwf_ax_width,
                self.hist_height,
            ],
            sharex=self.snr_vs_pwf_ax,
        )
        self.histx_ax.axis("off")

    def _init_histy_ax(self) -> None:
        """Initialises Axes into self.histy_ax."""
        self.histy_ax = self.fig.add_axes(
            [
                self.left_border + self.snr_vs_pwf_ax_width,
                self.bottom_border,
                self.hist_height,
                self.snr_vs_pwf_ax_height,
            ],
            sharey=self.snr_vs_pwf_ax,
        )
        self.histy_ax.axis("off")

    def _init_l_vs_pdt_ax(self) -> None:
        """Initialises Axes into self.l_vs_pdt_ax."""
        self.l_vs_pdt_ax = self.fig.add_axes(
            [
                self.left_border + self.snr_vs_pwf_ax_width + self.l_vs_pdt_spacing,
                self.bottom_border + self.snr_vs_pwf_ax_height + self.l_vs_pdt_spacing,
                1.0
                - self.left_border
                - self.snr_vs_pwf_ax_width
                - self.l_vs_pdt_spacing,
                1.0
                - self.bottom_border
                - self.snr_vs_pwf_ax_height
                - self.l_vs_pdt_spacing,
            ],
            ylabel="$L$ (pixels)",
            xlabel="$P∆t$ (pixels · s)",
        )

    def _apply_snr_thresh(self) -> None:
        """Splits self.polyline_regions into self.above_thresh and self.below_thresh."""
        above_thresh_mask = self.polyline_regions["snr"] >= self.snr_thresh
        self.above_thresh = self.polyline_regions[above_thresh_mask]
        self.below_thresh = self.polyline_regions[~above_thresh_mask]

    def _plot_snr_vs_pwf(self) -> None:
        """Plots scatter plot of SNR vs. Preliminary wingbeat frequency."""
        # Plot above-threshold data
        plot_herror_bars(
            self.snr_vs_pwf_ax,
            self.above_thresh["wb_freq_down"],
            self.above_thresh["wb_freq_up"],
            self.above_thresh["snr"],
            c=self.snr_vs_pwf_abovethresh_c,
            alpha=self.snr_vs_pwf_alpha,
            lw=errorbar_lw,
        )
        # Plot below-threshold data
        plot_herror_bars(
            self.snr_vs_pwf_ax,
            self.below_thresh["wb_freq_down"],
            self.below_thresh["wb_freq_up"],
            self.below_thresh["snr"],
            c=self.snr_vs_pwf_belowthresh_c,
            alpha=self.snr_vs_pwf_alpha,
            lw=errorbar_lw,
        )

        self.snr_vs_pwf_ax.axhline(
            self.snr_thresh, c=self.snr_thresh_line_c, zorder=0, label="SNR Threshold"
        )

    def _plot_marginal_hists(self) -> None:
        """Plots marginal histograms for SNR vs. Preliminary wingbeat frequency."""
        # Horizontal marginal
        hx, bx, p = self.histx_ax.hist(
            np.concatenate(
                [
                    self.polyline_regions["wb_freq_down"],
                    self.polyline_regions["wb_freq_up"],
                ]
            ),
            bins=np.logspace(
                np.log10(min(self.polyline_regions["wb_freq_down"])),
                np.log10(max(self.polyline_regions["wb_freq_up"])),
            ),
            facecolor=self.snr_vs_pwf_belowthresh_c,
            alpha=self.snr_vs_pwf_alpha,
        )

        hx_filt, bx_filt, p = self.histx_ax.hist(
            np.concatenate(
                [self.above_thresh["wb_freq_down"], self.above_thresh["wb_freq_up"]]
            ),
            bins=bx,
            facecolor=self.snr_vs_pwf_abovethresh_c,
            alpha=self.snr_vs_pwf_alpha,
        )

        # Vertial marginal
        # First need to pin bin edges to snr_thresh to avoid overlap
        min_snr = self.polyline_regions["snr"].min()
        max_snr = self.polyline_regions["snr"].max()
        nbins = 50
        by = np.linspace(
            min_snr - (max_snr - min_snr) / nbins,
            max_snr,
            num=nbins + 1,
        )
        by += self.snr_thresh - by[by <= self.snr_thresh][-1]

        hy, by, p = self.histy_ax.hist(
            self.polyline_regions["snr"],
            bins=by,
            orientation="horizontal",
            facecolor=self.snr_vs_pwf_belowthresh_c,
            alpha=self.snr_vs_pwf_alpha,
        )

        self.histy_ax.hist(
            self.above_thresh["snr"],
            bins=by,
            orientation="horizontal",
            facecolor=self.snr_vs_pwf_abovethresh_c,
            alpha=self.snr_vs_pwf_alpha,
        )

        # SNR threshold line should be continued into the marginal
        self.histy_ax.axhline(
            self.snr_thresh, c=self.snr_thresh_line_c, zorder=1, label="SNR Threshold"
        )

    def _plot_l_vs_pdt(self) -> None:
        """Plot blur length vs. pixel-period * ∆t for above thresh data only."""
        plot_herror_bars(
            self.l_vs_pdt_ax,
            self.above_thresh["best_peak"] * self.above_thresh["et_up"],
            self.above_thresh["best_peak"] * self.above_thresh["et_dn"],
            self.above_thresh["blur_length"],
            c="k",
            alpha=self.l_vs_pdt_alpha,
            lw=self.errorbar_lw,
        )
