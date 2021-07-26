from typing import Any, Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
import scipy.stats
from torch import Tensor

from camfi.datamodel.via import ViaRegionAttributes
from camfi.wingbeat import WingbeatSuppFigPlotter, BcesResult, WeightedGaussian


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
    class_mask: np.ndarray = None  # type: ignore[assignment]
    gmm_results: Optional[List[WeightedGaussian]] = None
    bces_results: Optional[List[BcesResult]] = None
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
    gmm_plot_range_stdevs: float = 4.0
    gmm_lw: float = 3
    l_vs_pdt_alpha: float = 0.5
    class_colours: List[str] = [
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:red",
        "tab:purple",
        "k",
    ]
    snr_vs_pwf_title: str = " (a)"
    snr_vs_pwf_title_y: float = 0.88
    l_vs_pdt_title: str = " (b)"
    l_vs_pdt_title_y: float = 0.88
    title_font_dict: Dict[str, Any] = {"fontweight": "bold"}
    fig: plt.Figure = None  # type: ignore[assignment]
    snr_vs_pwf_ax: plt.Axes = None  # type: ignore[assignment]
    histx_ax: plt.Axes = None  # type: ignore[assignment]
    histy_ax: plt.Axes = None  # type: ignore[assignment]
    l_vs_pdt_ax: plt.Axes = None  # type: ignore[assignment]
    above_thresh: pd.DataFrame = None  # type: ignore[assignment]
    below_thresh: pd.DataFrame = None  # type: ignore[assignment]

    class Config:
        arbitrary_types_allowed = True

    @validator("class_mask", pre=True, always=True)
    def class_mask_same_length_as_abovethresh(cls, v, values):
        n_abovethresh = np.count_nonzero(
            values["polyline_regions"]["snr"] >= values["snr_thresh"]
        )
        if v is None:
            v = np.zeros((n_abovethresh,), dtype="i4") - 1
        assert (
            len(v) == n_abovethresh
        ), "class_mask must have one value for each above-snr-thresh datapoint."
        return v

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
            lw=self.errorbar_lw,
        )
        # Plot below-threshold data
        plot_herror_bars(
            self.snr_vs_pwf_ax,
            self.below_thresh["wb_freq_down"],
            self.below_thresh["wb_freq_up"],
            self.below_thresh["snr"],
            c=self.snr_vs_pwf_belowthresh_c,
            alpha=self.snr_vs_pwf_alpha,
            lw=self.errorbar_lw,
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

        # Plot GMM
        if self.gmm_results is not None:
            scaling = np.mean(hx_filt * (bx_filt[1:] - bx_filt[:-1])) / 2
            for class_i in range(len(self.gmm_results)):
                pdf_x = np.logspace(
                    self.gmm_results[class_i].mean
                    - self.gmm_results[class_i].std * self.gmm_plot_range_stdevs,
                    self.gmm_results[class_i].mean
                    + self.gmm_results[class_i].std * self.gmm_plot_range_stdevs,
                    num=100,
                )
                self.histx_ax.plot(
                    pdf_x,
                    scaling
                    * self.gmm_results[class_i].weight
                    * scipy.stats.norm.pdf(
                        np.log10(pdf_x),
                        loc=self.gmm_results[class_i].mean,
                        scale=self.gmm_results[class_i].std,
                    ),
                    c=self.class_colours[class_i],
                    linewidth=self.gmm_lw,
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
        for class_i in np.unique(self.class_mask):
            mask = self.class_mask == class_i
            plot_herror_bars(
                self.l_vs_pdt_ax,
                self.above_thresh["best_peak"][mask] * self.above_thresh["et_up"][mask],
                self.above_thresh["best_peak"][mask] * self.above_thresh["et_dn"][mask],
                self.above_thresh["blur_length"][mask],
                c=self.class_colours[class_i],
                alpha=self.l_vs_pdt_alpha,
                lw=self.errorbar_lw,
            )

    def _plot_l_vs_pdt_regressions(self) -> None:
        """Plots regression lines."""
        if self.bces_results is not None:
            for i in range(len(self.bces_results)):
                xmax = (
                    self.above_thresh["best_peak"] * self.above_thresh["et_dn"]
                ).max()
                self.l_vs_pdt_ax.plot(
                    [0, xmax],
                    [
                        self.bces_results[i].y_intercept,
                        self.bces_results[i].y_intercept
                        + self.bces_results[i].gradient * xmax,
                    ],
                    c=self.class_colours[i],
                )

    def _add_titles(self) -> None:
        """Adds titles to subfigures."""
        title_y = 0.88
        a_title = self.snr_vs_pwf_ax.set_title(
            self.snr_vs_pwf_title,
            fontdict=self.title_font_dict,
            loc="left",
            y=self.snr_vs_pwf_title_y,
        )
        b_title = self.l_vs_pdt_ax.set_title(
            self.l_vs_pdt_title,
            fontdict=self.title_font_dict,
            loc="left",
            y=self.l_vs_pdt_title_y,
        )

    def plot(self) -> plt.Figure:
        """Produces plots."""
        # Initialise axes
        self._init_figure()

        # Apply snr threshold
        self._apply_snr_thresh()

        # Plot
        self._plot_snr_vs_pwf()
        self._plot_marginal_hists()
        self._plot_l_vs_pdt()
        self._plot_l_vs_pdt_regressions()
        self._add_titles()

        return self.fig
