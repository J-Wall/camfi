from matplotlib import pyplot as plt
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
