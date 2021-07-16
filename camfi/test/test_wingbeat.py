from pathlib import Path

from pytest import approx, fixture, raises
from torch import Tensor

from camfi import data, wingbeat


@fixture
def circle():
    return data.CircleShapeAttributes(cx=5.0, cy=10.0, r=20.0)


@fixture
def polyline():
    return data.PolylineShapeAttributes(
        all_points_x=[3589, 3913, 4140, 4264], all_points_y=[1107, 1095, 1083, 1079],
    )


@fixture
def via_region(polyline):
    return data.ViaRegion(
        region_attributes=data.ViaRegionAttributes(), shape_attributes=polyline
    )


@fixture
def via_region_circle(circle):
    return data.ViaRegion(
        region_attributes=data.ViaRegionAttributes(), shape_attributes=circle
    )


@fixture
def via_file_attributes():
    return data.ViaFileAttributes()


@fixture
def via_metadata(via_file_attributes, via_region):
    return data.ViaMetadata(
        file_attributes=via_file_attributes,
        filename="data/DSCF0010.JPG",
        regions=[via_region],
    )


@fixture
def wingbeat_extractor(via_metadata):
    return wingbeat.WingbeatExtractor(
        metadata=via_metadata,
        root="camfi/test",
        line_rate=9.05e04,
        location="cabramurra",
    )


class MockWingbeatSuppFigPlotter(wingbeat.WingbeatSuppFigPlotter):
    def __call__(
        self,
        region_attributes: data.ViaRegionAttributes,
        region_of_interest: Tensor,
        mean_autocorrelation: Tensor,
    ) -> None:
        self.get_filepath()
        return None


class TestWingbeatExtractor:
    def test_eq(self, wingbeat_extractor):
        assert wingbeat_extractor == wingbeat_extractor
        assert wingbeat_extractor != wingbeat_extractor.copy()

    def test_loading_image(self, wingbeat_extractor):
        image = wingbeat_extractor.image
        assert image.shape == (3456, 4608), "image should be converted to greyscale."
        assert image is wingbeat_extractor.image, "image should be cached."

    def test_loading_image_fails(self, wingbeat_extractor):
        wingbeat_extractor.root = Path("foo/bar")
        with raises(RuntimeError, match="No such file or directory"):
            wingbeat_extractor.image

    def test_loading_exposure_time(self, wingbeat_extractor):
        assert wingbeat_extractor.exposure_time == approx(1 / 9)
        wingbeat_extractor.metadata.file_attributes.exposure_time = 0.5
        assert wingbeat_extractor.exposure_time == approx(
            1 / 9
        ), "Value should be cached"

    def test_exposure_time_preloaded(self, wingbeat_extractor):
        wingbeat_extractor.metadata.file_attributes.exposure_time = 0.5
        assert (
            wingbeat_extractor.exposure_time == 0.5
        ), "Use pre-existing value if available"

    def test_exposure_time_forced(self, wingbeat_extractor):
        wingbeat_extractor.metadata.file_attributes.exposure_time = 0.5
        wingbeat_extractor.force_load_exif_metadata = True
        assert wingbeat_extractor.exposure_time == approx(
            1 / 9
        ), "Value should be force-loaded"

    def test_process_blur(self, wingbeat_extractor, polyline):
        region_attributes = wingbeat_extractor.process_blur(polyline, score=1.0)
        # assert region_attributes == data.ViaRegionAttributes(), region_attributes
        assert region_attributes.score == 1.0
        assert region_attributes.best_peak == 165
        assert region_attributes.blur_length == polyline.length()
        assert (
            region_attributes.snr > 0.0
        ), f"Peak has negative snr ({region_attributes.snr})."
        assert (
            region_attributes.wb_freq_up > region_attributes.wb_freq_down
        ), f"Estimate of wingbeat frequency should be higher if moth is flying up"
        assert (
            region_attributes.et_up < region_attributes.et_dn
        ), f"Effective exposure time is longer if moth is flying down"

    def test_process_blur_with_max_pixel_period(self, wingbeat_extractor, polyline):
        wingbeat_extractor.max_pixel_period = 100
        region_attributes = wingbeat_extractor.process_blur(polyline)
        assert (
            region_attributes.best_peak == 90
        ), "Actual best peak shoudn't be considered"

    def test_process_blur_fails_to_find_peak(self, wingbeat_extractor, polyline):
        wingbeat_extractor.max_pixel_period = 20
        region_attributes = wingbeat_extractor.process_blur(polyline, score=1.0)
        assert region_attributes.score == 1.0, "Score is kept even when no peak found"
        assert region_attributes.best_peak is None, "No peak should be found."
        assert region_attributes.blur_length == polyline.length()
        assert region_attributes.snr is None
        assert region_attributes.wb_freq_up is None
        assert region_attributes.wb_freq_down is None
        assert region_attributes.et_up is None
        assert region_attributes.et_dn is None

    def test_process_blur_with_suppfigplotter(self, wingbeat_extractor, polyline):
        wingbeat_extractor.supplementary_figure_plotter = MockWingbeatSuppFigPlotter(
            root="foo", image_filename="bar/baz.jpg"
        )
        region_attributes = wingbeat_extractor.process_blur(polyline)
        assert wingbeat_extractor.supplementary_figure_plotter.annotation_idx == 1

    def test_process_all_blurs(
        self, wingbeat_extractor, via_region_circle, via_region, polyline
    ):
        wingbeat_extractor.metadata.regions.append(
            via_region_circle
        )  # Shouldn't be processed
        wingbeat_extractor.metadata.regions.append(
            via_region.copy()
        )  # should be processed

        # Extraction should not modify the regions, just replace them
        via_region_circle = via_region_circle.copy()
        via_region = via_region.copy()

        # Expected
        region_attributes = wingbeat_extractor.process_blur(polyline)

        # Observed
        wingbeat_extractor.extract_wingbeats()  # Operates in place

        assert (
            wingbeat_extractor.metadata.regions[0].region_attributes
            == region_attributes
        )
        assert (
            wingbeat_extractor.metadata.regions[2].region_attributes
            == region_attributes
        )

        assert wingbeat_extractor.metadata.regions[1] == via_region_circle
