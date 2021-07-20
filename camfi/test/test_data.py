import bz2
from pathlib import Path
import random
from typing import Dict, Tuple

from pydantic import ValidationError
from pytest import approx, fixture, raises
from torch import float32, tensor, Tensor, zeros

from camfi import util, transform
from camfi.datamodel.autoannotation import (
    CamfiDataset,
    MaskMaker,
    Target,
    Prediction,
    ImageTransform,
)
from camfi.datamodel.geometry import (
    BoundingBox,
    PointShapeAttributes,
    CircleShapeAttributes,
    PolylineShapeAttributes,
)
from camfi.datamodel.locationtime import LocationTime
from camfi.datamodel.via import (
    ViaFileAttributes,
    ViaMetadata,
    ViaRegion,
    ViaRegionAttributes,
    ViaProject,
)


@fixture(params=[0.0, 50.0, 100.0, 150.0])
def cx(request):
    return request.param


@fixture(params=[0.0, 50.0, 100.0, 150.0])
def cy(request):
    return request.param


@fixture(params=[0.0, 50.0, 100.0, 150.0])
def r(request):
    return request.param


@fixture
def point_shape_attributes(cx, cy):
    return PointShapeAttributes(cx=cx, cy=cy)


@fixture
def circle_shape_attributes(cx, cy, r):
    return CircleShapeAttributes(cx=cx, cy=cy, r=r)


@fixture(
    params=[
        [0.0, 1.0],
        [0.5, 1.0, 2.0, 4.0, 5.0, 8.0],
        [10.0, 50.0, 20.0, 30.0, 45.0, 50.0],
        [100.0, 75.0, 28.0, 0.0, 15.0, 20.0],
    ]
)
def all_points_x(request):
    return request.param


@fixture(
    params=[
        [0.0, 1.0],
        [0.5, 1.0, 2.0, 4.0, 5.0, 8.0],
        [10.0, 50.0, 20.0, 30.0, 45.0, 50.0],
        [100.0, 75.0, 28.0, 0.0, 15.0, 20.0],
    ]
)
def all_points_y(request):
    return request.param


@fixture
def polyline_shape_attributes(all_points_x, all_points_y):
    min_size = min(len(all_points_x), len(all_points_y))
    all_points_x = all_points_x[:min_size]
    all_points_y = all_points_y[:min_size]

    return PolylineShapeAttributes(all_points_x=all_points_x, all_points_y=all_points_y)


@fixture(params=[None, 1, 5])
def mask_maker(request):
    return MaskMaker(shape=(100, 120), mask_dilate=request.param)


@fixture(params=[(0, 0, 1, 1), (10, 15, 12, 20)])
def bounding_box(request):
    x0, y0, x1, y1 = request.param
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


@fixture(params=[(0, 1, 1, 0), (1, 0, 0, 1), (0, 1, 1, 1), (1, 0, 1, 1)])
def invalid_bounding_box_params(request):
    return request.param


@fixture(params=[0, 10, 100])
def margin(request):
    return request.param


@fixture(params=[(50, 50), (50, 200), (200, 50)])
def shape(request):
    return request.param


@fixture
def boxes():
    bounding_box_params = [(0, 0, 1, 1), (0, 1, 1, 2)]
    return [
        BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
        for x0, y0, x1, y1 in bounding_box_params
    ]


@fixture
def labels():
    return [1, 2]


@fixture
def image_id():
    return 0


@fixture
def masks():
    return [zeros((2, 3)), tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]


@fixture
def scores():
    return [0.0, 1.0]


@fixture
def target(boxes, labels, masks, image_id):
    return Target(boxes=boxes, labels=labels, masks=masks, image_id=image_id)


@fixture
def prediction(boxes, labels, masks, scores):
    return Prediction(boxes=boxes, labels=labels, masks=masks, scores=scores)


@fixture
def target_dict(target):
    return target.to_tensor_dict()


@fixture
def prediction_dict(prediction):
    return prediction.to_tensor_dict()


@fixture
def via_metadata():
    regions = [
        ViaRegion(
            region_attributes=ViaRegionAttributes(),
            shape_attributes=PointShapeAttributes(cx=1, cy=2),
        ),
        ViaRegion(
            region_attributes=ViaRegionAttributes(),
            shape_attributes=CircleShapeAttributes(cx=5, cy=10, r=2),
        ),
        ViaRegion(
            region_attributes=ViaRegionAttributes(),
            shape_attributes=PolylineShapeAttributes(
                all_points_x=[0, 1, 5], all_points_y=[2, 3, 7]
            ),
        ),
    ]
    return ViaMetadata(
        file_attributes=ViaFileAttributes(), filename="foo/bar.jpg", regions=regions,
    )


@fixture
def via_project(scope="module"):
    with open("camfi/test/data/sample_project_no_metadata.json") as f:
        via_project_raw = f.read()

    return ViaProject.parse_raw(via_project_raw)


class TestPointShapeAttributes:
    def test_get_bounding_box(self, point_shape_attributes):
        bounding_box = point_shape_attributes.get_bounding_box()
        assert (
            point_shape_attributes.cx >= bounding_box.x0
        ), f"{point_shape_attributes!r} is not bound by {bounding_box!r}"
        assert (
            point_shape_attributes.cy >= bounding_box.y0
        ), f"{point_shape_attributes!r} is not bound by {bounding_box!r}"
        assert (
            point_shape_attributes.cx < bounding_box.x1
        ), f"{point_shape_attributes!r} is not bound by {bounding_box!r}"
        assert (
            point_shape_attributes.cy < bounding_box.y1
        ), f"{point_shape_attributes!r} is not bound by {bounding_box!r}"


class TestCircleShapeAttributes:
    def test_as_point(self, circle_shape_attributes):
        point = circle_shape_attributes.as_point()
        assert point.cx == circle_shape_attributes.cx
        assert point.cy == circle_shape_attributes.cy

    def test_get_bounding_box(self, circle_shape_attributes):
        bounding_box = circle_shape_attributes.get_bounding_box()
        assert (
            circle_shape_attributes.cx >= bounding_box.x0
        ), f"{circle_shape_attributes!r} is not bound by {bounding_box!r}"
        assert (
            circle_shape_attributes.cy >= bounding_box.y0
        ), f"{circle_shape_attributes!r} is not bound by {bounding_box!r}"
        assert (
            circle_shape_attributes.cx < bounding_box.x1
        ), f"{circle_shape_attributes!r} is not bound by {bounding_box}"
        assert (
            circle_shape_attributes.cy < bounding_box.y1
        ), f"{circle_shape_attributes!r} is not bound by {bounding_box!r}"


class TestPolylineShapeAttributes:
    def test_validator(self, all_points_x, all_points_y):
        if len(all_points_x) == len(all_points_y):
            shape = PolylineShapeAttributes(
                all_points_x=all_points_x, all_points_y=all_points_y
            )
            assert shape.all_points_x == all_points_x
            assert shape.all_points_y == all_points_y
        else:
            with raises(ValidationError):
                PolylineShapeAttributes(
                    all_points_x=all_points_x, all_points_y=all_points_y
                )

    def test_as_circle(self, polyline_shape_attributes):
        points = zip(
            polyline_shape_attributes.all_points_x,
            polyline_shape_attributes.all_points_y,
        )
        x, y, r = util.smallest_enclosing_circle(points)

        circle = polyline_shape_attributes.as_circle()

        tol = 1e-3

        assert circle.cx == approx(x, abs=tol, rel=tol)
        assert circle.cy == approx(y, abs=tol, rel=tol)
        assert circle.r == approx(r, abs=tol, rel=tol)

    def test_get_bounding_box(self, polyline_shape_attributes):
        bounding_box = polyline_shape_attributes.get_bounding_box()
        for x in polyline_shape_attributes.all_points_x:
            assert (
                x >= bounding_box.x0
            ), f"{polyline_shape_attributes!r} is not bound by {bounding_box!r}"
            assert (
                x < bounding_box.x1
            ), f"{polyline_shape_attributes!r} is not bound by {bounding_box!r}"

        for y in polyline_shape_attributes.all_points_y:
            assert (
                y >= bounding_box.y0
            ), f"{polyline_shape_attributes!r} is not bound by {bounding_box!r}"
            assert (
                y < bounding_box.y1
            ), f"{polyline_shape_attributes!r} is not bound by {bounding_box!r}"

    def test_extract_region_of_interest_gets_right_shape(self):
        random.seed(1234567890, version=2)  # Want arbitrary values, not random ones
        h, w = 100, 200
        n_segments = 10
        scan_distance = 10
        polyline = PolylineShapeAttributes(
            all_points_x=[random.uniform(0.0, w) for _ in range(n_segments + 1)],
            all_points_y=[random.uniform(0.0, h) for _ in range(n_segments + 1)],
        )
        image = zeros(h, w)
        roi = polyline.extract_region_of_interest(image, scan_distance)
        expected_shape = (scan_distance * 2 - 1, int(round(polyline.length())))
        assert roi.shape == expected_shape, f"{roi.shape} != {expected_shape}"


class TestViaRegion:
    def test_get_bounding_box(self, point_shape_attributes):
        region = ViaRegion(
            region_attributes=ViaRegionAttributes(),
            shape_attributes=point_shape_attributes,
        )
        assert region.get_bounding_box() == point_shape_attributes.get_bounding_box()


class TestViaMetadata:
    def test_get_bounding_boxes(self, via_metadata):
        bounding_boxes = via_metadata.get_bounding_boxes()
        assert len(bounding_boxes) == len(via_metadata.regions)
        for item in bounding_boxes:
            assert isinstance(item, BoundingBox)

    def test_get_labels(self, via_metadata):
        labels = via_metadata.get_labels()
        assert len(labels) == len(via_metadata.regions)
        for item in labels:
            assert isinstance(item, int)
            assert item > 0


class TestViaProject:
    def test_parse_empty_file(self):
        with open("camfi/test/data/empty_project.json") as f:
            via_project_raw = f.read()

        ViaProject.parse_raw(via_project_raw)

    def test_parse_no_annotations(self):
        with open("camfi/test/data/sample_project_no_annotations.json") as f:
            via_project_raw = f.read()

        ViaProject.parse_raw(via_project_raw)

    def test_parse_no_metadata(self):
        with open("camfi/test/data/sample_project_no_metadata.json") as f:
            via_project_raw = f.read()

        ViaProject.parse_raw(via_project_raw)

    def test_parse_bz2(self):
        with bz2.open("examples/data/cabramurra_all_annotations.json.bz2") as f:
            via_project_raw = f.read()

        ViaProject.parse_raw(via_project_raw)

    def test_parse_fails(self):
        with open("camfi/test/data/sample_project_no_metadata_malformed.json") as f:
            via_project_raw = f.read()

        with raises(ValidationError):
            ViaProject.parse_raw(via_project_raw)


class TestLocationTimeZone:
    def test_all_offset_aware_or_naive(self):
        assert LocationTime(
            camera_start_time="2021-07-15T14:00",
            actual_start_time=None,
            camera_end_time=None,
            actual_end_time=None,
        ) == LocationTime(camera_start_time="2021-07-15T14:00")
        assert LocationTime(
            camera_start_time="2021-07-15T14:00+10",
            actual_start_time=None,
            camera_end_time=None,
            actual_end_time=None,
        ) == LocationTime(camera_start_time="2021-07-15T14:00+10")
        LocationTime(
            camera_start_time="2021-07-15T14:00",
            actual_start_time="2021-07-15T14:00",
            camera_end_time="2021-07-15T15:00",
            actual_end_time="2021-07-15T15:00",
        )
        LocationTime(
            camera_start_time="2021-07-15T14:00+10",
            actual_start_time="2021-07-15T14:00+10",
            camera_end_time="2021-07-15T15:00+11",  # Mixed offsets are allowed
            actual_end_time="2021-07-15T15:00+11",
        )
        with raises(ValidationError):
            LocationTime(
                camera_start_time="2021-07-15T14:00+10",
                actual_start_time="2021-07-15T14:00",  # Mixed offset-awareness banned
            )
        with raises(ValidationError):
            LocationTime(
                camera_start_time="2021-07-15T14:00",
                actual_start_time="2021-07-15T14:00+10",  # Mixed offset-awareness banned
            )


class TestMaskMaker:
    def test_get_mask_point(self, mask_maker, point_shape_attributes):
        if (
            int(point_shape_attributes.cx) < mask_maker.shape[1]
            and int(point_shape_attributes.cy) < mask_maker.shape[0]
        ):
            mask = mask_maker.get_mask(point_shape_attributes)
            assert (
                mask.shape == mask_maker.shape
            ), f"mask shape {mask.shape} doesn't match mask_maker {mask_maker.shape}"
            assert (
                mask.min() >= 0.0 and mask.max() <= 1.0
            ), f"mask value range ({mask.min()}, {mask.max()}) not within (0.0, 1.0)"
            assert (
                mask[int(point_shape_attributes.cy), int(point_shape_attributes.cx)]
                > 0.0
            ), "Mask should be positive at point"
        else:
            with raises(ValueError):
                mask_maker.get_mask(point_shape_attributes)

    def test_get_mask_circle(self, mask_maker, circle_shape_attributes):
        if (
            int(circle_shape_attributes.cx) < mask_maker.shape[1]
            and int(circle_shape_attributes.cy) < mask_maker.shape[0]
        ):
            mask = mask_maker.get_mask(circle_shape_attributes)
            assert (
                mask.shape == mask_maker.shape
            ), f"mask shape {mask.shape} doesn't match mask_maker {mask_maker.shape}"
            assert (
                mask.min() >= 0.0 and mask.max() <= 1.0
            ), f"mask value range ({mask.min()}, {mask.max()}) not within (0.0, 1.0)"
            assert (
                mask[int(circle_shape_attributes.cy), int(circle_shape_attributes.cx)]
                > 0.0
            ), "Mask should be positive at centre"
        else:
            with raises(ValueError):
                mask_maker.get_mask(circle_shape_attributes)

    def test_get_mask_polyline(self, mask_maker, polyline_shape_attributes):
        polyline_valid = True
        for x, y in zip(
            polyline_shape_attributes.all_points_x,
            polyline_shape_attributes.all_points_y,
        ):
            if int(x) >= mask_maker.shape[1] or int(y) >= mask_maker.shape[0]:
                polyline_valid = False

        if polyline_valid:
            mask = mask_maker.get_mask(polyline_shape_attributes)
            assert (
                mask.shape == mask_maker.shape
            ), f"mask shape {mask.shape} doesn't match mask_maker {mask_maker.shape}"
            assert (
                mask.min() >= 0.0 and mask.max() <= 1.0
            ), f"mask value range ({mask.min()}, {mask.max()}) not within (0.0, 1.0)"
            for x, y in zip(
                polyline_shape_attributes.all_points_x,
                polyline_shape_attributes.all_points_y,
            ):
                assert (
                    mask[int(y), int(x)] > 0.0
                ), "Mask should be positive at polyline vertices"
        else:
            with raises(ValueError):
                mask_maker.get_mask(polyline_shape_attributes)

    def test_get_masks(self, mask_maker, via_metadata):
        masks = mask_maker.get_masks(via_metadata)
        assert len(masks) == len(via_metadata.regions)
        for mask in masks:
            assert isinstance(mask, Tensor)


class TestBoundingBox:
    def test_validator(self, invalid_bounding_box_params):
        x0, y0, x1, y1 = invalid_bounding_box_params

        with raises(ValueError, match="x1 and y1 must be larger than x0 and y0"):
            BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def test_add_margin_no_shape(self, bounding_box, margin):
        bounding_box_copy = bounding_box.copy()
        bounding_box.add_margin(margin)
        assert bounding_box.x0 <= bounding_box_copy.x0, "Should not decrease bounds"
        assert bounding_box.y0 <= bounding_box_copy.y0, "Should not decrease bounds"
        assert bounding_box.x1 >= bounding_box_copy.x1, "Should not decrease bounds"
        assert bounding_box.y1 >= bounding_box_copy.y1, "Should not decrease bounds"

    def test_add_margin_with_shape(self, bounding_box, margin, shape):
        bounding_box_copy = bounding_box.copy()
        bounding_box.add_margin(margin, shape=shape)
        assert bounding_box.x0 <= bounding_box_copy.x0, "Should not decrease bounds"
        assert bounding_box.y0 <= bounding_box_copy.y0, "Should not decrease bounds"
        assert (
            bounding_box.x1 >= bounding_box_copy.x1 or bounding_box.x1 == shape[1]
        ), "Should not decrease bounds unless bounding_box was already outside shape"
        assert (
            bounding_box.y1 >= bounding_box_copy.y1 or bounding_box.y1 == shape[0]
        ), "Should not decrease bounds unless bounding_box was already outside shape"
        assert (
            bounding_box.x0 <= shape[1]
        ), "{bounding_box!r} should not extend past {shape}"
        assert (
            bounding_box.y0 <= shape[0]
        ), "{bounding_box!r} should not extend past {shape}"
        assert (
            bounding_box.x1 <= shape[1]
        ), "{bounding_box!r} should not extend past {shape}"
        assert (
            bounding_box.y1 <= shape[0]
        ), "{bounding_box!r} should not extend past {shape}"

    def test_get_area(self, bounding_box):
        area = bounding_box.get_area()
        assert isinstance(area, int)
        assert area > 0


class TestTarget:
    def test_validator_passes(self, bounding_box):
        kwargs = dict(
            boxes=[bounding_box], labels=[1], image_id=0, masks=[zeros((2, 2))],
        )
        target = Target(**kwargs)
        for key, value in kwargs.items():
            assert getattr(target, key) == value

    def test_validator_fails(self, bounding_box):
        kwargs = dict(
            boxes=[bounding_box], labels=[1, 1], image_id=0, masks=[zeros((2, 2))],
        )
        with raises(ValueError):
            Target(**kwargs)

    def test_mask_validator_fails(self):
        for s0, s1 in [
            ((2, 2), (2, 3)),
            ((2, 2), (3, 2)),
            ((2, 3), (2, 2)),
            ((3, 2), (2, 2)),
        ]:
            kwargs = dict(
                boxes=[bounding_box, bounding_box],
                labels=[1, 1],
                image_id=0,
                masks=[zeros(s0), zeros(s1)],
            )
            with raises(ValueError):
                Target(**kwargs)

    def test_to_tensor_dict(self, target):
        target_dict = target.to_tensor_dict()
        fields = [
            "boxes",
            "labels",
            "image_id",
            "masks",
        ]
        for field in fields:
            assert field in target_dict

    def test_from_tensor_dict(self, target):
        target_dict = target.to_tensor_dict()
        fields = [
            "boxes",
            "labels",
            "image_id",
            "masks",
        ]
        target_from_dict = Target.from_tensor_dict(target_dict)
        assert target_from_dict.boxes == target.boxes
        assert target_from_dict.labels == target.labels
        assert target_from_dict.image_id == target.image_id
        for i in range(len(target_from_dict.masks)):
            assert target_from_dict.masks[i].allclose(target.masks[i])


class TestPrediction:
    def test_validator_passes(self, bounding_box):
        kwargs = dict(
            boxes=[bounding_box], labels=[1], masks=[zeros((2, 2))], scores=[0.0],
        )
        prediction = Prediction(**kwargs)
        for key, value in kwargs.items():
            assert getattr(prediction, key) == value

    def test_validator_fails(self, bounding_box):
        kwargs = dict(
            boxes=[bounding_box], labels=[1], masks=[zeros((2, 2))], scores=[0.0, 1.0],
        )
        with raises(ValueError):
            Prediction(**kwargs)

    def test_mask_validator_fails(self):
        for s0, s1 in [
            ((2, 2), (2, 3)),
            ((2, 2), (3, 2)),
            ((2, 3), (2, 2)),
            ((3, 2), (2, 2)),
        ]:
            kwargs = dict(
                boxes=[bounding_box, bounding_box],
                labels=[1, 1],
                masks=[zeros(s0), zeros(s1)],
                scores=[0.0, 1.0],
            )
            with raises(ValueError):
                Target(**kwargs)

    def test_to_tensor_dict(self, prediction):
        prediction_dict = prediction.to_tensor_dict()
        fields = [
            "boxes",
            "labels",
            "masks",
            "scores",
        ]
        for field in fields:
            assert field in prediction_dict

    def test_from_tensor_dict(self, prediction):
        prediction_dict = prediction.to_tensor_dict()
        fields = [
            "boxes",
            "labels",
            "masks",
            "scores",
        ]
        prediction_from_dict = Prediction.from_tensor_dict(prediction_dict)
        assert prediction_from_dict.boxes == prediction.boxes
        assert prediction_from_dict.labels == prediction.labels
        assert prediction_from_dict.scores == prediction.scores
        for i in range(len(prediction_from_dict.masks)):
            assert prediction_from_dict.masks[i].allclose(prediction.masks[i])

    def test_filter_by_score(self, prediction):
        filtered_prediction = prediction.filter_by_score(0.5)
        assert len(filtered_prediction) == 1
        assert len(filtered_prediction.boxes) == 1
        assert len(filtered_prediction.labels) == 1
        assert len(filtered_prediction.masks) == 1
        assert len(filtered_prediction.scores) == 1
        assert filtered_prediction.scores[0] >= 0.5


class MockImageTransform(ImageTransform):
    def apply_to_tensor_dict(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        image = image + 1.0
        return image, target


class TestCamfiDataset:
    def test_default_exclude_set(self, via_project):
        dataset = CamfiDataset(
            root="foo/bar",
            via_project=via_project,
            inference_mode=True,
            exclude={"baz"},
        )
        assert dataset.exclude == {Path("baz")}

    def test_default_exclude_unset(self, via_project):
        dataset = CamfiDataset(
            root="foo/bar", via_project=via_project, inference_mode=True
        )
        assert dataset.exclude == set()

    def test_only_set_if_not_inference_mode_transform_fails(self, via_project):
        with raises(ValidationError):
            CamfiDataset(
                root="foo/bar",
                via_project=via_project,
                inference_mode=True,
                transform=MockImageTransform(),
            )

    def test_only_set_if_not_inference_mode_min_annotations(self, via_project):
        with raises(ValidationError):
            CamfiDataset(
                root="foo/bar",
                via_project=via_project,
                inference_mode=True,
                min_annotations=1,
            )

    def test_only_set_if_not_inference_mode_max_annotations(self, via_project):
        with raises(ValidationError):
            CamfiDataset(
                root="foo/bar",
                via_project=via_project,
                inference_mode=True,
                max_annotations=10,
            )

    def test_only_set_if_not_inference_mode_box_margin(self, via_project):
        with raises(ValidationError):
            CamfiDataset(
                root="foo/bar",
                via_project=via_project,
                inference_mode=True,
                box_margin=20,
            )

    def test_set_iff_not_inference_mode_true(self, via_project, mask_maker):
        with raises(ValidationError):
            CamfiDataset(
                root="foo/bar",
                via_project=via_project,
                inference_mode=True,
                mask_maker=mask_maker,
            )

    def test_set_iff_not_inference_mode_false_fails(self, via_project):
        with raises(ValidationError):
            CamfiDataset(
                root="foo/bar", via_project=via_project, inference_mode=False,
            )

    def test_init_all_set_passes(self, via_project, mask_maker):
        dataset = CamfiDataset(
            root="foo/bar",
            via_project=via_project,
            inference_mode=False,
            transform=MockImageTransform(),
            min_annotations=1,
            max_annotations=5,
            mask_maker=mask_maker,
            box_margin=20,
        )
        assert isinstance(dataset.transform, ImageTransform)
        assert dataset.min_annotations == 1
        assert dataset.max_annotations == 5
        assert isinstance(dataset.mask_maker, MaskMaker)
        assert dataset.box_margin == 20

    def test_generate_filtered_keys_minmax_unset(self, via_project):
        dataset = CamfiDataset(
            root="foo/bar", via_project=via_project, inference_mode=True,
        )
        assert len(dataset.keys) == 4
        assert len(dataset) == 4

    def test_generate_filtered_keys_min4_max8(self, via_project, mask_maker):
        dataset = CamfiDataset(
            root="foo/bar",
            via_project=via_project,
            inference_mode=False,
            min_annotations=4,
            max_annotations=8,
            mask_maker=mask_maker,
        )
        assert len(dataset.keys) == 2
        assert len(dataset) == 2

    def test_generate_filtered_keys_min1_max4(self, via_project, mask_maker):
        dataset = CamfiDataset(
            root="foo/bar",
            via_project=via_project,
            inference_mode=False,
            min_annotations=1,
            max_annotations=4,
            mask_maker=mask_maker,
        )
        assert len(dataset.keys) == 1
        assert len(dataset) == 1

    def test_getitem_crop_none_inference_true(self, via_project):
        dataset = CamfiDataset(
            root="camfi/test/data", via_project=via_project, inference_mode=True,
        )
        image, target = dataset[0]
        assert image.shape == (3, 3456, 4608)
        assert image.dtype == float32

    def test_getitem_cropped_inference_true(self, via_project):
        dataset = CamfiDataset(
            root="camfi/test/data",
            via_project=via_project,
            inference_mode=True,
            crop=BoundingBox(x0=5, y0=10, x1=55, y1=50),
        )
        image, target = dataset[0]
        assert image.shape == (3, 40, 50)
        assert image.dtype == float32

    def test_getitem_cropped_inference_false(self, via_project, mask_maker):
        dataset = CamfiDataset(
            root="camfi/test/data",
            via_project=via_project,
            inference_mode=False,
            crop=BoundingBox(x0=0, y0=0, x1=4608, y1=3312),
            mask_maker=MaskMaker(shape=(3312, 4608)),
        )
        dataset_transformed = CamfiDataset(
            root="camfi/test/data",
            via_project=via_project,
            inference_mode=False,
            crop=BoundingBox(x0=0, y0=0, x1=4608, y1=3312),
            mask_maker=MaskMaker(shape=(3312, 4608), mask_dilate=1),
            transform=MockImageTransform(),
        )
        image, target = dataset[0]
        image_transformed, target_transformed = dataset_transformed[0]
        assert image_transformed.allclose(image + 1.0)
        assert image_transformed.dtype == float32
        assert target_transformed.masks[0].sum() > target.masks[0].sum()

    def test_metadata(self, via_project):
        dataset = CamfiDataset(
            root="foo/bar", via_project=via_project, inference_mode=True
        )
        assert isinstance(dataset.metadata(0), ViaMetadata)
