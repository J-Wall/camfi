import bz2

from pydantic import ValidationError
from pytest import approx, fixture, raises
from torch import tensor, Tensor, zeros

from camfi import data, util, transform


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
    return data.PointShapeAttributes(cx=cx, cy=cy)


@fixture
def circle_shape_attributes(cx, cy, r):
    return data.CircleShapeAttributes(cx=cx, cy=cy, r=r)


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

    return data.PolylineShapeAttributes(
        all_points_x=all_points_x, all_points_y=all_points_y
    )


@fixture(params=[None, 1, 5])
def mask_maker(request):
    return data.MaskMaker(shape=(100, 120), mask_dilate=request.param)


@fixture(params=[(0, 0, 1, 1), (10, 15, 12, 20)])
def bounding_box(request):
    x0, y0, x1, y1 = request.param
    return data.BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


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
        data.BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
        for x0, y0, x1, y1 in bounding_box_params
    ]


@fixture
def labels():
    return [1, 2]


@fixture
def image_id():
    return 0


@fixture
def area():
    return [1, 10]


@fixture
def iscrowd():
    return [0, 0]


@fixture
def masks():
    return [zeros((2, 3)), tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]


@fixture
def target(boxes, labels, image_id, area, iscrowd, masks):
    return data.Target(
        boxes=boxes,
        labels=labels,
        image_id=image_id,
        area=area,
        iscrowd=iscrowd,
        masks=masks,
    )


@fixture
def target_dict(target):
    return target.to_tensor_dict()


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
            shape = data.PolylineShapeAttributes(
                all_points_x=all_points_x, all_points_y=all_points_y
            )
            assert shape.all_points_x == all_points_x
            assert shape.all_points_y == all_points_y
        else:
            with raises(ValidationError):
                data.PolylineShapeAttributes(
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


class TestViaRegion:
    def test_get_bounding_box(self, point_shape_attributes):
        region = data.ViaRegion(
            region_attributes=data.ViaRegionAttributes(),
            shape_attributes=point_shape_attributes,
        )
        assert region.get_bounding_box() == point_shape_attributes.get_bounding_box()


class TestViaMetadata:
    def test_get_bounding_boxes(self):
        pass

    def test_get_labels(self):
        pass

    def test_get_iscrowd(self):
        pass


class TestViaProject:
    def test_parse_empty_file(self):
        with open("camfi/test/data/empty_project.json") as f:
            via_project_raw = f.read()

        data.ViaProject.parse_raw(via_project_raw)

    def test_parse_no_annotations(self):
        with open("camfi/test/data/sample_project_no_annotations.json") as f:
            via_project_raw = f.read()

        data.ViaProject.parse_raw(via_project_raw)

    def test_parse_no_metadata(self):
        with open("camfi/test/data/sample_project_no_metadata.json") as f:
            via_project_raw = f.read()

        data.ViaProject.parse_raw(via_project_raw)

    def test_parse_bz2(self):
        with bz2.open("examples/data/cabramurra_all_annotations.json.bz2") as f:
            via_project_raw = f.read()

        data.ViaProject.parse_raw(via_project_raw)

    def test_parse_fails(self):
        with open("camfi/test/data/sample_project_no_metadata_malformed.json") as f:
            via_project_raw = f.read()

        with raises(ValidationError):
            data.ViaProject.parse_raw(via_project_raw)


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


class TestBoundingBox:
    def test_validator(self, invalid_bounding_box_params):
        x0, y0, x1, y1 = invalid_bounding_box_params

        with raises(ValueError, match="x1 and y1 must be larger than x0 and y0"):
            data.BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

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


class TestTarget:
    def test_validator_passes(self, bounding_box):
        kwargs = dict(
            boxes=[bounding_box],
            labels=[1],
            image_id=0,
            area=[1],
            iscrowd=[0],
            masks=[zeros((2, 2))],
        )
        target = data.Target(**kwargs)
        for key, value in kwargs.items():
            assert getattr(target, key) == value

    def test_validator_fails(self, bounding_box):
        kwargs = dict(
            boxes=[bounding_box],
            labels=[1, 1],
            image_id=0,
            area=[1],
            iscrowd=[0],
            masks=[zeros((2, 2))],
        )
        with raises(ValueError):
            data.Target(**kwargs)

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
                area=[1, 1],
                iscrowd=[0, 0],
                masks=[zeros(s0), zeros(s1)],
            )
            with raises(ValueError):
                data.Target(**kwargs)

    def test_to_tensor_dict(self, target):
        target_dict = target.to_tensor_dict()
        fields = [
            "boxes",
            "labels",
            "image_id",
            "area",
            "iscrowd",
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
            "area",
            "iscrowd",
            "masks",
        ]
        target_from_dict = data.Target.from_tensor_dict(target_dict)
        assert target_from_dict.boxes == target.boxes
        assert target_from_dict.labels == target.labels
        assert target_from_dict.image_id == target.image_id
        assert target_from_dict.area == target.area
        assert target_from_dict.iscrowd == target.iscrowd
        for i in range(len(target_from_dict.masks)):
            assert target_from_dict.masks[i].allclose(target.masks[i])