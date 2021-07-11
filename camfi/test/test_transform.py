from math import sqrt
from typing import Dict, Tuple

from pytest import approx, fixture, raises
from torch import Tensor

from camfi import transform


@fixture(params=[0.0, 0.5, 1.0])
def random_horizontal_flip(request):
    return transform.RandomHorizontalFlip(prob=request.param)


@fixture
def image_tensor():
    return Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],])


class MockImageTransform(transform.ImageTransform):
    def __call__(
        self, image: Tensor, target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        image = image + 1.0
        return image, target


class TestCompose:
    def test_call(self, image_tensor):
        compose = transform.Compose(transforms=[MockImageTransform() for _ in range(5)])
        transformed_image_tensor, _ = compose(image_tensor, {})
        assert transformed_image_tensor.allclose(image_tensor + 5.0)


class TestRandomHorizontalFlip:
    def test_call(self, random_horizontal_flip):
        pass


def test_sec_trivial_colinear():
    points = [(0.0, 0.0), (1.0, 1.0,), (4.0, 4.0)]

    assert transform._sec_trivial(points) == approx((2.0, 2.0, 2 * sqrt(2)))


def test_sec_trivial_too_many_points():
    points = [(-1.0, -1.0), (0.0, 0.0,), (4.0, 4.0), (1.0, 2.0)]

    with raises(ValueError):
        transform._sec_trivial(points)
