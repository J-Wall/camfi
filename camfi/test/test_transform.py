from pytest import approx, fixture, raises
from torch import Tensor, zeros

from camfi.data import Target
from camfi import transform
from camfi.test.test_data import (
    target_dict,
    target,
    boxes,
    labels,
    image_id,
    area,
    iscrowd,
    masks,
    MockImageTransform,
)


@fixture(params=[0.0, 0.5, 1.0])
def random_horizontal_flip(request):
    return transform.RandomHorizontalFlip(prob=request.param)


@fixture
def image_tensor():
    return Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class TestCompose:
    def test_apply_to_tensor_dict(self, image_tensor):
        compose = transform.Compose(transforms=[MockImageTransform() for _ in range(5)])
        transformed_image_tensor, _ = compose.apply_to_tensor_dict(image_tensor, {})
        assert transformed_image_tensor.allclose(image_tensor + 5.0)

    def test_call(self, image_tensor, target):
        """Just test that no errors are raised, since this just wraps
        apply_to_tensor_dict."""
        compose = transform.Compose(transforms=[MockImageTransform() for _ in range(5)])
        transformed_image_tensor, target = compose(image_tensor, target)
        assert transformed_image_tensor.shape == image_tensor.shape
        assert isinstance(target, Target)


class TestRandomHorizontalFlip:
    def test_apply_to_tensor_dict_empty_dict(
        self, random_horizontal_flip, image_tensor
    ):
        height, width = image_tensor.shape

        non_flipped_image_tensor = image_tensor

        flipped_image_tensor = image_tensor.flip(-1)

        image_tensor, _ = random_horizontal_flip.apply_to_tensor_dict(image_tensor, {})

        if random_horizontal_flip.prob == 1.0:
            assert image_tensor.allclose(flipped_image_tensor)

        elif random_horizontal_flip.prob == 0.0:
            assert image_tensor.allclose(non_flipped_image_tensor)

        else:
            assert image_tensor.allclose(flipped_image_tensor) or image_tensor.allclose(
                non_flipped_image_tensor
            )

    def test_apply_to_tensor_dict(
        self, random_horizontal_flip, image_tensor, target_dict
    ):
        boxes = target_dict["boxes"]
        height, width = image_tensor.shape

        non_flipped_image_tensor = image_tensor
        non_flipped_masks = target_dict["masks"]
        non_flipped_boxes = boxes

        flipped_image_tensor = image_tensor.flip(-1)
        flipped_masks = target_dict["masks"].flip(-1)
        flipped_x0 = width - boxes[:, 2]
        flipped_y0 = boxes[:, 1]
        flipped_x1 = width - boxes[:, 0]
        flipped_y1 = boxes[:, 3]

        image_tensor, target_dict = random_horizontal_flip.apply_to_tensor_dict(
            image_tensor, target_dict
        )

        def flipped():
            assert image_tensor.allclose(flipped_image_tensor)
            assert target_dict["masks"].allclose(flipped_masks)
            assert target_dict["boxes"][:, 0].allclose(flipped_x0)
            assert target_dict["boxes"][:, 1].allclose(flipped_y0)
            assert target_dict["boxes"][:, 2].allclose(flipped_x1)
            assert target_dict["boxes"][:, 3].allclose(flipped_y1)

        def not_flipped():
            assert image_tensor.allclose(non_flipped_image_tensor)
            assert target_dict["masks"].allclose(non_flipped_masks)
            assert target_dict["boxes"].allclose(non_flipped_boxes)

        if random_horizontal_flip.prob == 1.0:
            flipped()

        elif random_horizontal_flip.prob == 0.0:
            not_flipped()

        else:
            try:
                flipped()
            except AssertionError:
                not_flipped()

    def test_call(self, random_horizontal_flip, image_tensor, target):
        """Just test that no errors are raised, since this just wraps
        apply_to_tensor_dict."""
        flipped_image_tensor, target = random_horizontal_flip(image_tensor, target)
        assert flipped_image_tensor.shape == image_tensor.shape
        assert isinstance(target, Target)
