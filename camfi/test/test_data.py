from camfi import data

from pydantic import ValidationError
import pytest


class TestPolylineShapeAttributes():

    def test_validator_success(self):
        all_points_x = [1.0, 0.0]
        all_points_y = [0.0, 1.0]
        shape = data.PolylineShapeAttributes(all_points_x=all_points_x, all_points_y=all_points_y)
        assert shape.all_points_x == all_points_x
        assert shape.all_points_y == all_points_y

    def test_validator_fail(self):
        all_points_x = [1.0, 0.0]
        all_points_y = [0.0, 1.0, 1.0]
        with pytest.raises(ValidationError):
            data.PolylineShapeAttributes(all_points_x=all_points_x, all_points_y=all_points_y)

