import pytest

def test_sum():
    assert (0.1 + 0.2) == 0.3

def test_sum_approx():
    assert (0.1 + 0.2) == pytest.approx(0.3)
