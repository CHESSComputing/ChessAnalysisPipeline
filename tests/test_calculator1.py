import pytest

# A standard, simple test case
def test_add_positive_numbers():
    assert 2+3 == 5

# A test case that handles expected errors
def test_divide_by_zero_raises_error():
    with pytest.raises(ValueError):
        10/0

