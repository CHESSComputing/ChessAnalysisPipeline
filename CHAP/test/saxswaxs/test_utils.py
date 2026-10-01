"""pytest-style unittests for CHAP.saxswaxs.utils module."""

# Third party modules
import numpy as np
import pytest
import zarr

# Local modules
from CHAP.saxswaxs.utils import dict_to_zarr


class TestDictToZarr:

    def test_returns_zarr_group(self):
        result = dict_to_zarr({})
        assert isinstance(result, zarr.Group)

    def test_empty_tree(self):
        result = dict_to_zarr({})
        assert list(result.keys()) == []

    def test_top_level_attributes(self):
        tree = {'attributes': {'foo': 'bar', 'num': 42}}
        result = dict_to_zarr(tree)
        assert result.attrs['foo'] == 'bar'
        assert result.attrs['num'] == 42

    def test_dataset_with_shape(self):
        tree = {
            'children': {
                'my_array': {'dtype': 'float64', 'shape': (10, 5)},
            }
        }
        result = dict_to_zarr(tree)
        assert 'my_array' in result
        assert result['my_array'].shape == (10, 5)
        assert result['my_array'].dtype == np.float64

    def test_dataset_with_data(self):
        data = np.array([1.0, 2.0, 3.0])
        tree = {
            'children': {
                'vals': {'data': data, 'dtype': 'float64'}
            }
        }
        result = dict_to_zarr(tree)
        assert 'vals' in result
        np.testing.assert_array_equal(result['vals'][:], data)

    def test_dataset_attributes(self):
        tree = {
            'children': {
                'arr': {
                    'dtype': 'float32',
                    'shape': (3,),
                    'attributes': {'units': 'nm', 'long_name': 'Length'},
                }
            }
        }
        result = dict_to_zarr(tree)
        assert result['arr'].attrs['units'] == 'nm'
        assert result['arr'].attrs['long_name'] == 'Length'

    def test_group_attributes(self):
        tree = {
            'children': {
                'grp': {
                    'attributes': {'NX_class': 'NXprocess'},
                    'children': {},
                }
            }
        }
        result = dict_to_zarr(tree)
        assert result['grp'].attrs['NX_class'] == 'NXprocess'

    def test_nested_groups(self):
        tree = {
            'children': {
                'outer': {
                    'children': {
                        'inner': {
                            'children': {
                                'deep_arr': {'dtype': 'int32', 'shape': (4,)},
                            }
                        }
                    }
                }
            }
        }
        result = dict_to_zarr(tree)
        assert 'outer' in result
        assert 'inner' in result['outer']
        assert 'deep_arr' in result['outer']['inner']
        assert result['outer']['inner']['deep_arr'].shape == (4,)

    def test_multiple_datasets_and_groups(self):
        tree = {
            'children': {
                'arr1': {'dtype': 'float64', 'shape': (5,)},
                'arr2': {'dtype': 'int32', 'shape': (3, 3)},
                'grp': {
                    'children': {
                        'nested': {'dtype': 'float32', 'shape': (2,)},
                    }
                },
            }
        }
        result = dict_to_zarr(tree)
        assert 'arr1' in result
        assert 'arr2' in result
        assert 'grp' in result
        assert 'nested' in result['grp']

    def test_root_and_child_attributes(self):
        tree = {
            'attributes': {'root_attr': 'root_val'},
            'children': {
                'grp': {
                    'attributes': {'child_attr': 'child_val'},
                    'children': {},
                }
            },
        }
        result = dict_to_zarr(tree)
        assert result.attrs['root_attr'] == 'root_val'
        assert result['grp'].attrs['child_attr'] == 'child_val'

    def test_with_logger(self):
        import logging
        logger = logging.getLogger('test_dict_to_zarr')
        tree = {
            'children': {
                'arr': {'dtype': 'float64', 'shape': (2,)},
            }
        }
        result = dict_to_zarr(tree, logger=logger)
        assert 'arr' in result

    def test_dataset_dtype_preserved(self):
        for dtype in ('float32', 'float64', 'int16', 'int32'):
            tree = {'children': {'a': {'dtype': dtype, 'shape': (3,)}}}
            result = dict_to_zarr(tree)
            assert str(result['a'].dtype) == dtype

    def test_dataset_multidim_shape(self):
        shape = (7, 4, 3)
        tree = {'children': {'cube': {'dtype': 'float64', 'shape': shape}}}
        result = dict_to_zarr(tree)
        assert result['cube'].shape == shape
