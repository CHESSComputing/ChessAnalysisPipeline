#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Functions to compare objects for some commonly used data formats."""

# Third party modules
import numpy as np


def nxs_match(nxobject_a, nxobject_b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return true if the two NeXus style
    `NXobject <https://manual.nexusformat.org/classes/base_classes/NXobject.html#index-0>`__
    objects "match" -- they must be exactly identical or within the
    rtol and atol limits for floats, with one exception: any attributes
    or fields related to code version or file datetime may differ and
    still be considered a "match". List-like fields are compared with
    [`np.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html).

    :param nxobject_a, nxobject_b: Input NXobjects to compare.
    :type nxobject_a, nxobject_b: nexus.nexusformat.NXobject
   :param rtol: Relative tolerance parameter for float comparisons,
        defaults to `1e-05`.
    :type rtol: float, optional
    :param atol: Absolute tolerance parameter for float comparisons,
        defaults to `1e-08`.
    :type atol: float, optional
    :param equal_nan: If True, NaNs will compare as equal in
        `np.testing.assert_allclose`, defaults to `False`.
    :type equal_nan: bool, optional
    :return: Whether nxobject_a matches nxobject_b or not and the error
        message (`''` upon success).
    :rtype: tuple[bool, str]
    """
    _SKIP_ATTRS = frozenset({
        'date', 'datetime', 'timestamp',
        'file_name','file_time', 'file_update_time',
        'version', 'program_version', 'CHAP_version',
        'HDF5_Version', 'h5py_version',
        'creator', 'creator_version',
    })
    _SKIP_FIELDS = frozenset({
        'start_time', 'end_time', 'duration', 'timestamp',
        'program_name', 'program_version', 'configuration',
    })

    def _filtered_attrs(obj):
        return {k: v for k, v in obj.attrs.items()
                if k not in _SKIP_ATTRS}

    def _match(a, b):
        if type(a) != type(b):
            raise ValueError(
                f'Type mismatch {a.nxname}: {type(a)} vs {type(b)}')
        a_attrs = _filtered_attrs(a)
        b_attrs = _filtered_attrs(b)
        if set(a_attrs) != set(b_attrs):
            raise ValueError(
                f'Attributes mismatch {a.nxname}: {a_attrs} vs {b_attrs}')
        for k, va in a_attrs.items():
            vb = b_attrs[k]
            if (isinstance(va, np.ndarray) or isinstance(va, float)
                    or isinstance(vb, float)):
                np.testing.assert_allclose(
                    va, vb, rtol=rtol, atol=atol, equal_nan=equal_nan)
                return
            elif va != vb:
                raise ValueError(
                    f'Value mismatch: {va}, {type(va)} != {vb} {type(vb)}')
        if a.nxclass == 'NXfield':
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(
                    f'{a.nxname} NXfield mismatch: {a.shape} {a.dtype} != '
                    f'{b.shape} {b.dtype}')
            np.testing.assert_allclose(
                a.nxdata, b.nxdata, rtol=rtol, atol=atol, equal_nan=equal_nan)
            return
        elif a.nxclass == 'NXgroup':
            a_keys = {k for k in a.keys() if k not in _SKIP_FIELDS}
            b_keys = {k for k in b.keys() if k not in _SKIP_FIELDS}
            if a_keys != b_keys:
                raise ValueError(
                    f'{a.nxname} key mismatch: {list(a.keys())} vs '
                    f'{list(b.keys())}')
            for k in a_keys:
                _match(a[k], b[k])
                return
    try:
        _match(nxobject_a, nxobject_b)
    except (ValueError, AssertionError) as e:
        return False, str(e)
    return True, ''


def dict_or_listlike_match(
        object_a, object_b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return true if the two dictionary or list-like style objects
    "match" -- they must be exactly identical or within the rtol and
    atol limits for floats. List-like fields are compared with
    [`np.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html).

    :param object_a, object_b: Input objects to compare.
    :type object_a, object_b: dict or list or tuple or numpy.ndarray
    :param rtol: Relative tolerance parameter for float comparisons,
        defaults to `1e-05`.
    :type rtol: float, optional
    :param atol: Absolute tolerance parameter for float comparisons,
        defaults to `1e-08`.
    :type atol: float, optional
    :param equal_nan: If True, NaNs will compare as equal in
        `np.testing.assert_allclose`, defaults to `False`.
    :type equal_nan: bool, optional
    :return: Whether object_a matches object_b or not and the error
        message (`''` upon success).
    :rtype: tuple[bool, str]
    """
    def _match(a, b, skip_numpy=False, key=''):
        if (not skip_numpy and isinstance(a, (list, tuple, np.ndarray))
                and isinstance(b, (list, tuple, np.ndarray))):
            try:
                va = np.asarray(a)
                vb = np.asarray(b)
                if va.dtype == np.object_ or vb.dtype == np.object_:
                    # Older versions of Numpy don't throw a ValueError
                    _match(a, b, skip_numpy=True, key=key)
                    return
                if va.shape != vb.shape:
                    raise ValueError(
                        f'Shape mismatch{key}: {va.shape} vs {vb.shape}')
                np.testing.assert_allclose(
                    va, vb, rtol=rtol, atol=atol, equal_nan=equal_nan)
                return
            except (TypeError, ValueError):
                _match(a, b, skip_numpy=True, key=key)
                return
        if type(a) is not type(b):
            raise ValueError(f'Type mismatch{key}: {type(a)} vs {type(b)}')
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                raise ValueError(
                    f'Key mismatch{key}: {list(a.keys())} vs {list(b.keys())}')
            for k in a:
                next_key = f'{key}["{k}"]' if key else f'["{k}"]'
                _match(a[k], b[k], key=next_key)
        elif isinstance(a, float) or isinstance(b, float):
            np.testing.assert_allclose(
                a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        elif a != b:
            raise ValueError(
                f'Value mismatch: {a}, {type(a)} != {b} {type(b)}')

    try:
        _match(object_a, object_b)
    except (ValueError, AssertionError) as e:
        return False, str(e)
    return True, ''

#from numba import njit
#
#@njit(parallel=False)
#def numba_allclose(a, b, rtol=1e-5, atol=1e-8):
#    """Compiled element-by-element comparison that exits instantly on failure."""
#    # Ensure shapes match
#    if a.shape != b.shape:
#        return False
#        
#    a_flat = a.ravel()
#    b_flat = b.ravel()
#    
#    for i in range(a_flat.size):
#        # Standard allclose formula
#        if abs(a_flat[i] - b_flat[i]) > (atol + rtol * abs(b_flat[i])):
#            return False  # Stops instantly at C-speed
#            
#    return True
