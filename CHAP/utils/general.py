#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint: disable=
"""
File       : general.py
Author     : Rolf Verberg <rolfverberg AT gmail dot com>
Description: A collection of general modules
"""
# RV write function that returns a list of peak indices for a given plot
# RV use raise_error concept on more functions

# System modules
from ast import literal_eval
from logging import getLogger
from os import path as os_path
from os import (
    access,
    R_OK,
)
from re import compile as re_compile
from re import split as re_split
from re import sub as re_sub
from sys import float_info

# Third party modules
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.widgets import Button
except ImportError:
    pass

logger = getLogger(__name__)


def depth_list(_list):
    """Return the depth of a list."""
    return isinstance(_list, list) and 1+max(map(depth_list, _list))


def depth_tuple(_tuple):
    """Return the depth of a tuple."""
    return isinstance(_tuple, tuple) and 1+max(map(depth_tuple, _tuple))


def unwrap_tuple(_tuple):
    """Unwrap a tuple."""
    if depth_tuple(_tuple) > 1 and len(_tuple) == 1:
        _tuple = unwrap_tuple(*_tuple)
    return _tuple


def illegal_value(value, name, location=None, raise_error=False, log=True):
    """Print illegal value message and/or raise error."""
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name, str):
        error_msg = \
            f'Illegal value for {name} {location}({value}, {type(value)})'
    else:
        error_msg = f'Illegal value {location}({value}, {type(value)})'
    if log:
        logger.error(error_msg)
    if raise_error:
        raise ValueError(error_msg)


def illegal_combination(
        value1, name1, value2, name2, location=None, raise_error=False,
        log=True):
    """Print illegal combination message and/or raise error."""
    if not isinstance(location, str):
        location = ''
    else:
        location = f'in {location} '
    if isinstance(name1, str):
        error_msg = f'Illegal combination for {name1} and {name2} {location}' \
            f'({value1}, {type(value1)} and {value2}, {type(value2)})'
    else:
        error_msg = f'Illegal combination {location}' \
            f'({value1}, {type(value1)} and {value2}, {type(value2)})'
    if log:
        logger.error(error_msg)
    if raise_error:
        raise ValueError(error_msg)


def test_ge_gt_le_lt(
        ge, gt, le, lt, func, location=None, raise_error=False, log=True):
    """
    Check individual and mutual validity of ge, gt, le, lt qualifiers.

    :param func: Test for integers or numbers
    :type func: callable: is_int, is_num
    :return: True upon success or False when mutually exlusive
    :rtype: bool
    """
    if ge is None and gt is None and le is None and lt is None:
        return True
    if ge is not None:
        if not func(ge):
            illegal_value(ge, 'ge', location, raise_error, log)
            return False
        if gt is not None:
            illegal_combination(ge, 'ge', gt, 'gt', location, raise_error, log)
            return False
    elif gt is not None and not func(gt):
        illegal_value(gt, 'gt', location, raise_error, log)
        return False
    if le is not None:
        if not func(le):
            illegal_value(le, 'le', location, raise_error, log)
            return False
        if lt is not None:
            illegal_combination(le, 'le', lt, 'lt', location, raise_error, log)
            return False
    elif lt is not None and not func(lt):
        illegal_value(lt, 'lt', location, raise_error, log)
        return False
    if ge is not None:
        if le is not None and ge > le:
            illegal_combination(ge, 'ge', le, 'le', location, raise_error, log)
            return False
        if lt is not None and ge >= lt:
            illegal_combination(ge, 'ge', lt, 'lt', location, raise_error, log)
            return False
    elif gt is not None:
        if le is not None and gt >= le:
            illegal_combination(gt, 'gt', le, 'le', location, raise_error, log)
            return False
        if lt is not None and gt >= lt:
            illegal_combination(gt, 'gt', lt, 'lt', location, raise_error, log)
            return False
    return True


def range_string_ge_gt_le_lt(ge=None, gt=None, le=None, lt=None):
    """
    Return a range string representation matching the ge, gt, le, lt
    qualifiers. Does not validate the inputs, do that as needed before
    calling.
    """
    range_string = ''
    if ge is not None:
        if le is None and lt is None:
            range_string += f'>= {ge}'
        else:
            range_string += f'[{ge}, '
    elif gt is not None:
        if le is None and lt is None:
            range_string += f'> {gt}'
        else:
            range_string += f'({gt}, '
    if le is not None:
        if ge is None and gt is None:
            range_string += f'<= {le}'
        else:
            range_string += f'{le}]'
    elif lt is not None:
        if ge is None and gt is None:
            range_string += f'< {lt}'
        else:
            range_string += f'{lt})'
    return range_string


def is_int(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """
    Value is an integer in range ge <= v <= le or gt < v < lt or some
    combination.

    :return: True if yes or False is no
    :rtype: bool
    """
    return _is_int_or_num(v, 'int', ge, gt, le, lt, raise_error, log)


def is_num(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """
    Value is a number in range ge <= v <= le or gt < v < lt or some
    combination.

    :return: True if yes or False is no
    :rtype: bool
    """
    return _is_int_or_num(v, 'num', ge, gt, le, lt, raise_error, log)


def _is_int_or_num(
        v, type_str, ge=None, gt=None, le=None, lt=None, raise_error=False,
        log=True):
    if type_str == 'int':
        if not isinstance(v, int):
            illegal_value(v, 'v', '_is_int_or_num', raise_error, log)
            return False
        if not test_ge_gt_le_lt(
                ge, gt, le, lt, is_int, '_is_int_or_num', raise_error, log):
            return False
    elif type_str == 'num':
        if not isinstance(v, (int, float)):
            illegal_value(v, 'v', '_is_int_or_num', raise_error, log)
            return False
        if not test_ge_gt_le_lt(
                ge, gt, le, lt, is_num, '_is_int_or_num', raise_error, log):
            return False
    else:
        illegal_value(type_str, 'type_str', '_is_int_or_num', raise_error, log)
        return False
    if ge is None and gt is None and le is None and lt is None:
        return True
    error = False
    if ge is not None and v < ge:
        error = True
        error_msg = f'Value {v} out of range: {v} !>= {ge}'
    if not error and gt is not None and v <= gt:
        error = True
        error_msg = f'Value {v} out of range: {v} !> {gt}'
    if not error and le is not None and v > le:
        error = True
        error_msg = f'Value {v} out of range: {v} !<= {le}'
    if not error and lt is not None and v >= lt:
        error = True
        error_msg = f'Value {v} out of range: {v} !< {lt}'
    if error:
        if log:
            logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return False
    return True


def is_int_pair(
        v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """
    Value is an integer pair, each in range ge <= v[i] <= le or
    gt < v[i] < lt or ge[i] <= v[i] <= le[i] or gt[i] < v[i] < lt[i]
    or some combination.

    :return: True if yes or False is no
    :rtype: bool
    """
    return _is_int_or_num_pair(v, 'int', ge, gt, le, lt, raise_error, log)


def is_num_pair(
        v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """
    Value is a number pair, each in range ge <= v[i] <= le or
    gt < v[i] < lt or ge[i] <= v[i] <= le[i] or gt[i] < v[i] < lt[i]
    or some combination.

    :return: True if yes or False is no
    :rtype: bool
    """
    return _is_int_or_num_pair(v, 'num', ge, gt, le, lt, raise_error, log)


def _is_int_or_num_pair(
        v, type_str, ge=None, gt=None, le=None, lt=None, raise_error=False,
        log=True):
    if type_str == 'int':
        if not (isinstance(v, (tuple, list)) and len(v) == 2
                and isinstance(v[0], int) and isinstance(v[1], int)):
            illegal_value(v, 'v', '_is_int_or_num_pair', raise_error, log)
            return False
        func = is_int
    elif type_str == 'num':
        if not (isinstance(v, (tuple, list)) and len(v) == 2
                and isinstance(v[0], (int, float))
                and isinstance(v[1], (int, float))):
            illegal_value(v, 'v', '_is_int_or_num_pair', raise_error, log)
            return False
        func = is_num
    else:
        illegal_value(
            type_str, 'type_str', '_is_int_or_num_pair', raise_error, log)
        return False
    if ge is None and gt is None and le is None and lt is None:
        return True
    if ge is None or func(ge, log=True):
        ge = 2*[ge]
    elif not _is_int_or_num_pair(
            ge, type_str, raise_error=raise_error, log=log):
        return False
    if gt is None or func(gt, log=True):
        gt = 2*[gt]
    elif not _is_int_or_num_pair(
            gt, type_str, raise_error=raise_error, log=log):
        return False
    if le is None or func(le, log=True):
        le = 2*[le]
    elif not _is_int_or_num_pair(
            le, type_str, raise_error=raise_error, log=log):
        return False
    if lt is None or func(lt, log=True):
        lt = 2*[lt]
    elif not _is_int_or_num_pair(
            lt, type_str, raise_error=raise_error, log=log):
        return False
    if (not func(v[0], ge[0], gt[0], le[0], lt[0], raise_error, log)
            or not func(v[1], ge[1], gt[1], le[1], lt[1], raise_error, log)):
        return False
    return True


def is_int_series(
        t_or_l, ge=None, gt=None, le=None, lt=None, raise_error=False,
        log=True):
    """
    Value is a tuple or list of integers, each in range
    ge <= l[i] <= le or gt < l[i] < lt or some combination.
    """
    if not test_ge_gt_le_lt(
            ge, gt, le, lt, is_int, 'is_int_series', raise_error, log):
        return False
    if not isinstance(t_or_l, (tuple, list)):
        illegal_value(t_or_l, 't_or_l', 'is_int_series', raise_error, log)
        return False
    if any(not is_int(v, ge, gt, le, lt, raise_error, log) for v in t_or_l):
        return False
    return True


def is_num_series(
        t_or_l, ge=None, gt=None, le=None, lt=None, raise_error=False,
        log=True):
    """
    Value is a tuple or list of numbers, each in range ge <= l[i] <= le
    or gt < l[i] < lt or some combination.
    """
    if not test_ge_gt_le_lt(
            ge, gt, le, lt, is_int, 'is_int_series', raise_error, log):
        return False
    if not isinstance(t_or_l, (tuple, list)):
        illegal_value(t_or_l, 't_or_l', 'is_num_series', raise_error, log)
        return False
    if any(not is_num(v, ge, gt, le, lt, raise_error, log) for v in t_or_l):
        return False
    return True


def is_str_series(t_or_l, raise_error=False, log=True):
    """
    Value is a tuple or list of strings.
    """
    if (not isinstance(t_or_l, (tuple, list))
            or any(not isinstance(s, str) for s in t_or_l)):
        illegal_value(t_or_l, 't_or_l', 'is_str_series', raise_error, log)
        return False
    return True


def is_dict_series(t_or_l, raise_error=False, log=True):
    """
    Value is a tuple or list of dictionaries.
    """
    if (not isinstance(t_or_l, (tuple, list))
            or any(not isinstance(d, dict) for d in t_or_l)):
        illegal_value(t_or_l, 't_or_l', 'is_dict_series', raise_error, log)
        return False
    return True


def is_dict_nums(d, raise_error=False, log=True):
    """
    Value is a dictionary with single number values
    """
    if (not isinstance(d, dict)
            or any(not is_num(v, log=False) for v in d.values())):
        illegal_value(d, 'd', 'is_dict_nums', raise_error, log)
        return False
    return True


def is_dict_strings(d, raise_error=False, log=True):
    """
    Value is a dictionary with single string values
    """
    if (not isinstance(d, dict)
            or any(not isinstance(v, str) for v in d.values())):
        illegal_value(d, 'd', 'is_dict_strings', raise_error, log)
        return False
    return True


def is_index(v, ge=0, lt=None, raise_error=False, log=True):
    """
    Value is an array index in range ge <= v < lt. NOTE lt IS NOT
    included!
    """
    if isinstance(lt, int):
        if lt <= ge:
            illegal_combination(
                ge, 'ge', lt, 'lt', 'is_index', raise_error, log)
            return False
    return is_int(v, ge=ge, lt=lt, raise_error=raise_error, log=log)


def is_index_range(v, ge=0, le=None, lt=None, raise_error=False, log=True):
    """
    Value is an array index range in range ge <= v[0] <= v[1] <= le or
    ge <= v[0] <= v[1] < lt. NOTE le IS included!
    """
    if not is_int_pair(v, raise_error=raise_error, log=log):
        return False
    if not test_ge_gt_le_lt(
            ge, None, le, lt, is_int, 'is_index_range', raise_error, log):
        return False
    if (not ge <= v[0] <= v[1] or (le is not None and v[1] > le)
            or (lt is not None and v[1] >= lt)):
        if le is not None:
            error_msg = \
                f'Value {v} out of range: !({ge} <= {v[0]} <= {v[1]} <= {le})'
        else:
            error_msg = \
                f'Value {v} out of range: !({ge} <= {v[0]} <= {v[1]} < {lt})'
        if log:
            logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return False
    return True


def index_nearest(a, value):
    """Return index of nearest array value."""
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(
            f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    # Round up for .5
    value *= 1.0+float_info.epsilon
    return (int)(np.argmin(np.abs(a-value)))


def index_nearest_low(a, value):
    """Return index of nearest array value, rounded down"""
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(
            f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    index = int(np.argmin(np.abs(a-value)))
    if value < a[index] and index > 0:
        index -= 1
    return index


def index_nearest_upp(a, value):
    """Return index of nearest array value, rounded upp."""
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(
            f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    index = int(np.argmin(np.abs(a-value)))
    if value > a[index] and index < a.size-1:
        index += 1
    return index


def round_to_n(x, n=1):
    """Round to a specific number of decimals."""
    if x == 0.0:
        return 0
    return type(x)(round(x, n-1-int(np.floor(np.log10(abs(x))))))


def round_up_to_n(x, n=1):
    """Round up to a specific number of decimals."""
    x_round = round_to_n(x, n)
    if abs(x/x_round) > 1.0:
        x_round += np.sign(x) * 10**(np.floor(np.log10(abs(x)))+1-n)
    return type(x)(x_round)


def trunc_to_n(x, n=1):
    """Truncate to a specific number of decimals."""
    x_round = round_to_n(x, n)
    if abs(x_round/x) > 1.0:
        x_round -= np.sign(x) * 10**(np.floor(np.log10(abs(x)))+1-n)
    return type(x)(x_round)


def almost_equal(a, b, sig_figs):
    """
    Check if equal to within a certain number of significant digits.
    """
    if is_num(a) and is_num(b):
        return abs(round_to_n(a-b, sig_figs)) < pow(10, 1-sig_figs)
    raise ValueError(
        f'Invalid value for a or b in almost_equal (a: {a}, {type(a)}, '
        f'b: {b}, {type(b)})')


def string_to_list(s, split_on_dash=True, remove_duplicates=True, sort=True):
    """
    Return a list of numbers by splitting/expanding a string on any
    combination of commas, whitespaces, or dashes (when
    split_on_dash=True).
    e.g: '1, 3, 5-8, 12 ' -> [1, 3, 5, 6, 7, 8, 12]
    """
    if not isinstance(s, str):
        illegal_value(s, 's', location='string_to_list')
        return None
    if not s:
        return []
    try:
        list1 = re_split(r'\s+,\s+|\s+,|,\s+|\s+|,', s.strip())
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    if split_on_dash:
        try:
            l_of_i = []
            for v in list1:
                list2 = [
                    literal_eval(x)
                    for x in re_split(r'\s+-\s+|\s+-|-\s+|\s+|-', v)]
                if len(list2) == 1:
                    l_of_i += list2
                elif len(list2) == 2 and list2[1] > list2[0]:
                    l_of_i += list(range(list2[0], 1+list2[1]))
                else:
                    raise ValueError
        except (ValueError, TypeError, SyntaxError, MemoryError,
                RecursionError):
            return None
    else:
        l_of_i = [literal_eval(x) for x in list1]
    if remove_duplicates:
        l_of_i = list(dict.fromkeys(l_of_i))
    if sort:
        l_of_i = sorted(l_of_i)
    return l_of_i


def get_trailing_int(string):
    """Get the trailing integer in a string."""
    index_regex = re_compile(r'\d+$')
    match = index_regex.search(string)
    if match is None:
        return None
    return int(match.group())


def input_int(
        s=None, ge=None, gt=None, le=None, lt=None, default=None, inset=None,
        raise_error=False, log=True):
    """Interactively prompt the user to enter an integer."""
    return _input_int_or_num(
        'int', s, ge, gt, le, lt, default, inset, raise_error, log)


def input_num(
        s=None, ge=None, gt=None, le=None, lt=None, default=None,
        raise_error=False, log=True):
    """Interactively prompt the user to enter a number."""
    return _input_int_or_num(
        'num', s, ge, gt, le, lt, default, None, raise_error,log)


def _input_int_or_num(
        type_str, s=None, ge=None, gt=None, le=None, lt=None, default=None,
        inset=None, raise_error=False, log=True):
    """Interactively prompt the user to enter an integer or number."""
    if type_str == 'int':
        if not test_ge_gt_le_lt(
                ge, gt, le, lt, is_int, '_input_int_or_num', raise_error, log):
            return None
    elif type_str == 'num':
        if not test_ge_gt_le_lt(
                ge, gt, le, lt, is_num, '_input_int_or_num', raise_error, log):
            return None
    else:
        illegal_value(
            type_str, 'type_str', '_input_int_or_num', raise_error, log)
        return None
    if default is not None:
        if not _is_int_or_num(
                default, type_str, raise_error=raise_error, log=log):
            return None
        if ge is not None and default < ge:
            illegal_combination(
                ge, 'ge', default, 'default', '_input_int_or_num', raise_error,
                log)
            return None
        if gt is not None and default <= gt:
            illegal_combination(
                gt, 'gt', default, 'default', '_input_int_or_num', raise_error,
                log)
            return None
        if le is not None and default > le:
            illegal_combination(
                le, 'le', default, 'default', '_input_int_or_num', raise_error,
                log)
            return None
        if lt is not None and default >= lt:
            illegal_combination(
                lt, 'lt', default, 'default', '_input_int_or_num', raise_error,
                log)
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if inset is not None:
        if (not isinstance(inset, (tuple, list))
                or any(not isinstance(i, int) for i in inset)):
            illegal_value(
                inset, 'inset', '_input_int_or_num', raise_error, log)
            return None
    v_range = f'{range_string_ge_gt_le_lt(ge, gt, le, lt)}'
    if v_range:
        v_range = f' {v_range}'
    if s is None:
        if type_str == 'int':
            print(f'Enter an integer{v_range}{default_string}: ')
        else:
            print(f'Enter a number{v_range}{default_string}: ')
    else:
        print(f'{s}{v_range}{default_string}: ')
    try:
        i = input()
        if isinstance(i, str) and not i:
            v = default
            print(f'{v}')
        else:
            v = literal_eval(i)
        if inset and v not in inset:
            raise ValueError(f'{v} not part of the set {inset}')
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        v = None
    if not _is_int_or_num(v, type_str, ge, gt, le, lt):
        v = _input_int_or_num(
            type_str, s, ge, gt, le, lt, default, inset, raise_error, log)
    return v


def input_int_list(
        s=None, ge=None, le=None, split_on_dash=True, remove_duplicates=True,
        sort=True, raise_error=False, log=True):
    """
    Prompt the user to input a list of interger and split the entered
    string on any combination of commas, whitespaces, or dashes (when
    split_on_dash is True).
    e.g: '1 3,5-8 , 12 ' -> [1, 3, 5, 6, 7, 8, 12]

    remove_duplicates: removes duplicates if True (may also change the
        order)
    sort: sort in ascending order if True
    return None upon an illegal input
    """
    return _input_int_or_num_list(
        'int', s, ge, le, split_on_dash, remove_duplicates, sort, raise_error,
        log)


def input_num_list(
        s=None, ge=None, le=None, remove_duplicates=True, sort=True,
        raise_error=False, log=True):
    """
    Prompt the user to input a list of numbers and split the entered
    string on any combination of commas or whitespaces.
    e.g: '1.0, 3, 5.8, 12 ' -> [1.0, 3.0, 5.8, 12.0]

    remove_duplicates: removes duplicates if True (may also change the
        order)
    sort: sort in ascending order if True
    return None upon an illegal input
    """
    return _input_int_or_num_list(
        'num', s, ge, le, False, remove_duplicates, sort, raise_error, log)


def _input_int_or_num_list(
        type_str, s=None, ge=None, le=None, split_on_dash=True,
        remove_duplicates=True, sort=True, raise_error=False, log=True):
    # RV do we want a limit on max dimension?
    if type_str == 'int':
        if not test_ge_gt_le_lt(
                ge, None, le, None, is_int, 'input_int_or_num_list',
                raise_error, log):
            return None
    elif type_str == 'num':
        if not test_ge_gt_le_lt(
                ge, None, le, None, is_num, 'input_int_or_num_list',
                raise_error, log):
            return None
    else:
        illegal_value(type_str, 'type_str', '_input_int_or_num_list')
        return None
    v_range = f'{range_string_ge_gt_le_lt(ge=ge, le=le)}'
    if v_range:
        v_range = f' (each value in {v_range})'
    if s is None:
        print(f'Enter a series of integers{v_range}: ')
    else:
        print(f'{s}{v_range}: ')
    try:
        _list = string_to_list(input(), split_on_dash, remove_duplicates, sort)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        _list = None
    except:
        print('Unexpected error')
        raise
    if (not isinstance(_list, list) or any(
            not _is_int_or_num(v, type_str, ge=ge, le=le) for v in _list)):
        if split_on_dash:
            print('Invalid input: enter a valid set of dash/comma/whitespace '
                  'separated integers e.g. 1 3,5-8 , 12')
        else:
            print('Invalid input: enter a valid set of comma/whitespace '
                  'separated integers e.g. 1 3,5 8 , 12')
        _list = _input_int_or_num_list(
            type_str, s, ge, le, split_on_dash, remove_duplicates, sort,
            raise_error, log)
    return _list


def input_yesno(s=None, default=None):
    """Interactively prompt the user to enter a y/n question."""
    if default is not None:
        if not isinstance(default, str):
            illegal_value(default, 'default', 'input_yesno')
            return None
        if default.lower() in 'yes':
            default = 'y'
        elif default.lower() in 'no':
            default = 'n'
        else:
            illegal_value(default, 'default', 'input_yesno')
            return None
        default_string = f' [{default}]'
    else:
        default_string = ''
    if s is None:
        print(f'Enter yes or no{default_string}: ')
    else:
        print(f'{s}{default_string}: ')
    i = input()
    if isinstance(i, str) and not i:
        i = default
        print(f'{i}')
    if i is not None and i.lower() in 'yes':
        v = True
    elif i is not None and i.lower() in 'no':
        v = False
    else:
        print('Invalid input, enter yes or no')
        v = input_yesno(s, default)
    return v


def input_menu(items, default=None, header=None):
    """Interactively prompt the user to select from a menu."""
    if (not isinstance(items, (tuple, list))
            or any(not isinstance(i, str) for i in items)):
        illegal_value(items, 'items', 'input_menu')
        return None
    if default is not None:
        if not (isinstance(default, str) and default in items):
            logger.error(
                f'Invalid value for default ({default}), must be in {items}')
            return None
        default_string = f' [{1+items.index(default)}]'
    else:
        default_string = ''
    if header is None:
        print('Choose one of the following items '
              f'(1, {len(items)}){default_string}:')
    else:
        print(f'{header} (1, {len(items)}){default_string}:')
    for i, choice in enumerate(items):
        print(f'  {i+1}: {choice}')
    try:
        choice = input()
        if isinstance(choice, str) and not choice:
            choice = items.index(default)
            print(f'{1+choice}')
        else:
            choice = literal_eval(choice)
            if isinstance(choice, int) and 1 <= choice <= len(items):
                choice -= 1
            else:
                raise ValueError
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        choice = None
    except:
        print('Unexpected error')
        raise
    if choice is None:
        print(f'Invalid choice, enter a number between 1 and {len(items)}')
        choice = input_menu(items, default)
    return choice


def assert_no_duplicates_in_list_of_dicts(_list, raise_error=False):
    """
    Assert that there are no duplicates in a list of dictionaries.
    """
    if not isinstance(_list, list):
        illegal_value(
            _list, '_list', 'assert_no_duplicates_in_list_of_dicts',
            raise_error)
        return None
    if any(not isinstance(d, dict) for d in _list):
        illegal_value(
            _list, '_list', 'assert_no_duplicates_in_list_of_dicts',
            raise_error)
        return None
    if (len(_list) != len([dict(_tuple) for _tuple in
                          {tuple(sorted(d.items())) for d in _list}])):
        if raise_error:
            raise ValueError(f'Duplicate items found in {_list}')
        logger.error(f'Duplicate items found in {_list}')
        return None
    return _list


def assert_no_duplicate_key_in_list_of_dicts(_list, key, raise_error=False):
    """
    Assert that there are no duplicate keys in a list of dictionaries.
    """
    if not isinstance(key, str):
        illegal_value(
            key, 'key', 'assert_no_duplicate_key_in_list_of_dicts',
            raise_error)
        return None
    if not isinstance(_list, list):
        illegal_value(
            _list, '_list', 'assert_no_duplicate_key_in_list_of_dicts',
            raise_error)
        return None
    if any(isinstance(d, dict) for d in _list):
        illegal_value(
            _list, '_list', 'assert_no_duplicates_in_list_of_dicts',
            raise_error)
        return None
    keys = [d.get(key, None) for d in _list]
    if None in keys or len(set(keys)) != len(_list):
        if raise_error:
            raise ValueError(
                f'Duplicate or missing key ({key}) found in {_list}')
        logger.error(f'Duplicate or missing key ({key}) found in {_list}')
        return None
    return _list


def assert_no_duplicate_attr_in_list_of_objs(_list, attr, raise_error=False):
    """
    Assert that there are no duplicate attributes in a list of objects.
    """
    if not isinstance(attr, str):
        illegal_value(
            attr, 'attr', 'assert_no_duplicate_attr_in_list_of_objs',
            raise_error)
        return None
    if not isinstance(_list, list):
        illegal_value(
            _list, '_list', 'assert_no_duplicate_key_in_list_of_objs',
            raise_error)
        return None
    attrs = [getattr(obj, attr, None) for obj in _list]
    if None in attrs or len(set(attrs)) != len(_list):
        if raise_error:
            raise ValueError(
                f'Duplicate or missing attr ({attr}) found in {_list}')
        logger.error(f'Duplicate or missing attr ({attr}) found in {_list}')
        return None
    return _list


def file_exists_and_readable(f):
    """Check if a file exists and is readable."""
    if not os_path.isfile(f):
        raise ValueError(f'{f} is not a valid file')
    if not access(f, R_OK):
        raise ValueError(f'{f} is not accessible for reading')
    return f


def draw_mask_1d(
        ydata, xdata=None, label=None, ref_data=[],
        current_index_ranges=None, current_mask=None,
        select_mask=True, num_index_ranges_max=None,
        title=None, xlabel=None, ylabel=None,
        test_mode=False, return_figure=False):
    """Display a 2D plot and have the user select a mask.

    :param ydata: data array for which a mask will be constructed
    :type ydata: numpy.ndarray
    :param xdata: x-coordinates of the reference data, defaults to
        None
    :type xdata: numpy.ndarray, optional
    :param label: legend label for the reference data, defaults to
        None
    :type label: str, optional
    :param ref_data: a list of additional reference data to
        plot. Items in the list should be tuples of positional
        arguments and keyword arguments to unpack and pass directly to
        `matplotlib.axes.Axes.plot`, defaults to []
    :type ref_data: list[tuple[tuple, dict]]
    :param current_index_ranges: list of preselected index ranges to
        mask, defaults to None
    :type current_index_ranges: list[tuple[int, int]]
    :param current_mask: preselected boolean mask array, defaults to
        None
    :type current_mask: numpy.ndarray, optional
    :param select_mask: if True, user-selected ranges will be included
        when the returned mask is applied to `ydata`. If False, they
        will be excluded. Defaults to True.
    :type select_mask: bool, optional
    :param title: title for the displayed figure, defaults to None
    :type title: str, optional
    :param xlabel: label for the x-axis of the displayed figure,
        defaults to None
    :type xlabel: str, optional
    :param ylabel: label for the y-axis of the displayed figure,
        defaults to None
    :type ylabel: str, optional
    :param test_mode: if True, run as a non-interactive test
        case. Defaults to False
    :type test_mode: bool, optional
    :param return_figure: if True, also return a matplotlib figure of
        the drawn mask, defaults to False
    :type return_figure: bool, optional
    :return: a boolean mask array and the list of selected index
        ranges (and a matplotlib figure, if `return_figure` was True).
    :rtype: numpy.ndarray, list[tuple[int, int]] [, matplotlib.figure.Figure]
    """
    raise RuntimeError(
        'draw_mask_1d is obsolete, update to use widgets')
    # RV make color blind friendly
    def draw_selections(
            ax, current_include, current_exclude, selected_index_ranges):
        """Draw the selections."""
        ax.clear()
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.plot(xdata, ydata, 'k', label=label)
        for data in ref_data:
            ax.plot(*data[0], **data[1])
        ax.legend()
        for low, upp in current_include:
            xlow = 0.5 * (xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5 * (xdata[upp]+xdata[min(num_data-1, 1+upp)])
            ax.axvspan(xlow, xupp, facecolor='green', alpha=0.5)
        for low, upp in current_exclude:
            xlow = 0.5 * (xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5 * (xdata[upp]+xdata[min(num_data-1, 1+upp)])
            ax.axvspan(xlow, xupp, facecolor='red', alpha=0.5)
        for low, upp in selected_index_ranges:
            xlow = 0.5 * (xdata[max(0, low-1)]+xdata[low])
            xupp = 0.5 * (xdata[upp]+xdata[min(num_data-1, 1+upp)])
            ax.axvspan(xlow, xupp, facecolor=selection_color, alpha=0.5)
        ax.get_figure().canvas.draw()

    def onclick(event):
        """Action taken on clicking the mouse button."""
        if event.inaxes in [fig.axes[0]]:
            selected_index_ranges.append(index_nearest_upp(xdata, event.xdata))

    def onrelease(event):
        """Action taken on releasing the mouse button."""
        if selected_index_ranges:
            if isinstance(selected_index_ranges[-1], int):
                if event.inaxes in [fig.axes[0]]:
                    event.xdata = index_nearest_low(xdata, event.xdata)
                    if selected_index_ranges[-1] <= event.xdata:
                        selected_index_ranges[-1] = \
                            (selected_index_ranges[-1], event.xdata)
                    else:
                        selected_index_ranges[-1] = \
                            (event.xdata, selected_index_ranges[-1])
                    draw_selections(
                        event.inaxes, current_include, current_exclude,
                        selected_index_ranges)
                else:
                    selected_index_ranges.pop(-1)

    def confirm_selection(event):
        """Action taken on hitting the confirm button."""
        plt.close()

    def clear_last_selection(event):
        """Action taken on hitting the clear button."""
        if selected_index_ranges:
            selected_index_ranges.pop(-1)
        else:
            while current_include:
                current_include.pop()
            while current_exclude:
                current_exclude.pop()
            selected_mask.fill(False)
        draw_selections(
            ax, current_include, current_exclude, selected_index_ranges)

    def update_mask(mask, selected_index_ranges, unselected_index_ranges):
        """Update the plot with the selected mask."""
        for low, upp in selected_index_ranges:
            selected_mask = np.logical_and(
                xdata >= xdata[low], xdata <= xdata[upp])
            mask = np.logical_or(mask, selected_mask)
        for low, upp in unselected_index_ranges:
            unselected_mask = np.logical_and(
                xdata >= xdata[low], xdata <= xdata[upp])
            mask[unselected_mask] = False
        return mask

    def update_index_ranges(mask):
        """
        Update the currently included index ranges (where mask = True).
        """
        current_include = []
        for i, m in enumerate(mask):
            if m:
                if (not current_include
                        or isinstance(current_include[-1], tuple)):
                    current_include.append(i)
            else:
                if current_include and isinstance(current_include[-1], int):
                    current_include[-1] = (current_include[-1], i-1)
        if current_include and isinstance(current_include[-1], int):
            current_include[-1] = (current_include[-1], num_data-1)
        return current_include

    # Check inputs
    ydata = np.asarray(ydata)
    if ydata.ndim > 1:
        logger.warning(f'Invalid ydata dimension ({ydata.ndim})')
        return None, None
    num_data = ydata.size
    if xdata is None:
        xdata = np.arange(num_data)
    else:
        xdata = np.asarray(xdata, dtype=np.float64)
        if xdata.ndim > 1 or xdata.size != num_data:
            logger.warning(f'Invalid xdata shape ({xdata.shape})')
            return None, None
        if not np.all(xdata[:-1] < xdata[1:]):
            logger.warning('Invalid xdata: must be monotonically increasing')
            return None, None
    if current_index_ranges is not None:
        if not isinstance(current_index_ranges, (tuple, list)):
            logger.warning(
                'Invalid current_index_ranges parameter '
                f'({current_index_ranges}, {type(current_index_ranges)})')
            return None, None
    if not isinstance(select_mask, bool):
        logger.warning(
            f'Invalid select_mask parameter ({select_mask}, '
            f'{type(select_mask)})')
        return None, None
    if num_index_ranges_max is not None:
        logger.warning(
            'num_index_ranges_max input not yet implemented in draw_mask_1d')
    if title is None:
        title = 'select ranges of data'
    elif not isinstance(title, str):
        illegal_value(title, 'title')
        title = ''

    if select_mask:
        selection_color = 'green'
    else:
        selection_color = 'red'

    # Set initial selected mask and the selected/unselected index
    #     ranges as needed
    selected_index_ranges = []
    unselected_index_ranges = []
    selected_mask = np.full(xdata.shape, False, dtype=bool)
    if current_index_ranges is None:
        if current_mask is None:
            if not select_mask:
                selected_index_ranges = [(0, num_data-1)]
                selected_mask = np.full(xdata.shape, True, dtype=bool)
        else:
            selected_mask = np.copy(np.asarray(current_mask, dtype=bool))
    if current_index_ranges is not None and current_index_ranges:
        current_index_ranges = sorted(list(current_index_ranges))
        for low, upp in current_index_ranges:
            if low > upp or low >= num_data or upp < 0:
                continue
            low = max(low, 0)
            upp = min(upp, num_data-1)
            selected_index_ranges.append((low, upp))
        selected_mask = update_mask(
            selected_mask, selected_index_ranges, unselected_index_ranges)
    if current_index_ranges is not None and current_mask is not None:
        selected_mask = np.logical_and(current_mask, selected_mask)
    if current_mask is not None:
        selected_index_ranges = update_index_ranges(selected_mask)

    # Set up range selections for display
    current_include = selected_index_ranges
    current_exclude = []
    selected_index_ranges = []
    if not current_include:
        if select_mask:
            current_exclude = [(0, num_data-1)]
        else:
            current_include = [(0, num_data-1)]
    else:
        if current_include[0][0] > 0:
            current_exclude.append((0, current_include[0][0]-1))
        for i in range(1, len(current_include)):
            current_exclude.append(
                (1+current_include[i-1][1], current_include[i][0]-1))
        if current_include[-1][1] < num_data-1:
            current_exclude.append((1+current_include[-1][1], num_data-1))

    # Set up matplotlib figure
    plt.close('all')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    draw_selections(
        ax, current_include, current_exclude, selected_index_ranges)

    if not test_mode:
        # Set up event handling for click-and-drag range selection
        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_release = fig.canvas.mpl_connect('button_release_event', onrelease)

        # Set up confirm / clear range selection buttons
        confirm_b = Button(plt.axes([0.75, 0.015, 0.15, 0.075]), 'Confirm')
        clear_b = Button(plt.axes([0.59, 0.015, 0.15, 0.075]), 'Clear')
        cid_confirm = confirm_b.on_clicked(confirm_selection)
        cid_clear = clear_b.on_clicked(clear_last_selection)

        # Show figure
        plt.show(block=True)

        # Disconnect callbacks when figure is closed
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_release)
        confirm_b.disconnect(cid_confirm)
        clear_b.disconnect(cid_clear)

        # Remove buttons & readjust axes before returning a figure
        if return_figure:
            confirm_b.ax.remove()
            clear_b.ax.remove()
            plt.subplots_adjust(bottom=0.0)

    # Swap selection depending on select_mask
    if not select_mask:
        selected_index_ranges, unselected_index_ranges = \
            unselected_index_ranges, selected_index_ranges

    # Update the mask with the currently selected/unselected x-ranges
    selected_mask = update_mask(
        selected_mask, selected_index_ranges, unselected_index_ranges)

    # Update the currently included index ranges (where mask is True)
    current_include = update_index_ranges(selected_mask)

    if return_figure:
        return selected_mask, current_include, fig
    return selected_mask, current_include

def select_roi_2d(
        a, preselected_roi=None, title=None, title_a=None,
        row_label='row_index', column_label='column_index', interactive=True):
    """Display a 2D image and have the user select a single rectangular
       region of interest.

    :param a: Two-dimensional image data array for which a region of
        interest will be selected.
    :type a: numpy.ndarray
    :param preselected_roi: Preselected region of interest,
        defaults to `None`.
    :type preselected_roi: tuple(int, int, int, int)
    :param title: Title for the displayed figure, defaults to `None`.
    :type title: str, optional
    :param title_a: Title for the image of a, defaults to `None`.
    :type title_a: str, optional
    :param row_label: Label for the y-axis of the displayed figure,
        defaults to `row_index`.
    :type row_label: str, optional
    :param column_label: Label for the x-axis of the displayed figure,
        defaults to `column_index`.
    :type column_label: str, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `True`.
    :type interactive: bool, optional
    :return: The selected region of interest as array indices and a
        matplotlib figure.
    :rtype: matplotlib.figure.Figure, tuple(int, int, int, int)
    """
    # Third party modules
    from matplotlib.widgets import RectangleSelector

    # Local modules
    from CHAP.utils.general import index_nearest

    def on_rect_select(eclick, erelease):
        """Callback function for the RectangleSelector widget."""
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        error_texts.append(
            plt.figtext(
                *error_pos,
                f'Selected ROI: {tuple(int(v) for v in rects[0].extents)}',
                **error_props))
        plt.draw()

    def reset(event):
        rects[0].set_visible(False)
        rects.pop()
        rects.append(
            RectangleSelector(
                ax, on_rect_select, props=rect_props, useblit=True,
                interactive=interactive, drag_from_anywhere=True,
                ignore_event_outside=False))
        plt.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        fig_title[0].remove()
        fig_title.pop()
        roi = tuple(int(v) for v in rects[0].extents)
        if roi[1]-roi[0] < 1 or roi[3]-roi[2] < 1:
            roi = None
        fig_title.append(
            plt.figtext(*title_pos, f'Selected ROI: {roi}', **title_props))
        plt.close()

    fig_title = []
    error_texts = []

    # Check inputs
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f'Invalid image dimension ({a.ndim})')
    if preselected_roi is not None:
        if (not is_int_series(preselected_roi, ge=0, log=False)
                or len(preselected_roi) != 4):
            raise ValueError('Invalid parameter preselected_roi '
                             f'({preselected_roi})')
    if title is None:
        title = 'Click and drag to select or adjust a region of interest (ROI)'

    title_pos = (0.5, 0.95)
    title_props = {'fontsize': 'x-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    error_pos = (0.5, 0.90)
    error_props = {'fontsize': 'x-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    rect_props = {
        'alpha': 0.5, 'facecolor': 'tab:blue', 'edgecolor': 'blue'}

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(a)
    ax.set(title=title_a, xlabel=column_label, ylabel=row_label)
    ax.set_xlim(0, a.shape[1])
    ax.set_ylim(a.shape[0], 0)
    fig.subplots_adjust(bottom=0.0, top=0.85)

    # Setup the preselected range of interest if provided
    rects = [RectangleSelector(
        ax, on_rect_select, props=rect_props, useblit=True,
        interactive=interactive, drag_from_anywhere=True,
        ignore_event_outside=True)]
    if preselected_roi is not None:
        rects[0].extents = preselected_roi

    if not interactive:

        fig_title.append(
            plt.figtext(
                *title_pos,
                f'Selected ROI: {[int(v) for v in preselected_roi]}',
                **title_props))

    else:

        fig_title.append(plt.figtext(*title_pos, title, **title_props))
        if preselected_roi is not None:
            error_texts.append(
                plt.figtext(
                    *error_pos,
                    f'Preselected ROI: {[int(v) for v in preselected_roi]}',
                    **error_props))
        fig.subplots_adjust(bottom=0.2)

        # Setup "Reset" button
        reset_btn = Button(plt.axes([0.125, 0.05, 0.15, 0.075]), 'Reset')
        reset_cid = reset_btn.on_clicked(reset)

        # Setup "Confirm" button
        confirm_btn = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        confirm_cid = confirm_btn.on_clicked(confirm)

        # Show figure for user interaction
        plt.show()

        # Disconnect all widget callbacks when figure is closed
        reset_btn.disconnect(reset_cid)
        confirm_btn.disconnect(confirm_cid)

        # ... and remove the buttons before returning the figure
        reset_btn.ax.remove()
        confirm_btn.ax.remove()

    fig_title[0].set_in_layout(True)
    fig.tight_layout(rect=(0,0,1,0.95))

    # Remove the handles before returning the figure
    rects[0]._center_handle.set_visible(False)
    rects[0]._corner_handles.set_visible(False)
    rects[0]._edge_handles.set_visible(False)

    roi = tuple(int(v) for v in rects[0].extents)
    if roi[1]-roi[0] < 1 or roi[3]-roi[2] < 1:
        roi = None
#    else:
#        # Remove the handles before returning the figure
#        rects[0]._center_handle.set_visible(False)
#        rects[0]._corner_handles.set_visible(False)
#        rects[0]._edge_handles.set_visible(False)

    return fig, roi


def select_peaks(
        ydata, xdata, peak_locations,
        peak_labels=None,
        mask=None,
        pre_selected_peak_indices=[],
        return_sorted=True,
        title='', xlabel='', ylabel='',
        interactive=True):
    """
    Show a plot of the 1D data provided with user-selectable markers
    at the given locations. Return the locations of the markers that
    the user selected with their mouse interactions.

    :param ydata: 1D array of values to plot
    :type ydata: numpy.ndarray
    :param xdata: values of the independent dimension corresponding to
        ydata, defaults to None
    :type xdata: numpy.ndarray, optional
    :param peak_locations: locations of selectable markers in the same
        units as xdata.
    :type peak_locations: list
    :param peak_labels: list of annotations for each peak, defaults to
        None
    :type peak_labels: list[str], optional
    :param mask: boolean array representing a mask that will be
        applied to the data at some later point, defaults to None
    :type mask: np.ndarray, optional
    :param pre_selected_peak_indices: indices of markers that should
        already be selected when the figure shows up, defaults to []
    :type pre_selected_peak_indices: list[int], optional
    :param return_sorted: sort the indices of selected markers before
        returning (otherwise: return them in the same order that the
        user selected them), defaults to True
    :type return_sorted: bool, optional
    :param title: title for the plot, defaults to ''
    :type title: str, optional
    :param xlabel: x-axis label for the plot, defaults to ''
    :type xlabel: str, optional
    :param ylabel: y-axis label for the plot, defaults to ''
    :type ylabel: str, optional
    :param interactive: show the plot and allow user interactions with
        the matplotlib figure, defults to True
    :type interactive: bool, optional
    :return: the locations of the user-selected peaks
    :rtype: list
    """
    raise RuntimeError(
        'select_peaks is obsolete, use version in CHAP.edd.utils')

    if ydata.size != xdata.size:
        raise ValueError('x and y data must have the same size')
    if mask is not None and mask.size != ydata.size:
        raise ValueError('mask must have the same size as data')


    excluded_peak_props = {
        'color': 'black', 'linestyle': '--','linewidth': 1,
        'marker': 10, 'markersize': 5, 'fillstyle': 'none'}
    included_peak_props = {
        'color': 'green', 'linestyle': '-', 'linewidth': 2,
        'marker': 10, 'markersize': 10, 'fillstyle': 'full'}
    masked_peak_props = {
        'color': 'gray', 'linestyle': ':', 'linewidth': 1}

    # Setup reference data & plot
    if mask is None:
        mask = np.full(ydata.shape, True, dtype=bool)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    handles = ax.plot(xdata, ydata, label='Reference data')
    handles.append(mlines.Line2D(
        [], [], label='Excluded / unselected', **excluded_peak_props))
    handles.append(mlines.Line2D(
        [], [], label='Included / selected', **included_peak_props))
    handles.append(mlines.Line2D(
        [], [], label='In masked region (unselectable)', **masked_peak_props))
    ax.legend(handles=handles, loc='center right')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()

    # Plot a vertical line marker at each peak location
    peak_vlines = []
    x_indices = np.arange(ydata.size)
    if peak_labels is None:
        peak_labels = [''] * len(peak_locations)
    for i, (loc, lbl) in enumerate(zip(peak_locations, peak_labels)):
        nearest_index = np.searchsorted(xdata, loc)
        if nearest_index in x_indices[mask]:
            if i in pre_selected_peak_indices:
                peak_vline = ax.axvline(loc, **included_peak_props)
            else:
                peak_vline = ax.axvline(loc, **excluded_peak_props)
            peak_vline.set_picker(5)
        else:
            if i in pre_selected_peak_indices:
                logger.warning(
                    f'Pre-selected peak index {i} is in a masked region and '
                    'will not be selectable.')
                pre_selected_peak_indices.remove(i)
            peak_vline = ax.axvline(loc, **masked_peak_props)
        ax.text(loc, 1, lbl, ha='right', va='top', rotation=90,
                transform=ax.get_xaxis_transform())
        peak_vlines.append(peak_vline)

    # Indicate masked regions by gray-ing out the axes facecolor
    exclude_bounds = []
    for i, m in enumerate(mask):
        if not m:
            if (not exclude_bounds) or isinstance(exclude_bounds[-1], tuple):
                exclude_bounds.append(i)
        else:
            if exclude_bounds and isinstance(exclude_bounds[-1], int):
                exclude_bounds[-1] = (exclude_bounds[-1], i-1)
    if exclude_bounds and isinstance(exclude_bounds[-1], int):
        exclude_bounds[-1] = (exclude_bounds[-1], mask.size-1)
    for (low, upp) in exclude_bounds:
        xlow = xdata[low]
        xupp = xdata[upp]
        ax.axvspan(xlow, xupp, facecolor='gray', alpha=0.5)

    selected_peak_indices = pre_selected_peak_indices
    if interactive:
        # Setup interative peak selection
        def onpick(event):
            try:
                peak_index = peak_vlines.index(event.artist)
            except:
                pass
            else:
                peak_vline = event.artist
                if peak_index in selected_peak_indices:
                    peak_vline.set(**excluded_peak_props)
                    selected_peak_indices.remove(peak_index)
                else:
                    peak_vline.set(**included_peak_props)
                    selected_peak_indices.append(peak_index)
                plt.draw()
        cid_pick_peak = fig.canvas.mpl_connect('pick_event', onpick)

        # Setup "Confirm" button
        def confirm_selection(event):
            plt.close()
        plt.subplots_adjust(bottom=0.2)
        confirm_b = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        cid_confirm = confirm_b.on_clicked(confirm_selection)

        # Show figure for user interaction
        plt.show()

        # Disconnect all widget callbacks when figure is closed
        fig.canvas.mpl_disconnect(cid_pick_peak)
        confirm_b.disconnect(cid_confirm)

        # ...and remove the buttons before returning the figure
        confirm_b.ax.remove()
        plt.subplots_adjust(bottom=0.0)

    selected_peaks = peak_locations[selected_peak_indices]
    if return_sorted:
        selected_peaks.sort()
    return selected_peaks, fig


def select_image_indices(
        a, axis, b=None, preselected_indices=None, min_range=None,
        min_num_indices=2, max_num_indices=2, title=None,
        title_a=None, title_b=None, row_label='row index',
        column_label='column index', interactive=True):
    """Display a 2D image and have the user select a set of image
       indices in either row or column direction. 

    :param a: Two-dimensional image data array for which a region of
        interest will be selected.
    :type a: numpy.ndarray
    :param axis: The selection direction (0: row, 1: column)
    :type axis: int
    :param b: A secondary two-dimensional image data array for which
        a shared region of interest will be selected,
        defaults to `None`.
    :type b: numpy.ndarray, optional
    :param preselected_indices: Preselected image indices,
        defaults to `None`.
    :type preselected_roi: tuple(int), list(int)
    :param min_range: The minimal range spanned by the selected 
        indices, defaults to `None`
    :type min_range: int, optional
    :param title: Title for the displayed figure, defaults to `None`.
    :type title: str, optional
    :param title_a: Title for the image of a, defaults to `None`.
    :type title_a: str, optional
    :param title_b: Title for the image of b, defaults to `None`.
    :type title_b: str, optional
    :param row_label: Label for the y-axis of the displayed figure,
        defaults to `row_index`.
    :type row_label: str, optional
    :param column_label: Label for the x-axis of the displayed figure,
        defaults to `column_index`.
    :type column_label: str, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `True`.
    :type interactive: bool, optional
    :return: The selected region of interest as array indices and a
        matplotlib figure.
    :rtype: matplotlib.figure.Figure, tuple(int, int, int, int)
    """
    # Third party modules
    from matplotlib.widgets import TextBox, Button

    def add_index(index):
        if index in indices:
            raise ValueError(f'Ignoring duplicate of selected {row_column}s')
        elif len(indices) >= max_num_indices:
            raise ValueError(
                f'Exceeding maximum number of selected {row_column}s, click '
                'either "Reset" or "Confirm"')
        elif (len(indices) and min_range is not None
                and abs(max(index, max(indices)) - min(index, min(indices)))
                    < min_range):
            raise ValueError(
                f'Selected {row_column} range is smaller than the required '
                'minimal range of {min_range}: ignoring last selection')
        else:
            indices.append(index)
            if not axis:
                for ax in axs:
                    lines.append(ax.axhline(indices[-1], c='r', lw=2))
            else:
                for ax in axs:
                    lines.append(ax.axvline(indices[-1], c='r', lw=2))

    def select_index(expression):
        """Callback function for the "Select row/column index" TextBox.
        """
        if not len(expression):
            return
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        try:
            index = int(expression)
            if not 0 <= index <= a.shape[axis]:
                raise ValueError
        except ValueError:
            error_texts.append(
                plt.figtext(
                    *error_pos,
                    f'Invalid {row_column} index ({expression}), enter an '
                    f'integer value between 0 and {a.shape[axis]-1}',
                    **error_props))
        else:
            try:
                add_index(index)
            except ValueError as e:
                error_texts.append(plt.figtext(*error_pos, e, **error_props))
        index_input.set_val('')
        for ax in axs:
            ax.get_figure().canvas.draw()

    def reset(event):
        """Callback function for the "Reset" button."""
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        for line in reversed(lines):
            line.remove()
        indices.clear()
        lines.clear()
        for ax in axs:
            ax.get_figure().canvas.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if len(indices) < min_num_indices:
            if not error_texts:
                error_texts.append(
                    plt.figtext(
                        *error_pos,
                        f'Select at least {min_num_indices} unique '
                        f'{row_column}s',
                        **error_props))
            for ax in axs:
                ax.get_figure().canvas.draw()
        else:
            # Remove error texts and add selected indices if set
            if error_texts:
                error_texts[0].remove()
                error_texts.pop()
            fig_title[0].remove()
            fig_title.pop()
            fig_title.append(
                plt.figtext(
                    *title_pos,
                    f'Selected row indices: {sorted(indices)}',
                    **title_props))
            plt.close()

    # Check inputs
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f'Invalid image dimension ({a.ndim})')
    if axis < 0 or axis >= a.ndim:
        raise ValueError(f'Invalid parameter axis ({axis})')
    if not axis:
        row_column = 'row'
    else:
        row_column = 'column'
    if preselected_indices is not None:
        if not is_int_series(
                preselected_indices, ge=0, le=a.shape[axis], log=False):
            if interactive:
                logger.warning(
                    'Invalid parameter preselected_indices '
                    f'({preselected_indices}), ignoring preselected_indices')
                preselected_indices = None
            else:
                raise ValueError('Invalid parameter preselected_indices '
                                 f'({preselected_indices})')
    if min_range is not None and not 2 <= min_range <= a.shape[axis]:
        raise ValueError('Invalid parameter min_range ({min_range})')
    if title is None:
        title = f'Select or adjust image {row_column} indices'
    if b is not None:
        b = np.asarray(b)
        if b.ndim != 2:
            raise ValueError(f'Invalid image dimension ({b.ndim})')
        if a.shape[0] != b.shape[0]:
            raise ValueError(f'Inconsistent image shapes({a.shape} vs '
                             f'{b.shape})')
        
    indices = []
    lines = []
    fig_title = []
    error_texts = []

    title_pos = (0.5, 0.95)
    title_props = {'fontsize': 'x-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    error_pos = (0.5, 0.90)
    error_props = {'fontsize': 'x-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    if b is None:
       fig, axs = plt.subplots(figsize=(11, 8.5))
       axs = [axs]
    else:
       if a.shape[0]+b.shape[0] > max(a.shape[1], b.shape[1]):
           fig, axs = plt.subplots(1, 2, figsize=(11, 8.5))
       else:
           fig, axs = plt.subplots(2, 1, figsize=(11, 8.5))
    axs[0].imshow(a)
    if b is None:
        axs[0].set(title=title_a, xlabel=column_label, ylabel=row_label)
    else:
        axs[1].imshow(b)
        if a.shape[0]+b.shape[0] > max(a.shape[1], b.shape[1]):
            axs[0].set(title=title_a, xlabel=column_label, ylabel=row_label)
            axs[1].set(title=title_b, xlabel=column_label)
        else:
            axs[0].set(title=title_a, ylabel=row_label)
            axs[1].set(title=title_b, xlabel=column_label, ylabel=row_label)
    for ax in axs:
        ax.set_xlim(0, a.shape[1])
        ax.set_ylim(a.shape[0], 0)
    fig.subplots_adjust(bottom=0.0, top=0.85)

    # Setup the preselected indices if provided
    if preselected_indices is not None:
        preselected_indices = sorted(list(preselected_indices))
        for index in preselected_indices:
            add_index(index)   

    if not interactive:

        fig_title.append(
            plt.figtext(
                *title_pos,
                f'Selected row indices: {preselected_indices}',
                **title_props))

    else:

        fig_title.append(plt.figtext(*title_pos, title, **title_props))
        if preselected_indices is not None:
            error_texts.append(
                plt.figtext(
                    *error_pos,
                    f'Preselected row indices: {preselected_indices}',
                    **error_props))
        fig.subplots_adjust(bottom=0.2)

        # Setup TextBox
        index_input = TextBox(
            plt.axes([0.25, 0.05, 0.15, 0.075]), f'Select {row_column} index ')
        indices_cid = index_input.on_submit(select_index)

        # Setup "Reset" button
        reset_btn = Button(plt.axes([0.5, 0.05, 0.15, 0.075]), 'Reset')
        reset_cid = reset_btn.on_clicked(reset)

        # Setup "Confirm" button
        confirm_btn = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        confirm_cid = confirm_btn.on_clicked(confirm)

        plt.show()

        # Disconnect all widget callbacks when figure is closed
        index_input.disconnect(indices_cid)
        reset_btn.disconnect(reset_cid)
        confirm_btn.disconnect(confirm_cid)

        # ... and remove the buttons before returning the figure
        index_input.ax.remove()
        reset_btn.ax.remove()
        confirm_btn.ax.remove()

    fig_title[0].set_in_layout(True)
    fig.tight_layout(rect=(0,0,1,0.95))

    if len(indices):
        return fig, tuple(sorted(indices))
    return fig, None


def select_one_image_bound(
        a, axis, bound=None, bound_name=None, title='select array bounds',
        default='y', raise_error=False):
    """
    Interactively select a data boundary for a 2D numpy array.
    """
    raise RuntimeError(
        'select_one_image_bound is obsolete, update to use widgets')
    a = np.asarray(a)
    if a.ndim != 2:
        illegal_value(
            a.ndim, 'array dimension', location='select_one_image_bound',
            raise_error=raise_error)
        return None
    if axis < 0 or axis >= a.ndim:
        illegal_value(
            axis, 'axis', location='select_one_image_bound',
            raise_error=raise_error)
        return None
    if bound_name is None:
        bound_name = 'data bound'
    if bound is None:
        min_ = 0
        max_ = a.shape[axis]
        bound_max = a.shape[axis]-1
        while True:
            if axis:
                quick_imshow(
                    a[:,min_:max_], title=title, aspect='auto',
                    extent=[min_,max_,a.shape[0],0])
            else:
                quick_imshow(
                    a[min_:max_,:], title=title, aspect='auto',
                    extent=[0,a.shape[1], max_,min_])
            zoom_flag = input_yesno(
                f'Set {bound_name} (y) or zoom in (n)?', 'y')
            if zoom_flag:
                bound = input_int(f'    Set {bound_name}', ge=0, le=bound_max)
                clear_imshow(title)
                break
            min_ = input_int('    Set lower zoom index', ge=0, le=bound_max)
            max_ = input_int(
                '    Set upper zoom index', ge=min_+1, le=bound_max+1)

    elif not is_int(bound, ge=0, le=a.shape[axis]-1):
        illegal_value(
            bound, 'bound', location='select_one_image_bound',
            raise_error=raise_error)
        return None
    else:
        print(f'Current {bound_name} = {bound}')
    a_tmp = np.copy(a)
    a_tmp_max = a.max()
    if axis:
        a_tmp[:,bound] = a_tmp_max
    else:
        a_tmp[bound,:] = a_tmp_max
    quick_imshow(a_tmp, title=title, aspect='auto')
    del a_tmp
    if not input_yesno(f'Accept this {bound_name} (y/n)?', default):
        bound = select_one_image_bound(
            a, axis, bound_name=bound_name, title=title)
    clear_imshow(title)
    return bound


def clear_imshow(title=None):
    """Clear an image opened by quick_imshow()."""
    plt.ioff()
    if title is None:
        title = 'quick imshow'
    elif not isinstance(title, str):
        raise ValueError(f'Invalid parameter title ({title})')
    plt.close(fig=title)


def clear_plot(title=None):
    """Clear an image opened by quick_plot()."""
    plt.ioff()
    if title is None:
        title = 'quick plot'
    elif not isinstance(title, str):
        raise ValueError(f'Invalid parameter title ({title})')
    plt.close(fig=title)


def quick_imshow(
        a, title=None, path=None, name=None, save_fig=False, save_only=False,
        clear=True, extent=None, show_grid=False, grid_color='w',
        grid_linewidth=1, block=False, **kwargs):
    """Display a 2D image."""
    if title is not None and not isinstance(title, str):
        raise ValueError(f'Invalid parameter title ({title})')
    if path is not None and not isinstance(path, str):
        raise ValueError(f'Invalid parameter path ({path})')
    if not isinstance(save_fig, bool):
        raise ValueError(f'Invalid parameter save_fig ({save_fig})')
    if not isinstance(save_only, bool):
        raise ValueError(f'Invalid parameter save_only ({save_only})')
    if not isinstance(clear, bool):
        raise ValueError(f'Invalid parameter clear ({clear})')
    if not isinstance(block, bool):
        raise ValueError(f'Invalid parameter block ({block})')
    if not title:
        title = 'quick imshow'
    if name is None:
        ttitle = re_sub(r'\s+', '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if ('cmap' in kwargs and a.ndim == 3
            and (a.shape[2] == 3 or a.shape[2] == 4)):
        use_cmap = True
        if a.shape[2] == 4 and a[:,:,-1].min() != a[:,:,-1].max():
            use_cmap = False
        if any(
                a[i,j,0] != a[i,j,1] and a[i,j,0] != a[i,j,2]
                for i in range(a.shape[0])
                for j in range(a.shape[1])):
            use_cmap = False
        if use_cmap:
            a = a[:,:,0]
        else:
            logger.warning('Image incompatible with cmap option, ignore cmap')
            kwargs.pop('cmap')
    if extent is None:
        extent = (0, a.shape[1], a.shape[0], 0)
    if clear:
        try:
            plt.close(fig=title)
        except:
            pass
    if not save_only:
        if block:
            plt.ioff()
        else:
            plt.ion()
    plt.figure(title)
    plt.imshow(a, extent=extent, **kwargs)
    if show_grid:
        ax = plt.gca()
        ax.grid(color=grid_color, linewidth=grid_linewidth)
#    if title != 'quick imshow':
#        plt.title = title
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)
        if block:
            plt.show(block=block)


def quick_plot(
        *args, xerr=None, yerr=None, vlines=None, title=None, xlim=None,
        ylim=None, xlabel=None, ylabel=None, legend=None, path=None, name=None,
        show_grid=False, save_fig=False, save_only=False, clear=True,
        block=False, **kwargs):
    """Display a 2D line plot."""
    if title is not None and not isinstance(title, str):
        illegal_value(title, 'title', 'quick_plot')
        title = None
    if (xlim is not None and not isinstance(xlim, (tuple, list))
            and len(xlim) != 2):
        illegal_value(xlim, 'xlim', 'quick_plot')
        xlim = None
    if (ylim is not None and not isinstance(ylim, (tuple, list))
            and len(ylim) != 2):
        illegal_value(ylim, 'ylim', 'quick_plot')
        ylim = None
    if xlabel is not None and not isinstance(xlabel, str):
        illegal_value(xlabel, 'xlabel', 'quick_plot')
        xlabel = None
    if ylabel is not None and not isinstance(ylabel, str):
        illegal_value(ylabel, 'ylabel', 'quick_plot')
        ylabel = None
    if legend is not None and not isinstance(legend, (tuple, list)):
        illegal_value(legend, 'legend', 'quick_plot')
        legend = None
    if path is not None and not isinstance(path, str):
        illegal_value(path, 'path', 'quick_plot')
        return
    if not isinstance(show_grid, bool):
        illegal_value(show_grid, 'show_grid', 'quick_plot')
        return
    if not isinstance(save_fig, bool):
        illegal_value(save_fig, 'save_fig', 'quick_plot')
        return
    if not isinstance(save_only, bool):
        illegal_value(save_only, 'save_only', 'quick_plot')
        return
    if not isinstance(clear, bool):
        illegal_value(clear, 'clear', 'quick_plot')
        return
    if not isinstance(block, bool):
        illegal_value(block, 'block', 'quick_plot')
        return
    if title is None:
        title = 'quick plot'
    if name is None:
        ttitle = re_sub(r'\s+', '_', title)
        if path is None:
            path = f'{ttitle}.png'
        else:
            path = f'{path}/{ttitle}.png'
    else:
        if path is None:
            path = name
        else:
            path = f'{path}/{name}'
    if clear:
        try:
            plt.close(fig=title)
        except:
            pass
    args = unwrap_tuple(args)
    if depth_tuple(args) > 1 and (xerr is not None or yerr is not None):
        logger.warning('Error bars ignored form multiple curves')
    if not save_only:
        if block:
            plt.ioff()
        else:
            plt.ion()
    plt.figure(title)
    if depth_tuple(args) > 1:
        for y in args:
            plt.plot(*y, **kwargs)
    else:
        if xerr is None and yerr is None:
            plt.plot(*args, **kwargs)
        else:
            plt.errorbar(*args, xerr=xerr, yerr=yerr, **kwargs)
    if vlines is not None:
        if isinstance(vlines, (int, float)):
            vlines = [vlines]
        for v in vlines:
            plt.axvline(v, color='r', linestyle='--', **kwargs)
#    if vlines is not None:
#        for s in tuple(
#                ([x, x], list(plt.gca().get_ylim())) for x in vlines):
#            plt.plot(*s, color='red', **kwargs)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if show_grid:
        ax = plt.gca()
        ax.grid(color='k')  # , linewidth=1)
    if legend is not None:
        plt.legend(legend)
    if save_only:
        plt.savefig(path)
        plt.close(fig=title)
    else:
        if save_fig:
            plt.savefig(path)
        plt.show(block=block)
