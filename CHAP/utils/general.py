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
except ImportError:
    pass

logger = getLogger(__name__)

tiny = np.finfo(np.float64).resolution

def gformat(val, length=11):
    """
    Format a number with '%g'-like format, while:
      - the length of the output string will be of the requested length
      - positive numbers will have a leading blank
      - the precision will be as high as possible
      - trailing zeros will not be trimmed
    """
    # Code taken from lmfit library
    if val is None or isinstance(val, bool):
        return f'{repr(val):>{length}s}'
    try:
        expon = int(np.log10(abs(val)))
    except (OverflowError, ValueError):
        expon = 0
    except TypeError:
        return f'{repr(val):>{length}s}'

    length = max(length, 7)
    form = 'e'
    prec = length - 7
    if abs(expon) > 99:
        prec -= 1
    elif 0 < expon < prec+4 or -expon < prec-1 <= 0:
        form = 'f'
        prec += 4
        if expon > 0:
            prec -= expon
    return f'{val:{length}.{prec}{form}}'


def getfloat_attr(obj, attr, length=11):
    """Format an attribute of an object for printing."""
    # Code taken from lmfit library
    val = getattr(obj, attr, None)
    if val is None:
        return 'unknown'
    if isinstance(val, int):
        return f'{val}'
    if isinstance(val, float):
        return gformat(val, length=length).strip()
    return repr(val)


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


def not_zero(value):
    """Return value with a minimal absolute size of tiny,
    preserving the sign.
    """
    return float(np.copysign(max(tiny, abs(value)), value))


def test_ge_gt_le_lt(
        ge, gt, le, lt, func, location=None, raise_error=False, log=True):
    """Check individual and mutual validity of ge, gt, le, lt
    qualifiers.

    :param func: Test for integers or numbers.
    :type func: callable: is_int, is_num
    :return: True upon success or False when mutually exlusive.
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
    """Return a range string representation matching the ge, gt, le, lt
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
    """Value is an integer in range ge <= v <= le or gt < v < lt or
    some combination.

    :return: True if yes or False is no.
    :rtype: bool
    """
    return _is_int_or_num(v, 'int', ge, gt, le, lt, raise_error, log)


def is_num(v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is a number in range ge <= v <= le or gt < v < lt or some
    combination.

    :return: True if yes or False is no.
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
    """Value is an integer pair, each in range ge <= v[i] <= le or
    gt < v[i] < lt or ge[i] <= v[i] <= le[i] or gt[i] < v[i] < lt[i]
    or some combination.

    :return: True if yes or False is no.
    :rtype: bool
    """
    return _is_int_or_num_pair(v, 'int', ge, gt, le, lt, raise_error, log)


def is_num_pair(
        v, ge=None, gt=None, le=None, lt=None, raise_error=False, log=True):
    """Value is a number pair, each in range ge <= v[i] <= le or
    gt < v[i] < lt or ge[i] <= v[i] <= le[i] or gt[i] < v[i] < lt[i]
    or some combination.

    :return: True if yes or False is no.
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
    """Value is a tuple or list of integers, each in range
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
    """Value is a tuple or list of numbers, each in range
    ge <= l[i] <= le or gt < l[i] < lt or some combination.
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
    """Value is a tuple or list of strings."""
    if (not isinstance(t_or_l, (tuple, list))
            or any(not isinstance(s, str) for s in t_or_l)):
        illegal_value(t_or_l, 't_or_l', 'is_str_series', raise_error, log)
        return False
    return True


def is_dict_series(t_or_l, raise_error=False, log=True):
    """Value is a tuple or list of dictionaries."""
    if (not isinstance(t_or_l, (tuple, list))
            or any(not isinstance(d, dict) for d in t_or_l)):
        illegal_value(t_or_l, 't_or_l', 'is_dict_series', raise_error, log)
        return False
    return True


def is_dict_nums(d, raise_error=False, log=True):
    """Value is a dictionary with single number values."""
    if (not isinstance(d, dict)
            or any(not is_num(v, log=False) for v in d.values())):
        illegal_value(d, 'd', 'is_dict_nums', raise_error, log)
        return False
    return True


def is_dict_strings(d, raise_error=False, log=True):
    """Value is a dictionary with single string values."""
    if (not isinstance(d, dict)
            or any(not isinstance(v, str) for v in d.values())):
        illegal_value(d, 'd', 'is_dict_strings', raise_error, log)
        return False
    return True


def is_index(v, ge=0, lt=None, raise_error=False, log=True):
    """Value is an array index in range ge <= v < lt. NOTE lt IS NOT
    included!
    """
    if isinstance(lt, int):
        if lt <= ge:
            illegal_combination(
                ge, 'ge', lt, 'lt', 'is_index', raise_error, log)
            return False
    return is_int(v, ge=ge, lt=lt, raise_error=raise_error, log=log)


def is_index_range(v, ge=0, le=None, lt=None, raise_error=False, log=True):
    """Value is an array index range in range ge <= v[0] <= v[1] <= le
    or ge <= v[0] <= v[1] < lt. NOTE le IS included!
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


def index_nearest_down(a, value):
    """Return index of nearest array value, rounded down."""
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(
            f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    index = int(np.argmin(np.abs(a-value)))
    if value < a[index] and index > 0:
        index -= 1
    return index


def index_nearest_up(a, value):
    """Return index of nearest array value, rounded up."""
    a = np.asarray(a)
    if a.ndim > 1:
        raise ValueError(
            f'Invalid array dimension for parameter a ({a.ndim}, {a})')
    index = int(np.argmin(np.abs(a-value)))
    if value > a[index] and index < a.size-1:
        index += 1
    return index


def get_consecutive_int_range(a):
    """Return a list of pairs of integers marking consecutive ranges
    of integers.
    """
    a.sort()
    i = 0
    int_ranges = []
    while i < len(a):
        j = i
        while j < len(a)-1:
            if a[j+1] > 1 + a[j]:
                break
            j += 1
        int_ranges.append([a[i], a[j]])
        i = j+1
    return int_ranges


def round_to_n(x, n=1):
    """Round to a specific number of sig figs."""
    if x == 0.0:
        return 0
    return type(x)(round(x, n-1-int(np.floor(np.log10(abs(x))))))


def round_up_to_n(x, n=1):
    """Round up to a specific number of sig figs."""
    x_round = round_to_n(x, n)
    if abs(x/x_round) > 1.0:
        x_round += np.sign(x) * 10**(np.floor(np.log10(abs(x)))+1-n)
    return type(x)(x_round)


def trunc_to_n(x, n=1):
    """Truncate to a specific number of sig figs."""
    x_round = round_to_n(x, n)
    if abs(x_round/x) > 1.0:
        x_round -= np.sign(x) * 10**(np.floor(np.log10(abs(x)))+1-n)
    return type(x)(x_round)


def almost_equal(a, b, sig_figs):
    """Check if equal to within a certain number of significant digits.
    """
    if is_num(a) and is_num(b):
        return abs(round_to_n(a-b, sig_figs)) < pow(10, 1-sig_figs)
    raise ValueError(
        f'Invalid value for a or b in almost_equal (a: {a}, {type(a)}, '
        f'b: {b}, {type(b)})')


def string_to_list(
        s, split_on_dash=True, remove_duplicates=True, sort=True,
        raise_error=False):
    """Return a list of numbers by splitting/expanding a string on any
    combination of commas, whitespaces, or dashes (when
    split_on_dash=True).
    e.g: '1, 3, 5-8, 12 ' -> [1, 3, 5, 6, 7, 8, 12]

    :param s: Input string.
    :type s: str
    :param split_on_dash: Allow dashes in input string,
        defaults to `True`.
    :type split_on_dash: bool, optional
    :param remove_duplicates: Removes duplicates (may also change the
        order), defaults to `True`.
    :type remove_duplicates: bool, optional
    :param sort: Sort in ascending order, defaults to `True`.
    :type sort: bool, optional
    :param raise_error: Raise an exception upon any error,
        defaults to `False`.
    :type raise_error: bool, optional
    :return: Input list or none upon an illegal input.
    :rtype: list
    """
    if not isinstance(s, str):
        illegal_value(s, 's', location='string_to_list')
        return None
    if not s:
        return []
    try:
        list1 = re_split(r'\s+,\s+|\s+,|,\s+|\s+|,', s.strip())
    except (ValueError, TypeError, SyntaxError, MemoryError,
            RecursionError) as e:
        if not raise_error:
            return None
        raise e
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
                RecursionError) as e:
            if not raise_error:
                return None
            raise e
    else:
        l_of_i = [literal_eval(x) for x in list1]
    if remove_duplicates:
        l_of_i = list(dict.fromkeys(l_of_i))
    if sort:
        l_of_i = sorted(l_of_i)
    return l_of_i


def list_to_string(a):
    """Return a list of pairs of integers marking consecutive ranges
    of integers in string notation."""
    int_ranges = get_consecutive_int_range(a)
    if not int_ranges:
        return ''
    if int_ranges[0][0] == int_ranges[0][1]:
        s = f'{int_ranges[0][0]}'
    else:
        s = f'{int_ranges[0][0]}-{int_ranges[0][1]}'
    for int_range in int_ranges[1:]:
        if int_range[0] == int_range[1]:
            s += f', {int_range[0]}'
        else:
            s += f', {int_range[0]}-{int_range[1]}'
    return s


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
        s=None, num_max=None, ge=None, le=None, split_on_dash=True,
        remove_duplicates=True, sort=True, raise_error=False, log=True):
    """Prompt the user to input a list of integers and split the
    entered string on any combination of commas, whitespaces, or
    dashes (when split_on_dash is True).
    e.g: '1 3,5-8 , 12 ' -> [1, 3, 5, 6, 7, 8, 12]

    :param s: Interactive user prompt, defaults to `None`.
    :type s: str, optional
    :param num_max: Maximum number of inputs in list.
    :type num_max: int, optional
    :param ge: Minimum value of inputs in list.
    :type ge: int, optional
    :param le: Minimum value of inputs in list.
    :type le: int, optional
    :param split_on_dash: Allow dashes in input string,
        defaults to `True`.
    :type split_on_dash: bool, optional
    :param remove_duplicates: Removes duplicates (may also change the
        order), defaults to `True`.
    :type remove_duplicates: bool, optional
    :param sort: Sort in ascending order, defaults to `True`.
    :type sort: bool, optional
    :param raise_error: Raise an exception upon any error,
        defaults to `False`.
    :type raise_error: bool, optional
    :param log: Print an error message upon any error,
        defaults to `True`.
    :type log: bool, optional
    :return: Input list or none upon an illegal input.
    :rtype: list
    """
    return _input_int_or_num_list(
        'int', s, num_max, ge, le, split_on_dash, remove_duplicates, sort,
        raise_error, log)


def input_num_list(
        s=None, num_max=None, ge=None, le=None, remove_duplicates=True,
        sort=True, raise_error=False, log=True):
    """Prompt the user to input a list of numbers and split the entered
    string on any combination of commas or whitespaces.
    e.g: '1.0, 3, 5.8, 12 ' -> [1.0, 3.0, 5.8, 12.0]

    :param s: Interactive user prompt.
    :type s: str, optional
    :param num_max: Maximum number of inputs in list.
    :type num_max: int, optional
    :param ge: Minimum value of inputs in list.
    :type ge: float, optional
    :param le: Minimum value of inputs in list.
    :type le: float, optional
    :param remove_duplicates: Removes duplicates (may also change the
        order), defaults to `True`.
    :type remove_duplicates: bool, optional
    :param sort: Sort in ascending order, defaults to `True`.
    :type sort: bool, optional
    :param raise_error: Raise an exception upon any error,
        defaults to `False`.
    :type raise_error: bool, optional
    :param log: Print an error message upon any error,
        defaults to `True`.
    :type log: bool, optional
    :return: Input list or none upon an illegal input.
    :rtype: list
    """
    return _input_int_or_num_list(
        'num', s, num_max, ge, le, False, remove_duplicates, sort, raise_error,
        log)


def _input_int_or_num_list(
        type_str, s=None, num_max=None, ge=None, le=None, split_on_dash=True,
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
    if (num_max is not None
            and not is_int(num_max, gt=0, raise_error=raise_error, log=log)):
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
    if (not isinstance(_list, list)
            or (num_max is not None and len(_list) > num_max)
            or any(
                not _is_int_or_num(v, type_str, ge=ge, le=le) for v in _list)):
        num = '' if num_max is None else f'up to {num_max} '
        if split_on_dash:
            print(f'Invalid input: enter a valid set of {num}dash/comma/'
                  'whitespace separated numbers e.g. 1 3,5-8 , 12')
        else:
            print(f'Invalid input: enter a valid set of {num}comma/whitespace '
                  'separated numbers e.g. 1 3,5 8 , 12')
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
    """Assert that there are no duplicates in a list of dictionaries.
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
    """Assert that there are no duplicate keys in a list of
    dictionaries.
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
    """Assert that there are no duplicate attributes in a list of
    objects.
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


def rolling_average(
        y, x=None, dtype=None, start=0, end=None, width=None,
        stride=None, num=None, average=True, mode='valid',
        use_convolve=None):
    """Returns the rolling sum or average of an array over the last
    dimension.
    """
    y = np.asarray(y)
    y_shape = y.shape
    if y.ndim == 1:
        y = np.expand_dims(y, 0)
    else:
        y = y.reshape((np.prod(y.shape[0:-1]), y.shape[-1]))
    if x is not None:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError('Parameter "x" must be a 1D array-like')
        if x.size != y.shape[1]:
            raise ValueError(f'Dimensions of "x" and "y[1]" do not '
                             f'match ({x.size} vs {y.shape[1]})')
    if dtype is None:
        if average:
            dtype = y.dtype
        else:
            dtype = np.float32
    if width is None and stride is None and num is None:
        raise ValueError('Invalid input parameters, specify at least one of '
                         '"width", "stride" or "num"')
    if width is not None and not is_int(width, ge=1):
        raise ValueError(f'Invalid "width" parameter ({width})')
    if stride is not None and not is_int(stride, ge=1):
        raise ValueError(f'Invalid "stride" parameter ({stride})')
    if num is not None and not is_int(num, ge=1):
        raise ValueError(f'Invalid "num" parameter ({num})')
    if not isinstance(average, bool):
        raise ValueError(f'Invalid "average" parameter ({average})')
    if mode not in ('valid', 'full'):
        raise ValueError(f'Invalid "mode" parameter ({mode})')
    size = y.shape[1]
    if size < 2:
        raise ValueError(f'Invalid y[1] dimension ({size})')
    if not is_int(start, ge=0, lt=size):
        raise ValueError(f'Invalid "start" parameter ({start})')
    if end is None:
        end = size
    elif not is_int(end, gt=start, le=size):
        raise ValueError(f'Invalid "end" parameter ({end})')
    if use_convolve is None:
        if len(y_shape) == 1:
            use_convolve = True
        else:
            use_convolve = False
    if use_convolve and (start or end < size):
        y = np.take(y, range(start, end), axis=1)
        if x is not None:
            x = x[start:end]
        size = y.shape[1]
    else:
        size = end-start

    if stride is None:
        if width is None:
            width = max(1, int(size/num))
            stride = width
        else:
            width = min(width, size)
            if num is None:
                stride = width
            else:
                stride = max(1, int((size-width) / (num-1)))
    else:
        stride = min(stride, size-stride)
        if width is None:
            width = stride

    if mode == 'valid':
        num = 1 + max(0, int((size-width) / stride))
    else:
        num = int(size/stride)
        if num*stride < size:
            num += 1

    if use_convolve:
        n_start = 0
        n_end = width
        weight = np.empty((num))
        for n in range(num):
            n_num = n_end-n_start
            weight[n] = n_num
            n_start += stride
            n_end = min(size, n_end+stride)

        window = np.ones((width))
        if x is not None:
            if mode == 'valid':
                rx = np.convolve(x, window)[width-1:1-width:stride]
            else:
                rx = np.convolve(x, window)[width-1::stride]
            rx /= weight

        ry = []
        if mode == 'valid':
            for i in range(y.shape[0]):
                ry.append(np.convolve(y[i], window)[width-1:1-width:stride])
        else:
            for i in range(y.shape[0]):
                ry.append(np.convolve(y[i], window)[width-1::stride])
        ry = np.reshape(ry, (*y_shape[0:-1], num))
        if len(y_shape) == 1:
            ry = np.squeeze(ry)
        if average:
            ry = (np.asarray(ry).astype(np.float32)/weight).astype(dtype)
        elif mode != 'valid':
            weight = np.where(weight < width, width/weight, 1.0)
            ry = (np.asarray(ry).astype(np.float32)*weight).astype(dtype)
    else:
        ry = np.zeros((num, y.shape[0]), dtype=y.dtype)
        if x is not None:
            rx = np.zeros(num, dtype=x.dtype)
        n_start = start
        n_end = n_start+width
        for n in range(num):
            y_sum = np.sum(y[:,n_start:n_end], 1)
            n_num = n_end-n_start
            if n_num < width:
                y_sum *= width/n_num
            ry[n] = y_sum
            if x is not None:
                rx[n] = np.sum(x[n_start:n_end])/n_num
            n_start += stride
            n_end = min(start+size, n_end+stride)
        ry = np.reshape(ry.T, (*y_shape[0:-1], num))
        if len(y_shape) == 1:
            ry = np.squeeze(ry)
        if average:
            ry = (ry.astype(np.float32)/width).astype(dtype)

    if x is None:
        return ry
    return ry, rx


def baseline_arPLS(
        y, mask=None, w=None, tol=1.e-8, lam=1.e6, max_iter=20,
        full_output=False):
    """Returns the smoothed baseline estimate of a spectrum.

    Based on S.-J. Baek, A. Park, Y.-J Ahn, and J. Choo,
    "Baseline correction using asymmetrically reweighted penalized
    least squares smoothing", Analyst, 2015,140, 250-257

    :param y: The spectrum.
    :type y: array-like
    :param mask: A mask to apply to the spectrum before baseline
       construction.
    :type mask: array-like, optional
    :param w: The weights (allows restart for additional ieterations).
    :type w: numpy.array, optional
    :param tol: The convergence tolerence, defaults to `1.e-8`.
    :type tol: float, optional
    :param lam: The &lambda (smoothness) parameter (the balance
        between the residual of the data and the baseline and the
        smoothness of the baseline). The suggested range is between
        100 and 10^8, defaults to `10^6`.
    :type lam: float, optional
    :param max_iter: The maximum number of iterations,
        defaults to `20`.
    :type max_iter: int, optional
    :param full_output: Whether or not to also output the baseline
        corrected spectrum, the number of iterations and error in the
        returned result, defaults to `False`.
    :type full_output: bool, optional
    :return: The smoothed baseline, with optionally the baseline
        corrected spectrum, the weights, the number of iterations and
        the error in the returned result.
    :rtype: numpy.array [, numpy.array, int, float]
    """
    # With credit to: Daniel Casas-Orozco
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    # Third party modules
    from scipy.sparse import (
        spdiags,
        linalg,
    )

    if not is_num(tol, gt=0):
        raise ValueError(f'Invalid tol parameter ({tol})')
    if not is_num(lam, gt=0):
        raise ValueError(f'Invalid lam parameter ({lam})')
    if not is_int(max_iter, gt=0):
        raise ValueError(f'Invalid max_iter parameter ({max_iter})')
    if not isinstance(full_output, bool):
        raise ValueError(f'Invalid full_output parameter ({max_iter})')
    y = np.asarray(y)
    if mask is not None:
        mask = mask.astype(bool)
        y_org = y
        y = y[mask]
    num = y.size

    diag = np.ones((num-2))
    D = spdiags([diag, -2*diag, diag], [0, -1, -2], num, num-2)

    H = lam * D.dot(D.T)

    if w is None:
        w = np.ones(num)
    W = spdiags(w, 0, num, num)

    error = 1
    num_iter = 0

    exp_max = int(np.log(float_info.max))
    while error > tol and num_iter < max_iter:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1.0 / (1.0 + np.exp(
            np.clip(2.0 * (d - (2.0*s - m))/s, None, exp_max)))
        error = np.linalg.norm(w_new - w) / np.linalg.norm(w)
        num_iter += 1
        w = w_new
        W.setdiag(w)

    if mask is not None:
        zz = np.zeros(y_org.size)
        zz[mask] = z
        z = zz
        if full_output:
            d = y_org - z
    if full_output:
        return z, d, w, num_iter, float(error)
    return z


def fig_to_iobuf(fig):
    """Return an in-memory object as a byte stream represention of
    a Matplotlib figure.

    :param fig: Matplotlib figure object.
    :type fig: matplotlib.figure.Figure
    :return: Byte stream representation of the Matplotlib figure.
    :rtype: _io.BytesIO
    """
    # System modules
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format='png')
    return buf


def save_iobuf_fig(buf, filename, fileformat=None, force_overwrite=False):
    """Save a byte stream represention of a Matplotlib figure to file.

    :param buf: Byte stream representation of the Matplotlib figure.
    :type buf: _io.BytesIO
    :param filename: Filename (with or without extension).
    :type filename: str
    :param fileformat: The file format (ignored if filename has an
        extension), defaults to 'png'.
    :type fileformat: str, optional
    :param force_overwrite: Flag to allow `filename` to be overwritten
        if it already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    :raises RuntimeError: If a file already exists and
        `force_overwrite` is `False`.
    """
    # Third party modules
    from PIL import Image

    exts = Image.registered_extensions()
    exts = {ex for ex, f in exts.items() if f in Image.SAVE}

    # Validate filename and extension
    basename, ext = os_path.splitext(filename)
    if fileformat is None:
        if not ext or ext not in exts:
            ext = '.png'
    else:
        if not ext or ext not in exts:
            try:
                assert isinstance(fileformat, str)
                ext = fileformat
                if fileformat[0] != '.':
                    ext = f'.{ext}'
                assert ext in  exts
            except:
                ext = '.png'
    filename = basename + ext
    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    # Write image to file
    buf.seek(0)
    img = Image.open(buf)
    img.save(basename + ext)


def select_mask_1d(
        y, x=None, preselected_index_ranges=None, preselected_mask=None,
        title=None, xlabel=None, ylabel=None, min_num_index_ranges=None,
        max_num_index_ranges=None, interactive=True, filename=None,
        return_buf=False):
    """Display a lineplot and have the user select a mask.

    :param y: One-dimensional data array for which a mask will be
        constructed.
    :type y: numpy.ndarray
    :param x: x-coordinates of the reference data.
    :type x: numpy.ndarray, optional
    :param preselected_index_ranges: List of preselected index ranges
        to mask (bounds are inclusive).
    :type preselected_index_ranges: Union(list[tuple(int, int)],
        list[list[int]], list[tuple(float, float)], list[list[float]]),
        optional
    :param preselected_mask: Preselected boolean mask array.
    :type preselected_mask: numpy.ndarray, optional
    :param title: Title for the displayed figure.
    :type title: str, optional
    :param xlabel: Label for the x-axis of the displayed figure.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis of the displayed figure.
    :type ylabel: str, optional
    :param min_num_index_ranges: The minimum number of selected index
        ranges.
    :type min_num_index_ranges: int, optional
    :param max_num_index_ranges: The maximum number of selected index
        ranges.
    :type max_num_index_ranges: int, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `True`.
    :type interactive: bool, optional
    :param filename: Save a .png of the plot to filename, defaults to
        `None`, in which case the plot is not saved.
    :type filename: str, optional
    :param return_buf: Return an in-memory object as a byte stream
        represention of the Matplotlib figure, defaults to `False`.
    :type return_buf: bool, optional
    :return: A byte stream represention of the Matplotlib figure if
        return_buf is `True` (`None` otherwise), a boolean mask array,
        and the list of selected index ranges.
    :rtype: Union[io.BytesIO, None], numpy.ndarray, list[list[int, int]]
    """
    # System modules
    from copy import deepcopy

    # Third party modules
    if interactive or filename is not None or return_buf:
        from matplotlib.patches import Patch
        from matplotlib.widgets import Button, SpanSelector

    def change_fig_title(title):
        if fig_title:
            fig_title[0].remove()
            fig_title.pop()
        fig_title.append(plt.figtext(*title_pos, title, **title_props))

    def change_error_text(error):
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        error_texts.append(plt.figtext(*error_pos, error, **error_props))

    def get_selected_index_ranges(change_fnc=None, title=''):
        selected_index_ranges = sorted(
            [[index_nearest(x, span.extents[0]),
              index_nearest(x, span.extents[1])+1]
             for span in spans])
        if change_fnc is not None:
            if len(selected_index_ranges) > 1:
                change_fnc(
                    f'{title}Selected ROIs: {selected_index_ranges}')
            elif selected_index_ranges:
                change_fnc(
                    f'{title}Selected ROI: {tuple(selected_index_ranges[0])}')
            else:
                change_fnc(f'{title}Selected ROI: None')
        return selected_index_ranges

    def add_span(event, xrange_init=None):
        """Callback function for the "Add span" button."""
        if (max_num_index_ranges is not None
                and len(spans) >= max_num_index_ranges):
            change_error_text(
                'Exceeding max number of ranges, adjust an existing '
                'range or click "Reset"/"Confirm"')
        else:
            spans.append(
                SpanSelector(
                    ax, select_span, 'horizontal', props=included_props,
                    useblit=True, interactive=interactive,
                    drag_from_anywhere=True, ignore_event_outside=True,
                    grab_range=5))
            if xrange_init is None:
                xmin_init, xmax_init = min(x), 0.05*(max(x)-min(x))
            else:
                xmin_init, xmax_init = xrange_init
            spans[-1]._selection_completed = True
            spans[-1].extents = (xmin_init, xmax_init)
            spans[-1].onselect(xmin_init, xmax_init)
        plt.draw()

    def select_span(xmin, xmax):
        """Callback function for the SpanSelector widget."""
        combined_spans = True
        while combined_spans:
            combined_spans = False
            for i, span1 in enumerate(spans):
                for span2 in spans[i+1:]:
                    if (span1.extents[1] >= span2.extents[0]
                            and span1.extents[0] <= span2.extents[1]):
                        change_error_text(
                            'Combined overlapping spans in currently '
                            'selected mask')
                        span2.extents = (
                            min(span1.extents[0], span2.extents[0]),
                            max(span1.extents[1], span2.extents[1]))
                        span1.set_visible(False)
                        spans.remove(span1)
                        combined_spans = True
                        break
                if combined_spans:
                    break
        get_selected_index_ranges(change_error_text)
        plt.draw()

    def reset(event):
        """Callback function for the "Reset" button."""
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        for span in reversed(spans):
            span.set_visible(False)
            spans.remove(span)
        get_selected_index_ranges(change_error_text)
        plt.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if (min_num_index_ranges is not None
                and len(spans) < min_num_index_ranges):
            change_error_text(
                f'Select at least {min_num_index_ranges} unique index ranges')
            plt.draw()
        else:
            if error_texts:
                error_texts[0].remove()
                error_texts.pop()
            get_selected_index_ranges(change_fig_title, title)
            plt.close()

    def update_mask(mask, selected_index_ranges):
        """Update the mask with the selected index ranges."""
        for min_, max_ in selected_index_ranges:
            mask = np.logical_or(
                mask,
                np.logical_and(x >= x[min_], x <= x[min(max_, num_data-1)]))
        return mask

    def update_index_ranges(mask):
        """Update the selected index ranges (where mask = True)."""
        selected_index_ranges = []
        for i, m in enumerate(mask):
            if m:
                if (not selected_index_ranges
                        or isinstance(selected_index_ranges[-1], tuple)):
                    selected_index_ranges.append(i)
            else:
                if (selected_index_ranges
                        and isinstance(selected_index_ranges[-1], int)):
                    selected_index_ranges[-1] = \
                        (selected_index_ranges[-1], i-1)
        if (selected_index_ranges
                and isinstance(selected_index_ranges[-1], int)):
            selected_index_ranges[-1] = (selected_index_ranges[-1], num_data-1)
        return selected_index_ranges

    # Check inputs
    y = np.asarray(y)
    if y.ndim > 1:
        raise ValueError(f'Invalid y dimension ({y.ndim})')
    num_data = y.size
    if x is None:
        x = np.arange(num_data)+0.5
    else:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1 or x.size != num_data:
            raise ValueError(f'Invalid x shape ({x.shape})')
        if not np.all(x[:-1] < x[1:]):
            raise ValueError('Invalid x: must be monotonically increasing')
    if title is None:
        title = ''
    else:
        title = f'{title}: '
    if preselected_index_ranges is None:
        preselected_index_ranges = []
    else:
        if not isinstance(preselected_index_ranges, list):
            raise ValueError('Invalid parameter preselected_index_ranges '
                             f'({preselected_index_ranges})')
        index_ranges = []
        for i, v in enumerate(preselected_index_ranges):
            if not is_num_pair(v):
                raise ValueError('Invalid parameter preselected_index_ranges '
                                 f'({preselected_index_ranges})')
            index_ranges.append(
                (max(0, int(v[0])), min(num_data, int(v[1])-1)))
        preselected_index_ranges = index_ranges

    # Setup the preselected mask and index ranges if provided
    if preselected_mask is not None:
        preselected_index_ranges = update_index_ranges(
            update_mask(
                np.copy(np.asarray(preselected_mask, dtype=bool)),
                preselected_index_ranges))

    if not interactive and filename is None and not return_buf:

        # Update the mask with the preselected index ranges
        selected_mask = update_mask(len(x)*[False], preselected_index_ranges)

        return None, selected_mask, preselected_index_ranges

    spans = []
    fig_title = []
    error_texts = []

    # Setup the Matplotlib figure
    title_pos = (0.5, 0.95)
    title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    error_pos = (0.5, 0.90)
    error_props = {'fontsize': 'x-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    excluded_props = {
        'facecolor': 'white', 'edgecolor': 'gray', 'linestyle': ':'}
    included_props = {
        'alpha': 0.5, 'facecolor': 'tab:blue', 'edgecolor': 'blue'}

    fig, ax = plt.subplots(figsize=(11, 8.5))
    handles = ax.plot(x, y, color='k', label='Reference Data')
    handles.append(Patch(
        label='Excluded / unselected ranges', **excluded_props))
    handles.append(Patch(
        label='Included / selected ranges', **included_props))
    ax.legend(handles=handles)
    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.set_xlim(x[0], x[-1])
    fig.subplots_adjust(bottom=0.0, top=0.85)

    # Add the preselected index ranges
    for min_, max_ in preselected_index_ranges:
        add_span(None, xrange_init=(x[min_], x[min(max_, num_data-1)]))

    if not interactive:

        get_selected_index_ranges(change_fig_title, title)
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()

    else:

        change_fig_title(f'{title}Click and drag to select ranges')
        get_selected_index_ranges(change_error_text)
        fig.subplots_adjust(bottom=0.2)

        # Setup "Add span" button
        add_span_btn = Button(
            plt.axes([0.15, 0.05, 0.15, 0.075]), 'Add span')
        add_span_cid = add_span_btn.on_clicked(add_span)

        # Setup "Reset" button
        reset_btn = Button(plt.axes([0.45, 0.05, 0.15, 0.075]), 'Reset')
        reset_cid = reset_btn.on_clicked(reset)

        # Setup "Confirm" button
        confirm_btn = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
        confirm_cid = confirm_btn.on_clicked(confirm)

        # Show figure for user interaction
        plt.show()

        # Disconnect all widget callbacks when figure is closed
        add_span_btn.disconnect(add_span_cid)
        reset_btn.disconnect(reset_cid)
        confirm_btn.disconnect(confirm_cid)

        # ...and remove the buttons before returning the figure
        add_span_btn.ax.remove()
        reset_btn.ax.remove()
        confirm_btn.ax.remove()
        plt.subplots_adjust(bottom=0.0)

    selected_index_ranges = get_selected_index_ranges()

    # Update the mask with the currently selected index ranges
    selected_mask = update_mask(len(x)*[False], selected_index_ranges)

    if filename is not None or return_buf:
        if interactive:
            if len(selected_index_ranges) > 1:
                title += f'Selected ROIs: {selected_index_ranges}'
            else:
                title += f'Selected ROI: {tuple(selected_index_ranges[0])}'
            fig_title[0]._text = title
        fig_title[0].set_in_layout(True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if filename is not None:
            fig.savefig(filename)
        if return_buf:
            buf = fig_to_iobuf(fig)
    plt.close()
    if return_buf:
        return buf, selected_mask, selected_index_ranges
    return None, selected_mask, selected_index_ranges


def select_roi_1d(
        y, x=None, preselected_roi=None, title=None, xlabel=None, ylabel=None,
        interactive=True, filename=None, return_buf=False):
    """Display a 2D plot and have the user select a single region
    of interest.

    :param y: One-dimensional data array for which a for which a region
        of interest will be selected.
    :type y: numpy.ndarray
    :param x: x-coordinates of the data
    :type x: numpy.ndarray, optional
    :param preselected_roi: Preselected region of interest.
    :type preselected_roi: tuple(int, int), optional
    :param title: Title for the displayed figure.
    :type title: str, optional
    :param xlabel: Label for the x-axis of the displayed figure.
    :type xlabel: str, optional
    :param ylabel: Label for the y-axis of the displayed figure.
    :type ylabel: str, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `True`.
    :type interactive: bool, optional
    :param filename: Save a .png of the plot to filename, defaults to
        `None`, in which case the plot is not saved.
    :type filename: str, optional
    :param return_buf: Return an in-memory object as a byte stream
        represention of the Matplotlib figure, defaults to `False`.
    :type return_buf: bool, optional
    :return: A byte stream represention of the Matplotlib figure if
        return_buf is `True` (`None` otherwise), and the selected
        region of interest.
    :rtype: Union[io.BytesIO, None], tuple(int, int)
    """
    # Check inputs
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f'Invalid image dimension ({y.ndim})')
    if preselected_roi is not None:
        if not is_int_pair(preselected_roi, ge=0, le=y.size, log=False):
            raise ValueError('Invalid parameter preselected_roi '
                             f'({preselected_roi})')
        preselected_roi = [preselected_roi]

    buf, _, roi = select_mask_1d(
        y, x=x, preselected_index_ranges=preselected_roi, title=title,
        xlabel=xlabel, ylabel=ylabel, min_num_index_ranges=1,
        max_num_index_ranges=1, interactive=interactive, filename=filename,
        return_buf=return_buf)

    return buf, tuple(roi[0])

def select_roi_2d(
        a, preselected_roi=None, title=None, title_a=None,
        row_label='row index', column_label='column index', interactive=True,
        filename=None, return_buf=False):
    """Display a 2D image and have the user select a single rectangular
       region of interest.

    :param a: Two-dimensional image data array for which a region of
        interest will be selected.
    :type a: numpy.ndarray
    :param preselected_roi: Preselected region of interest.
    :type preselected_roi: tuple(int, int, int, int), optional
    :param title: Title for the displayed figure.
    :type title: str, optional
    :param title_a: Title for the image of a.
    :type title_a: str, optional
    :param row_label: Label for the y-axis of the displayed figure,
        defaults to `row index`.
    :type row_label: str, optional
    :param column_label: Label for the x-axis of the displayed figure,
        defaults to `column index`.
    :type column_label: str, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `True`.
    :type interactive: bool, optional
    :param filename: Save a .png of the plot to filename, defaults to
        `None`, in which case the plot is not saved.
    :type filename: str, optional
    :param return_buf: Return an in-memory object as a byte stream
        represention of the Matplotlib figure, defaults to `False`.
    :type return_buf: bool, optional
    :return: A byte stream represention of the Matplotlib figure if
        return_buf is `True` (`None` otherwise), and the selected
        region of interest.
    :rtype: Union[io.BytesIO, None], tuple(int, int, int, int)
    """
    # Third party modules
    if interactive or filename is not None or return_buf:
        from matplotlib.widgets import Button, RectangleSelector

    def change_fig_title(title):
        if fig_title:
            fig_title[0].remove()
            fig_title.pop()
        fig_title.append(plt.figtext(*title_pos, title, **title_props))

    def change_subfig_title(error):
        if subfig_title:
            subfig_title[0].remove()
            subfig_title.pop()
        subfig_title.append(plt.figtext(*error_pos, error, **error_props))

    def clear_selection():
        rects[0].set_visible(False)
        rects.pop()
        rects.append(
            RectangleSelector(
                ax, on_rect_select, props=rect_props,
                useblit=True, interactive=interactive, drag_from_anywhere=True,
                ignore_event_outside=False))

    def on_rect_select(eclick, erelease):
        """Callback function for the RectangleSelector widget."""
        if (not int(rects[0].extents[1]) - int(rects[0].extents[0])
                or not int(rects[0].extents[3]) - int(rects[0].extents[2])):
            clear_selection()
            change_subfig_title(
                'Selected ROI too small, try again')
        else:
            change_subfig_title(
                f'Selected ROI: {tuple(int(v) for v in rects[0].extents)}')
        plt.draw()

    def reset(event):
        """Callback function for the "Reset" button."""
        if subfig_title:
            subfig_title[0].remove()
            subfig_title.pop()
        clear_selection()
        plt.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if subfig_title:
            subfig_title[0].remove()
            subfig_title.pop()
        roi = tuple(int(v) for v in rects[0].extents)
        if roi[1]-roi[0] < 1 or roi[3]-roi[2] < 1:
            roi = None
        change_fig_title(f'Selected ROI: {roi}')
        plt.close()

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

    if not interactive and filename is None and not return_buf:
        return None, preselected_roi

    fig_title = []
    subfig_title = []

    title_pos = (0.5, 0.95)
    title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    error_pos = (0.5, 0.90)
    error_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                   'verticalalignment': 'bottom'}
    rect_props = {
        'alpha': 0.5, 'facecolor': 'tab:blue', 'edgecolor': 'blue'}

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(a)
    ax.set_title(title_a, fontsize='xx-large')
    ax.set_xlabel(column_label, fontsize='x-large')
    ax.set_ylabel(row_label, fontsize='x-large')
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

        if preselected_roi is not None:
            change_fig_title(
                f'Selected ROI: {tuple(int(v) for v in preselected_roi)}')

    else:

        change_fig_title(title)
        if preselected_roi is not None:
            change_subfig_title(
                f'Preselected ROI: {tuple(int(v) for v in preselected_roi)}')
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

    if filename is not None or return_buf:
        if fig_title:
            fig_title[0].set_in_layout(True)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            fig.tight_layout(rect=(0, 0, 1, 1))

        # Remove the handles
        if interactive:
            rects[0]._center_handle.set_visible(False)
            rects[0]._corner_handles.set_visible(False)
            rects[0]._edge_handles.set_visible(False)
        if filename is not None:
            fig.savefig(filename)
        if return_buf:
            buf = fig_to_iobuf(fig)
    plt.close()

    roi = tuple(int(v) for v in rects[0].extents)
    if roi[1]-roi[0] < 1 or roi[3]-roi[2] < 1:
        roi = None

    if return_buf:
        return buf, roi
    return None, roi


def select_image_indices(
        a, axis, b=None, preselected_indices=None, axis_index_offset=0,
        min_range=None, min_num_indices=2, max_num_indices=2, title=None,
        title_a=None, title_b=None, row_label='row index',
        column_label='column index', interactive=True, return_buf=False):
    """Display a 2D image and have the user select a set of image
       indices in either row or column direction. 

    :param a: Two-dimensional image data array for which a region of
        interest will be selected.
    :type a: numpy.ndarray
    :param axis: The selection direction (0: row, 1: column)
    :type axis: int
    :param b: A secondary two-dimensional image data array for which
        a shared region of interest will be selected.
    :type b: numpy.ndarray, optional
    :param preselected_indices: Preselected image indices.
    :type preselected_indices: tuple(int), list(int), optional
    :param axis_index_offset: Offset in axis index range and
        preselected indices, defaults to `0`.
    :type axis_index_offset: int, optional
    :param min_range: The minimal range spanned by the selected 
        indices.
    :type min_range: int, optional
    :param min_num_indices: The minimum number of selected indices.
    :type min_num_indices: int, optional
    :param max_num_indices: The maximum number of selected indices.
    :type max_num_indices: int, optional
    :param title: Title for the displayed figure.
    :type title: str, optional
    :param title_a: Title for the image of a.
    :type title_a: str, optional
    :param title_b: Title for the image of b.
    :type title_b: str, optional
    :param row_label: Label for the y-axis of the displayed figure,
        defaults to `row index`.
    :type row_label: str, optional
    :param column_label: Label for the x-axis of the displayed figure,
        defaults to `column index`.
    :type column_label: str, optional
    :param interactive: Show the plot and allow user interactions with
        the matplotlib figure, defaults to `True`.
    :type interactive: bool, optional
    :param return_buf: Return an in-memory object as a byte stream
        represention of the Matplotlib figure instead of the
        matplotlib figure, defaults to `False`.
    :type return_buf: bool, optional
    :return: The selected region of interest as array indices and a
        matplotlib figure.
    :rtype: Union[matplotlib.figure.Figure, io.BytesIO],
        tuple(int, int, int, int)
    """
    # Third party modules
    from matplotlib.widgets import TextBox, Button

    def change_fig_title(title):
        if fig_title:
            fig_title[0].remove()
            fig_title.pop()
        fig_title.append(plt.figtext(*title_pos, title, **title_props))

    def change_error_text(error):
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        error_texts.append(plt.figtext(*error_pos, error, **error_props))

    def get_selected_indices(change_fnc=None):
        selected_indices = tuple(sorted(indices))
        if change_fnc is not None:
            num_indices = len(indices)
            if len(selected_indices) > 1:
                text = f'Selected {row_column} indices: {selected_indices}'
            elif selected_indices:
                text = f'Selected {row_column} index: {selected_indices[0]}'
            else:
                text = f'Selected {row_column} indices: None'
            if min_num_indices is not None and num_indices < min_num_indices:
                if min_num_indices == max_num_indices:
                    text += \
                        f', select another {max_num_indices-num_indices}'
                else:
                    text += \
                        f', select at least {max_num_indices-num_indices} more'
            change_fnc(text)
        return selected_indices

    def add_index(index):
        if index in indices:
            raise ValueError(f'Ignoring duplicate of selected {row_column}s')
        if max_num_indices is not None and len(indices) >= max_num_indices:
            raise ValueError(
                f'Exceeding maximum number of selected {row_column}s, click '
                'either "Reset" or "Confirm"')
        elif (indices and min_range is not None
                and abs(max(index, *indices) - min(index, *indices))
                    < min_range):
            raise ValueError(
                f'Selected {row_column} range is smaller than required '
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
        if not expression:
            return
        if error_texts:
            error_texts[0].remove()
            error_texts.pop()
        try:
            index = int(expression)
            if (index < axis_index_offset
                    or index > axis_index_offset+a.shape[axis]):
                raise ValueError
        except ValueError:
            change_error_text(
                f'Invalid {row_column} index ({expression}), enter an integer '
                f'between {axis_index_offset} and '
                f'{axis_index_offset+a.shape[axis]-1}')
        else:
            try:
                add_index(index)
                get_selected_indices(change_error_text)
            except ValueError as e:
                change_error_text(e)
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
        get_selected_indices(change_error_text)
        for ax in axs:
            ax.get_figure().canvas.draw()

    def confirm(event):
        """Callback function for the "Confirm" button."""
        if len(indices) < min_num_indices:
            change_error_text(
                f'Select at least {min_num_indices} unique {row_column}s')
            for ax in axs:
                ax.get_figure().canvas.draw()
        else:
            # Remove error texts and add selected indices if set
            if error_texts:
                error_texts[0].remove()
                error_texts.pop()
            get_selected_indices(change_fig_title)
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
    if not is_int(axis_index_offset, ge=0, log=False):
        raise ValueError(
            'Invalid parameter axis_index_offset ({axis_index_offset})')
    if preselected_indices is not None:
        if not is_int_series(
                preselected_indices, ge=axis_index_offset,
                le=axis_index_offset+a.shape[axis], log=False):
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
    title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
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
    extent = (0, a.shape[1], axis_index_offset+a.shape[0], axis_index_offset)
    axs[0].imshow(a, extent=extent)
    axs[0].set_title(title_a, fontsize='xx-large')
    if b is not None:
        axs[1].imshow(b, extent=extent)
        axs[1].set_title(title_b, fontsize='xx-large')
        if a.shape[0]+b.shape[0] > max(a.shape[1], b.shape[1]):
            axs[0].set_xlabel(column_label, fontsize='x-large')
            axs[0].set_ylabel(row_label, fontsize='x-large')
            axs[1].set_xlabel(column_label, fontsize='x-large')
        else:
            axs[0].set_ylabel(row_label, fontsize='x-large')
            axs[1].set_xlabel(column_label, fontsize='x-large')
            axs[1].set_ylabel(row_label, fontsize='x-large')
    for ax in axs:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    fig.subplots_adjust(bottom=0.0, top=0.85)

    # Setup the preselected indices if provided
    if preselected_indices is not None:
        preselected_indices = sorted(list(preselected_indices))
        for index in preselected_indices:
            add_index(index)

    if not interactive:

        get_selected_indices(change_fig_title)

    else:

        change_fig_title(title)
        get_selected_indices(change_error_text)
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
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if return_buf:
        buf = fig_to_iobuf(fig)
    else:
        buf = None
    plt.close()
    if indices:
        return buf, tuple(sorted(indices))
    return buf, None


def quick_imshow(
        a, title=None, row_label='row index', column_label='column index',
        path=None, name=None, show_fig=True, save_fig=False,
        return_fig=False, block=None, extent=None, show_grid=False,
        grid_color='w', grid_linewidth=1, **kwargs):
    """Display and or save a 2D image and or return an in-memory object
    as a byte stream represention.
    """
    if title is not None and not isinstance(title, str):
        raise ValueError(f'Invalid parameter title ({title})')
    if path is not None and not isinstance(path, str):
        raise ValueError(f'Invalid parameter path ({path})')
    if not isinstance(show_fig, bool):
        raise ValueError(f'Invalid parameter show_fig ({show_fig})')
    if not isinstance(save_fig, bool):
        raise ValueError(f'Invalid parameter save_fig ({save_fig})')
    if not isinstance(return_fig, bool):
        raise ValueError(f'Invalid parameter return_fig ({return_fig})')
    if block is not None and not isinstance(block, bool):
        raise ValueError(f'Invalid parameter block ({block})')
    if not title:
        title = 'quick imshow'
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
    plt.ioff()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.imshow(a, extent=extent, **kwargs)
    ax.set_title(title, fontsize='xx-large')
    ax.set_xlabel(column_label, fontsize='x-large')
    ax.set_ylabel(row_label, fontsize='x-large')
    if show_grid:
        ax.grid(color=grid_color, linewidth=grid_linewidth)
    if show_fig:
        plt.show(block=block)
    if save_fig:
        if name is None:
            title = re_sub(r'\s+', '_', title)
            if path is None:
                path = title
            else:
                path = f'{path}/{title}'
        else:
            if path is None:
                path = name
            else:
                path = f'{path}/{name}'
        if (os_path.splitext(path)[1]
                not in plt.gcf().canvas.get_supported_filetypes()):
            path += '.png'
        plt.savefig(path)
    if return_fig:
        buf = fig_to_iobuf(fig)
    else:
        buf = None
    plt.close()
    return buf 


def quick_plot(
        *args, xerr=None, yerr=None, vlines=None, title=None, xlim=None,
        ylim=None, xlabel=None, ylabel=None, legend=None, path=None, name=None,
        show_grid=False, save_fig=False, save_only=False, block=False,
        **kwargs):
    """Display a 2D line plot."""
    #RV FIX: Update with return_buf
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
    args = unwrap_tuple(args)
    if depth_tuple(args) > 1 and (xerr is not None or yerr is not None):
        logger.warning('Error bars ignored for multiple curves')
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
    plt.close()


def nxcopy(
        nxobject, exclude_nxpaths=None, nxpath_prefix=None,
        nxpathabs_prefix=None, nxpath_copy_abspath=None):
    """Function that returns a copy of a nexus object, optionally
    exluding certain child items.

    :param nxobject: The input nexus object to "copy".
    :type nxobject: nexusformat.nexus.NXobject
    :param exlude_nxpaths: A list of relative paths to child nexus
        objects that should be excluded from the returned "copy".
    :type exclude_nxpaths: str, list[str], optional
    :param nxpath_prefix: For use in recursive calls from inside this
        function only.
    :type nxpath_prefix: str
    :param nxpathabs_prefix: For use in recursive calls from inside
        this function only.
    :type nxpathabs_prefix: str
    :param nxpath_copy_abspath: For use in recursive calls from inside
        this function only.
    :type nxpath_copy_abspath: str
    :return: Copy of the input `nxobject` with some children optionally
        exluded.
    :rtype: nexusformat.nexus.NXobject
    """
    # Third party modules
    from nexusformat.nexus import (
        NXentry,
        NXfield,
        NXgroup,
        NXlink,
        NXlinkgroup,
        NXroot,
    )


    if isinstance(nxobject, NXlinkgroup):
        # The top level nxobject is a linked group
        # Create a group with the same name as the top level's target
        nxobject_copy = nxobject[nxobject.nxtarget].__class__(
            name=nxobject.nxname)
    elif isinstance(nxobject, (NXlink, NXfield)):
        # The top level nxobject is a (linked) field: return a copy
        attrs = nxobject.attrs
        attrs.pop('target', None)
        nxobject_copy = NXfield(
            value=nxobject.nxdata, name=nxobject.nxname,
            attrs=attrs)
        return nxobject_copy
    else:
        # Create a group with the same type/name as the nxobject
        nxobject_copy = nxobject.__class__(name=nxobject.nxname)

    # Copy attributes
    if isinstance(nxobject, NXroot):
        if 'default' in nxobject.attrs:
            nxobject_copy.attrs['default'] = nxobject.default
    else:
        for k, v in nxobject.attrs.items():
            nxobject_copy.attrs[k] = v

    # Setup paths
    if exclude_nxpaths is None:
        exclude_nxpaths = []
    elif isinstance(exclude_nxpaths, str):
        exclude_nxpaths = [exclude_nxpaths]
    for exclude_nxpath in exclude_nxpaths:
        if exclude_nxpath[0] == '/':
            raise ValueError(
                f'Invalid parameter in exclude_nxpaths ({exclude_nxpaths}), '
                'excluded paths should be relative')
    if nxpath_prefix is None:
        nxpath_prefix = ''
    if nxpathabs_prefix is None:
        if isinstance(nxobject, NXentry):
            nxpathabs_prefix = nxobject.nxpath
        else:
            nxpathabs_prefix = nxobject.nxpath.removesuffix(nxobject.nxname)
    if nxpath_copy_abspath is None:
        nxpath_copy_abspath = ''

    # Loop over all nxobject's children
    for k, v in nxobject.items():
        nxpath = os_path.join(nxpath_prefix, k)
        nxpathabs = os_path.join(nxpathabs_prefix, nxpath)
        if nxpath in exclude_nxpaths:
            if 'default' in nxobject_copy.attrs and nxobject_copy.default == k:
                nxobject_copy.attrs.pop('default')
            continue
        if isinstance(v, NXlinkgroup):
            if nxpathabs == v.nxpath and not any(
                    v.nxtarget.startswith(os_path.join(nxpathabs_prefix, p))
                    for p in exclude_nxpaths):
                nxobject_copy[k] = NXlink(v.nxtarget)
            else:
                nxobject_copy[k] = nxcopy(
                    v, exclude_nxpaths=exclude_nxpaths,
                    nxpath_prefix=nxpath, nxpathabs_prefix=nxpathabs_prefix,
                    nxpath_copy_abspath=os_path.join(nxpath_copy_abspath, k))
        elif isinstance(v, NXlink):
            if nxpathabs == v.nxpath and not any(
                    v.nxtarget.startswith(os_path.join(nxpathabs_prefix, p))
                    for p in exclude_nxpaths):
                nxobject_copy[k] = v
            else:
                nxobject_copy[k] = v.nxdata
                for kk, vv in v.attrs.items():
                    nxobject_copy[k].attrs[kk] = vv
                nxobject_copy[k].attrs.pop('target', None)
        elif isinstance(v, NXgroup):
            nxobject_copy[k] = nxcopy(
                v, exclude_nxpaths=exclude_nxpaths,
                nxpath_prefix=nxpath, nxpathabs_prefix=nxpathabs_prefix,
                nxpath_copy_abspath=os_path.join(nxpath_copy_abspath, k))
        else:
            nxobject_copy[k] = v.nxdata
            for kk, vv in v.attrs.items():
                nxobject_copy[k].attrs[kk] = vv
            if nxpathabs != os_path.join(nxpath_copy_abspath, k):
                nxobject_copy[k].attrs.pop('target', None)

    return nxobject_copy
