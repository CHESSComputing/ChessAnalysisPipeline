from CHAP.common.utils.fit import (Fit,
                                   FitMap,
                                   FitMultipeak)
from CHAP.common.utils.material import Material

# def create_mask(data, bounds, exclude_bounds=True, current_mask=None):
#     '''Return a boolean array that masks out the values in `bounds` when applied
#     to `data`.

#     :param data: the array for which a mask will be constructed
#     :type data: Union[list, numpy.ndarray]
#     :param bounds: a range of values in `data` (min, max) that the mask will
#         exclude (or include if `exclude_bounds=False`).
#     :type bounds: tuple
#     :param exclude_bounds: should applying the mask to `data`  exclude (`True`)
#         or include (`False`) the value ranges in `bounds`, defaults to `True`
#     :type exclude_bounds: True, optional
#     :param current_mask: an existing mask array for `data` that will be "or"-ed
#         with the mask constructed from `bounds` before returning, defaults to
#         None
#     :type current_mask: numpy.ndarray(dtype=numpy.bool_), optional
#     :return: a boolean mask array for `data`.
#     :rtype: numpy.ndarray(dtype=numpy.bool_)
#     '''

#     import numpy as np

#     min_, max_ = bounds
#     if exclude_bounds:
#         mask = np.logical_or(data < min_, data > max_)
#     else:
#         mask = np.logical_and(data > min_, data < max_)

#     if current_mask is not None:
#         mask = np.logical_or(mask, current_mask)

#     return(mask)
