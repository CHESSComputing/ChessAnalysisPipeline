"""Common Pydantic model classes."""

# System modules
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
from pydantic import (
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
)
#from typing_extensions import Annotated

# Local modules
from CHAP.models import CHAPBaseModel


class BinarizeProcessorConfig(CHAPBaseModel):
    """Configuration class to binarize a dataset in a 2D or 3D
    array-like object or a NeXus NXdata or NXfield object.

    :param method: Binarization method, defaults to `'CHAP'`
        (CHAP's internal implementation of Otzu's method).
    :type method: Literal['CHAP', 'isodata', 'minimum', 'otsu', 'yen']
    :param num_bin: The number of bins used to calculate the
        histogram in the binarization algorithms, defaults to `256`.
    :type num_bin: int, optional
    :param nxpath: The path to a specific NeXus NXdata or NXfield
        object in the NeXus file tree to read the input data from
        (ignored for non-NeXus input objects).
    :type nxpath: str, optional
    :param remove_original_data: Removes the original data field
        (ignored for non-NeXus input objects), defaults to `False`.
    :type remove_original_data: bool, optional
    """
    method: Optional[Literal[
        'CHAP', 'isodata', 'minimum', 'otsu', 'yen']] = 'CHAP'
    num_bin: Optional[conint(ge=0)] = 256
    nxpath: Optional[str] = None
    remove_original_data: Optional[bool] = False


class ImageProcessorConfig(CHAPBaseModel):
    """Class representing the configuration of various image selection
    and visualization types of processors.

    :param animation: Create an animation for an image stack
        (ignored for a single image), defaults to `False`.
    :type animation: bool, optional
    :param axis: Axis direction or name for the image slice(s),
        defaults to `0`.
    :type axis: Union[int, str], optional
    :param coord_range: Coordinate value range of the selected image
        slice(s), up to three floating point numbers (start, end,
        step), defaults to `None`, which enables index_range to select
        the image slice(s). Include only `coord_range` or
        `index_range`, not both.
    :type coord_range: Union[float, list[float]], optional
    :param index_range: Array index range of the selected image
        slice(s), up to three integers (start, end, step).
        Set index_range to -1 to select the center image slice
        of an image stack in the `axis` direction. Only used when
        coord_range = `None`. Defaults to `None`, which will include
        all slices.
    :type index_range: Union[int, list[int]], optional
    :ivar fileformat: Image (stack) return file type, defaults to
        'png' for a single image, 'tif' for an image stack, or
        'gif' for an animation.
    :type fileformat: Literal['gif', 'jpeg', 'png', 'tif'], optional
    :param vrange: Data value range in image slice(s), defaults to
        `None`, which uses the full data value range in the slice(s).
    :type vrange: list[float, float]
    """
    animation: Optional[bool] = False
    axis: Optional[Union[conint(ge=0), constr(min_length=1)]] = 0
    coord_range: Optional[Union[
        confloat(allow_inf_nan=False),
        conlist(min_length=2, max_length=3,
                item_type=confloat(allow_inf_nan=False))]] = None
    index_range: Optional[Union[
        int,
        conlist(
            min_length=2, max_length=3, item_type=Union[None, int])]] = None
    fileformat: Optional[Literal['gif', 'jpeg', 'png', 'tif']] = None
    vrange: Optional[
        conlist(min_length=2, max_length=2,
                item_type=confloat(allow_inf_nan=False))] = None

    @field_validator('index_range', mode='before')
    @classmethod
    def validate_index_range(cls, index_range):
        """Validate the index_range.

        :ivar index_range: Array index range of the selected image
            slice(s), defaults to `None`..
        :type index_range: Union[float, list[float]], optional
        :return: Validated index_range.
        :rtype: list[int]
        """
        if isinstance(index_range, int):
            return index_range
        return [None if isinstance(i, str) and i.lower() == 'none' else i
                for i in index_range]

    @field_validator('vrange', mode='before')
    @classmethod
    def validate_vrange(cls, vrange):
        """Validate the vrange.

        :ivar vrange: Data value range in image slice(s),
            defaults to `None`..
        :type vrange: list[float, float], optional
        :return: Validated vrange.
        :rtype: list[float, float]
        """
        if isinstance(vrange, (list, tuple)) and len(vrange) == 2:
            return [min(vrange), max(vrange)]
        return vrange
