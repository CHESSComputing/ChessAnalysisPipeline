"""Common Pydantic model classes."""

# System modules
import os
from typing import (
    Optional,
    Union,
)

# Third party modules
from pydantic import (
    DirectoryPath,
    Field,
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)
#from typing_extensions import Annotated

# Local modules
from CHAP.models import CHAPBaseModel


class ImageProcessorConfig(CHAPBaseModel):
    """Class representing the configuration of various image selection
    and visualization types of processors.

    :param animation: Create an animation, defaults to `False`.
    :type animation: bool, optional
    :param axis: Axis direction or name of the image slice(s),
        defaults to `0`.
    :type axis: Union[int, str], optional
    :param coord_range: Coordinate value range of the selected image
        slice(s), up to three floats (start, end, step),
        defaults to `None`, which will include all slices. Include
        only `coord_range` or `index_range`, not both.
    :type coord_range: Union[float, list[float]], optional
    :param index_range: Array index range of the selected image
        slice(s), up to three integers (start, end, step),
        defaults to `None`, which will include all slices.
    :type index_range: Union[int, list[int]], optional
    :ivar filename: Image (stack) output filename, defaults to
        `imagestack.tif`.
    :type filename: str, optional
    :param vrange: Data value range in image slice(s), defaults to
        `None`, which uses the full data value range in the slice(s).
    :type vrange: list[float, float]
    :type vmax: float

    """
    outputdir: Optional[DirectoryPath] = Field(None, exclude=True)
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
    filename: Optional[str]
    vrange: Optional[
        conlist(min_length=2, max_length=2,
                item_type=confloat(allow_inf_nan=False))] = None

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        filename path.

        :param data: Pydantic validator data object.
        :type data: ImageProcessorConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        if isinstance(data, dict):
            outputdir = data.get('outputdir')
            if outputdir is not None:
                filename = data.get('filename')
                if filename is None:
                    data['filename'] = os.path.join(
                        outputdir, 'imagestack.tif')
                elif not os.path.isabs(filename):
                    data['filename'] = os.path.join(outputdir, filename)
        return data

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
