"""Common `Pydantic <https://github.com/pydantic/pydantic>`__ model
configuration classes.
"""

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


class BinarizeConfig(CHAPBaseModel):
    """Configuration class to binarize a dataset in a 2D or 3D
    array-like object or a NeXus style
    `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html>`__
    or
    `NXfield <https://nexpy.github.io/nexpy/treeapi.html#nexusformat.nexus.tree.NXfield>`__
    object.

    :ivar method: Binarization method, defaults to `'CHAP'`
        (CHAP's internal implementation of Otzu's method).
    :vartype method: Literal[
        'CHAP', 'isodata', 'minimum', 'otsu', 'yen']
    :ivar num_bin: The number of bins used to calculate the
        histogram in the binarization algorithms, defaults to `256`.
    :vartype num_bin: int, optional
    :ivar nxpath: The path to a specific NeXus style NXdata or NXfield
        object in the NeXus file tree to read the input data from
        (ignored for non-NeXus input objects).
    :vartype nxpath: str, optional
    :ivar remove_original_data: Removes the original data field
        (ignored for non-NeXus input objects), defaults to `False`.
    :vartype remove_original_data: bool, optional
    """

    method: Optional[Literal[
        'CHAP', 'isodata', 'minimum', 'otsu', 'yen']] = 'CHAP'
    num_bin: Optional[conint(ge=0)] = 256
    nxpath: Optional[str] = None
    remove_original_data: Optional[bool] = False


class ImageProcessorConfig(CHAPBaseModel):
    """Class representing the configuration of various image selection
    and visualization types of processors.

    :ivar animation: Create an animation for an image stack
        (ignored for a single image), defaults to `False`.
    :vartype animation: bool, optional
    :ivar axis: Axis direction or name for the image slice(s),
        defaults to `0`.
    :vartype axis: int or str, optional
    :ivar basename: Basename of each file when saving a set of 'tif'
        images (only used when 'fileformat' = 'fit'), defaults to
        'image'.
    :vartype basename: str, optional
    :ivar coord_range: Coordinate value range of the selected image
        slice(s), up to three floating point numbers (start, end,
        step), defaults to `None`, which enables index_range to select
        the image slice(s). Include only `coord_range` or
        `index_range`, not both.
    :vartype coord_range: float or list[float], optional
    :ivar index_range: Array index range of the selected image
        slice(s), up to three integers (start, end, step).
        Set index_range to -1 to select the center image slice
        of an image stack in the `axis` direction. Only used when
        coord_range = `None`. Defaults to `None`, which will include
        all slices.
    :vartype index_range: int or list[int], optional
    :ivar fileformat: Image (stack) return file type, defaults to
        'png' for a single image, 'tif' for a (set of) 'tif' image(s),
        or 'gif' for an animation. Set to 'tifstack' for a single 'tif'
        image stack.
    :vartype fileformat: Literal[
        'gif', 'jpeg', 'png', 'tif' 'tifstack'], optional
    :ivar vrange: Data value range in image slice(s), defaults to
        `None`, which uses the full data value range in the slice(s).
        Specify as [None, float] or [float, None] to set only the upper
        or lower limit of the value range.
    :vartype vrange: list[float, float]
    """

    animation: Optional[bool] = False
    axis: Optional[Union[conint(ge=0), constr(min_length=1)]] = 0
    basename: Optional[constr(min_length=1)] = 'image'
    # FIX convert to using CHAPSlice
    coord_range: Optional[Union[
        confloat(allow_inf_nan=False),
        conlist(min_length=2, max_length=3,
                item_type=confloat(allow_inf_nan=False))]] = None
    index_range: Optional[Union[
        int,
        conlist(
            min_length=2, max_length=3, item_type=Union[None, int])]] = None
    fileformat: Optional[
        Literal['gif', 'jpeg', 'png', 'tif', 'tifstack']] = None
    vrange: Optional[
        conlist(min_length=2, max_length=2,
                item_type=Union[None, confloat(allow_inf_nan=False)])] = None

    @field_validator('index_range', mode='before')
    @classmethod
    def validate_index_range(cls, index_range):
        """Validate the index_range.

        :ivar index_range: Array index range of the selected image
            slice(s), defaults to `None`.
        :type index_range: int or list[int], optional
        :return: Validated index_range.
        :rtype: list[int]
        """
        if isinstance(index_range, int):
            return [index_range]
        return [None if isinstance(i, str) and i.lower() == 'none' else i
                for i in index_range]

    @field_validator('vrange', mode='before')
    @classmethod
    def validate_vrange(cls, vrange):
        """Validate the vrange.

        :ivar vrange: Data value range in image slice(s),
            defaults to `None`.
        :type vrange: list[float, float], optional
        :return: Validated vrange.
        :rtype: list[float, float]
        """
        if isinstance(vrange, (list, tuple)) and len(vrange) == 2:
            if None not in vrange:
                return [min(vrange), max(vrange)]
        return [None if isinstance(i, str) and i.lower() == 'none' else i
                for i in index_range]


class UnstructuredToStructuredConfig(CHAPBaseModel):
    """Configuration class to reshape data in an
    `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html>`__
    from an "unstructured" to a "structured" representation.

    :ivar nxpath: Path to a specific NeXus style NXdata object in the
        NeXus file tree to read the input data from.
    :vartype nxpath: str, optional
    :ivar signals: Paths to the dataset's signal-like fields to
        reshape (in addition to possible ones in the optional `nxpath`
        object).
    :vartype signals: str or list[str], optional
    :ivar unstructured_axes: Names of the dataset's unstructured axes
        fields. Defaults to the `'unstructured axis'` attribute of the
        default NeXus style NXdata object or that specified in `nxpath`
        if present. If `nxpath` is unspecified and there is no default
        NeXus style NXdata object, the `unstructured_axes` is required
        and has to contain full paths to the unstructured axes fields.
    :vartype unstructured_axes: str or list[str], optional
    """

    nxpath: Optional[str] = None
    signals: Optional[
        Union[str, conlist(min_length=1, item_type=str)]] = None
    unstructured_axes: Optional[
        Union[str, conlist(min_length=1, item_type=str)]] = None

    @field_validator('nxpath', mode='before')
    @classmethod
    def validate_nxpath(cls, nxpath):
        """Validate nxpath.

        :param nxpath: Path to a specific NeXus style NXdata object in
            the NeXus file tree to read the input data from.
        :type nxpath: str
        :return: Validated nxpath.
        :rtype: str
        """
        if nxpath[0] == '/':
            nxpath = nxpath[1:]
        return nxpath

    @field_validator('signals', mode='before')
    @classmethod
    def validate_signals(cls, signals):
        """Validate signals.

        :param signals: The (additional) dataset's signal-like fields.
        :type signals: str or list[str]
        :return: Validated signals.
        :rtype: list[str]
        """
        if isinstance(signals, str):
            signals = [signals]
        for i, signal in enumerate(signals):
            if signal[0] == '/':
                signals[i] = signal[1:]
        return signals

    @field_validator('unstructured_axes', mode='before')
    @classmethod
    def validate_unstructured_axes(cls, unstructured_axes):
        """Validate unstructured_axes.

        :param unstructured_axes: The dataset's unstructured axes.
        :type unstructured_axes: str or list[str]
        :return: Validated unstructured axes.
        :rtype: list[str]
        """
        if isinstance(unstructured_axes, str):
            unstructured_axes = [unstructured_axes]
        for i, axis in enumerate(unstructured_axes):
            if axis[0] == '/':
                unstructured_axes[i] = axis[1:]
        return unstructured_axes
