# system modules

# third party imports
from pydantic import (
        BaseModel,
        StrictBool,
        conint,
        conlist,
        confloat,
        constr,
)
from typing import Literal, Optional


class Detector(BaseModel):
    """
    Detector class to represent the detector used in the experiment.
    
    :ivar prefix: Prefix of the detector in the SPEC file.
    :type prefix: str
    :ivar rows: Number of pixel rows on the detector
    :type rows: int
    :ivar columns: Number of pixel columns on the detector
    :type columns: int
    :ivar pixel_size: Pixel size of the detector in mm
    :type pixel_size: int or list[int]
    :ivar lens_magnification: Lens magnification for the detector
    :type lens_magnification: float, optional
    """
    prefix: constr(strip_whitespace=True, min_length=1)
    rows: conint(gt=0)
    columns: conint(gt=0)
    pixel_size: conlist(item_type=confloat(gt=0, allow_inf_nan=False), min_items=1, max_items=2)
    lens_magnification: confloat(gt=0, allow_inf_nan=False) = 1.0


class TomoSetupConfig(BaseModel):
    """
    Class representing the configuration for the tomography reconstruction setup.

    :ivar detectors: Detector used in the tomography experiment
    :type detectors: Detector
    """
    detector: Detector.construct()
    include_raw_data: Optional[StrictBool] = False


class TomoReduceConfig(BaseModel):
    """
    Class representing the configuration for tomography image reductions.

    :ivar tool_type: Type of tomography reconstruction tool; always set to "reduce_data"
    :type tool_type: str, optional
    :ivar detectors: Detector used in the tomography experiment
    :type detectors: Detector
    :ivar img_x_bounds: Detector image bounds in the x-direction
    :type img_x_bounds: list[int], optional
    """
    tool_type: Literal['reduce_data'] = 'reduce_data'
    detector: Detector = Detector.construct()
    img_x_bounds: Optional[conlist(item_type=conint(ge=0), min_items=2, max_items=2)]


class TomoFindCenterConfig(BaseModel):
    """
    Class representing the configuration for tomography find center axis.

    :ivar tool_type: Type of tomography reconstruction tool; always set to "find_center"
    :type tool_type: str, optional
    :ivar center_stack_index: Stack index of tomography set to find center axis (offset 1)
    :type center_stack_index: int, optional
    :ivar lower_row: Lower row index for center finding
    :type lower_row: int, optional
    :ivar lower_center_offset: Center at lower row index
    :type lower_center_offset: float, optional
    :ivar upper_row: Upper row index for center finding
    :type upper_row: int, optional
    :ivar upper_center_offset: Center at upper row index
    :type upper_center_offset: float, optional
    """
    tool_type: Literal['find_center'] = 'find_center'
    center_stack_index: Optional[conint(ge=1)]
    lower_row: Optional[conint(ge=-1)]
    lower_center_offset: Optional[confloat(allow_inf_nan=False)]
    upper_row: Optional[conint(ge=-1)]
    upper_center_offset: Optional[confloat(allow_inf_nan=False)]


class TomoReconstructConfig(BaseModel):
    """
    Class representing the configuration for tomography image reconstruction.

    :ivar tool_type: Type of tomography reconstruction tool; always set to "reconstruct_data"
    :type tool_type: str, optional
    :ivar x_bounds: Reconstructed image bounds in the x-direction
    :type x_bounds: list[int], optional
    :ivar y_bounds: Reconstructed image bounds in the y-direction
    :type y_bounds: list[int], optional
    :ivar z_bounds: Reconstructed image bounds in the z-direction
    :type z_bounds: list[int], optional
    """
    tool_type: Literal['reconstruct_data'] = 'reconstruct_data'
    x_bounds: Optional[conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    y_bounds: Optional[conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    z_bounds: Optional[conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]


class TomoCombineConfig(BaseModel):
    """
    Class representing the configuration for combined tomography stacks.

    :ivar tool_type: Type of tomography reconstruction tool; always set to "combine_data"
    :type tool_type: str, optional
    :ivar x_bounds: Reconstructed image bounds in the x-direction
    :type x_bounds: list[int], optional
    :ivar y_bounds: Reconstructed image bounds in the y-direction
    :type y_bounds: list[int], optional
    :ivar z_bounds: Reconstructed image bounds in the z-direction
    :type z_bounds: list[int], optional
    """
    tool_type: Literal['combine_data'] = 'combine_data'
    x_bounds: Optional[conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    y_bounds: Optional[conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    z_bounds: Optional[conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]

