"""Tomography Pydantic model classes."""

# Third party imports
from typing import (
    Literal,
    Optional,
)
from pydantic import (
    BaseModel,
    StrictBool,
    conint,
    conlist,
    confloat,
    constr,
)

# Local modules
from CHAP.common.models.map import SpecScans

class Detector(BaseModel):
    """
    Detector class to represent the detector used in the experiment.

    :ivar prefix: Prefix of the detector in the SPEC file.
    :type prefix: str
    :ivar rows: Number of pixel rows on the detector.
    :type rows: int
    :ivar columns: Number of pixel columns on the detector.
    :type columns: int
    :ivar pixel_size: Pixel size of the detector in mm.
    :type pixel_size: int or list[int]
    :ivar lens_magnification: Lens magnification for the detector,
        defaults to 1.0.
    :type lens_magnification: float, optional
    """
    prefix: constr(strip_whitespace=True, min_length=1)
    rows: conint(gt=0)
    columns: conint(gt=0)
    pixel_size: conlist(
        item_type=confloat(gt=0, allow_inf_nan=False),
        min_items=1, max_items=2)
    lens_magnification: confloat(gt=0, allow_inf_nan=False) = 1.0


class TomoSetupConfig(BaseModel):
    """
    Class representing the configuration for the tomography
    reconstruction setup processor.

    :ivar detector: Detector used in the tomography experiment.
    :type detector: Detector
    :ivar include_raw_data: Flag to designate whether raw data will be
        included (True) or not (False), defaults to False.
    :type include_raw_data: bool, optional
    """
    detector: Detector.construct()
    include_raw_data: Optional[StrictBool] = False
    dark_field: Optional[conlist(item_type=SpecScans, min_items=1)]
    bright_field: Optional[conlist(item_type=SpecScans, min_items=1)]


class TomoReduceConfig(BaseModel):
    """
    Class representing the configuration for the tomography image
    reduction processor.

    :ivar detector: Detector used in the tomography experiment.
    :type detector: Detector
    :ivar img_x_bounds: Detector image bounds in the x-direction.
    :type img_x_bounds: list[int], optional
    :ivar delta_theta: Rotation angle increment in image reduction
        in degrees.
    :type delta_theta: float, optional
    """
    detector: Detector = Detector.construct()
    img_x_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    delta_theta: Optional[confloat(gt=1, allow_inf_nan=False)]
    dark_field: Optional[conlist(item_type=SpecScans, min_items=1)]
    bright_field: Optional[conlist(item_type=SpecScans, min_items=1)]


class TomoFindCenterConfig(BaseModel):
    """
    Class representing the configuration for the tomography center axis
    finding processor.

    :ivar center_stack_index: Stack index of the tomography set to find
        the center axis.
    :type center_stack_index: int, optional
    :ivar lower_row: Lower row index for the center finding processor.
    :type lower_row: int, optional
    :ivar lower_center_offset: Center at the lower row index.
    :type lower_center_offset: float, optional
    :ivar upper_row: Upper row index for the center finding processor.
    :type upper_row: int, optional
    :ivar upper_center_offset: Center at the upper row index.
    :type upper_center_offset: float, optional
    :ivar gaussian_sigma: Standard deviation for the Gaussian filter
        applied to image reconstruction visualizations, defaults to no
        filtering performed.
    :type gaussian_sigma: float, optional
    :ivar ring_width: Maximum width of rings to be filtered in the image
        reconstruction in pixels, defaults to no filtering performed.
    :type ring_width: float, optional
    """
    center_stack_index: Optional[conint(ge=0)]
    lower_row: Optional[conint(ge=-1)]
    lower_center_offset: Optional[confloat(allow_inf_nan=False)]
    upper_row: Optional[conint(ge=-1)]
    upper_center_offset: Optional[confloat(allow_inf_nan=False)]
    gaussian_sigma: Optional[confloat(gt=0, allow_inf_nan=False)]
    ring_width: Optional[confloat(gt=0, allow_inf_nan=False)]


class TomoReconstructConfig(BaseModel):
    """
    Class representing the configuration for the tomography image
    reconstruction processor.

    :ivar x_bounds: Reconstructed image bounds in the x-direction.
    :type x_bounds: list[int], optional
    :ivar y_bounds: Reconstructed image bounds in the y-direction.
    :type y_bounds: list[int], optional
    :ivar z_bounds: Reconstructed image bounds in the z-direction.
    :type z_bounds: list[int], optional
    :ivar secondary_iters: Number of secondary iterations in the tomopy
        image reconstruction algorithm, defaults to 0.
    :type secondary_iters: int, optional
    :ivar remove_stripe_sigma: Damping parameter in Fourier space in
        tomopy's horizontal stripe removal tool, defaults to no
        correction performed.
    :type remove_stripe_sigma: float, optional
    :ivar ring_width: Maximum width of rings to be filtered in the image
        reconstruction in pixels, defaults to no filtering performed.
    :type ring_width: float, optional
    """
    x_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    y_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    z_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    secondary_iters: conint(ge=0) = 0
    remove_stripe_sigma: Optional[confloat(gt=0, allow_inf_nan=False)]
    ring_width: Optional[confloat(gt=0, allow_inf_nan=False)]


class TomoCombineConfig(BaseModel):
    """
    Class representing the configuration for the combined tomography
    stacks processor.

    :ivar x_bounds: Combined image bounds in the x-direction.
    :type x_bounds: list[int], optional
    :ivar y_bounds: Combined image bounds in the y-direction.
    :type y_bounds: list[int], optional
    :ivar z_bounds: Combined image bounds in the z-direction.
    :type z_bounds: list[int], optional
    """
    x_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    y_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    z_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]


class TomoSimConfig(BaseModel):
    """
    Class representing the configuration for the tomography simulator.

    :ivar station: The station name (in 'idxx' format).
    :type station: str
    :ivar detector: Detector used in the tomography experiment.
    :type detector: Detector
    :ivar sample_type: Sample type for the tomography simulator, one of
        "square_rod", "square_pipe", "hollow_cube", or "hollow_brick".
    :type sample_type: str
    :ivar sample_size: Size of each sample dimension in mm (internally
        converted to an integer number of pixels).
    :type sample_size: list[float]
    :ivar wall_thickness: Wall thickness for pipe, cube, and brick in
        mm (internally converted to an integer number of pixels).
    :type wall_thickness: float
    :ivar mu: Linear attenuation coefficient in mm^-1, defaults to 0.05.
    :type mu: float, optional
    :ivar theta_step: Rotation angle increment in the tomography
        simulation in degrees.
    :type theta_step: float
    :ivar beam_intensity: Initial beam intensity in counts,
        defaults to 1.e9.
    :type beam_intensity: float, optional
    :ivar background_intensity: Background intensity in counts,
        defaults to 20.
    :type beam_intensity: float, optional
    :ivar slit_size: Vertical beam height in mm, defaults to 1.0.
    :type beam_intensity: float, optional
    """
    station: Literal['id1a3','id3a','id3b']
    detector: Detector.construct()
    sample_type: Literal[
        'square_rod', 'square_pipe', 'hollow_cube', 'hollow_brick']
    sample_size: conlist(
        item_type=confloat(gt=0, allow_inf_nan=False),
        min_items=1, max_items=2)
    wall_thickness: Optional[confloat(ge=0, allow_inf_nan=False)]
    mu: Optional[confloat(gt=0, allow_inf_nan=False)] = 0.05
    theta_step: confloat(gt=0, allow_inf_nan=False)
    beam_intensity: Optional[confloat(gt=0, allow_inf_nan=False)] = 1.e9
    background_intensity: Optional[confloat(gt=0, allow_inf_nan=False)] = 20
    slit_size: Optional[confloat(gt=0, allow_inf_nan=False)] = 1.0
