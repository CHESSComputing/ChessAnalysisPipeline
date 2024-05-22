"""Tomography Pydantic model classes."""

# Third party imports
from typing import (
    Literal,
    Optional,
)
from pydantic import (
    BaseModel,
    conint,
    conlist,
    confloat,
    constr,
)

class Detector(BaseModel):
    """
    Detector class to represent the detector used in the experiment.
    The image origin is assumed to be in the top-left corner, with
    rows down (-z in lab frame) and columns sideways (+x in lab frame).

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


class TomoReduceConfig(BaseModel):
    """
    Class representing the configuration for the tomography image
    reduction processor.

    :ivar img_row_bounds: Detector image bounds in the row-direction
        (ignored for id1a3 and id3a).
    :type img_row_bounds: list[int], optional
    :ivar delta_theta: Rotation angle increment in image reduction
        in degrees.
    :type delta_theta: float, optional
    """
    img_row_bounds: Optional[
        conlist(item_type=conint(ge=-1), min_items=2, max_items=2)]
    delta_theta: Optional[confloat(gt=0, allow_inf_nan=False)]


class TomoFindCenterConfig(BaseModel):
    """
    Class representing the configuration for the tomography center axis
    finding processor.

    :ivar center_stack_index: Stack index of the tomography set to find
        the center axis.
    :type center_stack_index: int, optional
    :ivar center_rows: Row indices for the center finding processor.
    :type center_rows: list[int, int], optional
    :ivar center_offsets: Centers at the center finding row indices in
        pixels.
    :type center_offsets: list[float, float], optional
    :ivar center_offset_min: Minimum value of center_offset in center
        axis finding search in pixels.
    :type center_offset_min: float, optional
    :ivar center_offset_max: Maximum value of center_offset in center
        axis finding search in pixels.
    :type center_offset_max: float, optional
    :ivar gaussian_sigma: Standard deviation for the Gaussian filter
        applied to image reconstruction visualizations, defaults to no
        filtering performed.
    :type gaussian_sigma: float, optional
    :ivar ring_width: Maximum width of rings to be filtered in the image
        reconstruction in pixels, defaults to no filtering performed.
    :type ring_width: float, optional
    """
    center_stack_index: Optional[conint(ge=0)]
    center_rows: Optional[conlist(
        item_type=conint(ge=0), min_items=2, max_items=2)]
    center_offsets: Optional[conlist(
        item_type=confloat(allow_inf_nan=False),
        min_items=2, max_items=2)]
    center_offset_min: Optional[confloat(allow_inf_nan=False)]
    center_offset_max: Optional[confloat(allow_inf_nan=False)]
    center_search_range: Optional[conlist(
        item_type=confloat(allow_inf_nan=False),
        min_items=1, max_items=3)]
    gaussian_sigma: Optional[confloat(ge=0, allow_inf_nan=False)]
    ring_width: Optional[confloat(ge=0, allow_inf_nan=False)]


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
    :ivar gaussian_sigma: Standard deviation for the Gaussian filter
        applied to image reconstruction visualizations, defaults to no
        filtering performed.
    :type gaussian_sigma: float, optional
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
    gaussian_sigma: Optional[confloat(ge=0, allow_inf_nan=False)]
    remove_stripe_sigma: Optional[confloat(ge=0, allow_inf_nan=False)]
    ring_width: Optional[confloat(ge=0, allow_inf_nan=False)]


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
    :type station: Literal['id1a3', 'id3a', 'id3b']
    :ivar detector: Detector used in the tomography experiment.
    :type detector: Detector
    :ivar sample_type: Sample type for the tomography simulator.
    :type sample_type: Literal['square_rod', 'square_pipe',
        'hollow_cube', 'hollow_brick', 'hollow_pyramid']
    :ivar sample_size: Size of each sample dimension in mm (internally
        converted to an integer number of pixels). Enter three values
        for sample_type == `'hollow_pyramid'`, the height and the side
        at the respective bottom and the top of the pyramid.
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
    :type background_intensity:: float, optional
    :ivar slit_size: Vertical beam height in mm, defaults to 1.0.
    :type slit_size:: float, optional
    """
    station: Literal['id1a3', 'id3a', 'id3b']
    detector: Detector.construct()
    sample_type: Literal[
        'square_rod', 'square_pipe', 'hollow_cube', 'hollow_brick',
        'hollow_pyramid']
    sample_size: conlist(
        item_type=confloat(gt=0, allow_inf_nan=False),
        min_items=1, max_items=3)
    wall_thickness: Optional[confloat(ge=0, allow_inf_nan=False)]
    mu: Optional[confloat(gt=0, allow_inf_nan=False)] = 0.05
    theta_step: confloat(gt=0, allow_inf_nan=False)
    beam_intensity: Optional[confloat(gt=0, allow_inf_nan=False)] = 1.e9
    background_intensity: Optional[confloat(gt=0, allow_inf_nan=False)] = 20
    slit_size: Optional[confloat(gt=0, allow_inf_nan=False)] = 1.0
