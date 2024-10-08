"""GIWAXS Pydantic model classes."""

# System modules
import os
from typing import (
    Literal,
    Optional,
)

# Third party modules
import numpy as np
from pydantic import (
    BaseModel,
    FilePath,
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
)


class Detector(BaseModel):
    """Detector class to represent a single detector used in the
    experiment.

    :param prefix: Prefix of the detector in the SPEC file.
    :type prefix: str
    :param poni_file: Path to the PONI file.
    :type poni_file: str
    """
    prefix: constr(strip_whitespace=True, min_length=1)
    poni_file: FilePath

    @field_validator('poni_file')
    @classmethod
    def validate_poni_file(cls, poni_file):
        """Validate the PONI file by checking if it's a valid PONI
        file.

        :param poni_file: Path to the PONI file.
        :type poni_file: str
        :raises ValueError: If poni_file is not a valid PONI file.
        :returns: Absolute path to the PONI file.
        :rtype: str
        """
        # Third party modules
        from pyFAI import load

        poni_file = os.path.abspath(poni_file)
        try:
            load(poni_file)
        except Exception as exc:
            raise ValueError(f'{poni_file} is not a valid PONI file') from exc
        return poni_file


class GiwaxsConversionConfig(BaseModel):
    """Class representing metadata required to locate GIWAXS image
    files for a single scan to convert to q_par/q_perp coordinates.

    :ivar detectors: List of detector configurations.
    :type detectors: list[Detector]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :type scan_step_indices: Union(int, list[int], str), optional
    :ivar save_raw_data: Save the raw data in the NeXus output,
        defaults to `False`.
    :type save_raw_data: bool, optional
    """
    detectors: conlist(item_type=Detector, min_length=1)
    scan_step_indices: Optional[
        conlist(item_type=conint(ge=0), min_length=1)] = None
    save_raw_data: Optional[bool] = False

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Validate the specified list of scan step indices.

        :param scan_step_indices: List of scan numbers.
        :type scan_step_indices: list of int
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list[int]
        """
        if isinstance(scan_step_indices, int):
            scan_step_indices = [scan_step_indices]
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)

        return scan_step_indices

class IntegrationConfig(BaseModel):
    """Class representing the configuration for a raw detector data
    integration.

    :ivar tool_type: Type of integration tool; always set to
        "integration".
    :type tool_type: str, optional
    :ivar title: Title of the integration.
    :type title: str
    :ivar integration_type: Type of integration.
    :type integration_type: Literal['azimuthal', 'radial', 'cake']
    :ivar detectors: List of detector configurations.
    :type detectors: list[Detector]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :type scan_step_indices: Union(int, list[int], str), optional
    :ivar radial_units: Radial units for the integration,
        defaults to `'q_A^-1'`.
    :type radial_units: str, optional
    :ivar radial_min: Minimum radial value for the integration range.
    :type radial_min: float
    :ivar radial_max: Maximum radial value for the integration range.
    :type radial_max: float
    :ivar radial_npt: Number of points in the radial range for the
        integration.
    :type radial_npt: int, optional
    :ivar azimuthal_units: Azimuthal units for the integration.
    :type azimuthal_units: str, optional
    :ivar azimuthal_min: Minimum azimuthal value for the integration
        range.
    :type azimuthal_min: float, optional
    :ivar azimuthal_max: Maximum azimuthal value for the integration
        range.
    :type azimuthal_max: float, optional
    :ivar azimuthal_npt: Number of points in the azimuthal range for
        the integration.
    :type azimuthal_npt: int, optional
    :ivar include_errors: option to include pyFAI's calculated Poisson
        errors with the integrtion results, defaults to `False`.
    :type include_errors: bool, optional
    :ivar right_handed: For radial and cake integration, reverse the
        direction of the azimuthal coordinate from pyFAI's convention,
        defaults to True.
    :type right_handed: bool, optional
    """
    tool_type: Literal['integration'] = 'integration'
    title: constr(strip_whitespace=True, min_length=1)
    integration_type: Literal['azimuthal', 'radial', 'cake']
    detectors: conlist(item_type=Detector, min_length=1)
    scan_step_indices: Optional[
        conlist(item_type=conint(ge=0), min_length=1)] = None
    radial_units: str = 'q_A^-1'
    radial_min: confloat(ge=0)
    radial_max: confloat(gt=0)
    radial_npt: conint(gt=0) = 1800
    azimuthal_units: str = 'chi_deg'
    azimuthal_min: confloat(ge=-180) = -180
    azimuthal_max: confloat(le=360) = 180
    azimuthal_npt: conint(gt=0) = 3600
    include_errors: bool = False
    right_handed: bool = True

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Validate the specified list of scan step indices.

        :param scan_step_indices: List of scan numbers.
        :type scan_step_indices: list of int
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list[int]
        """
        if isinstance(scan_step_indices, int):
            scan_step_indices = [scan_step_indices]
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)

        return scan_step_indices

    @field_validator('radial_units')
    @classmethod
    def validate_radial_units(cls, radial_units):
        """Validate the radial units for the integration.

        :param radial_units: Unvalidated radial units for the
            integration.
        :type radial_units: str
        :raises ValueError: If radial units are not one of the
            recognized radial units.
        :return: Validated radial units.
        :rtype: str
        """
        # Third party modules
        from pyFAI.units import RADIAL_UNITS

        if radial_units in RADIAL_UNITS.keys():
            return radial_units
        else:
            raise ValueError(
                f'Invalid radial units: {radial_units}. Must be one of '
                ', '.join(RADIAL_UNITS.keys()))

    @field_validator('azimuthal_units')
    def validate_azimuthal_units(cls, azimuthal_units):
        """Validate that `azimuthal_units` is one of the keys in the
        `pyFAI.units.AZIMUTHAL_UNITS` dictionary.

        :param azimuthal_units: The string representing the unit to be
            validated.
        :type azimuthal_units: str
        :raises ValueError: If `azimuthal_units` is not one of the
            keys in `pyFAI.units.AZIMUTHAL_UNITS`.
        :return: The original supplied value, if is one of the keys in
            `pyFAI.units.AZIMUTHAL_UNITS`.
        :rtype: str
        """
        # Third party modules
        from pyFAI.units import AZIMUTHAL_UNITS

        if azimuthal_units in AZIMUTHAL_UNITS.keys():
            return azimuthal_units
        else:
            raise ValueError(
                f'Invalid azimuthal units: {azimuthal_units}. Must be one of '
                ', '.join(AZIMUTHAL_UNITS.keys()))

    def validate_range_max(range_name):
        """Validate the maximum value of an integration range.

        :param range_name: The name of the integration range
            (e.g. radial, azimuthal).
        :type range_name: str
        :return: The callable that performs the validation.
        :rtype: callable
        """
        def _validate_range_max(cls, range_max, info):
            """Check if the maximum value of the integration range is
            greater than its minimum value.

            :param range_max: The maximum value of the integration
                range.
            :type range_max: float
            :param info: Pydantic validator info object.
            :type info: pydantic_core._pydantic_core.ValidationInfo
            :raises ValueError: If the maximum value of the
                integration range is not greater than its minimum
                value.
            :return: The validated maximum range value.
            :rtype: float
            """
            range_min = info.data.get(f'{range_name}_min')
            if range_min < range_max:
                return range_max
            raise ValueError(
                'Maximum value of integration range must be greater than'
                'its minimum value ({range_name}_min={range_min}).')
        return _validate_range_max

    _validate_radial_max = field_validator(
        'radial_max')(validate_range_max('radial'))
    _validate_azimuthal_max = field_validator(
        'azimuthal_max')(validate_range_max('azimuthal'))

