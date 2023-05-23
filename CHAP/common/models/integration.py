# System modules
import copy
from functools import cache
import os
from typing import Literal, Optional

# Third party modules
import numpy as np
from pydantic import (BaseModel,
                      validator,
                      constr,
                      conlist,
                      conint,
                      confloat,
                      FilePath)
from pyFAI import load as pyfai_load
from pyFAI.multi_geometry import MultiGeometry
from pyFAI.units import AZIMUTHAL_UNITS, RADIAL_UNITS


class Detector(BaseModel):
    """Detector class to represent a single detector used in the
    experiment.

    :param prefix: Prefix of the detector in the SPEC file.
    :type prefix: str
    :param poni_file: Path to the poni file.
    :type poni_file: str
    :param mask_file: Optional path to the mask file.
    :type mask_file: str, optional
    """
    prefix: constr(strip_whitespace=True, min_length=1)
    poni_file: FilePath
    mask_file: Optional[FilePath]

    @validator('poni_file', allow_reuse=True)
    def validate_poni_file(cls, poni_file):
        """Validate the poni file by checking if it's a valid PONI
        file.

        :param poni_file: Path to the poni file.
        :type poni_file: str
        :raises ValueError: If poni_file is not a valid PONI file.
        :returns: Absolute path to the poni file.
        :rtype: str
        """
        poni_file = os.path.abspath(poni_file)
        try:
            ai = azimuthal_integrator(poni_file)
        except Exception as exc:
            raise ValueError(f'{poni_file} is not a valid PONI file') from exc
        return poni_file

    @validator('mask_file', allow_reuse=True)
    def validate_mask_file(cls, mask_file, values):
        """Validate the mask file. If a mask file is provided, it
        checks if it's a valid TIFF file.

        :param mask_file: Path to the mask file.
        :type mask_file: str or None
        :param values: A dictionary of the Detector fields.
        :type values: dict
        :raises ValueError: If mask_file is provided and it's not a
            valid TIFF file.
        :raises ValueError: If `'poni_file'` is not provided in `values`.
        :returns: Absolute path to the mask file or None.
        :rtype: str or None
        """
        if mask_file is None:
            return mask_file

        mask_file = os.path.abspath(mask_file)
        poni_file = values.get('poni_file')
        if poni_file is None:
            raise ValueError(
                'Cannot validate mask file without a PONI file.')
        try:
            mask_array = get_mask_array(mask_file, poni_file)
        except BaseException as exc:
            raise ValueError(
                f'Unable to open {mask_file} as a TIFF file') from exc
        return mask_file

    @property
    def azimuthal_integrator(self):
        """Return the azimuthal integrator associated with this
        detector.
        """
        return azimuthal_integrator(self.poni_file)

    @property
    def mask_array(self):
        """Return the mask array assocated with this detector."""
        return get_mask_array(self.mask_file, self.poni_file)


@cache
def azimuthal_integrator(poni_file:str):
    """Return the azimuthal integrator from a PONI file

    :param poni_file: path to a PONI file
    :type poni_file: str
    :return: azimuthal integrator
    :rtype: pyFAI.azimuthal_integrator.AzimuthalIntegrator
    """
    if not isinstance(poni_file, str):
        poni_file = str(poni_file)
    return pyfai_load(poni_file)


@cache
def get_mask_array(mask_file:str, poni_file:str):
    """Return a mask array associated with a detector loaded from a
    tiff file.

    :param mask_file: path to a .tiff file
    :type mask_file: str
    :param poni_file: path to a PONI file
    :type poni_file: str
    :return: the mask array loaded from `mask_file`
    :rtype: numpy.ndarray
    """
    if mask_file is not None:
        # Third party modules
        from pyspec.file.tiff import TiffFile

        if not isinstance(mask_file, str):
            mask_file = str(mask_file)

        with TiffFile(mask_file) as tiff:
            mask_array = tiff.asarray()
    else:
        mask_array = np.zeros(azimuthal_integrator(poni_file).detector.shape)
    return mask_array


class IntegrationConfig(BaseModel):
    """Class representing the configuration for a raw detector data
    integration.

    :ivar tool_type: type of integration tool; always set to
        "integration"
    :type tool_type: str, optional
    :ivar title: title of the integration
    :type title: str
    :ivar integration_type: type of integration, one of "azimuthal",
        "radial", or "cake"
    :type integration_type: str
    :ivar detectors: list of detectors used in the integration
    :type detectors: List[Detector]
    :ivar radial_units: radial units for the integration, defaults to
        `'q_A^-1'`
    :type radial_units: str, optional
    :ivar radial_min: minimum radial value for the integration range
    :type radial_min: float, optional
    :ivar radial_max: maximum radial value for the integration range
    :type radial_max: float, optional
    :ivar radial_npt: number of points in the radial range for the
        integration
    :type radial_npt: int, optional
    :ivar azimuthal_units: azimuthal units for the integration
    :type azimuthal_units: str, optional
    :ivar azimuthal_min: minimum azimuthal value for the integration
        range
    :type azimuthal_min: float, optional
    :ivar azimuthal_max: maximum azimuthal value for the integration
        range
    :type azimuthal_max: float, optional
    :ivar azimuthal_npt: number of points in the azimuthal range for
        the integration
    :type azimuthal_npt: int, optional
    :ivar error_model: error model for the integration, one of
        "poisson" or "azimuthal"
    :type error_model: str, optional
    """
    tool_type: Literal['integration'] = 'integration'
    title: constr(strip_whitespace=True, min_length=1)
    integration_type: Literal['azimuthal', 'radial', 'cake']
    detectors: conlist(item_type=Detector, min_items=1)
    radial_units: str = 'q_A^-1'
    radial_min: confloat(ge=0)
    radial_max: confloat(gt=0)
    radial_npt: conint(gt=0) = 1800
    azimuthal_units: str = 'chi_deg'
    azimuthal_min: confloat(ge=-180) = -180
    azimuthal_max: confloat(le=360) = 180
    azimuthal_npt: conint(gt=0) = 3600
    error_model: Optional[Literal['poisson', 'azimuthal']]
    sequence_index: Optional[conint(gt=0)]

    @validator('radial_units', allow_reuse=True)
    def validate_radial_units(cls, radial_units):
        """Validate the radial units for the integration.

        :param radial_units: unvalidated radial units for the
            integration
        :type radial_units: str
        :raises ValueError: if radial units are not one of the
            recognized radial units
        :return: validated radial units
        :rtype: str
        """
        if radial_units in RADIAL_UNITS.keys():
            return radial_units
        raise ValueError(
            f'Invalid radial units: {radial_units}. '
            f'Must be one of {", ".join(RADIAL_UNITS.keys())}')

    @validator('azimuthal_units', allow_reuse=True)
    def validate_azimuthal_units(cls, azimuthal_units):
        """Validate that `azimuthal_units` is one of the keys in the
        `pyFAI.units.AZIMUTHAL_UNITS` dictionary.

        :param azimuthal_units: The string representing the unit to be
            validated.
        :type azimuthal_units: str
        :raises ValueError: If `azimuthal_units` is not one of the
            keys in `pyFAI.units.AZIMUTHAL_UNITS`
        :return: The original supplied value, if is one of the keys in
            `pyFAI.units.AZIMUTHAL_UNITS`.
        :rtype: str
        """
        if azimuthal_units in AZIMUTHAL_UNITS.keys():
            return azimuthal_units
        raise ValueError(
            f'Invalid azimuthal units: {azimuthal_units}. '
            f'Must be one of {", ".join(AZIMUTHAL_UNITS.keys())}')

    def validate_range_max(range_name:str):
        """Validate the maximum value of an integration range.

        :param range_name: The name of the integration range
            (e.g. radial, azimuthal).
        :type range_name: str
        :return: The callable that performs the validation.
        :rtype: callable
        """
        def _validate_range_max(cls, range_max, values):
            """Check if the maximum value of the integration range is
            greater than its minimum value.

            :param range_max: The maximum value of the integration
                range.
            :type range_max: float
            :param values: The values of the other fields being
                validated.
            :type values: dict
            :raises ValueError: If the maximum value of the
                integration range is not greater than its minimum
                value.
            :return: The validated maximum range value
            :rtype: float
            """
            range_min = values.get(f'{range_name}_min')
            if range_min < range_max:
                return range_max
            raise ValueError(
                'Maximum value of integration range must be '
                'greater than minimum value of integration range '
                f'({range_name}_min={range_min}).')
        return _validate_range_max

    _validate_radial_max = validator(
        'radial_max',
        allow_reuse=True)(validate_range_max('radial'))
    _validate_azimuthal_max = validator(
        'azimuthal_max',
        allow_reuse=True)(validate_range_max('azimuthal'))

    def validate_for_map_config(self, map_config:BaseModel):
        """Validate the existence of the detector data file for all
        scan points in `map_config`.

        :param map_config: The `MapConfig` instance to validate
            against.
        :type map_config: MapConfig
        :raises RuntimeError: If a detector data file could not be
            found for a scan point occurring in `map_config`.
        :return: None
        :rtype: None
        """
        for detector in self.detectors:
            for scans in map_config.spec_scans:
                for scan_number in scans.scan_numbers:
                    scanparser = scans.get_scanparser(scan_number)
                    for scan_step_index in range(scanparser.spec_scan_npts):
                        # Make sure the detector data file exists for
                        # all scan points
                        try:
                            detector_data_file = \
                                scanparser.get_detector_data_file(
                                    detector.prefix, scan_step_index)
                        except Exception as exc:
                            raise RuntimeError(
                                'Could not find data file for detector prefix '
                                f'{detector.prefix} '
                                f'on scan number {scan_number} '
                                f'in spec file {scans.spec_file}') from exc

    def get_azimuthal_adjustments(self):
        """To enable a continuous range of integration in the
        azimuthal direction for radial and cake integration, obtain
        adjusted values for this `IntegrationConfig`'s `azimuthal_min`
        and `azimuthal_max` values, the angle amount by which those
        values were adjusted, and the proper location of the
        discontinuity in the azimuthal direction.

        :return: Adjusted chi_min, adjusted chi_max, chi_offset,
            chi_discontinuity
        :rtype: tuple[float,float,float,float]
        """
        return get_azimuthal_adjustments(self.azimuthal_min,
                                         self.azimuthal_max)

    def get_azimuthal_integrators(self):
        """Get a list of `AzimuthalIntegrator`s that correspond to the
        detector configurations in this instance of
        `IntegrationConfig`.

        The returned `AzimuthalIntegrator`s are (if need be)
        artificially rotated in the azimuthal direction to achieve a
        continuous range of integration in the azimuthal direction.

        :returns: A list of `AzimuthalIntegrator`s appropriate for use
            by this `IntegrationConfig` tool
        :rtype: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
        """
        chi_offset = self.get_azimuthal_adjustments()[2]
        return get_azimuthal_integrators(
            tuple([detector.poni_file for detector in self.detectors]),
            chi_offset=chi_offset)

    def get_multi_geometry_integrator(self):
        """Get a `MultiGeometry` integrator suitable for use by this
        instance of `IntegrationConfig`.

        :return: A `MultiGeometry` integrator
        :rtype: pyFAI.multi_geometry.MultiGeometry
        """
        poni_files = tuple([detector.poni_file for detector in self.detectors])
        radial_range = (self.radial_min, self.radial_max)
        azimuthal_range = (self.azimuthal_min, self.azimuthal_max)
        return get_multi_geometry_integrator(poni_files, self.radial_units,
                                             radial_range, azimuthal_range)

    def get_azimuthally_integrated_data(self,
                                        spec_scans:BaseModel,
                                        scan_number:int,
                                        scan_step_index:int):
        """Return azimuthally-integrated data for the scan step
        specified.

        :param spec_scans: An instance of `SpecScans` containing the
            scan step requested.
        :type spec_scans: SpecScans
        :param scan_number: The number of the scan containing the scan
            step requested.
        :type scan_number: int
        :param scan_step_index: The index of the scan step requested.
        :type scan_step_index: int
        :return: A 1D array of azimuthally-integrated raw detector
            intensities.
        :rtype: np.ndarray
        """
        detector_data = spec_scans.get_detector_data(self.detectors,
                                                     scan_number,
                                                     scan_step_index)
        integrator = self.get_multi_geometry_integrator()
        lst_mask = [detector.mask_array for detector in self.detectors]
        result = integrator.integrate1d(detector_data,
                                        lst_mask=lst_mask,
                                        npt=self.radial_npt,
                                        error_model=self.error_model)
        if result.sigma is None:
            return result.intensity
        return result.intensity, result.sigma

    def get_radially_integrated_data(self,
                                     spec_scans:BaseModel,
                                     scan_number:int,
                                     scan_step_index:int):
        """Return radially-integrated data for the scan step
        specified.

        :param spec_scans: An instance of `SpecScans` containing the
            scan step requested.
        :type spec_scans: SpecScans
        :param scan_number: The number of the scan containing the scan
            step requested.
        :type scan_number: int
        :param scan_step_index: The index of the scan step requested.
        :type scan_step_index: int
        :return: A 1D array of radially-integrated raw detector
            intensities.
        :rtype: np.ndarray
        """
        # Handle idiosyncracies of azimuthal ranges in pyFAI Adjust
        # chi ranges to get a continuous range of iintegrated data
        chi_min, chi_max, *adjust = self.get_azimuthal_adjustments()
        # Perform radial integration on a detector-by-detector basis.
        intensity_each_detector = []
        variance_each_detector = []
        integrators = self.get_azimuthal_integrators()
        for integrator, detector in zip(integrators, self.detectors):
            detector_data = spec_scans.get_detector_data(
                [detector], scan_number, scan_step_index)[0]
            result = integrator.integrate_radial(
                detector_data,
                self.azimuthal_npt,
                unit=self.azimuthal_units,
                azimuth_range=(chi_min, chi_max),
                radial_unit=self.radial_units,
                radial_range=(self.radial_min, self.radial_max),
                mask=detector.mask_array)  # , error_model=self.error_model)
            intensity_each_detector.append(result.intensity)
            if result.sigma is not None:
                variance_each_detector.append(result.sigma**2)
        # Add the individual detectors' integrated intensities
        # together
        intensity = np.nansum(intensity_each_detector, axis=0)
        # Ignore data at values of chi for which there was no data
        intensity = np.where(intensity == 0, np.nan, intensity)
        if len(intensity_each_detector) != len(variance_each_detector):
            return intensity

        # Get the standard deviation of the summed detectors'
        # intensities
        sigma = np.sqrt(np.nansum(variance_each_detector, axis=0))
        return intensity, sigma

    def get_cake_integrated_data(self,
                                 spec_scans:BaseModel,
                                 scan_number:int,
                                 scan_step_index:int):
        """Return cake-integrated data for the scan step specified.

        :param spec_scans: An instance of `SpecScans` containing the
            scan step requested.
        :type spec_scans: SpecScans
        :param scan_number: The number of the scan containing the scan
            step requested.
        :type scan_number: int
        :param scan_step_index: The index of the scan step requested.
        :type scan_step_index: int
        :return: A 2D array of cake-integrated raw detector
            intensities.
        :rtype: np.ndarray
        """
        detector_data = spec_scans.get_detector_data(
            self.detectors, scan_number, scan_step_index)
        integrator = self.get_multi_geometry_integrator()
        lst_mask = [detector.mask_array for detector in self.detectors]
        result = integrator.integrate2d(
            detector_data,
            lst_mask=lst_mask,
            npt_rad=self.radial_npt,
            npt_azim=self.azimuthal_npt,
            method='bbox',
            error_model=self.error_model)
        if result.sigma is None:
            return result.intensity
        return result.intensity, result.sigma

    def get_integrated_data(self,
                            spec_scans:BaseModel,
                            scan_number:int,
                            scan_step_index:int):
        """Return integrated data for the scan step specified.

        :param spec_scans: An instance of `SpecScans` containing the
            scan step requested.
        :type spec_scans: SpecScans
        :param scan_number: The number of the scan containing the scan
            step requested.
        :type scan_number: int
        :param scan_step_index: The index of the scan step requested.
        :type scan_step_index: int
        :return: An array of integrated raw detector intensities.
        :rtype: np.ndarray
        """
        if self.integration_type == 'azimuthal':
            return self.get_azimuthally_integrated_data(spec_scans,
                                                        scan_number,
                                                        scan_step_index)
        if self.integration_type == 'radial':
            return self.get_radially_integrated_data(spec_scans,
                                                     scan_number,
                                                     scan_step_index)
        if self.integration_type == 'cake':
            return self.get_cake_integrated_data(spec_scans,
                                                 scan_number,
                                                 scan_step_index)
        return None

    @property
    def integrated_data_coordinates(self):
        """Return a dictionary of coordinate arrays for navigating the
        dimension(s) of the integrated data produced by this instance
        of `IntegrationConfig`.

        :return: A dictionary with either one or two keys: 'azimuthal'
            and/or 'radial', each of which points to a 1-D `numpy`
            array of coordinate values.
        :rtype: dict[str,np.ndarray]
        """
        if self.integration_type == 'azimuthal':
            return get_integrated_data_coordinates(
                radial_range=(self.radial_min, self.radial_max),
                radial_npt=self.radial_npt)
        if self.integration_type == 'radial':
            return get_integrated_data_coordinates(
                azimuthal_range=(self.azimuthal_min, self.azimuthal_max),
                azimuthal_npt=self.azimuthal_npt)
        if self.integration_type == 'cake':
            return get_integrated_data_coordinates(
                radial_range=(self.radial_min, self.radial_max),
                radial_npt=self.radial_npt,
                azimuthal_range=(self.azimuthal_min, self.azimuthal_max),
                azimuthal_npt=self.azimuthal_npt)
        return None

    @property
    def integrated_data_dims(self):
        """Return a tuple of the coordinate labels for the integrated
        data produced by this instance of `IntegrationConfig`.
        """
        directions = list(self.integrated_data_coordinates.keys())
        dim_names = [getattr(self, f'{direction}_units')
                     for direction in directions]
        return dim_names

    @property
    def integrated_data_shape(self):
        """Return a tuple representing the shape of the integrated
        data produced by this instance of `IntegrationConfig` for a
        single scan step.
        """
        return tuple([len(coordinate_values)
                      for coordinate_name, coordinate_values
                      in self.integrated_data_coordinates.items()])


@cache
def get_azimuthal_adjustments(chi_min:float, chi_max:float):
    """Fix chi discontinuity at 180 degrees and return the adjusted
    chi range, offset, and discontinuty.

    If the discontinuity is crossed, obtain the offset to artificially
    rotate detectors to achieve a continuous azimuthal integration
    range.

    :param chi_min: The minimum value of the azimuthal range.
    :type chi_min: float
    :param chi_max: The maximum value of the azimuthal range.
    :type chi_max: float
    :return: The following four values: the adjusted minimum value of
        the azimuthal range, the adjusted maximum value of the
        azimuthal range, the value by which the chi angle was
        adjusted, the position of the chi discontinuity.
    """
    # Fix chi discontinuity at 180 degrees for now.
    chi_disc = 180
    # If the discontinuity is crossed, artificially rotate the
    # detectors to achieve a continuous azimuthal integration range
    if chi_min < chi_disc and chi_max > chi_disc:
        chi_offset = chi_max - chi_disc
    else:
        chi_offset = 0
    return chi_min-chi_offset, chi_max-chi_offset, chi_offset, chi_disc


@cache
def get_azimuthal_integrators(poni_files:tuple, chi_offset=0):
    """Return a list of `AzimuthalIntegrator` objects generated from
    PONI files.

    :param poni_files: Tuple of strings, each string being a path to a
        PONI file.
    :type poni_files: tuple
    :param chi_offset: The angle in degrees by which the
        `AzimuthalIntegrator` objects will be rotated, defaults to 0.
    :type chi_offset: float, optional
    :return: List of `AzimuthalIntegrator` objects
    :rtype: list[pyFAI.azimuthalIntegrator.AzimuthalIntegrator]
    """
    ais = []
    for poni_file in poni_files:
        ai = copy.deepcopy(azimuthal_integrator(poni_file))
        ai.rot3 += chi_offset * np.pi/180
        ais.append(ai)
    return ais


@cache
def get_multi_geometry_integrator(poni_files:tuple, radial_unit:str,
                                  radial_range:tuple, azimuthal_range:tuple):
    """Return a `MultiGeometry` instance that can be used for
    azimuthal or cake integration.

    :param poni_files: Tuple of PONI files that describe the detectors
        to be integrated.
    :type poni_files: tuple
    :param radial_unit: Unit to use for radial integration range.
    :type radial_unit: str
    :param radial_range: Tuple describing the range for radial integration.
    :type radial_range: tuple[float,float]
    :param azimuthal_range:Tuple describing the range for azimuthal
        integration.
    :type azimuthal_range: tuple[float,float]
    :return: `MultiGeometry` instance that can be used for azimuthal
        or cake integration.
    :rtype: pyFAI.multi_geometry.MultiGeometry
    """
    chi_min, chi_max, chi_offset, chi_disc = \
        get_azimuthal_adjustments(*azimuthal_range)
    ais = copy.deepcopy(get_azimuthal_integrators(poni_files,
                                                  chi_offset=chi_offset))
    multi_geometry = MultiGeometry(
        ais,
        unit=radial_unit,
        radial_range=radial_range,
        azimuth_range=(chi_min, chi_max),
        wavelength=sum([ai.wavelength for ai in ais])/len(ais),
        chi_disc=chi_disc)
    return multi_geometry


@cache
def get_integrated_data_coordinates(azimuthal_range:tuple = None,
                                    azimuthal_npt:int = None,
                                    radial_range:tuple = None,
                                    radial_npt:int = None):
    """Return a dictionary of coordinate arrays for the specified
    radial and/or azimuthal integration ranges.

    :param azimuthal_range: Tuple specifying the range of azimuthal
        angles over which to generate coordinates, in the format (min,
        max), defaults to None.
    :type azimuthal_range: tuple[float,float], optional
    :param azimuthal_npt: Number of azimuthal coordinate points to
        generate, defaults to None.
    :type azimuthal_npt: int, optional
    :param radial_range: Tuple specifying the range of radial
        distances over which to generate coordinates, in the format
        (min, max), defaults to None.
    :type radial_range: tuple[float,float], optional
    :param radial_npt: Number of radial coordinate points to generate,
        defaults to None.
    :type radial_npt: int, optional
    :return: A dictionary with either one or two keys: 'azimuthal'
        and/or 'radial', each of which points to a 1-D `numpy` array
        of coordinate values.
    :rtype: dict[str,np.ndarray]
    """
    integrated_data_coordinates = {}
    if azimuthal_range is not None and azimuthal_npt is not None:
        integrated_data_coordinates['azimuthal'] = np.linspace(
            *azimuthal_range, azimuthal_npt)
    if radial_range is not None and radial_npt is not None:
        integrated_data_coordinates['radial'] = np.linspace(
            *radial_range, radial_npt)
    return integrated_data_coordinates
