# System modules
import os
from pathlib import PosixPath
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
from hexrd.material import Material
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    StrictBool,
    confloat,
    conint,
    conlist,
    constr,
    root_validator,
    validator,
)
from scipy.interpolate import interp1d

# Local modules
from CHAP.common.models.map import MapConfig
from CHAP.utils.parfile import ParFile
from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser


# Material configuration classes

class MaterialConfig(BaseModel):
    """Model for parameters to characterize a sample material.

    :ivar material_name: Sample material name.
    :type material_name: str, optional
    :ivar lattice_parameters: Lattice spacing(s) in angstroms.
    :type lattice_parameters: float, list[float], optional
    :ivar sgnum: Space group of the material.
    :type sgnum: int, optional
    """
    material_name: Optional[constr(strip_whitespace=True, min_length=1)]
    lattice_parameters: Optional[Union[
        confloat(gt=0),
        conlist(item_type=confloat(gt=0), min_items=1, max_items=6)]]
    sgnum: Optional[conint(ge=0)]

    _material: Optional[Material]

    class Config:
        underscore_attrs_are_private = False

    @root_validator
    def validate_material(cls, values):
        """Create and validate the private attribute _material.

        :param values: Dictionary of previously validated field values.
        :type values: dict
        :return: The validated list of `values`.
        :rtype: dict
        """
        # Local modules
        from CHAP.edd.utils import make_material

        values['_material'] = make_material(values.get('material_name'),
                                            values.get('sgnum'),
                                            values.get('lattice_parameters'))
        return values

    def unique_hkls_ds(self, tth_tol=0.15, tth_max=90.0):
        """Get a list of unique HKLs and their lattice spacings.

        :param tth_tol: Minimum resolvable difference in 2&theta
            between two unique HKL peaks, defaults to `0.15`.
        :type tth_tol: float, optional
        :param tth_max: Detector rotation about hutch x axis,
            defaults to `90.0`.
        :type tth_max: float, optional
        :return: Unique HKLs and their lattice spacings in angstroms.
        :rtype: np.ndarray, np.ndarray
        """
        # Local modules
        from CHAP.edd.utils import get_unique_hkls_ds

        return get_unique_hkls_ds([self._material])

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_material' in d:
            del d['_material']
        return d


class CeriaConfig(MaterialConfig):
    """Model for the sample material used in calibrations.

    :ivar material_name: Calibration material name,
        defaults to `'CeO2'`.
    :type material_name: str, optional
    :ivar lattice_parameters: Lattice spacing(s) for the calibration
        material in angstroms, defaults to `5.41153`.
    :type lattice_parameters: float, list[float], optional
    :ivar sgnum: Space group of the calibration material,
        defaults to `225`.
    :type sgnum: int, optional
    """
    #RV Name suggests it's always Ceria, why have material_name?
    material_name: constr(strip_whitespace=True, min_length=1) = 'CeO2'
    lattice_parameters: confloat(gt=0) = 5.41153
    sgnum: Optional[conint(ge=0)] = 225


# Detector configuration classes

class MCAElementConfig(BaseModel):
    """Class representing metadata required to configure a single MCA
    detector element.

    :ivar detector_name: Name of the MCA detector element in the scan,
        defaults to `'mca1'`.
    :type detector_name: str
    :ivar num_bins: Number of MCA channels.
    :type num_bins: int, optional
    """
    detector_name: constr(strip_whitespace=True, min_length=1) = 'mca1'
    num_bins: Optional[conint(gt=0)]

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        return d


class MCAElementCalibrationConfig(MCAElementConfig):
    """Class representing metadata required to calibrate a single MCA
    detector element.

    :ivar tth_max: Detector rotation about lab frame x axis,
       defaults to `90`.
    :type tth_max: float, optional
    :ivar hkl_tth_tol: Minimum resolvable difference in 2&theta between
        two unique HKL peaks, defaults to `0.15`.
    :type hkl_tth_tol: float, optional
    :ivar hkl_indices: List of unique HKL indices to fit peaks for in
        the calibration routine, defaults to `[]`.
    :type hkl_indices: list[int], optional
    :ivar background: Background model for peak fitting.
    :type background: str, list[str], optional
    :ivar energy_calibration_coeffs: Detector channel index to energy
        polynomial conversion coefficients ([a, b, c] with
        E_i = a*i^2 + b*i + c), defaults to `[0, 0, 1]`.
    :type energy_calibration_coeffs:
        list[float, float, float], optional
    :ivar tth_initial_guess: Initial guess for 2&theta,
        defaults to `5.0`.
    :type tth_initial_guess: float, optional
    :ivar tth_calibrated: Calibrated value for 2&theta.
    :type tth_calibrated: float, optional
    :ivar include_energy_ranges: List of MCA channel energy ranges
        in keV whose data should be included after applying a mask
        (bounds are inclusive), defaults to `[[50, 150]]`
    :type include_energy_ranges: list[[float, float]], optional
    """
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    hkl_indices: Optional[conlist(item_type=conint(ge=0))] = []
    background: Optional[Union[str, list]]
    tth_initial_guess: confloat(gt=0, le=tth_max, allow_inf_nan=False) = 5.0
    energy_calibration_coeffs: conlist(
        min_items=3, max_items=3,
        item_type=confloat(allow_inf_nan=False)) = [0, 0, 1]
    intercept_initial_guess: Optional[confloat(allow_inf_nan=False)]
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)]
    include_energy_ranges: conlist(
        min_items=1,
        item_type=conlist(
            item_type=confloat(ge=25),
            min_items=2,
            max_items=2)) = [[50, 150]]

    @validator('include_energy_ranges', each_item=True)
    def validate_include_energy_range(cls, value, values):
        """Ensure that no energy ranges are outside the boundary of the
        detector.

        :param value: Field value to validate (`include_energy_ranges`).
        :type values: dict
        :param values: Dictionary of previously validated field values.
        :type values: dict
        :return: The validated value of `include_energy_ranges`.
        :rtype: dict
        """
        value.sort()
        n_max = values.get('num_bins')
        if n_max is not None:
            n_max -= 1
            a, b, c = values.get('energy_calibration_coeffs')
            e_max = (a*n_max + b)*n_max +c
            if value[0] < c or value[1] > e_max:
                newvalue = [float(max(value[0], c)),
                        float(min(value[1], e_max))]
                print(
                    f'WARNING: include_energy_range out of range'
                    f' ({value}): adjusted to {newvalue}')
                value = newvalue
        return value

    @validator('hkl_indices', pre=True)
    def validate_hkl_indices(cls, hkl_indices):
        if isinstance(hkl_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            hkl_indices = string_to_list(hkl_indices)
        return sorted(hkl_indices)

    @property
    def energies(self):
        """Return calibrated bin energies."""
        a, b, c = self.energy_calibration_coeffs
        channels = np.arange(self.num_bins)
        return (a*channels + b)*channels + c

    @property
    def include_bin_ranges(self):
        """Return the value of `include_energy_ranges` represented in
        terms of channel indices instead of channel energies.
        """
        from CHAP.utils.general import (
            index_nearest_down,
            index_nearest_up,
        )

        include_bin_ranges = []
        energies = self.energies
        for e_min, e_max in self.include_energy_ranges:
            include_bin_ranges.append(
                [index_nearest_down(energies, e_min),
                 index_nearest_up(energies, e_max)])
        return include_bin_ranges

    def get_include_energy_ranges(self, include_bin_ranges):
        """Given a list of channel index ranges, return the
        corresponding list of channel energy ranges.

        :param include_bin_ranges: A list of channel bin ranges to convert to
            energy ranges.
        :type include_bin_ranges: list[list[int]]
        :returns: Energy ranges
        :rtype: list[list[float]]
        """
        energies = self.energies
        return [[float(energies[i]) for i in range_]
                 for range_ in include_bin_ranges]

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.
        Note that the bounds of self.include_energy_ranges are inclusive.

        :return: Boolean mask array.
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for min_, max_ in self.include_bin_ranges:
            mask = np.logical_or(
                mask, np.logical_and(bin_indices >= min_, bin_indices <= max_))
        return mask

#RV need def dict?
#        d['include_energy_ranges'] = [
#            [float(energy) for energy in d['include_energy_ranges'][i]]
#            for i in range(len(d['include_energy_ranges']))]


class MCAElementDiffractionVolumeLengthConfig(MCAElementConfig):
    """Class representing metadata required to perform a diffraction
    volume length measurement for a single MCA detector element.

    :ivar measurement_mode: Placeholder for recording whether the
        measured DVL value was obtained through the automated
        calculation or a manual selection, defaults to `'auto'`.
    :type measurement_mode: Literal['manual', 'auto'], optional
    :ivar sigma_to_dvl_factor: The DVL is obtained by fitting a reduced
        form of the MCA detector data. `sigma_to_dvl_factor` is a
        scalar value that converts the standard deviation of the
        gaussian fit to the measured DVL, defaults to `3.5`.
    :type sigma_to_dvl_factor: Literal[3.5, 2.0, 4.0], optional
    :ivar dvl_measured: Placeholder for the measured diffraction
        volume length before writing the data to file.
    :type dvl_measured: float, optional
    :ivar fit_amplitude: Placeholder for amplitude of the gaussian fit.
    :type fit_amplitude: float, optional
    include_energy_ranges: conlist(
        min_items=1,
        item_type=conlist(
            item_type=confloat(ge=25),
            min_items=2,
            max_items=2)) = [[50, 150]]
    :ivar fit_center: Placeholder for center of the gaussian fit.
    :type fit_center: float, optional
    :ivar fit_sigma: Placeholder for sigma of the gaussian fit.
    :type fit_sigma: float, optional
    """
    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
    sigma_to_dvl_factor: Optional[Literal[3.5, 2.0, 4.0]] = 3.5
    dvl_measured: Optional[confloat(gt=0)] = None
    fit_amplitude: Optional[float] = None
    fit_center: Optional[float] = None
    fit_sigma: Optional[float] = None
    #RV FIX does this rely on include_energy_ranges

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.
        Exclude `sigma_to_dvl_factor` from the dict representation if
        `measurement_mode` is `'manual'`.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        if self.measurement_mode == 'manual':
            del d['sigma_to_dvl_factor']
        for param in ('amplitude', 'center', 'sigma'):
            d[f'fit_{param}'] = float(d[f'fit_{param}'])
        return d


class MCAElementStrainAnalysisConfig(MCAElementConfig):
    """Class representing metadata required to perform a strain
    analysis fitting for a single MCA detector element.

    :param tth_max: Detector rotation about hutch x axis, defaults
            to `90.0`.
    :type tth_max: float, optional
    :ivar hkl_tth_tol: Minimum resolvable difference in 2&theta between
        two unique HKL peaks, defaults to `0.15`.
    :type hkl_tth_tol: float, optional
    :ivar hkl_indices: List of unique HKL indices to fit peaks for in
        the calibration routine, defaults to `[]`.
    :type hkl_indices: list[int], optional
    :ivar background: Background model for peak fitting.
    :type background: str, list[str], optional
    :ivar num_proc: Number of processors used for peak fitting.
    :type num_proc: int, optional
    :ivar peak_models: Peak model for peak fitting,
        defaults to `'gaussian'`.
    :type peak_models: Literal['gaussian', 'lorentzian']],
        list[Literal['gaussian', 'lorentzian']]], optional
    :ivar fwhm_min: Minimum FWHM for peak fitting, defaults to `1.0`.
    :type fwhm_min: float, optional
    :ivar fwhm_max: Maximum FWHM for peak fitting, defaults to `5.0`.
    :type fwhm_max: float, optional
    :ivar rel_amplitude_cutoff: Relative peak amplitude cutoff for
        peak fitting (any peak with an amplitude smaller than
        `rel_amplitude_cutoff` times the sum of all peak amplitudes
        gets removed from the fit model), defaults to `None`.
    :type rel_amplitude_cutoff: float, optional
    :ivar tth_calibrated: Calibrated value for 2&theta.
    :type tth_calibrated: float, optional
    :ivar energy_calibration_coeffs: Detector channel index to energy
        polynomial conversion coefficients ([a, b, c] with
        E_i = a*i^2 + b*i + c), defaults to `[0, 0, 1]`.
    :type energy_calibration_coeffs:
        list[float, float, float], optional
    :ivar calibration_bin_ranges: List of MCA channel index ranges
        whose data was included in the calibration.
    :type calibration_bin_ranges: list[[int, int]], optional
    :ivar tth_file: Path to the file with the 2&theta map.
    :type tth_file: FilePath, optional
    :ivar tth_map: Map of the 2&theta values.
    :type tth_map: np.ndarray, optional
    :ivar include_energy_ranges: List of MCA channel energy ranges
        in keV whose data should be included after applying a mask
        (bounds are inclusive), defaults to `[[50, 150]]`
    :type include_energy_ranges: list[[float, float]], optional
    """
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    hkl_indices: Optional[conlist(item_type=conint(ge=0))] = []
    background: Optional[Union[str, list]]
    num_proc: Optional[conint(gt=0)] = os.cpu_count()
    peak_models: Union[
        conlist(item_type=Literal['gaussian', 'lorentzian'], min_items=1),
        Literal['gaussian', 'lorentzian']] = 'gaussian'
    fwhm_min: confloat(gt=0, allow_inf_nan=False) = 0.25
    fwhm_max: confloat(gt=0, allow_inf_nan=False) = 2.0
    rel_amplitude_cutoff: Optional[confloat(gt=0, lt=1.0, allow_inf_nan=False)]

    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)]
    energy_calibration_coeffs: conlist(
        min_items=3, max_items=3,
        item_type=confloat(allow_inf_nan=False)) = [0, 0, 1]
    calibration_bin_ranges: Optional[
        conlist(
            min_items=1,
            item_type=conlist(
                item_type=conint(ge=0),
                min_items=2,
                max_items=2))]
    tth_file: Optional[FilePath]
    tth_map: Optional[np.ndarray] = None
    include_energy_ranges: conlist( 
        min_items=1,
        item_type=conlist(
            item_type=confloat(ge=25),
            min_items=2,
            max_items=2)) = [[50, 150]]

    #RV lots of overlap with MCAElementCalibrationConfig (only missing
    #   tth_initial_guess)
    #   Should we derive from MCAElementCalibrationConfig in some way
    #   or make a MCAElementEnergyCalibrationConfig with what's shared
    #   and derive MCAElementCalibrationConfig from this as well with 
    #   the unique fields tth_initial_guess added?
    #   Revisit when we redo the detectors

    @validator('hkl_indices', pre=True)
    def validate_hkl_indices(cls, hkl_indices):
        if isinstance(hkl_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            hkl_indices = string_to_list(hkl_indices)
        return sorted(hkl_indices)

    class Config:
        arbitrary_types_allowed = True

    @property
    def energies(self):
        """Return calibrated bin energies."""
        a, b, c = self.energy_calibration_coeffs
        channels = np.arange(self.num_bins)
        return (a*channels + b)*channels + c

    @property
    def include_bin_ranges(self):
        """Return the value of `include_energy_ranges` represented in
        terms of channel indices instead of channel energies.
        """
        from CHAP.utils.general import (
            index_nearest_down,
            index_nearest_up,
        )

        include_bin_ranges = []
        energies = self.energies
        for e_min, e_max in self.include_energy_ranges:
            include_bin_ranges.append(
                [index_nearest_down(energies, e_min),
                 index_nearest_up(energies, e_max)])
        return include_bin_ranges

    def get_include_energy_ranges(self, include_bin_ranges):
        """Given a list of channel index ranges, return the
        corresponding list of channel energy ranges.

        :param include_bin_ranges: A list of channel bin ranges to convert to
            energy ranges.
        :type include_bin_ranges: list[list[int]]
        :returns: Energy ranges
        :rtype: list[list[float]]
        """
        energies = self.energies
        return [[float(energies[i]) for i in range_]
                 for range_ in include_bin_ranges]

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.
        Note that the bounds of self.include_energy_ranges are inclusive.

        :return: Boolean mask array.
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for min_, max_ in self.include_bin_ranges:
            mask = np.logical_or(
                mask, np.logical_and(bin_indices >= min_, bin_indices <= max_))
        return mask

    def add_calibration(self, calibration):
        """Finalize values for some fields using a completed
        MCAElementCalibrationConfig that corresponds to the same
        detector.

        :param calibration: Existing calibration configuration to use
            by MCAElementStrainAnalysisConfig.
        :type calibration: MCAElementCalibrationConfig
        :return: None
        """
        add_fields = [
            'tth_calibrated', 'energy_calibration_coeffs', 'num_bins']
        for field in add_fields:
            setattr(self, field, getattr(calibration, field))
        self.calibration_bin_ranges = calibration.include_bin_ranges

    def get_tth_map(self, map_shape):
        """Return the map of 2&theta values to use -- may vary at each
        point in the map.

        :param map_shape: The shape of the suplied 2&theta map.
        :return: Map of 2&theta values.
        :rtype: np.ndarray
        """
        if getattr(self, 'tth_map', None) is not None:
            if self.tth_map.shape != map_shape:
                raise ValueError(
                    'Invalid "tth_map" field shape '
                    f'{self.tth_map.shape} (expected {map_shape})')
            return self.tth_map
        return np.full(map_shape, self.tth_calibrated)

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


# Processor configuration classes

class MCAScanDataConfig(BaseModel):
    """Class representing metadata required to locate raw MCA data for
    a single scan and construct a mask for it.

    :ivar inputdir: Input directory, used only if any file in the
            configuration is not an absolute path.
    :type inputdir: str, optional
    :ivar spec_file: Path to the SPEC file containing the scan.
    :type spec_file: str, optional
    :ivar scan_number: Number of the scan in `spec_file`.
    :type scan_number: int, optional
    :ivar par_file: Path to the par file associated with the scan.
    :type par_file: str, optional
    :ivar scan_column: Required column name in `par_file`.
    :type scan_column: str, optional
    :ivar detectors: List of MCA detector element metadata
        configurations.
    :type detectors: list[MCAElementConfig]
    """
    inputdir: Optional[DirectoryPath]
    spec_file: Optional[FilePath]
    scan_number: Optional[conint(gt=0)]
    par_file: Optional[FilePath]
    scan_column: Optional[str]
    detectors: conlist(min_items=1, item_type=MCAElementConfig)#RV FIX does this rely on include_energy_ranges

    _parfile: Optional[ParFile]
    _scanparser: Optional[ScanParser]

    class Config:
        underscore_attrs_are_private = False

    @root_validator(pre=True)
    def validate_scan(cls, values):
        """Finalize file paths for spec_file and par_file.

        :param values: Dictionary of class field values.
        :type values: dict
        :raises ValueError: Invalid SPEC or par file.
        :return: The validated list of `values`.
        :rtype: dict
        """
        inputdir = values.get('inputdir')
        spec_file = values.get('spec_file')
        par_file = values.get('par_file')
        if spec_file is not None and par_file is not None:
            raise ValueError('Use either spec_file or par_file, not both')
        elif spec_file is not None:
            if inputdir is not None and not os.path.isabs(spec_file):
                values['spec_file'] = os.path.join(inputdir, spec_file)
        elif par_file is not None:
            if inputdir is not None and not os.path.isabs(par_file):
                values['par_file'] = os.path.join(inputdir, par_file)
            if 'scan_column' not in values:
                raise ValueError(
                    'scan_column is required when par_file is used')
            if isinstance(values['scan_column'], str):
                parfile = ParFile(par_file)
                if values['scan_column'] not in parfile.column_names:
                    raise ValueError(
                        f'No column named {values["scan_column"]} in '
                        + '{values["par_file"]}. Options: '
                        + ', '.join(parfile.column_names))
        else:
            raise ValueError('Must use either spec_file or par_file')

        return values

    @root_validator
    def validate_detectors(cls, values):
        """Fill in values for _scanparser / _parfile (if applicable).
        Fill in each detector's num_bins field, if needed.
        Check each detector's include_energy_ranges field against the
        flux file, if available.

        :param values: Dictionary of previously validated field values.
        :type values: dict
        :raises ValueError: Unable to obtain a value for num_bins.
        :return: The validated list of `values`.
        :rtype: dict
        """
        spec_file = values.get('spec_file')
        par_file = values.get('par_file')
        detectors = values.get('detectors')
        flux_file = values.get('flux_file')
        if spec_file is not None:
            values['_scanparser'] = ScanParser(
                spec_file, values.get('scan_number'))
            values['_parfile'] = None
        elif par_file is not None:
            values['_parfile'] = ParFile(par_file)
            values['_scanparser'] = ScanParser(
                values['_parfile'].spec_file,
                values['_parfile'].good_scan_numbers()[0])
        for detector in detectors:
            if detector.num_bins is None:
                try:
                    detector.num_bins = values['_scanparser']\
                        .get_detector_num_bins(detector.detector_name)
                except Exception as e:
                    raise ValueError('No value found for num_bins') from e
        if flux_file is not None:
            # System modules
            from copy import deepcopy

            flux = np.loadtxt(flux_file)
            flux_file_energies = flux[:,0]/1.e3
            flux_e_min = flux_file_energies.min()
            flux_e_max = flux_file_energies.max()
            for detector in detectors:
                for i, (det_e_min, det_e_max) in enumerate(
                        deepcopy(detector.include_energy_ranges)):
                    if det_e_min < flux_e_min or det_e_max > flux_e_max:
                        energy_range = [float(max(det_e_min, flux_e_min)),
                                        float(min(det_e_max, flux_e_max))]
                        print(
                            f'WARNING: include_energy_ranges[{i}] out of range'
                            f' ({detector.include_energy_ranges[i]}): adjusted'
                            f' to {energy_range}')
                        detector.include_energy_ranges[i] = energy_range

        return values

    @property
    def scanparser(self):
        """Return the scanparser."""
        try:
            scanparser = self._scanparser
        except:
            scanparser = ScanParser(self.spec_file, self.scan_number)
            self._scanparser = scanparser
        return scanparser

    def mca_data(self, detector_config, scan_step_index=None):
        """Get the array of MCA data collected by the scan.

        :param detector_config: Detector for which data is returned.
        :type detector_config: MCAElementConfig
        :param scan_step_index: Only return the MCA spectrum for the
            given scan step index, defaults to `None`, which returns
            all the available MCA spectra.
        :type scan_step_index: int, optional
        :return: The current detectors's MCA data.
        :rtype: np.ndarray
        """
        detector_name = detector_config.detector_name
        if self._parfile is not None:
            if scan_step_index is None:
                data = np.asarray(
                    [ScanParser(self._parfile.spec_file, scan_number)\
                     .get_all_detector_data(detector_name)[0] \
                     for scan_number in self._parfile.good_scan_numbers()])
            else:
                data = ScanParser(
                    self._parfile.spec_file,
                    self._parfile.good_scan_numbers()[scan_step_index])\
                    .get_all_detector_data(detector_name)
        else:
            if scan_step_index is None:
                data = self.scanparser.get_all_detector_data(
                    detector_name)
            else:
                data = self.scanparser.get_detector_data(
                    detector_config.detector_name, scan_step_index)
        return data

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if d.get('_parfile') is None:
            del d['par_file']
            del d['scan_column']
        else:
            del d['spec_file']
            del d['scan_number']                
        for k in ('_scanparser', '_parfile', 'inputdir'):
            if k in d:
                del d[k]
        return d


class DiffractionVolumeLengthConfig(MCAScanDataConfig):
    """Class representing metadata required to perform a diffraction
    volume length calculation for an EDD setup using a steel-foil
    raster scan.

    :ivar sample_thickness: Thickness of scanned foil sample. Quantity
        must be provided in the same units as the values of the
        scanning motor.
    :type sample_thickness: float
    :ivar detectors: Individual detector element DVL
        measurement configurations
    :type detectors: list[MCAElementDiffractionVolumeLengthConfig]
    """
    sample_thickness: float
    detectors: conlist(min_items=1,
                       item_type=MCAElementDiffractionVolumeLengthConfig)

    @property
    def scanned_vals(self):
        """Return the list of values visited by the scanning motor
        over the course of the raster scan.

        :return: List of scanned motor values
        :rtype: np.ndarray
        """
        if self._parfile is not None:
            return self._parfile.get_values(
                self.scan_column,
                scan_numbers=self._parfile.good_scan_numbers())
        return self.scanparser.spec_scan_motor_vals[0]


class MCACeriaCalibrationConfig(MCAScanDataConfig):
    """
    Class representing metadata required to perform a Ceria calibration
    for an MCA detector.

    :ivar scan_step_indices: Optional scan step indices to use for the
        calibration. If not specified, the calibration will be
        performed on the average of all MCA spectra for the scan.
    :type scan_step_indices: list[int], optional
    :ivar flux_file: File name of the csv flux file containing station
        beam energy in eV (column 0) versus flux (column 1).
    :type flux_file: str, optional
    :ivar material: Material configuration for Ceria.
    :type material: CeriaConfig
    :ivar detectors: List of individual MCA detector element
        calibration configurations.
    :type detectors: list[MCAElementCalibrationConfig]
    :ivar max_iter: Maximum number of iterations of the calibration
        routine, defaults to `10`.
    :type detectors: int, optional
    :ivar tune_tth_tol: Cutoff error for tuning 2&theta. Stop iterating
        the calibration routine after an iteration produces a change in
        the tuned value of 2&theta that is smaller than this cutoff,
        defaults to `1e-8`.
    :ivar tune_tth_tol: float, optional
    """
    scan_step_indices: Optional[conlist(min_items=1, item_type=conint(ge=0))]
    material: CeriaConfig = CeriaConfig()
    detectors: conlist(min_items=1, item_type=MCAElementCalibrationConfig)
    flux_file: Optional[FilePath]
    max_iter: conint(gt=0) = 10
    tune_tth_tol: confloat(ge=0) = 1e-8

    @root_validator(pre=True)
    def validate_config(cls, values):
        """Ensure that a valid configuration was provided and finalize
        flux_file filepath.

        :param values: Dictionary of class field values.
        :type values: dict
        :return: The validated list of `values`.
        :rtype: dict
        """
        inputdir = values.get('inputdir')
        if inputdir is not None:
            flux_file = values.get('flux_file')
            if flux_file is not None and not os.path.isabs(flux_file):
                values['flux_file'] = os.path.join(inputdir, flux_file)

        return values

    @validator('scan_step_indices', pre=True, always=True)
    def validate_scan_step_indices(cls, scan_step_indices, values):
        """Validate the specified list of scan numbers.

        :ivar scan_step_indices: Optional scan step indices to use for the
            calibration. If not specified, the calibration will be
            performed on the average of all MCA spectra for the scan.
        :type scan_step_indices: list[int], optional
        :param values: Dictionary of validated class field values.
        :type values: dict
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list of int
        """
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(
                scan_step_indices, raise_error=True)
        return scan_step_indices

    @property
    def flux_file_energy_range(self):
        """Get the energy range in the flux corection file.

        :return: The energy range in the flux corection file.
        :rtype: tuple(float, float)
        """
        if self.flux_file is None:
            return None
        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        return energies.min(), energies.max()

    def mca_data(self, detector_config):
        """Get the array of MCA data to use for calibration.

        :param detector_config: Detector for which data is returned.
        :type detector_config: MCAElementConfig
        :return: The current detectors's MCA data.
        :rtype: np.ndarray
        """
        if self.scan_step_indices is None:
            data = super().mca_data(detector_config)
            if self.scanparser.spec_scan_npts > 1:
                data = np.average(data, axis=0)
            else:
                data = data[0]
        elif len(self.scan_step_indices) == 1:
            data = super().mca_data(
                detector_config, scan_step_index=self.scan_step_indices[0])
        else:
            data = []
            for scan_step_index in self.scan_step_indices:
                data.append(super().mca_data(
                    detector_config, scan_step_index=scan_step_index))
            data = np.average(data, axis=0)
        return data

    def flux_correction_interpolation_function(self):
        """
        Get an interpolation function to correct MCA data for the
        relative energy flux of the incident beam.

        :return: Energy flux correction interpolation function.
        :rtype: scipy.interpolate._polyint._Interpolator1D
        """

        if self.flux_file is None:
            return None
        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        relative_intensities = flux[:,1]/np.max(flux[:,1])
        interpolation_function = interp1d(energies, relative_intensities)
        return interpolation_function


class StrainAnalysisConfig(BaseModel):
    """Class representing input parameters required to perform a
    strain analysis.

    :ivar inputdir: Input directory, used only if any file in the
            configuration is not an absolute path.
    :type inputdir: str, optional
    :ivar map_config: The map configuration for the MCA data on which
        the strain analysis is performed.
    :type map_config: CHAP.common.models.map.MapConfig, optional
    :ivar par_file: Path to the par file associated with the scan.
    :type par_file: str, optional
    :ivar dataset_id: Integer ID of the SMB-style EDD dataset.
    :type dataset_id: int, optional
    :ivar par_dims: List of independent dimensions.
    :type par_dims: list[dict[str,str]], optional
    :ivar other_dims: List of other column names from `par_file`.
    :type other_dims: list[dict[str,str]], optional
    :ivar detectors: List of individual detector element strain
        analysis configurations
    :type detectors: list[MCAElementStrainAnalysisConfig]
    :ivar materials: Sample material configurations.
    :type materials: list[MaterialConfig]
    :ivar flux_file: File name of the csv flux file containing station
        beam energy in eV (column 0) versus flux (column 1).
    :type flux_file: str, optional
    :ivar sum_fly_axes: Whether to sum over the fly axis or not
        for EDD scan types not 0, defaults to `True`.
    :type sum_fly_axes: bool, optional
    """
    inputdir: Optional[DirectoryPath]
    map_config: Optional[MapConfig]
    par_file: Optional[FilePath]
    dataset_id: Optional[int]
    par_dims: Optional[list[dict[str,str]]]
    other_dims: Optional[list[dict[str,str]]]
    detectors: conlist(min_items=1, item_type=MCAElementStrainAnalysisConfig)
    materials: list[MaterialConfig]
    flux_file: Optional[FilePath]
    sum_fly_axes: Optional[StrictBool]
    oversampling: Optional[dict] = {'num': 10}

    _parfile: Optional[ParFile]

    @root_validator(pre=True)
    def validate_config(cls, values):
        """Ensure that a valid configuration was provided and finalize
        input filepaths.

        :param values: Dictionary of class field values.
        :type values: dict
        :raises ValueError: Missing par_dims value.
        :return: The validated list of `values`.
        :rtype: dict
        """
        inputdir = values.get('inputdir')
        flux_file = values.get('flux_file')
        par_file = values.get('par_file')
        if (inputdir is not None and flux_file is not None
                and not os.path.isabs(flux_file)):
            values['flux_file'] = os.path.join(inputdir, flux_file)
        if par_file is not None:
            if inputdir is not None and not os.path.isabs(par_file):
                values['par_file'] = os.path.join(inputdir, par_file)
            if 'dataset_id' in values:
                from CHAP.edd import EddMapReader
                values['_parfile'] = ParFile(values['par_file'])
                values['map_config'] = EddMapReader().read(
                    values['par_file'], values['dataset_id'])
            elif 'par_dims' in values:
                values['_parfile'] = ParFile(values['par_file'])
                values['map_config'] = values['_parfile'].get_map(
                    'EDD', 'id1a3', values['par_dims'],
                    other_dims=values.get('other_dims', []))
            else:
                raise ValueError(
                    'dataset_id or par_dims is required when using par_file')
        map_config = values.get('map_config')
        if isinstance(map_config, dict):
            for i, scans in enumerate(map_config.get('spec_scans')):
                spec_file = scans.get('spec_file')
                if inputdir is not None and not os.path.isabs(spec_file):
                    values['map_config']['spec_scans'][i]['spec_file'] = \
                        os.path.join(inputdir, spec_file)
        return values

    @validator('detectors', pre=True, each_item=True)
    def validate_tth_file(cls, detector, values):
        """Finalize value for tth_file for each detector"""
        inputdir = values.get('inputdir')
        tth_file = detector.get('tth_file')
        if tth_file:
            if not os.path.isabs(tth_file):
                detector['tth_file'] = os.path.join(inputdir, tth_file)
        return detector

    @validator('detectors', each_item=True)
    def validate_tth(cls, detector, values):
        """Validate detector element tth_file field. It may only be
        used if StrainAnalysisConfig used par_file.
        """
        if detector.tth_file is not None:
            if not values.get('par_file'):
                raise ValueError(
                    'variable tth angles may only be used with a '
                    + 'StrainAnalysisConfig that uses par_file.')
            else:
                try:
                    detector.tth_map = ParFile(values['par_file']).map_values(
                        values['map_config'], np.loadtxt(detector.tth_file))
                except Exception as e:
                    raise ValueError(
                        'Could not get map of tth angles from '
                        + f'{detector.tth_file}') from e
        return detector

    @validator('sum_fly_axes', always=True)
    def validate_sum_fly_axes(cls, value, values):
        """Validate the sum_fly_axes field.

        :param value: Field value to validate (`sum_fly_axes`).
        :type value: bool
        :param values: Dictionary of validated class field values.
        :type values: dict
        :return: The validated value for sum_fly_axes.
        :rtype: bool
        """
        if value is None:
            map_config = values.get('map_config')
            if map_config is not None:
                if map_config.attrs['scan_type'] < 3:
                    value = False
                else:
                    value = True
        return value

    @validator('oversampling', always=True)
    def validate_oversampling(cls, value, values):
        """Validate the oversampling field.

        :param value: Field value to validate (`oversampling`).
        :type value: bool
        :param values: Dictionary of validated class field values.
        :type values: dict
        :return: The validated value for oversampling.
        :rtype: bool
        """
        # Local modules
        from CHAP.utils.general import is_int

        map_config = values.get('map_config')
        if map_config is None or map_config.attrs['scan_type'] < 3:
            return None
        if value is None:
            return {'num': 10}
        if 'start' in value and not is_int(value['start'], ge=0):
            raise ValueError('Invalid "start" parameter in "oversampling" '
                             f'field ({value["start"]})')
        if 'end' in value and not is_int(value['end'], gt=0):
            raise ValueError('Invalid "end" parameter in "oversampling" '
                             f'field ({value["end"]})')
        if 'width' in value and not is_int(value['width'], gt=0):
            raise ValueError('Invalid "width" parameter in "oversampling" '
                             f'field ({value["width"]})')
        if 'stride' in value and not is_int(value['stride'], gt=0):
            raise ValueError('Invalid "stride" parameter in "oversampling" '
                             f'field ({value["stride"]})')
        if 'num' in value and not is_int(value['num'], gt=0):
            raise ValueError('Invalid "num" parameter in "oversampling" '
                             f'field ({value["num"]})')
        if 'mode' in value and 'mode' not in ('valid', 'full'):
            raise ValueError('Invalid "mode" parameter in "oversampling" '
                             f'field ({value["mode"]})')
        if not ('width' in value or 'stride' in value or 'num' in value):
            raise ValueError('Invalid input parameters, specify at least one '
                             'of "width", "stride" or "num"')
        return value

    def mca_data(self, detector=None, map_index=None):
        """Get MCA data for a single or multiple detector elements.

        :param detector: Detector(s) for which data is returned,
            defaults to `None`, which return MCA data for all 
            detector elements.
        :type detector: Union[int, MCAElementStrainAnalysisConfig],
            optional
        :param map_index: Index of a single point in the map, defaults
            to `None`, which returns MCA data for each point in the map.
        :type map_index: tuple, optional
        :return: A single MCA spectrum.
        :rtype: np.ndarray
        """
        if detector is None:
            mca_data = []
            for detector_config in self.detectors:
                mca_data.append(
                    self.mca_data(detector_config, map_index))
            return np.asarray(mca_data)
        else:
            if isinstance(detector, int):
                detector_config = self.detectors[detector]
            else:
                if not isinstance(detector, MCAElementStrainAnalysisConfig):
                    raise ValueError('Invalid parameter detector ({detector})')
                detector_config = detector
            if map_index is None:
                fly_axis_labels = self.map_config.attrs.get('fly_axis_labels')
                mca_data = []
                for map_index in np.ndindex(self.map_config.shape):
                    mca_data.append(self.mca_data(
                        detector_config, map_index))
                mca_data = np.reshape(
                    mca_data, (*self.map_config.shape, len(mca_data[0])))
                if self.sum_fly_axes and fly_axis_labels:
                    scan_type = self.map_config.attrs['scan_type']
                    if scan_type in (3, 5):
                        sum_indices = []
                        for axis in fly_axis_labels:
                            sum_indices.append(self.map_config.dims.index(axis))
                        return np.sum(mca_data, tuple(sorted(sum_indices)))
                    elif scan_type == 4:
                        # Local modules
                        from CHAP.edd.utils import get_rolling_sum_spectra

                        return get_rolling_sum_spectra(
                            mca_data,
                            self.map_config.dims.index(fly_axis_labels[0]),
                            self.oversampling.get('start', 0),
                            self.oversampling.get('end'),
                            self.oversampling.get('width'),
                            self.oversampling.get('stride'),
                            self.oversampling.get('num'),
                            self.oversampling.get('mode', 'valid'))
                    else:
                        raise ValueError(
                            f'scan_type {scan_type} not implemented yet '
                            'in StrainAnalysisConfig.mca_data()')
                else:
                    return np.asarray(mca_data)
            else:
                return self.map_config.get_detector_data(
                    detector_config.detector_name, map_index)

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_scanparser' in d:
            del d['_scanparser']
        return d
