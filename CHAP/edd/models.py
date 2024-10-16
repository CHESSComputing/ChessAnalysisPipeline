"""EDD Pydantic model classes."""

# System modules
from copy import deepcopy
import os
from pathlib import PosixPath
from typing import (
    Dict,
    Literal,
    Optional,
    Union,
)

# Third party modules
from chess_scanparsers import SMBMCAScanParser as ScanParser
import numpy as np
from hexrd.material import Material
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PrivateAttr,
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)
from scipy.interpolate import interp1d
from typing_extensions import Annotated

# Local modules
from CHAP.common.models.map import Detector
from CHAP.utils.parfile import ParFile

# Baseline configuration class

class BaselineConfig(BaseModel):
    """Baseline model configuration.

    :ivar tol: The convergence tolerence, defaults to `1.e-6`.
    :type tol: float, optional
    :ivar lam: The &lambda (smoothness) parameter (the balance
        between the residual of the data and the baseline and the
        smoothness of the baseline). The suggested range is between
        100 and 10^8, defaults to `10^6`.
    :type lam: float, optional
    :ivar max_iter: The maximum number of iterations,
        defaults to `100`.
    :type max_iter: int, optional
    """
    tol: confloat(gt=0, allow_inf_nan=False) = 1.e-6
    lam: confloat(gt=0, allow_inf_nan=False) = 1.e6
    max_iter: conint(gt=0) = 100
    attrs: Optional[dict] = {}


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
    material_name: Optional[constr(strip_whitespace=True, min_length=1)] = None
    lattice_parameters: Optional[Union[
        confloat(gt=0),
        conlist(min_length=1, max_length=6, item_type=confloat(gt=0))]] = None
    sgnum: Optional[conint(ge=0)] = None

    _material: Optional[Material]

    @model_validator(mode='after')
    def validate_material(self):
        """Create and validate the private attribute _material.

        :return: The validated list of class properties.
        :rtype: dict
        """
        # Local modules
        from CHAP.edd.utils import make_material

        self._material = make_material(
            self.material_name, self.sgnum, self.lattice_parameters)
        return self

    def unique_hkls_ds(self, tth_tol=0.15, tth_max=90.0):
        """Get a list of unique HKLs and their lattice spacings.

        :param tth_tol: Minimum resolvable difference in 2&theta
            between two unique HKL peaks, defaults to `0.15`.
        :type tth_tol: float, optional
        :param tth_max: Detector rotation about hutch x axis,
            defaults to `90.0`.
        :type tth_max: float, optional
        :return: Unique HKLs and their lattice spacings in angstroms.
        :rtype: numpy.ndarray, numpy.ndarray
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
        for k, v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_material' in d:
            del d['_material']
        return d


# Detector configuration classes

class MCAElementConfig(Detector):
    """Class representing metadata required to configure a single MCA
    detector element.

    :ivar id: The MCA detector id (name or channel index) in the scan,
        defaults to `'0'`.
    :type id: str
    :ivar num_bins: Number of MCA channels.
    :type num_bins: int, optional
    """
    num_bins: Optional[conint(gt=0)] = None

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import NXattr

        d = super().dict(*args, **kwargs)
        for k, v in d['attrs'].items():
            if isinstance(v, NXattr):
                d['attrs'][k] = v.nxdata
        return d


class MCAElementCalibrationConfig(MCAElementConfig):
    """Class representing metadata required to calibrate a single MCA
    detector element.

    :ivar tth_max: Detector rotation about lab frame x axis,
       defaults to `90`.
    :type tth_max: float, optional
    :ivar hkl_tth_tol: Minimum resolvable difference in 2&theta between
        two unique Bragg peaks, defaults to `0.15`.
    :type hkl_tth_tol: float, optional
    :ivar energy_calibration_coeffs: Detector channel index to energy
        polynomial conversion coefficients ([a, b, c] with
        E_i = a*i^2 + b*i + c), defaults to `[0, 0, 1]`.
    :type energy_calibration_coeffs:
        list[float, float, float], optional
    :ivar background: Background model for peak fitting.
    :type background: str, list[str], optional
    :ivar baseline: Automated baseline subtraction configuration,
        defaults to `False`.
    :type baseline: Union(bool, BaselineConfig), optional
    :ivar tth_initial_guess: Initial guess for 2&theta,
        defaults to `5.0`.
    :type tth_initial_guess: float, optional
    :ivar tth_calibrated: Calibrated value for 2&theta.
    :type tth_calibrated: float, optional
    :ivar include_energy_ranges: List of MCA channel energy ranges
        in keV whose data should be included after applying a mask
        (bounds are inclusive), defaults to `[[50, 150]]`.
    :type include_energy_ranges: list[[float, float]], optional
    """
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    energy_calibration_coeffs: conlist(
        min_length=3, max_length=3,
        item_type=confloat(allow_inf_nan=False)) = [0, 0, 1]
    background: Optional[Union[str, list]] = None
    baseline: Optional[Union[bool, BaselineConfig]] = False
    tth_initial_guess: confloat(gt=0, le=tth_max, allow_inf_nan=False) = 5.0
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    include_energy_ranges: Annotated[
        conlist(
            min_length=1,
            item_type=conlist(
                min_length=2,
                max_length=2,
                item_type=confloat(ge=25))),
        Field(validate_default=True)] = [[50, 150]]

    _hkl_indices: list = PrivateAttr()

    @field_validator('include_energy_ranges')
    @classmethod
    def validate_include_energy_range(cls, include_energy_ranges, info):
        """Ensure that no energy ranges are outside the boundary of the
        detector.

        :param include_energy_ranges: The value of
            `include_energy_ranges` to validate.
        :type include_energy_ranges: dict
        :param info: Pydantic validator info object.
        :type info: pydantic_core._pydantic_core.ValidationInfo
        :rtype: dict
        """
        n_max = info.data.get('num_bins')
        for i in range(len(include_energy_ranges)):
            include_energy_ranges[i].sort()
        if n_max is not None:
            n_max -= 1
            a, b, c = info.data.get('energy_calibration_coeffs')
            e_max = (a*n_max + b)*n_max +c
            for i, include_energy_range in enumerate(
                    deepcopy(include_energy_ranges)):
                if (include_energy_range[0] < c
                        or include_energy_range[1] > e_max):
                    include_energy_ranges[i] = [
                        float(max(include_energy_range[0], c)),
                        float(min(include_energy_range[1], e_max))]
                    print(
                        f'WARNING: include_energy_range out of range'
                        f' ({include_energy_range}): adjusted to '
                        f'{include_energy_ranges[i]}')
        return include_energy_ranges

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

    @property
    def hkl_indices(self):
        """Return the hkl_indices consistent with the selected energy
        ranges (include_energy_ranges).
        """
        if hasattr(self, '_hkl_indices'):
            return self._hkl_indices
        return []

    def get_include_energy_ranges(self, include_bin_ranges):
        """Given a list of channel index ranges, return the
        corresponding list of channel energy ranges.

        :param include_bin_ranges: A list of channel bin ranges to
            convert to energy ranges.
        :type include_bin_ranges: list[[int,int]]
        :returns: Energy ranges.
        :rtype: list[[float,float]]
        """
        energies = self.energies
        return [[float(energies[i]) for i in range_]
                for range_ in include_bin_ranges]

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.
        Note that the bounds of self.include_energy_ranges are
        inclusive.

        :return: Boolean mask array.
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for (min_, max_) in self.include_bin_ranges:
            mask = np.logical_or(
                mask, np.logical_and(bin_indices >= min_, bin_indices <= max_))
        return mask

    def set_hkl_indices(self, hkl_indices):
        """Set the private attribute `hkl_indices`."""
        self._hkl_indices = hkl_indices

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        if '_hkl_indices:' in d:
            del d['_hkl_indices:']
        return d


class MCAElementDiffractionVolumeLengthConfig(MCAElementConfig):
    """Class representing metadata required to perform a diffraction
    volume length measurement for a single MCA detector element.

    :ivar include_bin_ranges: List of MCA channel index ranges
        whose data is included in the measurement.
    :type include_bin_ranges: list[[int, int]], optional
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
    :ivar fit_center: Placeholder for center of the gaussian fit.
    :type fit_center: float, optional
    :ivar fit_sigma: Placeholder for sigma of the gaussian fit.
    :type fit_sigma: float, optional
    """
    include_bin_ranges: Optional[
        conlist(
            min_length=1,
            item_type=conlist(
                min_length=2,
                max_length=2,
                item_type=conint(ge=0)))] = None
    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
    sigma_to_dvl_factor: Optional[Literal[3.5, 2.0, 4.0]] = 3.5
    dvl_measured: Optional[confloat(gt=0)] = None
    fit_amplitude: Optional[float] = None
    fit_center: Optional[float] = None
    fit_sigma: Optional[float] = None

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.
        Note that the bounds of self.include_energy_ranges are
        inclusive.

        :return: Boolean mask array.
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for (min_, max_) in self.include_bin_ranges:
            mask = np.logical_or(
                mask, np.logical_and(bin_indices >= min_, bin_indices <= max_))
        return mask

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
    :ivar hkl_indices: List of unique HKL indices to use in the strain
        analysis, defaults to `[]`.
    :type hkl_indices: list[int], optional
    :ivar background: Background model for peak fitting.
    :type background: str, list[str], optional
    :ivar baseline: Automated baseline subtraction configuration,
        defaults to `False`.
    :type baseline: Union(bool, BaselineConfig), optional
    :ivar num_proc: Number of processors used for peak fitting.
    :type num_proc: int, optional
    :ivar peak_models: Peak model for peak fitting,
        defaults to `'gaussian'`.
    :type peak_models: Literal['gaussian', 'lorentzian']],
        list[Literal['gaussian', 'lorentzian']]], optional
    :ivar fwhm_min: Minimum FWHM for peak fitting, defaults to `0.25`.
    :type fwhm_min: float, optional
    :ivar fwhm_max: Maximum FWHM for peak fitting, defaults to `2.0`.
    :type fwhm_max: float, optional
    :ivar centers_range: Peak centers range for peak fitting.
        The allowed range the peak centers will be the initial
        values &pm; `centers_range`. Defaults to `2.0`.
    :type centers_range: float, optional
    :ivar rel_height_cutoff: Relative peak height cutoff for
        peak fitting (any peak with a height smaller than
        `rel_height_cutoff` times the maximum height of all peaks 
        gets removed from the fit model), defaults to `None`.
    :type rel_height_cutoff: float, optional
    :ivar tth_calibrated: Calibrated value for 2&theta.
    :type tth_calibrated: float, optional
    :ivar energy_calibration_coeffs: Detector channel index to energy
        polynomial conversion coefficients ([a, b, c] with
        E_i = a*i^2 + b*i + c), defaults to `[0, 0, 1]`.
    :type energy_calibration_coeffs:
        list[float, float, float], optional
    :ivar calibration_bin_ranges: List of MCA channel index ranges
        whose data is included in the calibration.
    :type calibration_bin_ranges: list[[int, int]], optional
    :ivar tth_file: Path to the file with the 2&theta map.
    :type tth_file: FilePath, optional
    :ivar tth_map: Map of the 2&theta values.
    :type tth_map: numpy.ndarray, optional
    :ivar include_energy_ranges: List of MCA channel energy ranges
        in keV whose data should be included after applying a mask
        (bounds are inclusive), defaults to `[[50, 150]]`.
    :type include_energy_ranges: list[[float, float]], optional
    """
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    hkl_indices: Optional[conlist(item_type=conint(ge=0))] = []
    background: Optional[Union[str, list]] = None
    baseline: Optional[Union[bool, BaselineConfig]] = False
    num_proc: Optional[conint(gt=0)] = max(1, os.cpu_count()//4)
    peak_models: Union[
        conlist(min_length=1, item_type=Literal['gaussian', 'lorentzian']),
        Literal['gaussian', 'lorentzian']] = 'gaussian'
    fwhm_min: confloat(gt=0, allow_inf_nan=False) = 0.25
    fwhm_max: confloat(gt=0, allow_inf_nan=False) = 2.0
    centers_range: confloat(gt=0, allow_inf_nan=False) = 2.0
    rel_height_cutoff: Optional[
        confloat(gt=0, lt=1.0, allow_inf_nan=False)] = None
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    energy_calibration_coeffs: conlist(
        min_length=3, max_length=3,
        item_type=confloat(allow_inf_nan=False)) = [0, 0, 1]
    calibration_bin_ranges: Optional[
        conlist(
            min_length=1,
            item_type=conlist(
                min_length=2,
                max_length=2,
                item_type=conint(ge=0)))] = None
    tth_file: Optional[FilePath] = None
    tth_map: Optional[np.ndarray] = None
    include_energy_ranges: conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=confloat(ge=25))) = [[50, 150]]

    #RV lots of overlap with MCAElementCalibrationConfig (only missing
    #   tth_initial_guess)
    #   Should we derive from MCAElementCalibrationConfig in some way
    #   or make a MCAElementEnergyCalibrationConfig with what's shared
    #   and derive MCAElementCalibrationConfig from this as well with
    #   the unique fields tth_initial_guess added?
    #   Revisit when we redo the detectors

    @field_validator('hkl_indices', mode='before')
    @classmethod
    def validate_hkl_indices(cls, hkl_indices):
        """Validate the HKL indices.

        :ivar hkl_indices: List of unique HKL indices.
        :type hkl_indices: list[int]
        :return: List of HKL indices.
        :rtype: list[int]
        """
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

        :param include_bin_ranges: A list of channel bin ranges to
            convert to energy ranges.
        :type include_bin_ranges: list[[int,int]]
        :returns: Energy ranges.
        :rtype: list[[float,float]]
        """
        energies = self.energies
        return [[float(energies[i]) for i in range_]
                for range_ in include_bin_ranges]

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.
        Note that the bounds of self.include_energy_ranges are
        inclusive.

        :return: Boolean mask array.
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for (min_, max_) in self.include_bin_ranges:
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
        :rtype: numpy.ndarray
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
        for k, v in d.items():
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
    inputdir: Optional[DirectoryPath] = None
    spec_file: Optional[FilePath] = None
    scan_number: Optional[conint(gt=0)] = None
    par_file: Optional[FilePath] = None
    scan_column: Optional[str] = None
    detectors: conlist(min_length=1, item_type=MCAElementConfig)

    _parfile: Optional[ParFile] = None
    _scanparser: Optional[ScanParser] = None

    @model_validator(mode='before')
    @classmethod
    def validate_scan(cls, data):
        """Finalize file paths for spec_file and par_file.

        :param data: Pydantic validator data object.
        :type data: MCAScanDataConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid SPEC or par file.
        :return: The validated list of class properties.
        :rtype: dict
        """
        inputdir = data.get('inputdir')
        spec_file = data.get('spec_file')
        par_file = data.get('par_file')
        if spec_file is not None and par_file is not None:
            raise ValueError('Use either spec_file or par_file, not both')
        if spec_file is not None:
            if inputdir is not None and not os.path.isabs(spec_file):
                data['spec_file'] = os.path.join(inputdir, spec_file)
        elif par_file is not None:
            if inputdir is not None and not os.path.isabs(par_file):
                data['par_file'] = os.path.join(inputdir, par_file)
            if 'scan_column' not in data:
                raise ValueError(
                    'scan_column is required when par_file is used')
            if isinstance(data['scan_column'], str):
                parfile = ParFile(par_file)
                if data['scan_column'] not in parfile.column_names:
                    raise ValueError(
                        f'No column named {data["scan_column"]} in '
                        + '{data["par_file"]}. Options: '
                        + ', '.join(parfile.column_names))
        else:
            raise ValueError('Must use either spec_file or par_file')

        return data

    @model_validator(mode='after')
    def validate_detectors(self):
        """Fill in values for _scanparser / _parfile (if applicable).
        Fill in each detector's num_bins field, if needed.
        Check each detector's include_energy_ranges field against the
        flux file, if available.

        :raises ValueError: Unable to obtain a value for num_bins.
        :return: The validated list of class properties.
        :rtype: dict
        """
        spec_file = self.spec_file
        par_file = self.par_file
        detectors = self.detectors
        flux_file = self.flux_file
        if spec_file is not None:
            self._scanparser = ScanParser(
                spec_file, self.scan_number)
            self._parfile = None
        elif par_file is not None:
            self._parfile = ParFile(par_file)
            self._scanparser = ScanParser(
                self._parfile.spec_file,
                self._parfile.good_scan_numbers()[0])
        for detector in detectors:
            if detector.num_bins is None:
                try:
                    detector.num_bins = \
                        self._scanparser.get_detector_num_bins(
                                detector.id)
                except Exception as e:
                    raise ValueError('No value found for num_bins') from e
        if flux_file is not None:
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

        return self

    @property
    def scanparser(self):
        """Return the scanparser."""
        try:
            scanparser = self._scanparser
        except:
            scanparser = ScanParser(self.spec_file, self.scan_number)
            self._scanparser = scanparser
        return scanparser

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k, v in d.items():
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
    :ivar detectors: Individual detector element DVL measurement
        configurations.
    :type detectors: list[MCAElementDiffractionVolumeLengthConfig]
    """
    sample_thickness: float
    detectors: conlist(
        min_length=1, item_type=MCAElementDiffractionVolumeLengthConfig)

    @property
    def scanned_vals(self):
        """Return the list of values visited by the scanning motor
        over the course of the raster scan.

        :return: List of scanned motor values.
        :rtype: numpy.ndarray
        """
        if self._parfile is not None:
            return self._parfile.get_values(
                self.scan_column,
                scan_numbers=self._parfile.good_scan_numbers())
        return self.scanparser.spec_scan_motor_vals[0]


class MCAEnergyCalibrationConfig(BaseModel):
    """Class representing metadata required to perform an energy
    calibration for an MCA detector.

    :ivar inputdir: Input directory, used only if any file in the
        configuration is not an absolute path.
    :type inputdir: str, optional
    :ivar scan_step_indices: Optional scan step indices to use for the
        calibration. If not specified, the calibration will be
        performed on the average of all MCA spectra for the scan.
    :type scan_step_indices: list[int], optional
    :ivar detectors: List of individual MCA detector element
        calibration configurations.
    :type detectors: list[MCAElementCalibrationConfig], optional
    :ivar flux_file: File name of the csv flux file containing station
        beam energy in eV (column 0) versus flux (column 1).
    :type flux_file: str, optional
    :ivar material: Material configuration for the calibration,
        defaults to `Ceria`.
    :type material: MaterialConfig, optional
    :ivar peak_energies: Theoretical locations of peaks in keV to use
        for calibrating the MCA channel energies. It is _strongly_
        recommended to use fluorescence peaks for the energy
        calibration.
    :type peak_energies: list[float]
    :ivar max_peak_index: Index of the peak in `peak_energies`
        with the highest amplitude.
    :type max_peak_index: int
    :ivar fit_index_ranges: Explicit ranges of uncalibrated MCA
        channel index ranges to include during energy calibration
        when the given peaks are fitted to the provied MCA spectrum.
        Use this parameter or select it interactively by running a
        pipeline with `config.interactive: True`.
    :type fit_index_ranges: list[[int, int]], optional

    Note: Fluorescence data:
        https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
    """
    inputdir: Optional[DirectoryPath] = None
    scan_step_indices: Optional[Annotated[conlist(
        min_length=1, item_type=conint(ge=0)),
        Field(validate_default=True)]] = None
    detectors: Optional[conlist(item_type=MCAElementCalibrationConfig)] = None
    flux_file: Optional[FilePath] = None
    material: Optional[MaterialConfig] = MaterialConfig(
        material_name='CeO2', lattice_parameters=5.41153, sgnum=225)
    peak_energies: conlist(min_length=2, item_type=confloat(gt=0))
    max_peak_index: conint(ge=0)
    fit_index_ranges: Optional[
        conlist(
            min_length=1,
            item_type=conlist(
                min_length=2,
                max_length=2,
                item_type=conint(ge=0)))] = None

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        flux_file filepath.

        :param data: Pydantic validator data object.
        :type data: MCAEnergyCalibrationConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        inputdir = data.get('inputdir')
        if inputdir is not None:
            flux_file = data.get('flux_file')
            if flux_file is not None and not os.path.isabs(flux_file):
                data['flux_file'] = os.path.join(inputdir, flux_file)

        return data

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Validate the specified list of scan numbers.

        :ivar scan_step_indices: Optional scan step indices to use for
            the calibration. If not specified, the calibration will be
            performed on the average of all MCA spectra for the scan.
        :type scan_step_indices: list[int], optional
        :raises ValueError: Invalid experiment type.
        :return: List of step indices.
        :rtype: list[int]
        """
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(
                scan_step_indices, raise_error=True)
        return scan_step_indices

    @field_validator('max_peak_index')
    @classmethod
    def validate_max_peak_index(cls, max_peak_index, info):
        """Validate the specified index of the XRF peak with the
        highest amplitude.

        :ivar max_peak_index: The index of the XRF peak with the
            highest amplitude.
        :type max_peak_index: int
        :param info: Pydantic validator info object.
        :type info: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid max_peak_index.
        :return: The validated value of `max_peak_index`.
        :rtype: int
        """
        peak_energies = info.data.get('peak_energies')
        if not 0 <= max_peak_index < len(peak_energies):
            raise ValueError('max_peak_index out of bounds')
        return max_peak_index

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

    def flux_correction_interpolation_function(self):
        """Get an interpolation function to correct MCA data for the
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

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        if 'inputdir' in d:
            del d['inputdir']
        return d


class MCATthCalibrationConfig(MCAEnergyCalibrationConfig):
    """Class representing metadata required to perform a tth
    calibration for an MCA detector.

    :ivar calibration_method: Type of calibration method,
        defaults to `'direct_fit_residual'`.
    :type calibration_method:
        Literal['direct_fit_residual', 'iterate_tth'], optional
    :ivar max_iter: Maximum number of iterations of the calibration
        routine (only used for `'iterate_tth'`), defaults to `10`.
    :type max_iter: int, optional
    :ivar tune_tth_tol: Cutoff error for tuning 2&theta (only used for
        `'iterate_tth'`). Stop iterating the calibration routine after
        an iteration produces a change in the tuned value of 2&theta
        that is smaller than this cutoff, defaults to `1e-8`.
    :ivar tune_tth_tol: float, optional
    """
    calibration_method: Optional[Literal[
        'direct_fit_residual',
        'direct_fit_peak_energies',
        'direct_fit_combined',
        'iterate_tth']] = 'iterate_tth'
    max_iter: conint(gt=0) = 10
    tune_tth_tol: confloat(ge=0) = 1e-8

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


class StrainAnalysisConfig(BaseModel):
    """Class representing input parameters required to perform a
    strain analysis.

    :ivar inputdir: Input directory, used only if any file in the
        configuration is not an absolute path.
    :type inputdir: str, optional
    :ivar detectors: List of individual detector element strain
        analysis configurations, defaults to `None` (use all).
    :type detectors: list[MCAElementStrainAnalysisConfig], optional
    :ivar materials: Sample material configurations.
    :type materials: list[MaterialConfig]
    :ivar flux_file: File name of the csv flux file containing station
        beam energy in eV (column 0) versus flux (column 1).
    :type flux_file: str, optional
    :ivar sum_axes: Whether to sum over the fly axis or not
        for EDD scan types not 0, defaults to `True`.
    :type sum_axes: Union[bool, list[str]], optional
    :ivar oversampling: FIX
    :type oversampling: FIX
    """
    inputdir: Optional[DirectoryPath] = None
    detectors: Optional[conlist(
        min_length=1, item_type=MCAElementStrainAnalysisConfig)] = None
    materials: conlist(item_type=MaterialConfig)
    flux_file: Optional[FilePath] = None
    sum_axes: Optional[
        Union[bool, conlist(min_length=1, item_type=str)]] = True
    oversampling: Optional[
        Annotated[Dict, Field(validate_default=True)]] = {'num': 10}

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        flux_file filepath.

        :param data: Pydantic validator data object.
        :type data: MCAEnergyCalibrationConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        inputdir = data.get('inputdir')
        if inputdir is not None:
            flux_file = data.get('flux_file')
            if flux_file is not None and not os.path.isabs(flux_file):
                data['flux_file'] = os.path.join(inputdir, flux_file)

        return data

    @field_validator('detectors', mode='before')
    @classmethod
    def validate_tth_file(cls, detectors, info):
        """Finalize value for tth_file for each detector."""
        inputdir = info.data.get('inputdir')
        for detector in detectors:
            tth_file = detector.get('tth_file')
            if tth_file is not None:
                if not os.path.isabs(tth_file):
                    detector['tth_file'] = os.path.join(inputdir, tth_file)
        return detectors

# FIX tth_file/tth_map not updated
#    @field_validator('detectors')
#    @classmethod
#    def validate_tth(cls, detectors, info):
#        """Validate detector element tth_file field. It may only be
#        used if StrainAnalysisConfig used par_file.
#        """
#        for detector in detectors:
#            tth_file = detector.tth_file
#            if tth_file is not None:
#                if not info.data.get('par_file'):
#                    raise ValueError(
#                        'variable tth angles may only be used with a '
#                        'StrainAnalysisConfig that uses par_file.')
#                else:
#                    try:
#                        detector.tth_map = ParFile(
#                            info.data['par_file']).map_values(
#                                info.data['map_config'],
#                                np.loadtxt(tth_file))
#                    except Exception as e:
#                        raise ValueError(
#                            'Could not get map of tth angles from '
#                            f'{tth_file}') from e
#        return detectors

    @field_validator('oversampling')
    @classmethod
    def validate_oversampling(cls, oversampling, info):
        """Validate the oversampling field.

        :param oversampling: The value of `oversampling` to validate.
        :type oversampling: dict
        :param info: Pydantic validator info object.
        :type info: StrainAnalysisConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The validated value for oversampling.
        :rtype: bool
        """
        # Local modules
        from CHAP.utils.general import is_int

        raise ValueError('oversampling not updated yet')
        map_config = info.data.get('map_config')
        if map_config is None or map_config.attrs['scan_type'] < 3:
            return None
        if oversampling is None:
            return {'num': 10}
        if 'start' in oversampling and not is_int(oversampling['start'], ge=0):
            raise ValueError('Invalid "start" parameter in "oversampling" '
                             f'field ({oversampling["start"]})')
        if 'end' in oversampling and not is_int(oversampling['end'], gt=0):
            raise ValueError('Invalid "end" parameter in "oversampling" '
                             f'field ({oversampling["end"]})')
        if 'width' in oversampling and not is_int(oversampling['width'], gt=0):
            raise ValueError('Invalid "width" parameter in "oversampling" '
                             f'field ({oversampling["width"]})')
        if ('stride' in oversampling
                and not is_int(oversampling['stride'], gt=0)):
            raise ValueError('Invalid "stride" parameter in "oversampling" '
                             f'field ({oversampling["stride"]})')
        if 'num' in oversampling and not is_int(oversampling['num'], gt=0):
            raise ValueError('Invalid "num" parameter in "oversampling" '
                             f'field ({oversampling["num"]})')
        if 'mode' in oversampling and 'mode' not in ('valid', 'full'):
            raise ValueError('Invalid "mode" parameter in "oversampling" '
                             f'field ({oversampling["mode"]})')
        if not ('width' in oversampling or 'stride' in oversampling
                or 'num' in oversampling):
            raise ValueError('Invalid input parameters, specify at least one '
                             'of "width", "stride" or "num"')
        return oversampling

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k, v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if 'inputdir' in d:
            del d['inputdir']
        return d
