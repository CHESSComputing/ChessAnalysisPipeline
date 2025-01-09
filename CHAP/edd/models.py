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
    """Baseline model configuration class.

    :ivar lam: The &lambda (smoothness) parameter (the balance
        between the residual of the data and the baseline and the
        smoothness of the baseline). The suggested range is between
        100 and 10^8, defaults to `10^6`.
    :type lam: float, optional
    :ivar max_iter: The maximum number of iterations,
        defaults to `100`.
    :type max_iter: int, optional
    :ivar tol: The convergence tolerence, defaults to `1.e-6`.
    :type tol: float, optional
    """
    attrs: Optional[dict] = {}
    lam: confloat(gt=0, allow_inf_nan=False) = 1.e6
    max_iter: conint(gt=0) = 100
    tol: confloat(gt=0, allow_inf_nan=False) = 1.e-6


# Fit configuration class

class FitConfig(BaseModel):
    """Fit parameters configuration class for peak fitting.

    :ivar background: Background model for peak fitting,
        defaults to `constant`.
    :type background: str, list[str], optional
    :ivar baseline: Automated baseline subtraction configuration,
        defaults to `False`.
    :type baseline: Union(bool, BaselineConfig), optional
    :ivar centers_range: Peak centers range for peak fitting.
        The allowed range for the peak centers will be the initial
        values &pm; `centers_range` (in MCA channels for calibration
        or keV for strain analysis). Defaults to `20` for calibration
        and `2.0` for strain analysis.
    :type centers_range: float, optional
    :ivar energy_mask_ranges: List of MCA energy mask ranges in keV
        for selecting the data to be included after applying a mask
        (bounds are inclusive). Specify either energy_mask_ranges or
        mask_ranges, not both.
    :type energy_mask_ranges: list[[float, float]], optional
    :ivar fwhm_min: Minimum FWHM for peak fitting (in MCA channels
        for calibration or keV for strain analysis). Defaults to `3`
        for calibration and `0.25` for analysis.
    :type fwhm_min: float, optional
    :ivar fwhm_max: Maximum FWHM for peak fitting (in MCA channels
        for calibration or keV for strain analysis). Defaults to `25`
        for calibration and `2.0` for analysis.
    :type fwhm_max: float, optional
    :ivar mask_ranges: List of MCA channel bin ranges for selecting
        the data to be included in the energy calibration after
        applying a mask (bounds are inclusive). Specify for
        energy calibration only.
    :type mask_ranges: list[[int, int]], optional
    """
    background: Optional[conlist(item_type=constr(
        strict=True, strip_whitespace=True, to_lower=True))] = None
    baseline: Optional[Union[bool, BaselineConfig]] = None
    centers_range: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    energy_mask_ranges: Optional[conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=confloat(allow_inf_nan=False)))] = None
    fwhm_min: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    fwhm_max: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    mask_ranges: Optional[conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=conint(ge=0)))] = None

    @field_validator('background', mode='before')
    @classmethod
    def validate_background(cls, background):
        """Validate the background model.

        :ivar background: Background model for peak fitting.
        :type background: str, list[str], optional
        :return: List of validated background models.
        :rtype: list[str]
        """
        if isinstance(background, str):
            return [background]
        return sorted(background)

    @field_validator('baseline', mode='before')
    @classmethod
    def validate_baseline(cls, baseline):
        """Validate the baseline configuration.

        :ivar baseline: Automated baseline subtraction configuration.
        :type baseline: Union(bool, BaselineConfig), optional
        :return: Validated baseline subtraction configuration.
        :rtype: bool, BaselineConfig
        """
        if isinstance(baseline, bool) and baseline:
            return BaselineConfig()
        return baseline

    @field_validator('energy_mask_ranges', mode='before')
    @classmethod
    def validate_energy_mask_ranges(cls, energy_mask_ranges):
        """Validate the mask ranges for selecting the data to include.

        :ivar energy_mask_ranges: List of MCA energy mask ranges in keV
            for selecting the data to be included after applying a mask
            (bounds are inclusive).
        :type energy_mask_ranges: list[[float, float]], optional
        :return: Validated energy mask ranges.
        :rtype: list[[float, float]]
        """
        if energy_mask_ranges:
            return sorted([sorted(v) for v in energy_mask_ranges])
        return energy_mask_ranges

    @field_validator('mask_ranges', mode='before')
    @classmethod
    def validate_mask_ranges(cls, mask_ranges):
        """Validate the mask ranges for selecting the data to include.

        :ivar mask_ranges: List of MCA channel bin ranges for selecting
            the data to be included after applying a mask
            (bounds are inclusive).
        :type mask_ranges: list[[int, int]], optional
        :return: Validated mask ranges.
        :rtype: list[[int, int]]
        """
        if mask_ranges:
            return sorted([sorted(v) for v in mask_ranges])
        return mask_ranges

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
#        d['baseline'] = self.baseline.dict()
        return d


# Material configuration class

class MaterialConfig(BaseModel):
    """Sample material parameters configuration class.

    :ivar material_name: Sample material name.
    :type material_name: str, optional
    :ivar lattice_parameters: Lattice spacing(s) in angstroms.
    :type lattice_parameters: float, list[float], optional
    :ivar sgnum: Space group of the material.
    :type sgnum: int, optional
    """
    material_name: Optional[constr(strip_whitespace=True, min_length=1)] = None
    lattice_parameters: Optional[Union[
        confloat(gt=0, allow_inf_nan=False),
        conlist(
            min_length=1, max_length=6,
            item_type=confloat(gt=0, allow_inf_nan=False))]] = None
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

#    def unique_hkls_ds(self, tth_max=None, tth_tol=None):
#        """Get a list of unique HKLs and their lattice spacings.
#
#        :param tth_max: Detector rotation about hutch x axis,
#            defaults to `90.0`.
#        :type tth_max: float, optional
#        :param tth_tol: Minimum resolvable difference in 2&theta
#            between two unique HKL peaks, defaults to `0.15`.
#        :type tth_tol: float, optional
#        :return: Unique HKLs and their lattice spacings in angstroms.
#        :rtype: numpy.ndarray, numpy.ndarray
#        """
#        # Local modules
#        from CHAP.edd.utils import get_unique_hkls_ds
#
#        if tth_max is None:
#            tth_max = 90.0
#        if tth_tol is None:
#            tth_tol = 0.15
#        return get_unique_hkls_ds(
#            [self._material], tth_max=tth_max, tth_tol=tth_tol)

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        if '_material' in d:
            del d['_material']
        return d


# Detector configuration classes

class MCAElementConfig(Detector, FitConfig):
    """Class representing metadata required to configure a single MCA
    detector element.

    :ivar energy_calibration_coeffs: Detector channel index to energy
        polynomial conversion coefficients ([a, b, c] with
        E_i = a*i^2 + b*i + c).
    :type energy_calibration_coeffs:
        list[float, float, float], optional
    :ivar num_bins: Number of MCA channels.
    :type num_bins: int, optional
    :ivar tth_max: Detector rotation about lab frame x axis.
    :type tth_max: float, optional
    :ivar tth_tol: Minimum resolvable difference in 2&theta between
        two unique Bragg peaks,
    :type tth_tol: float, optional
    :ivar tth_calibrated: Calibrated value for 2&theta.
    :type tth_calibrated: float, optional
    :ivar tth_initial_guess: Initial guess for 2&theta superseding
        the global one in MCATthCalibrationConfig.
    :type tth_initial_guess: float, optional
    """
    energy_calibration_coeffs: Optional[conlist(
        min_length=3, max_length=3,
        item_type=confloat(allow_inf_nan=False))] = None
    num_bins: Optional[conint(gt=0)] = None
    tth_max: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    tth_tol: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    tth_initial_guess: Optional[confloat(gt=0, allow_inf_nan=False)] = None

    _hkl_indices: list = PrivateAttr()

    @model_validator(mode='after')
    def validate_fit_config(self):
        """Set default fit parameters for centers_range, fwhm_min,
        and fwhm_max.

        :return: Validated configuration class.
        :rtype: FitConfig
        """
        if self.centers_range is None:
            self.centers_range = 20
        if self.fwhm_min is None:
            self.fwhm_min = 3
        if self.fwhm_max is None:
            self.fwhm_max = 25
        return self

    @property
    def energies(self):
        """Return calibrated bin energies."""
        a, b, c = self.energy_calibration_coeffs
        channel_bins = np.arange(self.num_bins)
        return (a*channel_bins + b)*channel_bins + c

    @property
    def hkl_indices(self):
        """Return the hkl_indices consistent with the selected energy
        ranges (include_energy_ranges).
        """
        if hasattr(self, '_hkl_indices'):
            return self._hkl_indices
        return []

    @hkl_indices.setter
    def hkl_indices(self, value):
        """Set the private attribute `hkl_indices`."""
        self._hkl_indices = value

    def convert_mask_ranges(self, mask_ranges):
        """Given a list of mask ranges in channel bins, set the
        corresponding list of channel energy mask ranges.

        :param mask_ranges: A list of mask ranges to
            convert to energy mask ranges.
        :type mask_ranges: list[[int,int]]
        """
        energies = self.energies
        self.energy_mask_ranges = [
            [float(energies[i]) for i in range_] for range_ in mask_ranges]

    def get_mask_ranges(self):
        """Return the value of `mask_ranges` if set or convert the
        `energy_mask_ranges` from channel energies to channel indices.
        """
        if self.mask_ranges:
            return self.mask_ranges
        if self.energy_mask_ranges is None:
            return None

        # Local modules
        from CHAP.utils.general import (
            index_nearest_down,
            index_nearest_up,
        )

        mask_ranges = []
        energies = self.energies
        for e_min, e_max in self.energy_mask_ranges:
            mask_ranges.append(
                [index_nearest_down(energies, e_min),
                 index_nearest_up(energies, e_max)])
        return mask_ranges

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.
        Note that the bounds of the mask ranges are inclusive.

        :return: Boolean mask array.
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        mask_ranges = self.get_mask_ranges()
        channel_bins = np.arange(self.num_bins, dtype=np.int32)
        for (min_, max_) in mask_ranges:
            mask = np.logical_or(
                mask,
                np.logical_and(channel_bins >= min_, channel_bins <= max_))
        return mask

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


#class MCAElementDiffractionVolumeLengthConfig(MCAElementConfig):
#    """Class representing metadata required to perform a diffraction
#    volume length measurement for a single MCA detector element.
#
#    :ivar include_bin_ranges: List of MCA channel index ranges
#        whose data is included in the measurement.
#    :type include_bin_ranges: list[[int, int]], optional
#    :ivar measurement_mode: Placeholder for recording whether the
#        measured DVL value was obtained through the automated
#        calculation or a manual selection, defaults to `'auto'`.
#    :type measurement_mode: Literal['manual', 'auto'], optional
#    :ivar sigma_to_dvl_factor: The DVL is obtained by fitting a reduced
#        form of the MCA detector data. `sigma_to_dvl_factor` is a
#        scalar value that converts the standard deviation of the
#        gaussian fit to the measured DVL, defaults to `3.5`.
#    :type sigma_to_dvl_factor: Literal[3.5, 2.0, 4.0], optional
#    :ivar dvl_measured: Placeholder for the measured diffraction
#        volume length before writing the data to file.
#    :type dvl_measured: float, optional
#    :ivar fit_amplitude: Placeholder for amplitude of the gaussian fit.
#    :type fit_amplitude: float, optional
#    :ivar fit_center: Placeholder for center of the gaussian fit.
#    :type fit_center: float, optional
#    :ivar fit_sigma: Placeholder for sigma of the gaussian fit.
#    :type fit_sigma: float, optional
#    """
#    include_bin_ranges: Optional[
#        conlist(
#            min_length=1,
#            item_type=conlist(
#                min_length=2,
#                max_length=2,
#                item_type=conint(ge=0)))] = None
#    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
#    sigma_to_dvl_factor: Optional[Literal[3.5, 2.0, 4.0]] = 3.5
#    dvl_measured: Optional[confloat(gt=0, allow_inf_nan=False)] = None
#    fit_amplitude: Optional[float] = None
#    fit_center: Optional[float] = None
#    fit_sigma: Optional[float] = None
#
#    def mca_mask(self):
#        """Get a boolean mask array to use on this MCA element's data.
#        Note that the bounds of self.include_energy_ranges are
#        inclusive.
#
#        :return: Boolean mask array.
#        :rtype: numpy.ndarray
#        """
#        mask = np.asarray([False] * self.num_bins)
#        bin_indices = np.arange(self.num_bins)
#        for (min_, max_) in self.include_bin_ranges:
#            mask = np.logical_or(
#                mask, np.logical_and(bin_indices >= min_, bin_indices <= max_))
#        return mask
#
#    def dict(self, *args, **kwargs):
#        """Return a representation of this configuration in a
#        dictionary that is suitable for dumping to a YAML file.
#        Exclude `sigma_to_dvl_factor` from the dict representation if
#        `measurement_mode` is `'manual'`.
#
#        :return: Dictionary representation of the configuration.
#        :rtype: dict
#        """
#        d = super().dict(*args, **kwargs)
#        if self.measurement_mode == 'manual':
#            del d['sigma_to_dvl_factor']
#        for param in ('amplitude', 'center', 'sigma'):
#            d[f'fit_{param}'] = float(d[f'fit_{param}'])
#        return d


class MCAElementStrainAnalysisConfig(MCAElementConfig):
    """Class representing metadata required to perform a strain
    analysis.

    :ivar num_proc: Number of processors used for peak fitting.
    :type num_proc: int, optional
    :ivar peak_models: Peak model for peak fitting,
        defaults to `'gaussian'`.
    :type peak_models: Literal['gaussian', 'lorentzian']],
        list[Literal['gaussian', 'lorentzian']]], optional
    :ivar rel_height_cutoff: Relative peak height cutoff for
        peak fitting (any peak with a height smaller than
        `rel_height_cutoff` times the maximum height of all peaks 
        gets removed from the fit model), defaults to `None`.
    :type rel_height_cutoff: float, optional
    :ivar tth_file: Path to the file with the 2&theta map.
    :type tth_file: FilePath, optional
    :ivar tth_map: Map of the 2&theta values.
    :type tth_map: numpy.ndarray, optional
    """
    num_proc: Optional[conint(gt=0)] = max(1, os.cpu_count()//4)
    peak_models: Union[
        conlist(min_length=1, item_type=Literal['gaussian', 'lorentzian']),
        Literal['gaussian', 'lorentzian']] = 'gaussian'
    rel_height_cutoff: Optional[
        confloat(gt=0, lt=1.0, allow_inf_nan=False)] = None
#    tth_file: Optional[FilePath] = None
#    tth_map: Optional[np.ndarray] = None

    _calibration_energy_mask_ranges: conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=confloat(allow_inf_nan=False))) = PrivateAttr()

    @model_validator(mode='after')
    def validate_fit_config(self):
        """Set default fit parameters for centers_range, fwhm_min,
        and fwhm_max.

        :return: Validated configuration class.
        :rtype: MCAElementStrainAnalysisConfig
        """
        if self.centers_range is None:
            self.centers_range = 0.25
        if self.fwhm_min is None:
            self.fwhm_min = 0.25
        if self.fwhm_max is None:
            self.fwhm_max = 2.0
        return self

    def add_calibration(self, calibration):
        """Finalize values for some fields using a tth calibration
        MCAElementConfig corresponding to the same detector.

        :param calibration: Existing calibration configuration to use
            by MCAElementStrainAnalysisConfig.
        :type calibration: MCAElementConfig
        """
        for field in ['energy_calibration_coeffs', 'num_bins',
                      'tth_calibrated']:
            setattr(self, field, getattr(calibration, field))
        if self.energy_mask_ranges is None:
            self.energy_mask_ranges = calibration.energy_mask_ranges
        self._calibration_energy_mask_ranges = calibration.energy_mask_ranges

    def get_calibration_mask_ranges(self):
        """Return the `_calibration_energy_mask_ranges` converted from
        channel energies to channel indices.
        """
        if not hasattr(self, '_calibration_energy_mask_ranges'):
            return None

        # Local modules
        from CHAP.utils.general import (
            index_nearest_down,
            index_nearest_up,
        )

        energy_mask_ranges = []
        energies = self.energies
        for e_min, e_max in self._calibration_energy_mask_ranges:
            energy_mask_ranges.append(
                [index_nearest_down(energies, e_min),
                 index_nearest_up(energies, e_max)])
        return energy_mask_ranges

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
        return d


# Processor configuration classes

#class MCAScanDataConfig(BaseModel):
#    """Class representing metadata required to locate raw MCA data for
#    a single scan and construct a mask for it.
#
#    :ivar inputdir: Input directory, used only if any file in the
#        configuration is not an absolute path.
#    :type inputdir: str, optional
#    :ivar spec_file: Path to the SPEC file containing the scan.
#    :type spec_file: str, optional
#    :ivar scan_number: Number of the scan in `spec_file`.
#    :type scan_number: int, optional
#    :ivar par_file: Path to the par file associated with the scan.
#    :type par_file: str, optional
#    :ivar scan_column: Required column name in `par_file`.
#    :type scan_column: str, optional
#    :ivar detectors: List of MCA detector element metadata
#        configurations.
#    :type detectors: list[MCAElementConfig]
#    """
#    inputdir: Optional[DirectoryPath] = None
#    spec_file: Optional[FilePath] = None
#    scan_number: Optional[conint(gt=0)] = None
#    par_file: Optional[FilePath] = None
#    scan_column: Optional[str] = None
#    detectors: conlist(min_length=1, item_type=MCAElementConfig)
#
#    _parfile: Optional[ParFile] = None
#    _scanparser: Optional[ScanParser] = None
#
#    @model_validator(mode='before')
#    @classmethod
#    def validate_scan(cls, data):
#        """Finalize file paths for spec_file and par_file.
#
#        :param data: Pydantic validator data object.
#        :type data: MCAScanDataConfig,
#            pydantic_core._pydantic_core.ValidationInfo
#        :raises ValueError: Invalid SPEC or par file.
#        :return: The validated list of class properties.
#        :rtype: dict
#        """
#        inputdir = data.get('inputdir')
#        spec_file = data.get('spec_file')
#        par_file = data.get('par_file')
#        if spec_file is not None and par_file is not None:
#            raise ValueError('Use either spec_file or par_file, not both')
#        if spec_file is not None:
#            if inputdir is not None and not os.path.isabs(spec_file):
#                data['spec_file'] = os.path.join(inputdir, spec_file)
#        elif par_file is not None:
#            if inputdir is not None and not os.path.isabs(par_file):
#                data['par_file'] = os.path.join(inputdir, par_file)
#            if 'scan_column' not in data:
#                raise ValueError(
#                    'scan_column is required when par_file is used')
#            if isinstance(data['scan_column'], str):
#                parfile = ParFile(par_file)
#                if data['scan_column'] not in parfile.column_names:
#                    raise ValueError(
#                        f'No column named {data["scan_column"]} in '
#                        + '{data["par_file"]}. Options: '
#                        + ', '.join(parfile.column_names))
#        else:
#            raise ValueError('Must use either spec_file or par_file')
#
#        return data
#
#    @model_validator(mode='after')
#    def validate_detectors(self):
#        """Fill in values for _scanparser / _parfile (if applicable).
#        Fill in each detector's num_bins field, if needed.
#        Check each detector's include_energy_ranges field against the
#        flux file, if available.
#
#        :raises ValueError: Unable to obtain a value for num_bins.
#        :return: The validated list of class properties.
#        :rtype: dict
#        """
#        spec_file = self.spec_file
#        par_file = self.par_file
#        detectors = self.detectors
#        flux_file = self.flux_file
#        if spec_file is not None:
#            self._scanparser = ScanParser(
#                spec_file, self.scan_number)
#            self._parfile = None
#        elif par_file is not None:
#            self._parfile = ParFile(par_file)
#            self._scanparser = ScanParser(
#                self._parfile.spec_file,
#                self._parfile.good_scan_numbers()[0])
#        for detector in detectors:
#            if detector.num_bins is None:
#                try:
#                    detector.num_bins = \
#                        self._scanparser.get_num_detector_bins()
#                except Exception as e:
#                    raise ValueError('No value found for num_bins') from e
#        if flux_file is not None:
#            flux = np.loadtxt(flux_file)
#            flux_file_energies = flux[:,0]/1.e3
#            flux_e_min = flux_file_energies.min()
#            flux_e_max = flux_file_energies.max()
#            for detector in detectors:
#                for i, (det_e_min, det_e_max) in enumerate(
#                        deepcopy(detector.include_energy_ranges)):
#                    if det_e_min < flux_e_min or det_e_max > flux_e_max:
#                        energy_range = [float(max(det_e_min, flux_e_min)),
#                                        float(min(det_e_max, flux_e_max))]
#                        print(
#                            f'WARNING: include_energy_ranges[{i}] out of range'
#                            f' ({detector.include_energy_ranges[i]}): adjusted'
#                            f' to {energy_range}')
#                        detector.include_energy_ranges[i] = energy_range
#
#        return self
#
#    @property
#    def scanparser(self):
#        """Return the scanparser."""
#        try:
#            scanparser = self._scanparser
#        except:
#            scanparser = ScanParser(self.spec_file, self.scan_number)
#            self._scanparser = scanparser
#        return scanparser
#
#    def dict(self, *args, **kwargs):
#        """Return a representation of this configuration in a
#        dictionary that is suitable for dumping to a YAML file.
#
#        :return: Dictionary representation of the configuration.
#        :rtype: dict
#        """
#        d = super().dict(*args, **kwargs)
#        for k, v in d.items():
#            if isinstance(v, PosixPath):
#                d[k] = str(v)
#        if d.get('_parfile') is None:
#            del d['par_file']
#            del d['scan_column']
#        else:
#            del d['spec_file']
#            del d['scan_number']
#        for k in ('_scanparser', '_parfile', 'inputdir'):
#            if k in d:
#                del d[k]
#        return d
#
#
#class DiffractionVolumeLengthConfig(MCAScanDataConfig):
#    """Class representing metadata required to perform a diffraction
#    volume length calculation for an EDD setup using a steel-foil
#    raster scan.
#
#    :ivar sample_thickness: Thickness of scanned foil sample. Quantity
#        must be provided in the same units as the values of the
#        scanning motor.
#    :type sample_thickness: float
#    :ivar detectors: Individual detector element DVL measurement
#        configurations.
#    :type detectors: list[MCAElementDiffractionVolumeLengthConfig]
#    """
#    sample_thickness: float
#    detectors: conlist(
#        min_length=1, item_type=MCAElementDiffractionVolumeLengthConfig)
#
#    @property
#    def scanned_vals(self):
#        """Return the list of values visited by the scanning motor
#        over the course of the raster scan.
#
#        :return: List of scanned motor values.
#        :rtype: numpy.ndarray
#        """
#        if self._parfile is not None:
#            return self._parfile.get_values(
#                self.scan_column,
#                scan_numbers=self._parfile.good_scan_numbers())
#        return self.scanparser.spec_scan_motor_vals[0]


class MCACalibrationConfig(FitConfig):
    """Base class representing metadata required to perform an energy
    or 2&theta calibration of an MCA detector.

    :ivar inputdir: Input directory, used only if any file in the
        configuration is not an absolute path.
    :type inputdir: str, optional
    :ivar flux_file: File name of the csv flux file containing station
        beam energy in eV (column 0) versus flux (column 1).
    :type flux_file: str, optional
    :ivar materials: Material configurations for the calibration,
        defaults to [`Ceria`].
    :type materials: list[MaterialConfig], optional
    :ivar scan_step_indices: Optional scan step indices to use for the
        calibration. If not specified, the calibration will be
        performed on the average of all MCA spectra for the scan.
    :type scan_step_indices: int, str, list[int], optional

    Note: Fluorescence data:
        https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
    """
    inputdir: Optional[DirectoryPath] = None
    flux_file: Optional[FilePath] = None
    materials: Optional[conlist(item_type=MaterialConfig)] = [MaterialConfig(
        material_name='CeO2', lattice_parameters=5.41153, sgnum=225)]
    scan_step_indices: Optional[Annotated[conlist(
        min_length=1, item_type=conint(ge=0)),
        Field(validate_default=True)]] = None

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        flux_file filepath.

        :param data: Pydantic validator data object.
        :type data: MCACalibrationConfig,
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

    @model_validator(mode='after')
    def validate_fit_config(self):
        """Set default fit parameters for centers_range, fwhm_min,
        and fwhm_max.

        :return: Validated configuration class.
        :rtype: FitConfig
        """
        if self.centers_range is None:
            self.centers_range = 20
        if self.fwhm_min is None:
            self.fwhm_min = 3
        if self.fwhm_max is None:
            self.fwhm_max = 25
        return self

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Validate the specified list of scan numbers.

        :ivar scan_step_indices: Optional scan step indices to use for
            the calibration. If not specified, the calibration will be
            performed on the average of all MCA spectra for the scan.
        :type scan_step_indices: int, str, list[int], optional
        :raises ValueError: Invalid experiment type.
        :return: List of step indices.
        :rtype: list[int]
        """
        if isinstance(scan_step_indices, int):
            scan_step_indices = [scan_step_indices]
        elif isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)
        return scan_step_indices

    def flux_file_energy_range(self):
        """Get the energy range in the flux correction file.

        :return: The energy range in the flux correction file.
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

    def update_detectors(self):
        for detector in self.detectors:
            for k, v in self:
                if hasattr(detector, k) and getattr(detector, k) is None:
                    setattr(detector, k, v)

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.
        Exclude inputdir and the inherited FitConfig parameters.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        if 'inputdir' in d:
            del d['inputdir']
        for k in vars(FitConfig()).keys():
            del d[k]
        for k, v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        return d


class MCAEnergyCalibrationConfig(MCACalibrationConfig):
    """Base class representing metadata required to perform an energy
    calibration of an MCA detector.

    :ivar detectors: List of individual MCA detector element
        calibration configurations.
    :type detectors: list[MCAElementConfig], optional
    :ivar max_energy_kev: Maximum channel energy of the MCA in
        keV, defaults to `200.0`.
    :type max_energy_kev: float, optional
    :ivar max_peak_index: Index of the peak in `peak_energies`
        with the highest amplitude.
    :type max_peak_index: int
    :ivar peak_energies: Theoretical locations of peaks in keV to use
        for calibrating the MCA channel energies. It is _strongly_
        recommended to use fluorescence peaks for the energy
        calibration.
    :type peak_energies: list[float]
    """
    detectors: Optional[conlist(item_type=MCAElementConfig)] = None
    max_energy_kev: Optional[confloat(gt=0, allow_inf_nan=False)] = 200.0
    max_peak_index: conint(ge=0)
    peak_energies: conlist(
        min_length=2, item_type=confloat(gt=0, allow_inf_nan=False))

    @model_validator(mode='after')
    def validate_detectors(self):
        """Validate the detector (energy) mask ranges.

        :return: Updated energy calibration configuration class.
        :rtype: MCAEnergyCalibrationConfig
        """
        if self.detectors is None:
            return self
        warning = False
        if self.energy_mask_ranges:
            self.energy_mask_ranges = None
            warning = True
        for detector in self.detectors:
            if detector.energy_mask_ranges:
                detector.energy_mask_ranges = None
                warning = True
        if warning:
            print('Ignoring energy_mask_ranges parameter for energy '
                  'calibration')
        self.update_detectors()
        return self

    @model_validator(mode='after')
    def validate_max_peak_index(self):
        """Validate the specified index of the XRF peak with the
        highest amplitude against the number of peak energies.

        :return: Validated energy calibration configuration class.
        :rtype: MCAEnergyCalibrationConfig
        """
        if not 0 <= self.max_peak_index < len(self.peak_energies):
            raise ValueError('max_peak_index out of bounds')
        return self


class MCATthCalibrationConfig(MCACalibrationConfig):
    """Class representing metadata required to perform a 2&theta
    calibration of an MCA detector.

    :ivar detectors: List of individual MCA detector element
        calibration configurations.
    :type detectors: list[MCAElementConfig], optional
    :ivar quadratic_energy_calibration: Adds a quadratic term to
        the detector channel index to energy conversion, defaults
        to `False` (linear only).
    :type quadratic_energy_calibration: bool, optional
    :ivar tth_initial_guess: Initial guess for 2&theta.
    :type tth_initial_guess: float, optional
    """
    detectors: Optional[conlist(item_type=MCAElementConfig)] = None
    quadratic_energy_calibration: bool = False
    tth_initial_guess: Optional[confloat(gt=0, allow_inf_nan=False)] = None

    @model_validator(mode='after')
    def validate_detectors(self):
        """Validate the detector (energy) mask ranges.

        :return: Updated tth calibration configuration class.
        :rtype: MCATthCalibrationConfig
        """
        if self.detectors is None:
            return self
        warning = False
        if self.mask_ranges:
            self.mask_ranges = None
            warning = True
        for detector in self.detectors:
            if detector.mask_ranges:
                detector.mask_ranges = None
                warning = True
        if warning:
            print('Ignoring mask_ranges parameter for tth calibration')
        self.update_detectors()
        return self

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

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        if 'tth_initial_guess' in d:
            del d['tth_initial_guess']
        return d


class StrainAnalysisConfig(MCACalibrationConfig):
    """Class representing input parameters required to perform a
    strain analysis.

    :ivar detectors: List of individual detector element strain
        analysis configurations, defaults to `None` (use all).
    :type detectors: list[MCAElementStrainAnalysisConfig], optional
    :ivar find_peaks: Exclude peaks where the average spectrum
        is below the `rel_height_cutoff` (in the detector
        configuration) cutoff relative to the maximum value of the
        average spectrum, defaults to `True`.
    :type find_peaks: bool, optional
    :ivar oversampling: FIX
    :type oversampling: FIX
    :ivar skip_animation: Skip the animation and plotting of
        the strain analysis fits, defaults to `False`.
    :type skip_animation: bool, optional
    :ivar sum_axes: Whether to sum over the fly axis or not
        for EDD scan types not 0, defaults to `True`.
    :type sum_axes: Union[bool, list[str]], optional
    """
    detectors: Optional[conlist(
        min_length=1, item_type=MCAElementStrainAnalysisConfig)] = None
    find_peaks: Optional[bool] = True
    oversampling: Optional[
        Annotated[Dict, Field(validate_default=True)]] = {'num': 10}
    skip_animation: Optional[bool] = False
    sum_axes: Optional[
        Union[bool, conlist(min_length=1, item_type=str)]] = True

    @model_validator(mode='after')
    def validate_detectors(self):
        """Validate the detector (energy) mask ranges.

        :return: Updated strain analysis configuration class.
        :rtype: StrainAnalysisConfig
        """
        if self.detectors is None:
            return self
        warning = False
        if self.mask_ranges:
            self.mask_ranges = None
            warning = True
        for detector in self.detectors:
            if detector.mask_ranges:
                detector.mask_ranges = None
                warning = True
        if warning:
            print('Ignoring mask_ranges parameter for strain analysis')
        self.update_detectors()
        return self

    @model_validator(mode='after')
    def validate_fit_config(self):
        """Set default fit parameters for centers_range, fwhm_min,
        and fwhm_max.

        :return: Validated configuration class.
        :rtype: StrainAnalysisConfig
        """
        if self.centers_range is None:
            self.centers_range = 0.25
        if self.fwhm_min is None:
            self.fwhm_min = 0.25
        if self.fwhm_max is None:
            self.fwhm_max = 2.0
        return self

#    @field_validator('detectors', mode='before')
#    @classmethod
#    def validate_detectors(cls, detectors, info):
#        """Finalize value for tth_file for each detector."""
#        inputdir = info.data.get('inputdir')
#        for detector in detectors:
#            tth_file = detector.get('tth_file')
#            if tth_file is not None:
#                if not os.path.isabs(tth_file):
#                    detector['tth_file'] = os.path.join(inputdir, tth_file)
#        return detectors

# FIX tth_file/tth_map not updated
#    @field_validator('detectors')
#    @classmethod
#    def validate_detectors(cls, detectors, info):
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

