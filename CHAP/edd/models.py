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
import numpy as np
from hexrd.material import Material
from pydantic import (
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
from CHAP.models import CHAPBaseModel
from CHAP.common.models.map import Detector
from CHAP.utils.models import Multipeak
#from CHAP.utils.parfile import ParFile


# Baseline configuration class

class BaselineConfig(CHAPBaseModel):
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

class FitConfig(CHAPBaseModel):
    """Fit parameters configuration class for peak fitting.

    :ivar background: Background model for peak fitting, defaults
        to `constant`.
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
    :ivar backgroundpeaks: Additional background peaks (their
        associated fit parameters in units of keV).
    :type backgroundpeaks: CHAP.utils.models.Multipeak, optional
    """
    background: Optional[conlist(item_type=constr(
        strict=True, strip_whitespace=True, to_lower=True))] = ['constant']
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
    backgroundpeaks: Optional[Multipeak] = None

    _default_centers_range: bool = PrivateAttr(default=False)
    _default_fwhm_min: bool = PrivateAttr(default=False)
    _default_fwhm_max: bool = PrivateAttr(default=False)

    @field_validator('background', mode='before')
    @classmethod
    def validate_background(cls, background):
        """Validate the background model.

        :ivar background: Background model for peak fitting.
        :type background: str, list[str], optional
        :return: List of validated background models.
        :rtype: list[str]
        """
        if background is None:
            return background
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

    def get_privateattr(self, field):
        """Return the default value for a model field if it has an
        associated default private attribute. 

        :ivar field: Model field name.
        :type field: str
        :return: Associated default private attribute if defined,
            otherwise `None`.
        :rtype: Any
        """
        if hasattr(self, f'_default_{field}'):
            return getattr(self, f'_default_{field}')
        return None

# Material configuration class

class MaterialConfig(CHAPBaseModel):
    """Sample material parameters configuration class.

    :ivar material_name: Sample material name.
    :type material_name: str, optional
    :ivar lattice_parameters: Lattice spacing(s) in angstroms.
    :type lattice_parameters: float, list[float], optional
    :ivar sgnum: Space group of the material.
    :type sgnum: int, optional
    """
    #RV FIX create a getter for lattice_parameters that always returns a list?
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

    _energy_calibration_mask_ranges: conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=conint(ge=0))) = PrivateAttr()
    _hkl_indices: list = PrivateAttr()

    @model_validator(mode='after')
    def validate_fitconfig(self):
        """Set the defaults for `FitConfig` parameters `centers_range`,
        `fwhm_min` and `fwhm_max`.

        :return: Updated `centers_range`, `fwhm_min` and `fwhm_max`
            parameters.
        :rtype: MCAEnergyCalibrationConfig
        """
        if self.centers_range is None:
            self.centers_range = 20
            self._default_centers_range = True
        if self.fwhm_min is None:
            self.fwhm_min = 3
            self._default_fwhm_min = True
        if self.fwhm_max is None:
            self.fwhm_max = 25
            self._default_fwhm_max = True
        return self

    def add_calibration(self, calibration):
        """Finalize values for some fields using a calibration
        MCAElementConfig corresponding to the same detector.

        :param calibration: Existing calibration configuration.
        :type calibration: MCAElementConfig
        """
        for field in ['energy_calibration_coeffs', 'num_bins',
                      '_energy_calibration_mask_ranges']:
            setattr(self, field, deepcopy(getattr(calibration, field)))
        if self.tth_calibrated is not None:
            self.logger.warning(
                'Ignoring tth_calibrated in calibration configuration')
            self.tth_calibrated = None

    @property
    def energies(self):
        """Return calibrated bin energies."""
        a, b, c = tuple(self.energy_calibration_coeffs)
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
            [float(energies[i]) for i in range_]
             for range_ in sorted([sorted(v) for v in mask_ranges])]

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


class MCAElementDiffractionVolumeLengthConfig(MCAElementConfig):
    """Class representing metadata required to perform a diffraction
    volume length measurement for a single MCA detector element.

    :ivar dvl: Measured diffraction volume length.
    :type dvl: float, optional
    :ivar fit_amplitude: Amplitude of the Gaussian fit.
    :type fit_amplitude: float, optional
    :ivar fit_center: Center of the Gaussian fit.
    :type fit_center: float, optional
    :ivar fit_sigma: Sigma of the Gaussian fit.
    :type fit_sigma: float, optional
    """
    dvl: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    fit_amplitude: Optional[float] = None
    fit_center: Optional[float] = None
    fit_sigma: Optional[float] = None


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

    def add_calibration(self, calibration):
        """Finalize values for some fields using a tth calibration
        MCAElementConfig corresponding to the same detector.

        :param calibration: Existing calibration configuration to use
            by MCAElementStrainAnalysisConfig.
        :type calibration: MCAElementConfig
        """
        for field in ['energy_calibration_coeffs', 'num_bins',
                      'tth_calibrated']:
            setattr(self, field, deepcopy(getattr(calibration, field)))
        if self.energy_mask_ranges is None:
            self.energy_mask_ranges = deepcopy(calibration.energy_mask_ranges)
        self._calibration_energy_mask_ranges = deepcopy(
            calibration.energy_mask_ranges)

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


# Processor configuration classes

class DiffractionVolumeLengthConfig(FitConfig):
    """Class representing metadata required to perform a diffraction
    volume length calculation for an EDD setup using a steel-foil
    raster scan.

    :ivar detectors: List of individual MCA detector element
        DVL measurement configurations.
    :type detectors: list[MCAElementDiffractionVolumeLengthConfig]
    :ivar max_energy_kev: Maximum channel energy of the MCA in
        keV, defaults to `200.0`.
    :type max_energy_kev: float, optional
    :ivar measurement_mode: Placeholder for recording whether the
        measured DVL value was obtained through the automated
        calculation or a manual selection, defaults to `'auto'`.
    :type measurement_mode: Literal['manual', 'auto'], optional
    :ivar sample_thickness: Thickness of scanned foil sample. Quantity
        must be provided in the same units as the values of the
        scanning motor.
    :type sample_thickness: float
    :ivar sigma_to_dvl_factor: The DVL is obtained by fitting a reduced
        form of the MCA detector data. `sigma_to_dvl_factor` is a
        scalar value that converts the standard deviation of the
        gaussian fit to the measured DVL, defaults to `3.5`.
    :type sigma_to_dvl_factor: Literal[2.0, 3.5, 4.0], optional
    """
    detectors: Optional[
        conlist(item_type=MCAElementDiffractionVolumeLengthConfig)] = None
    max_energy_kev: Optional[confloat(gt=0, allow_inf_nan=False)] = 200.0
    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
    sample_thickness: confloat(gt=0, allow_inf_nan=False)
    sigma_to_dvl_factor: Optional[Literal[2.0, 3.5, 4.0]] = 3.5

    _exclude = set(vars(FitConfig()).keys())

    @model_validator(mode='after')
    def validate_config(self):
        """Update the configuration with costum defaults after the
        normal native pydantic validation.

        :return: Updated energy calibration configuration class.
        :rtype: DiffractionVolumeLengthConfig
        """
        if self.detectors is not None:
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
        if self.measurement_mode == 'manual':
            self._exclude |= {'sigma_to_dvl_factor'}
        return self

#    def update_detectors(self):
#        """Update any detector configuration parameters not superseded
#        by individual detector values.
#        """
#        if self.detectors is not None:
#            for detector in self.detectors:
#                for k in self.__dict__:
#                    if hasattr(detector, k) and getattr(detector, k) is None:
#                        setattr(detector, k, deepcopy(getattr(self, k)))


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
    :ivar peak_energies: Theoretical locations of the fluorescence
        peaks in keV to use for calibrating the MCA channel energies.
    :type peak_energies: list[float], optional for energy calibration
    :ivar scan_step_indices: Optional scan step indices to use for the
        calibration. If not specified, the calibration will be
        performed on the average of all MCA spectra for the scan.
    :type scan_step_indices: int, str, list[int], optional

    Note: Fluorescence data:
        https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
    """
    inputdir: Optional[DirectoryPath] = Field(None, exclude=True)
    flux_file: Optional[FilePath] = None
    materials: Optional[conlist(item_type=MaterialConfig)] = [MaterialConfig(
        material_name='CeO2', lattice_parameters=5.41153, sgnum=225)]
    peak_energies: Optional[conlist(
        min_length=2, item_type=confloat(gt=0, allow_inf_nan=False))] = None
    scan_step_indices: Optional[Annotated[conlist(
        min_length=1, item_type=conint(ge=0)),
        Field(validate_default=True)]] = None

    _exclude = set(vars(FitConfig()).keys())

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
        if isinstance(data, dict):
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
        """Update any detector configuration parameters not superseded
        by individual detector values.
        """
        for detector in self.detectors:
            for k in self.__dict__:
                if hasattr(detector, k):
                    v = getattr(self, k)
                    have_default = detector.get_privateattr(k)
                    if have_default is None and v is not None:
                        setattr(detector, k, deepcopy(v))
                    elif have_default and v is not None:
                        setattr(detector, k, deepcopy(v))
                        detector._default_centers_range = False


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
    """
    detectors: Optional[conlist(item_type=MCAElementConfig)] = None
    max_energy_kev: Optional[confloat(gt=0, allow_inf_nan=False)] = 200.0
    max_peak_index: conint(ge=0)

    @model_validator(mode='after')
    def validate_detectors(self):
        """Validate the detector (energy) mask ranges and update any
        detector configuration parameters not superseded by their
        individual values.

        :return: Updated energy calibration configuration class.
        :rtype: MCAEnergyCalibrationConfig
        """
        if self.detectors is None:
            return self
        warning = False
        if self.energy_mask_ranges:
            self.energy_mask_ranges = None
            warning = True
        if self.detectors is not None:
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
    def validate_peak_energies(self):
        """Validate the specified index of the XRF peak with the
        highest amplitude against the number of peak energies.

        :return: Validated energy calibration configuration class.
        :rtype: MCAEnergyCalibrationConfig
        """
        if self.peak_energies is None:
            raise ValueError('peak_energies is required')
        if not 0 <= self.max_peak_index < len(self.peak_energies):
            raise ValueError('max_peak_index out of bounds')
        return self


class MCATthCalibrationConfig(MCACalibrationConfig):
    """Class representing metadata required to perform a 2&theta
    calibration of an MCA detector.

    :ivar calibration_method: Type of calibration method,
        defaults to `'direct_fit_bragg'`.
    :type calibration_method:
        Literal['direct_fit_bragg', 'direct_fit_tth_ecc'], optional
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
    calibration_method: Optional[Literal[
        'direct_fit_bragg', 'direct_fit_tth_ecc']] = 'direct_fit_bragg'
    detectors: Optional[conlist(item_type=MCAElementConfig)] = None
    quadratic_energy_calibration: Optional[bool] = False
    tth_initial_guess: Optional[
        confloat(gt=0, allow_inf_nan=False)] = Field(None, exclude=True)

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
        if self.detectors is not None:
            for detector in self.detectors:
                if detector.mask_ranges:
                    detector._energy_calibration_mask_ranges = deepcopy(
                        detector.mask_ranges)
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
        """Validate the detector (energy) mask ranges, set the defaults
        for `FitConfig` parameters `centers_range`, `fwhm_min` and
        `fwhm_max` and update any detector configuration parameters
        not superseded by their individual values.


        :return: Updated `centers_range`, `fwhm_min` and `fwhm_max`
            parameters.

        :return: Updated strain analysis configuration class.
        :rtype: StrainAnalysisConfig
        """
        if self.centers_range is None:
            self.centers_range = 2.0
            self._default_centers_range = True
        if self.fwhm_min is None:
            self.fwhm_min = 0.25
            self._default_fwhm_min = True
        if self.fwhm_max is None:
            self.fwhm_max = 2.0
            self._default_fwhm_max = True
        if self.detectors is None:
            return self
        warning = False
        if self.mask_ranges:
            self.mask_ranges = None
            warning = True
        if self.detectors is not None:
            for detector in self.detectors:
                if detector.mask_ranges:
                    detector.mask_ranges = None
                    warning = True
        if warning:
            print('Ignoring mask_ranges parameter for strain analysis')
        self.update_detectors()
        return self

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
