"""`Pydantic <https://github.com/pydantic/pydantic>`__ model
configuration classes unique to the the EDD workflow.
"""

# System modules
from copy import deepcopy
import os
import typing
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
#from hexrd.material import Material
from CHAP.utils.material import Material
from pydantic import (
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

    :ivar attrs: Additional baseline model configuration attributes.
    :vartype attrs: dict, optional
    :ivar lam: &lambda (smoothness) parameter (the balance between the
        residual of the data and the baseline and the smoothness of the
        baseline). The suggested range is between 100 and 10^8,
        defaults to `10^6`.
    :vartype lam: float, optional
    :ivar max_iter: Maximum number of iterations,
        defaults to `100`.
    :vartype max_iter: int, optional
    :ivar tol: Convergence tolerence, defaults to `1.e-6`.
    :vartype tol: float, optional
    """

    attrs: Optional[dict] = {}
    lam: confloat(gt=0, allow_inf_nan=False) = 1.e6
    max_iter: conint(gt=0) = 100
    tol: confloat(gt=0, allow_inf_nan=False) = 1.e-6


# Fit configuration class

class _FitConfig(CHAPBaseModel):
    """Fit parameters configuration class for peak fitting.

    :ivar background: Background model for peak fitting, defaults
        to `constant`.
    :vartype background: str, list[str], optional
    :ivar baseline: Automated baseline subtraction configuration,
        defaults to `False`.
    :vartype baseline: bool or BaselineConfig, optional
    :ivar centers_range: Peak centers range for peak fitting.
        The allowed range for the peak centers will be the initial
        values &pm; `centers_range` (in MCA channels for calibration
        or keV for strain analysis). Defaults to `20` for calibration
        and `2.0` for strain analysis.
    :vartype centers_range: float, optional
    :ivar energy_mask_ranges: MCA energy mask ranges in keV for
        selecting the data to be included after applying a mask (bounds
        are inclusive). Specify either energy_mask_ranges or
        mask_ranges, not both.
    :vartype energy_mask_ranges: list[[float, float]], optional
    :ivar fwhm_min: Minimum FWHM for peak fitting (in MCA channels
        for calibration or keV for strain analysis). Defaults to `3`
        for calibration and `0.25` for strain analysis.
    :vartype fwhm_min: float, optional
    :ivar fwhm_max: Maximum FWHM for peak fitting (in MCA channels
        for calibration or keV for strain analysis). Defaults to `25`
        for calibration and `2.0` for strain analysis.
    :vartype fwhm_max: float, optional
    :ivar mask_ranges: MCA channel bin ranges for selecting the data
        to be included in the energy calibration after applying a mask
        (bounds are inclusive). Specify for energy calibration only.
    :vartype mask_ranges: list[[int, int]], optional
    :ivar backgroundpeaks: Additional background peaks (their
        associated fit parameters in units of keV).
    :vartype backgroundpeaks: Multipeak, optional
    """

    background: Optional[conlist(item_type=constr(
        strict=True, strip_whitespace=True, to_lower=True))] = ['constant']
    baseline: Optional[Union[bool, BaselineConfig]] = None
    centers_range: Optional[confloat(gt=0, allow_inf_nan=False)] = 20
    energy_mask_ranges: Optional[conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=confloat(allow_inf_nan=False)))] = None
    fwhm_min: Optional[confloat(gt=0, allow_inf_nan=False)] = 3
    fwhm_max: Optional[confloat(gt=0, allow_inf_nan=False)] = 25
    mask_ranges: Optional[conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=conint(ge=0)))] = None
    backgroundpeaks: Optional[Multipeak] = None

    @field_validator('background', mode='before')
    @classmethod
    def validate_background(cls, background):
        """Validate the background model.

        :param background: Background model for peak fitting.
        :type background: str or list[str], optional
        :return: Validated background models.
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

        :param baseline: Automated baseline subtraction configuration.
        :type baseline: BaselineConfig, optional
        :return: Validated baseline subtraction configuration.
        :rtype: BaselineConfig or None
        """
        if isinstance(baseline, bool) and baseline:
            return BaselineConfig()
        return baseline

    @field_validator('energy_mask_ranges', mode='before')
    @classmethod
    def validate_energy_mask_ranges(cls, energy_mask_ranges):
        """Validate the mask ranges for selecting the data to include.

        :param energy_mask_ranges: MCA energy mask ranges in keV for
            selecting the data to be included after applying a mask
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

        :param mask_ranges: MCA channel bin ranges for selecting the
            data to be included after applying a mask (bounds are
            inclusive).
        :type mask_ranges: list[[int, int]], optional
        :return: Validated mask ranges.
        :rtype: list[[int, int]]
        """
        if mask_ranges:
            return sorted([sorted(v) for v in mask_ranges])
        return mask_ranges


# Material configuration class

class MaterialConfig(CHAPBaseModel):
    """Sample material parameters configuration class.

    :ivar material_name: Sample material name.
    :vartype material_name: str, optional
    :ivar lattice_parameters: Lattice spacing(s) in angstroms.
    :vartype lattice_parameters: float, list[float], optional
    :ivar sgnum: Space group of the material.
    :vartype sgnum: int, optional
    :ivar dmin: Minimum d-spacing for selecting the available HKLs,
        defaults to 0.35.
    :vartype dmin: float, optional
    """

    #RV FIX create a getter for lattice_parameters that always returns a list?
    material_name: Optional[constr(strip_whitespace=True, min_length=1)] = None
    lattice_parameters: Optional[Union[
        confloat(gt=0, allow_inf_nan=False),
        conlist(
            min_length=1, max_length=6,
            item_type=confloat(gt=0, allow_inf_nan=False))]] = None
    sgnum: Optional[conint(ge=0)] = None
    dmin: Optional[confloat(gt=0, allow_inf_nan=False)] = 0.35

    _material: Optional[Material]

    @model_validator(mode='after')
    def validate_materialconfig_after(self):
        """Create and validate the private attribute _material.

        :return: Validated configuration class.
        :rtype: MaterialConfig
        """
#        self._material = make_material(
#            self.material_name, self.sgnum, self.lattice_parameters, self.dmin)
        self._material = Material.make_material(
            self.material_name, sgnum=self.sgnum,
            lattice_parameters_angstroms=self.lattice_parameters)
#            pos=['4a', '8c'])
            #pos=[(0,0,0), (1/4, 1/4, 1/4), (3/4, 3/4, 3/4)])
        self.lattice_parameters = list([
            x.getVal('angstrom') if x.isLength() else x.getVal('radians')
            for x in self._material._lparms])
        return self


# Detector configuration classes

class MCADetectorCalibration(Detector, _FitConfig):
    """Class representing the configuration for a single MCA detector
    element to perform detector calibration.

    :ivar energy_calibration_coeffs: Detector channel index to energy
        polynomial conversion coefficients ([a, b, c] with
        E_i = a*i^2 + b*i + c).
    :vartype energy_calibration_coeffs:
        list[float, float, float], optional
    :ivar num_bins: Number of MCA channels.
    :vartype num_bins: int, optional
    :ivar tth_max: Detector rotation about lab frame x axis.
    :vartype tth_max: float, optional
    :ivar tth_tol: Minimum resolvable difference in 2&theta between
        two unique Bragg peaks,
    :vartype tth_tol: float, optional
    :ivar tth_calibrated: Calibrated value for 2&theta.
    :vartype tth_calibrated: float, optional
    :ivar tth_initial_guess: Initial guess for 2&theta superseding
        the global one in
        :class:`~CHAP.edd.models.MCATthCalibrationConfig`.
    :vartype tth_initial_guess: float, optional
    """

    processor_type: Literal['calibration']
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

#    def add_calibration(self, calibration):
#        """Finalize values for some fields using a calibration
#        MCADetectorCalibration corresponding to the same detector.
#
#        :param calibration: Existing calibration configuration.
#        :type calibration: MCADetectorCalibration
#        """
#        raise RuntimeError('To do')
#        for field in ['energy_calibration_coeffs', 'num_bins',
#                      '_energy_calibration_mask_ranges']:
#            setattr(self, field, deepcopy(getattr(calibration, field)))
#        if self.tth_calibrated is not None:
#            self.logger.warning(
#                'Ignoring tth_calibrated in calibration configuration')
#            self.tth_calibrated = None

    @property
    def energies(self):
        """Return the calibrated bin energies.

        :type: numpy.ndarray
        """
        a, b, c = tuple(self.energy_calibration_coeffs)
        channel_bins = np.arange(self.num_bins)
        return (a*channel_bins + b)*channel_bins + c

    @property
    def hkl_indices(self):
        """Return the HKL indices consistent with the selected energy
        ranges (include_energy_ranges).

        :type: list
        """
        if hasattr(self, '_hkl_indices'):
            return self._hkl_indices
        return []

    @hkl_indices.setter
    def hkl_indices(self, hkl_indices):
        """Set the HKL indices.

        :param hkl_indices: HKL indices.
        :type: list
        """
        self._hkl_indices = hkl_indices

    def convert_mask_ranges(self, mask_ranges):
        """Given a list of mask ranges in channel bins, set the
        corresponding list of channel energy mask ranges.

        :param mask_ranges: Mask ranges to convert to energy mask
            ranges.
        :type mask_ranges: list[[int, int]]
        """
        energies = self.energies
        self.energy_mask_ranges = [
            [float(energies[i]) for i in range_]
             for range_ in sorted([sorted(v) for v in mask_ranges])]

    def get_mask_ranges(self):
        """Return the list of mask ranges if set or convert the
        energy mask ranges from channel energies to channel indices
        and return those.

        :type: list[[float, float]]
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

    def set_energy_calibration_mask_ranges(self):
        """Set the value of the private attribite
        `_energy_calibration_mask_ranges` to value of `mask_ranges`.
        """
        self._energy_calibration_mask_ranges = deepcopy(self.mask_ranges)


class MCADetectorDiffractionVolumeLength(MCADetectorCalibration):
    """Class representing the configuration for a single MCA detector
    element to perform a diffraction volume length measurement.

    :ivar dvl: Measured diffraction volume length.
    :vartype dvl: float, optional
    :ivar fit_amplitude: Amplitude of the Gaussian fit.
    :vartype fit_amplitude: float, optional
    :ivar fit_center: Center of the Gaussian fit.
    :vartype fit_center: float, optional
    :ivar fit_sigma: Sigma of the Gaussian fit.
    :vartype fit_sigma: float, optional
    """

    processor_type: Literal['diffractionvolumelength']
    dvl: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    fit_amplitude: Optional[float] = None
    fit_center: Optional[float] = None
    fit_sigma: Optional[float] = None


class MCADetectorStrainAnalysis(MCADetectorCalibration):
    """Class representing the configuration to perform a strain
    analysis.

    :ivar centers_range: Peak centers range for peak fitting.
        The allowed range for the peak centers will be the initial
        values &pm; `centers_range` (in keV), defaults to `2.0`.
    :vartype centers_range: float, optional
    :ivar fwhm_min: Minimum FWHM for peak fitting (in keV),
        defaults to `0.25`.
    :vartype fwhm_min: float, optional
    :ivar fwhm_max: Maximum FWHM for peak fitting (in keV),
        defaults to `2.0`.
    :vartype fwhm_max: float, optional
    :ivar peak_models: Peak model(s) for peak fitting,
        defaults to `'gaussian'`.
    :vartype peak_models: Literal['gaussian', 'lorentzian', 'pvoigt']],
        list[Literal['gaussian', 'lorentzian', 'pvoigt']]], optional
    :ivar rel_height_cutoff: Relative peak height cutoff for
        peak fitting (any peak with a height smaller than
        `rel_height_cutoff` times the maximum height of all peaks 
        gets removed from the fit model), defaults to `None`.
    :vartype rel_height_cutoff: float, optional
    :ivar tth_map: Map of the 2&theta values.
    :vartype tth_map: numpy.ndarray, optional
    """

    #:ivar tth_file: Path to the file with the 2&theta map.
    #:vartype tth_file: FilePath, optional

    centers_range: Optional[confloat(gt=0, allow_inf_nan=False)] = 2
    fwhm_min: Optional[confloat(gt=0, allow_inf_nan=False)] = 0.25
    fwhm_max: Optional[confloat(gt=0, allow_inf_nan=False)] = 2.0
    processor_type: Literal['strainanalysis']
    peak_models: Union[
        conlist(
            min_length=1,
            item_type=Literal['gaussian', 'lorentzian', 'pvoigt']),
        Literal['gaussian', 'lorentzian', 'pvoigt']] = 'gaussian'
    rel_height_cutoff: Optional[
        confloat(gt=0, lt=1.0, allow_inf_nan=False)] = None
#    tth_file: Optional[FilePath] = None
    tth_map: Optional[np.ndarray] = None

    _calibration_energy_mask_ranges: conlist(
        min_length=1,
        item_type=conlist(
            min_length=2,
            max_length=2,
            item_type=confloat(allow_inf_nan=False))) = PrivateAttr()

    @field_validator('peak_models')
    @classmethod
    def validate_peak_models(cls, peak_models):
        """Validate the specified peak_models.

        :param peak_models: Peak model(s) for peak fitting.
        :type peak_models:
            Literal['gaussian', 'lorentzian', 'pvoigt']] or
            list[Literal['gaussian', 'lorentzian', 'pvoigt']]],
            optional
        :type peak_models:
        :return: Validated peak_models
        :rtype: Literal['gaussian', 'lorentzian', 'pvoigt']] or
            list[Literal['gaussian', 'lorentzian', 'pvoigt']]]
        """
        if isinstance(peak_models, list):
            raise NotImplementedError(
                'Multiple peak models not yet implemented')
        return peak_models

    def add_calibration(self, calibration):
        """Transfer certain 2&theta calibration parameters for use by
        :class:`~CHAP.edd.processor.LatticeParameterRefinementProcessor`
        or :class:`~CHAP.edd.processor.StrainAnalysisProcessor`.

        :param calibration: Existing calibration configuration.
        :type calibration: MCADetectorCalibration
        """
        for field in ['energy_calibration_coeffs', 'num_bins',
                      'tth_calibrated']:
            setattr(self, field, deepcopy(getattr(calibration, field)))
        if self.energy_mask_ranges is None:
            self.energy_mask_ranges = deepcopy(calibration.energy_mask_ranges)
        self._calibration_energy_mask_ranges = deepcopy(
            calibration.energy_mask_ranges)

    def get_calibration_mask_ranges(self):
        """Return the MCA channel bin ranges for the data used during
        the 2&theta calibration.

        :type: list[[int, int]]
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

        :param map_shape: Shape of the suplied 2&theta map.
        :type map_shape: tuple
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


MCADetector = Annotated[
    Union[
        MCADetectorCalibration,
        MCADetectorDiffractionVolumeLength,
        MCADetectorStrainAnalysis],
    Field(discriminator='processor_type')
]


class MCADetectorConfig(_FitConfig):
    """Class representing metadata required to configure a full MCA
    detector.

    :ivar detectors: Individual MCA detector elements.
    :vartype detectors: list[MCADetector], optional
    """

    processor_type: Literal[
        'calibration', 'diffractionvolumelength', 'strainanalysis']
    detectors: Optional[conlist(min_length=1, item_type=MCADetector)] = []

    _exclude = set(vars(_FitConfig()).keys())

    @model_validator(mode='before')
    @classmethod
    def validate_mcadetectorconfig_before(cls, data):
        """Validate the `MCADetectorConfig` class attributes.

        :param data:
            `Pydantic <https://github.com/pydantic/pydantic>`__
            validator data object.
        :type data: dict
        :return: Currently validated class attributes.
        :rtype: dict
        """
        if isinstance(data, dict):
            processor_type = data.get('processor_type').lower()
            if 'detectors' in data:
                detectors = data.pop('detectors')
                for d in detectors:
                    d['processor_type'] = processor_type
                    attrs = d.pop('attrs', {})
                    if 'default_fields' in attrs:
                        attrs.pop('default_fields')
                    if attrs:
                        d['attrs'] = attrs
                data['detectors'] = detectors
        return data

    @model_validator(mode='after')
    def validate_mcadetectorconfig_after(self):
        """Validate and update the detectors.

        :return: Validated detectors.
        :rtype: MCADetectorConfig
        """
        if self.detectors:
            self.update_detectors()
        return self

    def update_detectors(self):
        """Update individual detector parameters with any non-default
        values from the global detector configuration.
        """
        for k, v in self:
            if k in self.model_fields_set:
                for d in self.detectors:
                    if hasattr(d, k):
                        setattr(d, k, deepcopy(v))


# Processor configuration classes

class DiffractionVolumeLengthConfig(_FitConfig):
    """Configuration for the differential volume length processor
    :class:`~CHAP.edd.processor.DiffractionVolumeLengthProcessor`
    for an EDD setup using a steel-foil raster scan.

    :ivar max_energy_kev: Maximum channel energy of the MCA in
        keV, defaults to `200.0`.
    :vartype max_energy_kev: float, optional
    :ivar measurement_mode: Placeholder for recording whether the
        measured DVL value was obtained through the automated
        calculation or a manual selection, defaults to `'auto'`.
    :vartype measurement_mode: Literal['manual', 'auto'], optional
    :ivar sample_thickness: Thickness of scanned foil sample. Quantity
        must be provided in the same units as the values of the
        scanning motor.
    :vartype sample_thickness: float
    :ivar sigma_to_dvl_factor: The DVL is obtained by fitting a reduced
        form of the MCA detector data. `sigma_to_dvl_factor` is a
        scalar value that converts the standard deviation of the
        gaussian fit to the measured DVL, defaults to `3.5`.
    :vartype sigma_to_dvl_factor: Literal[2.0, 3.5, 4.0], optional
    """

    max_energy_kev: Optional[confloat(gt=0, allow_inf_nan=False)] = 200.0
    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
    sample_thickness: Optional[confloat(gt=0, allow_inf_nan=False)] = None
    sigma_to_dvl_factor: Optional[Literal[2.0, 3.5, 4.0]] = 3.5

    _exclude = set(vars(_FitConfig()).keys())

    @model_validator(mode='after')
    def validate_diffractionvolumelengthconfig_after(self):
        """Update the configuration with costum defaults after the
        normal native pydantic validation.

        :return: Updated DVL configuration class.
        :rtype: DiffractionVolumeLengthConfig
        """
        if self.measurement_mode == 'manual':
            self._exclude |= {'sigma_to_dvl_factor'}
        return self


class MCACalibrationConfig(CHAPBaseModel):
    """Base class configuration for energy and 2&theta calibration
    processors.

    :ivar flux_file: File name of the csv flux file containing station
        beam energy in eV (column 0) versus flux (column 1).
    :vartype flux_file: str, optional
    :ivar materials: Material configurations for the calibration,
        defaults to [`Ceria`].
    :vartype materials: list[MaterialConfig], optional
    :ivar peak_energies: Theoretical locations of the fluorescence
        peaks in keV to use for calibrating the MCA channel energies.
    :vartype peak_energies: list[float], optional for energy calibration
    :ivar scan_step_indices: Optional scan step indices to use for the
        calibration. If not specified, the calibration will be
        performed on the average of all MCA spectra for the scan.
    :vartype scan_step_indices: int, str, list[int], optional

    .. note::
       Fluorescence data:
       https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
    """

    flux_file: Optional[FilePath] = None
    materials: Optional[conlist(item_type=MaterialConfig)] = [MaterialConfig(
        material_name='CeO2', lattice_parameters=5.41153, sgnum=225)]
    peak_energies: Optional[conlist(
        min_length=2, item_type=confloat(gt=0, allow_inf_nan=False))] = [
            34.279, 34.720, 39.258, 40.233]
    scan_step_indices: Optional[
        conlist(min_length=1, item_type=conint(ge=0))] = None

    @model_validator(mode='before')
    @classmethod
    def validate_mcacalibrationconfig_before(cls, data):
        """Ensure that a valid configuration was provided and finalize
        flux_file filepath.

        :param data:
            `Pydantic <https://github.com/pydantic/pydantic>`__
            validator data object.
        :type data: dict
        :return: Currently validated class attributes.
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

        :param scan_step_indices: Optional scan step indices to use for
            the calibration. If not specified, the calibration will be
            performed on the average of all MCA spectra for the scan.
        :type scan_step_indices: int or str or list[int], optional
        :raises ValueError: Invalid experiment type.
        :return: Validated scan step indices.
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

        :type: tuple(float, float)
        """
        if self.flux_file is None:
            return None
        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        return energies.min(), energies.max()

    def flux_correction_interpolation_function(self):
        """Get an interpolation function to correct MCA data for the
        relative energy flux of the incident beam.

        :type: scipy.interpolate._polyint._Interpolator1D
        """
        if self.flux_file is None:
            return None
        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        relative_intensities = flux[:,1]/np.max(flux[:,1])
        interpolation_function = interp1d(energies, relative_intensities)
        return interpolation_function


class MCAEnergyCalibrationConfig(MCACalibrationConfig):
    """Configuration for the energy calibration processor
    :class:`~CHAP.edd.processor.MCAEnergyCalibrationProcessor`.

    :ivar max_energy_kev: Maximum channel energy of the MCA in
        keV, defaults to `200.0`.
    :vartype max_energy_kev: float, optional
    :ivar max_peak_index: Index of the peak in `peak_energies`
        with the highest amplitude, defaults to `1` (the second peak)
        for CeO2 calibration. Required for any other materials.
    :vartype max_peak_index: int, optional
    """

    max_energy_kev: Optional[confloat(gt=0, allow_inf_nan=False)] = 200.0
    max_peak_index: Optional[conint(ge=0)] = None

    @model_validator(mode='before')
    @classmethod
    def validate_mcaenergycalibrationconfig_before(cls, data):
        """Validate the `MCAEnergyCalibrationConfig` class attributes.

        :param data:
            `Pydantic <https://github.com/pydantic/pydantic>`__
            validator data object.
        :type data: dict
        :return: Currently validated class attributes.
        :rtype: dict
        """
        if isinstance(data, dict):
            detectors = data.pop('detectors', None)
            if detectors is not None:
                data['detector_config'] = {'detectors': detectors}
        return data

    @model_validator(mode='after')
    def validate_mcaenergycalibrationconfig_after(self):
        """Validate the detector (energy) mask ranges and update any
        detector configuration parameters not superseded by their
        individual values.

        :return: Validated energy calibration configuration class.
        :rtype: MCAEnergyCalibrationConfig
        """
        if self.peak_energies is None:
            raise ValueError('peak_energies is required')
        if (self.max_peak_index is not None
                and not 0 <= self.max_peak_index < len(self.peak_energies)):
            raise ValueError('max_peak_index out of bounds')
        return self

    @field_validator('max_peak_index', mode='before')
    @classmethod
    def validate_max_peak_index(cls, max_peak_index, info):
        """Validate max_peak_index.

        :param max_peak_index: Index of the peak in `peak_energies`
            with the highest amplitude, defaults to `1` (the second
            peak) for CeO2 calibration. Required for any other
            materials.
        :type max_peak_index: int, optional
        :param info: Model parameter validation information.
        :type info: pydantic.ValidationInfo
        :return: Validated max_peak_index.
        :rtype: int
        """
        if max_peak_index is None:
            materials = info.data.get('materials', [])
            if len(materials) != 1 or materials[0].material_name != 'CeO2':
                raise ValueError('max_peak_index is required unless the '
                                 'calibration material is CeO2')
            max_peak_index = 1
        return max_peak_index

class MCATthCalibrationConfig(MCACalibrationConfig):
    """Configuration for the 2&theta calibration and the reduced data
    processors, :class:`~CHAP.edd.processor.MCATthCalibrationProcessor`
    and :class:`~CHAP.edd.processor.ReducedDataProcessor`,
    respectively.

    :ivar calibration_method: Type of calibration method,
        defaults to `'direct_fit_bragg'`.
    :vartype calibration_method:
        Literal['direct_fit_bragg', 'direct_fit_tth_ecc'], optional
    :ivar quadratic_energy_calibration: Adds a quadratic term to
        the detector channel index to energy conversion, defaults
        to `False` (linear only).
    :vartype quadratic_energy_calibration: bool, optional
    :ivar tth_initial_guess: Initial guess for 2&theta.
    :vartype tth_initial_guess: float, optional
    """

    calibration_method: Optional[Literal[
        'direct_fit_bragg', 'direct_fit_tth_ecc']] = 'direct_fit_bragg'
    quadratic_energy_calibration: Optional[bool] = False
    tth_initial_guess: Optional[
        confloat(gt=0, allow_inf_nan=False)] = Field(None, exclude=True)

    def flux_file_energy_range(self):
        """Get the energy range in the flux corection file.

        :type: tuple(float, float)
        """
        if self.flux_file is None:
            return None
        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        return energies.min(), energies.max()


class StrainAnalysisConfig(MCACalibrationConfig):
    """Configuration for the lattice parameter refinement and strain
    analysis processors,
    :class:`~CHAP.edd.processor.LatticeParameterRefinementProcessor`
    and :class:`~CHAP.edd.processor.StrainAnalysisProcessor`,
    respectively.

    :ivar find_peak_cutoff: Use scipy.signal.find_peaks to exclude
        peaks for all spectra for a given detector and user specified
        mask. A particular HKL peak is removed from the set of HKLs,
        when its mean peak height is  below `find_peak_cutoff` times
        the maximum mean intensity for that detector. Defaults to `0`
        in which case this step is ignored.
    :vartype find_peak_cutoff: float, optional
    :ivar num_proc: Number of processors to be used by the strain
        analysis peak fitting routine.
    :vartype num_proc: int
    :ivar rel_height_cutoff: Used to excluded peaks based on the
        `find_peak` parameter as well as for peak fitting exclusion
        of the individual detector spectra (see the strain detector
        configuration
        :class:`~CHAP.edd.models.MCADetectorStrainAnalysis`).
        Defaults to `None`.
    :vartype rel_height_cutoff: float, optional
    :ivar skip_animation: Skip the animation and plotting of
        the strain analysis fits, defaults to `False`.
    :vartype skip_animation: bool, optional
    :ivar sum_axes: Whether to sum over the fly axis or not
        for EDD scan types not 0, defaults to `True`.
    :vartype sum_axes: bool or list[str], optional
    """
    #:ivar oversampling: FIX
    #:vartype oversampling: FIX

    find_peak_cutoff: Optional[confloat(ge=0.0, allow_inf_nan=False)] = 0.0
    num_proc: Optional[conint(gt=0)] = max(1, os.cpu_count()//4)
    #oversampling: dict = {'num': 10}
    rel_height_cutoff: Optional[
        confloat(gt=0.0, lt=1.0, allow_inf_nan=False)] = None
    skip_animation: Optional[bool] = False
    sum_axes: Optional[
        Union[bool, conlist(min_length=1, item_type=str)]] = True

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

#    @field_validator('oversampling')
#    @classmethod
#    def validate_oversampling(cls, oversampling, info):
#        """Validate the oversampling field.
#
#        :param oversampling: Value of `oversampling` to validate.
#        :type oversampling: dict
#        :param info: Model parameter validation information.
#        :type info: pydantic.ValidationInfo
#        :return: Validated oversampling value.
#        :rtype: bool
#        """
#        # Local modules
#        from CHAP.utils.general import is_int
#
#        raise ValueError('oversampling not updated yet')
#        map_config = info.data.get('map_config')
#        if map_config is None or map_config.attrs['scan_type'] < 3:
#            return None
#        if oversampling is None:
#            return {'num': 10}
#        if 'start' in oversampling and not is_int(oversampling['start'], ge=0):
#            raise ValueError('Invalid "start" parameter in "oversampling" '
#                             f'field ({oversampling["start"]})')
#        if 'end' in oversampling and not is_int(oversampling['end'], gt=0):
#            raise ValueError('Invalid "end" parameter in "oversampling" '
#                             f'field ({oversampling["end"]})')
#        if 'width' in oversampling and not is_int(oversampling['width'], gt=0):
#            raise ValueError('Invalid "width" parameter in "oversampling" '
#                             f'field ({oversampling["width"]})')
#        if ('stride' in oversampling
#                and not is_int(oversampling['stride'], gt=0)):
#            raise ValueError('Invalid "stride" parameter in "oversampling" '
#                             f'field ({oversampling["stride"]})')
#        if 'num' in oversampling and not is_int(oversampling['num'], gt=0):
#            raise ValueError('Invalid "num" parameter in "oversampling" '
#                             f'field ({oversampling["num"]})')
#        if 'mode' in oversampling and 'mode' not in ('valid', 'full'):
#            raise ValueError('Invalid "mode" parameter in "oversampling" '
#                             f'field ({oversampling["mode"]})')
#        if not ('width' in oversampling or 'stride' in oversampling
#                or 'num' in oversampling):
#            raise ValueError('Invalid input parameters, specify at least one '
#                             'of "width", "stride" or "num"')
#        return oversampling

MCADetectorCalibration.model_rebuild(_types_namespace=vars(typing))
MCADetectorConfig.model_rebuild(_types_namespace=vars(typing))
DiffractionVolumeLengthConfig.model_rebuild(_types_namespace=vars(typing))
