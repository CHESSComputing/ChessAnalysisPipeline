# third party modules
import numpy as np
from pathlib import PosixPath
from pydantic import (BaseModel,
                      confloat,
                      conint,
                      conlist,
                      constr,
                      FilePath,
                      root_validator,
                      validator)
from scipy.interpolate import interp1d
from typing import Literal, Optional, Union

# local modules
from CHAP.common.models.map import MapConfig
from CHAP.utils.material import Material
from CHAP.utils.parfile import ParFile
from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser


class MCAElementConfig(BaseModel):
    """Class representing metadata required to configure a single MCA
    detector element.

    :ivar detector_name: name of the MCA used with the scan
    :type detector_name: str
    :ivar num_bins: number of channels on the MCA
    :type num_bins: int
    :ivar include_bin_ranges: list of MCA channel index ranges whose
        data should be included after applying a mask
    :type include_bin_ranges: list[list[int]]
    """
    detector_name: constr(strip_whitespace=True, min_length=1) = 'mca1'
    num_bins: Optional[conint(gt=0)]
    include_bin_ranges: Optional[
        conlist(
            min_items=1,
            item_type=conlist(
                item_type=conint(ge=0),
                min_items=2,
                max_items=2))] = None

    @validator('include_bin_ranges', each_item=True)
    def validate_include_bin_range(cls, value, values):
        """Ensure no bin ranges are outside the boundary of the detector"""
        num_bins = values.get('num_bins')
        if num_bins is not None:
            value[1] = min(value[1], num_bins)
        return value

    def mca_mask(self):
        """Get a boolean mask array to use on this MCA element's data.

        :return: boolean mask array
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for min_, max_ in self.include_bin_ranges:
            _mask = np.logical_and(bin_indices > min_, bin_indices < max_)
            mask = np.logical_or(mask, _mask)
        return mask

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        d['include_bin_ranges'] = [
            list(d['include_bin_ranges'][i]) \
            for i in range(len(d['include_bin_ranges']))]
        return d


class MCAScanDataConfig(BaseModel):
    """Class representing metadata required to locate raw MCA data for
    a single scan and construct a mask for it.

    :ivar spec_file: Path to the SPEC file containing the scan
    :ivar scan_number: Number of the scan in `spec_file`
    :ivar detectors: list of detector element metadata configurations

    :ivar detector_name: name of the MCA used with the scan
    :ivar num_bins: number of channels on the MCA

    :ivar include_bin_ranges: list of MCA channel index ranges whose
        data should be included after applying a mask
    """
    spec_file: Optional[FilePath]
    scan_number: Optional[conint(gt=0)]
    par_file: Optional[FilePath]
    scan_column: Optional[Union[conint(ge=0), str]]

    detectors: conlist(min_items=1, item_type=MCAElementConfig)

    _parfile: Optional[ParFile]
    _scanparser: Optional[ScanParser]

    class Config:
        underscore_attrs_are_private = False

    @root_validator
    def validate_root(cls, values):
        """Validate the `values` dictionary. Fill in a value for
        `_scanparser` and `num_bins` (if the latter was not already
        provided)

        :param values: dictionary of field values to validate
        :type values: dict
        :return: the validated form of `values`
        :rtype: dict
        """
        spec_file = values.get('spec_file')
        par_file = values.get('par_file')
        if spec_file and par_file:
            raise ValueError('Use either spec_file or par_file, not both')
        elif spec_file:
            values['_scanparser'] = ScanParser(values.get('spec_file'),
                                               values.get('scan_number'))
            values['_parfile'] = None
        elif par_file:
            if 'scan_column' not in values:
                raise ValueError(
                    'When par_file is used, scan_column must be used, too')
            values['_parfile'] = ParFile(values.get('par_file'))
            if isinstance(values['scan_column'], str):
                if values['scan_column'] not in values['_parfile'].column_names:
                    raise ValueError(
                        f'No column named {values["scan_column"]} in '
                        + '{values["par_file"]}. Options: '
                        + ', '.join(values['_parfile'].column_names))
            #values['spec_file'] = values['_parfile'].spec_file
            values['_scanparser'] = ScanParser(
                values['_parfile'].spec_file,
                values['_parfile'].good_scan_numbers()[0])
        else:
            raise ValueError('Must use either spec_file or par_file')

        for detector in values.get('detectors'):
            if detector.num_bins is None:
                try:
                    detector.num_bins = values['_scanparser']\
                        .get_detector_num_bins(detector.detector_name)
                except Exception as exc:
                    raise ValueError('No value found for num_bins') from exc
        return values

    @property
    def scanparser(self):
        try:
            scanparser = self._scanparser
        except:
            scanparser = ScanParser(self.spec_file, self.scan_number)
            self._scanparser = scanparser
        return scanparser

    def mca_data(self, detector_config, scan_step_index=None):
        """Get the array of MCA data collected by the scan.

        :param detector_config: detector for which data will be returned
        :type detector_config: MCAElementConfig
        :return: MCA data
        :rtype: np.ndarray
        """
        detector_name = detector_config.detector_name
        if self._parfile is not None:
            if scan_step_index is None:
                import numpy as np
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
                    detector_config.detector_name, self.scan_step_index)
        return data

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
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
        for k in ('_scanparser', '_parfile'):
            if k in d:
                del d[k]
        return d


class MaterialConfig(BaseModel):
    """Model for parameters to characterize a sample material

    :ivar hexrd_h5_material_file: path to a HEXRD materials.h5 file containing
        an entry for the material properties.
    :ivar hexrd_h5_material_name: Name of the material entry in
        `hexrd_h5_material_file`.
    :ivar lattice_parameter_angstrom: lattice spacing in angstrom to use for
        a cubic crystal.
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
        from CHAP.edd.utils import make_material
        values['_material'] = make_material(values.get('material_name'),
                                            values.get('sgnum'),
                                            values.get('lattice_parameters'))
        return values

    def unique_ds(self, tth_tol=0.15, tth_max=90.0):
        """Get a list of unique HKLs and their lattice spacings

        :param tth_tol: minimum resolvable difference in 2&theta
            between two unique HKL peaks, defaults to `0.15`.
        :type tth_tol: float, optional
        :param tth_max: detector rotation about hutch x axis, defaults
            to `90.0`.
        :type tth_max: float, optional
        :return: unique HKLs and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """
        from CHAP.edd.utils import get_unique_hkls_ds
        return get_unique_hkls_ds([self._material])

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_material' in d:
            del d['_material']
        return d


class MCAElementCalibrationConfig(MCAElementConfig):
    """Class representing metadata & parameters required for
    calibrating a single MCA detector element.

    :ivar max_energy_kev: maximum channel energy of the MCA in keV
    :ivar tth_max: detector rotation about hutch x axis, defaults to `90`.
    :ivar hkl_tth_tol: minimum resolvable difference in 2&theta between two
        unique HKL peaks, defaults to `0.15`.
    :ivar fit_hkls: list of unique HKL indices to fit peaks for in the
        calibration routine
    :ivar tth_initial_guess: initial guess for 2&theta
    :ivar slope_initial_guess: initial guess for detector channel energy
        correction linear slope, defaults to `1.0`.
    :ivar intercept_initial_guess: initial guess for detector channel energy
        correction y-intercept, defaults to `0.0`.
    :ivar tth_calibrated: calibrated value for 2&theta, defaults to None
    :ivar slope_calibrated: calibrated value for detector channel energy
        correction linear slope, defaults to `None`
    :ivar intercept_calibrated: calibrated value for detector channel energy
        correction y-intercept, defaluts to None
    """
    max_energy_kev: confloat(gt=0)
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    fit_hkls: Optional[conlist(item_type=conint(ge=0), min_items=1)] = None
    tth_initial_guess: confloat(gt=0, le=tth_max, allow_inf_nan=False)
    slope_initial_guess: float = 1.0
    intercept_initial_guess: float = 0.0
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)]
    slope_calibrated: Optional[confloat(allow_inf_nan=False)]
    intercept_calibrated: Optional[confloat(allow_inf_nan=False)]

    def fit_ds(self, materials):
        """Get a list of HKLs and their lattice spacings that will be
        fit in the calibration routine

        :return: HKLs to fit and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """
        if not isinstance(materials, list):
            materials = [materials]
        from CHAP.edd.utils import get_unique_hkls_ds
        unique_hkls, unique_ds = get_unique_hkls_ds(materials)

        fit_hkls = np.array([unique_hkls[i] for i in self.fit_hkls])
        fit_ds = np.array([unique_ds[i] for i in self.fit_hkls])

        return fit_hkls, fit_ds


class MCAElementDiffractionVolumeLengthConfig(MCAElementConfig):
    """Class representing input parameters required to perform a
    diffraction volume length measurement for a single MCA detector
    element.

    :ivar measurement_mode: placeholder for recording whether the
        measured DVL value was obtained through the automated
        calculation or a manual selection.
    :type measurement_mode: Literal['manual', 'auto']
    :ivar sigma_to_dvl_factor: to measure the DVL, a gaussian is fit
        to a reduced from of the raster scan MCA data. This variable
        is a scalar that converts the standard deviation of the
        gaussian fit to the measured DVL.
    :type sigma_to_dvl_factor: Optional[Literal[1.75, 1., 2.]]
    :ivar dvl_measured: placeholder for the measured diffraction
        volume length before writing data to file.
    """
    measurement_mode: Optional[Literal['manual', 'auto']] = 'auto'
    sigma_to_dvl_factor: Optional[Literal[3.5, 2., 4.]] = 3.5
    dvl_measured: Optional[confloat(gt=0)] = None

    def dict(self, *args, **kwargs):
        """If measurement_mode is 'manual', exclude
        sigma_to_dvl_factor from the dict representation.
        """
        d = super().dict(*args, **kwargs)
        if self.measurement_mode == 'manual':
            del d['sigma_to_dvl_factor']
        return d


class DiffractionVolumeLengthConfig(MCAScanDataConfig):
    """Class representing metadata required to perform a diffraction
    volume length calculation for an EDD setup using a steel-foil
    raster scan.

    :ivar detectors: list of individual detector elmeent DVL
        measurement configurations
    :type detectors: list[MCAElementDiffractionVolumeLengthConfig]
    """
    detectors: conlist(min_items=1,
                       item_type=MCAElementDiffractionVolumeLengthConfig)

    @property
    def scanned_vals(self):
        """Return the list of values visited by the scanning motor
        over the course of the raster scan.

        :return: list of scanned motor values
        :rtype: np.ndarray
        """
        if self._parfile is not None:
            return self._parfile.get_values(
                self.scan_column,
                scan_numbers=self._parfile.good_scan_numbers())
        return self.scanparser.spec_scan_motor_vals[0]

    @property
    def scanned_dim_lbl(self):
        """Return a label for plot axes corresponding to the scanned
        dimension

        :rtype: str
        """
        if self._parfile is not None:
            return self.scan_column
        return self.scanparser.spec_scan_motor_mnes[0]

class CeriaConfig(MaterialConfig):
    """Model for a Material representing CeO2 used in calibrations.

    :ivar hexrd_h5_material_name: Name of the material entry in
        `hexrd_h5_material_file`, defaults to `'CeO2'`.
    :ivar lattice_parameter_angstrom: lattice spacing in angstrom to use for
        the cubic CeO2 crystal, defaults to `5.41153`.
    """
    material_name: constr(strip_whitespace=True, min_length=1) = 'CeO2'
    sgnum: Optional[conint(ge=0)] = 225
    lattice_parameters: confloat(gt=0) = 5.41153


class MCACeriaCalibrationConfig(MCAScanDataConfig):
    """
    Class representing metadata required to perform a Ceria calibration for an
    MCA detector.

    :ivar scan_step_index: Index of the scan step to use for calibration,
        optional. If not specified, the calibration routine will be performed
        on the average of all MCA spectra for the scan.

    :ivar flux_file: csv file containing station beam energy in eV (column 0)
        and flux (column 1)

    :ivar material: material configuration for Ceria
    :type material: CeriaConfig

    :ivar detectors: list of individual detector element calibration
        configurations
    :type detectors: list[MCAElementCalibrationConfig]

    :ivar max_iter: maximum number of iterations of the calibration routine,
        defaults to `10`.
    :ivar tune_tth_tol: stop iteratively tuning 2&theta when an iteration
        produces a change in the tuned value of 2&theta that is smaller than
        this value, defaults to `1e-8`.
    """
    scan_step_index: Optional[conint(ge=0)]

    flux_file: FilePath

    material: CeriaConfig = CeriaConfig()

    detectors: conlist(min_items=1, item_type=MCAElementCalibrationConfig)

    max_iter: conint(gt=0) = 10
    tune_tth_tol: confloat(ge=0) = 1e-8

    def mca_data(self, detector_config):
        """Get the 1D array of MCA data to use for calibration.

        :param detector_config: detector for which data will be returned
        :type detector_config: MCAElementConfig
        :return: MCA data
        :rtype: np.ndarray
        """
        if self.scan_step_index is None:
            data = super().mca_data(detector_config)
            if self.scanparser.spec_scan_npts > 1:
                data = np.average(data, axis=1)
            else:
                data = data[0]
        else:
            data = super().mca_data(detector_config,
                                    scan_step_index=self.scan_step_index)
        return data

    def flux_correction_interpolation_function(self):
        """
        Get an interpolation function to correct MCA data for relative energy
        flux of the incident beam.

        :return: energy flux correction interpolation function
        :rtype: scipy.interpolate._polyint._Interpolator1D
        """

        flux = np.loadtxt(self.flux_file)
        energies = flux[:,0]/1.e3
        relative_intensities = flux[:,1]/np.max(flux[:,1])
        interpolation_function = interp1d(energies, relative_intensities)
        return interpolation_function


class MCAElementStrainAnalysisConfig(MCAElementConfig):
    """Model for parameters need to perform strain analysis fitting
    for one MCA element.
    """
    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15
    fit_hkls: Optional[conlist(item_type=conint(ge=0), min_items=1)] = None
    background: Optional[str]
    peak_models: Union[
        conlist(item_type=Literal['gaussian', 'lorentzian'], min_items=1),
        Literal['gaussian', 'lorentzian']] = 'gaussian'

    tth_file: Optional[FilePath]
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)]
    slope_calibrated: Optional[confloat(allow_inf_nan=False)]
    intercept_calibrated: Optional[confloat(allow_inf_nan=False)]
    max_energy_kev: Optional[confloat(gt=0)]
    num_bins: Optional[conint(gt=0)]

    _tth_map: Optional[np.ndarray]

    def add_calibration(self, calibration):
        """Finalize values for some fields using a completed
        MCAElementCalibrationConfig that corresponds to the same
        detector.

        :param calibration: MCAElementCalibrationConfig
        :return: None
        """
        add_fields = ['tth_calibrated', 'slope_calibrated',
                      'intercept_calibrated', 'num_bins', 'max_energy_kev']
        for field in add_fields:
            setattr(self, field, getattr(calibration, field))

    def fit_ds(self, materials):
        """Get a list of HKLs and their lattice spacings that will be
        fit in the strain analysis routine

        :return: HKLs to fit and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """
        if not isinstance(materials, list):
            materials = [materials]
        from CHAP.edd.utils import get_unique_hkls_ds
        unique_hkls, unique_ds = get_unique_hkls_ds(materials)

        # unique_hkls, unique_ds = material.unique_ds(
        #     tth_tol=self.hkl_tth_tol, tth_max=self.tth_max)

        fit_hkls = np.array([unique_hkls[i] for i in self.fit_hkls])
        fit_ds = np.array([unique_ds[i] for i in self.fit_hkls])

        return fit_hkls, fit_ds

    def tth_map(self, map_config):
        """Return a map of tth values to use -- may vary at each point
        in the map.

        :param map_config: the map configuration with which the
            returned map of tth values will be used.
        :type map_config: MapConfig
        :return: map of thh values
        :rtype: np.ndarray
        """
        if self._tth_map is not None:
            return self._tth_map
        return np.full(map_config.shape, self.tth_calibrated)


class StrainAnalysisConfig(BaseModel):
    """Model for inputs to CHAP.edd.StrainAnalysisProcessor"""
    map_config: Optional[MapConfig]
    par_file: Optional[FilePath]
    par_dims: Optional[list[dict[str,str]]]
    other_dims: Optional[list[dict[str,str]]]
    flux_file: FilePath
    detectors: conlist(min_items=1, item_type=MCAElementStrainAnalysisConfig)
    materials: list[MaterialConfig]

    _parfile: Optional[ParFile]

    @root_validator(pre=True)
    def validate_map(cls, values):
        """Ensure exactly one valid map configuration was provided."""
        if values.get('par_file') is not None:
            if 'par_dims' not in values:
                raise ValueError(
                    'If using par_file, must also use par_dims')
            values['_parfile'] = ParFile(values['par_file'])
            values['map_config'] = values['_parfile'].get_map(
                'EDD', 'id1a3', values['par_dims'],
                other_dims=values.get('other_dims', []))
        return values

    @validator('detectors', each_item=True)
    def validate_tth(cls, detector, values):
        """Validate detector element tth_file field. It may only be
        used if StrainAnalysisConfig used par_file.
        """
        if detector.tth_file is not None:
            if values['_par_file'] is None:
                raise ValueError(
                    'variable tth angles may only be used with a '
                    + 'StrainAnalysisConfig that uses par_file.')
            else:
                tth = np.loadtxt(detector.tth_file)
                try:
                    detector._tth_map = values['_par_file'].map_values(
                        values['map_config'], tth)
                except Exception as e:
                    raise ValueError(
                        'Could not get map of tth angles from '
                        + f'{detector.tth_file}') from e
        return detector

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_scanparser' in d:
            del d['_scanparser']
        return d

    def mca_data(self, detector_config, map_index):
        """Get MCA data for a single detector element.

        :param detector_config: the detector to get data for
        :type detector_config: MCAElementStrainAnalysisConfig
        :param map_index: index of a single point in the map
        :type map_index: tuple
        :return: one spectrum of MCA data
        :rtype: np.ndarray
        """
        map_coords = {dim: self.map_config.coords[dim][i]
                      for dim,i in zip(self.map_config.dims, map_index)}
        for scans in self.map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    _coords = {
                        dim.label:dim.get_value(
                            scans, scan_number, scan_step_index)
                        for dim in self.map_config.independent_dimensions}
                    if _coords == map_coords:
                        break
        scanparser = scans.get_scanparser(scan_number)
        return scanparser.get_detector_data(detector_config.detector_name,
                                            scan_step_index)
