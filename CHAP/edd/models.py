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
from typing import Literal, Optional

# local modules
from CHAP.common.utils.scanparsers import SMBMCAScanParser as ScanParser


class MCAScanDataConfig(BaseModel):
    """Class representing metadata required to locate raw MCA data for
    a single scan and construct a mask for it.

    :ivar spec_file: Path to the SPEC file containing the scan
    :ivar scan_number: Number of the scan in `spec_file`

    :ivar detector_name: name of the MCA used with the scan
    :ivar num_bins: number of channels on the MCA

    :ivar include_bin_ranges: list of MCA channel index ranges whose
        data should be included after applying a mask
    """
    spec_file: FilePath
    scan_number: conint(gt=0)

    detector_name: constr(strip_whitespace=True, min_length=1) = 'mca1'
    num_bins: Optional[conint(gt=0)]

    include_bin_ranges: Optional[
        conlist(
            min_items=1,
            item_type=conlist(
                item_type=conint(ge=0),
                min_items=2,
                max_items=2))] = None

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
        values['_scanparser'] = ScanParser(values.get('spec_file'),
                                           values.get('scan_number'))
        if values.get('num_bins') is None:
            try:
                values['num_bins'] = values['_scanparser']\
                    .get_detector_num_bins(
                        values.get('detector_name'))
            except exc:
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

    @validator('include_bin_ranges', each_item=True)
    def validate_include_bin_range(cls, value, values):
        """Ensure no bin ranges are outside the boundary of the detector"""
        num_bins = values.get('num_bins')
        if num_bins is not None:
            value[1] = min(value[1], num_bins)
        return value

    def mca_mask(self):
        """Get a boolean mask array to use on the MCA data.

        :return: boolean mask array
        :rtype: numpy.ndarray
        """
        mask = np.asarray([False] * self.num_bins)
        bin_indices = np.arange(self.num_bins)
        for min_, max_ in self.include_bin_ranges:
            _mask = np.logical_and(bin_indices > min_, bin_indices < max_)
            mask = np.logical_or(mask, _mask)
        return mask

    def mca_data(self, scan_step_index=None):
        """Get the array of MCA data collected by the scan.

        :return: MCA data
        :rtype: np.ndarray
        """
        if scan_step_index is None:
            data = self.scanparser.get_all_detector_data(self.detector_name)
        else:
            data = self.scanparser.get_detector_data(self.detector_name,
                                                     self.scan_step_index)
        return data

    def dict(self):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict()
        for k,v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        if '_scanparser' in d:
            del d['_scanparser']
        return d


class DiffractionVolumeLengthConfig(MCAScanDataConfig):
    """Class representing metadata required to perform a diffraction
    volume length calculation for an EDD setup using a steel-foil
    raster scan.

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
    sigma_to_dvl_factor: Optional[Literal[1.75, 1., 2.]] = 1.75
    dvl_measured: Optional[confloat(gt=0)] = None

    @property
    def motor_vals(self):
        """Return the list of values visited by the scanning motor
        over the course of the raster scan.

        :return: list of scanned motor values
        :rtype: np.ndarray
        """
        return self.scanparser.spec_scan_motor_vals[0]

    @property
    def motor_mne(self):
        """Return the mnenomic of the raster scan's motor"""
        return self.scanparser.spec_scan_motor_mnes[0]

    def dict(self):
        """If measurement_mode is 'manual', exclude
        sigma_to_dvl_factor from the dict representation.
        """
        d = super().dict()
        if self.measurement_mode == 'manual':
            del d['sigma_to_dvl_factor']
        return d


class MCACeriaCalibrationConfig(MCAScanDataConfig):
    """
    Class representing metadata required to perform a Ceria calibration for an
    MCA detector.

    :ivar scan_step_index: Index of the scan step to use for calibration,
        optional. If not specified, the calibration routine will be performed
        on the average of all MCA spectra for the scan.

    :ivar flux_file: csv file containing station beam energy in eV (column 0)
        and flux (column 1)

    :ivar max_energy_kev: maximum channel energy of the MCA in keV

    :ivar hexrd_h5_material_file: path to a HEXRD materials.h5 file containing
        an entry for the material properties.
    :ivar hexrd_h5_material_name: Name of the material entry in
        `hexrd_h5_material_file`, defaults to `'CeO2'`.
    :ivar lattice_parameter_angstrom: lattice spacing in angstrom to use for
        the cubic CeO2 crystal, defaults to `5.41153`.

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

    :ivar max_iter: maximum number of iterations of the calibration routine,
        defaults to `10`.
    :ivar tune_tth_tol: stop iteratively tuning 2&theta when an iteration
        produces a change in the tuned value of 2&theta that is smaller than
        this value, defaults to `1e-8`.
    """
    scan_step_index: Optional[conint(ge=0)]

    flux_file: FilePath

    max_energy_kev: confloat(gt=0)

    hexrd_h5_material_file: FilePath
    hexrd_h5_material_name: constr(
        strip_whitespace=True, min_length=1) = 'CeO2'
    lattice_parameter_angstrom: confloat(gt=0) = 5.41153

    tth_max: confloat(gt=0, allow_inf_nan=False) = 90.0
    hkl_tth_tol: confloat(gt=0, allow_inf_nan=False) = 0.15

    fit_hkls: Optional[conlist(item_type=conint(ge=0), min_items=1)] = None

    tth_initial_guess: confloat(gt=0, le=tth_max, allow_inf_nan=False)
    slope_initial_guess: float = 1.0
    intercept_initial_guess: float = 0.0
    tth_calibrated: Optional[confloat(gt=0, allow_inf_nan=False)]
    slope_calibrated: Optional[confloat(allow_inf_nan=False)]
    intercept_calibrated: Optional[confloat(allow_inf_nan=False)]

    max_iter: conint(gt=0) = 10
    tune_tth_tol: confloat(ge=0) = 1e-8

    def mca_data(self):
        """Get the 1D array of MCA data to use for calibration.

        :return: MCA data
        :rtype: np.ndarray
        """
        if self.scan_step_index is None:
            data = super().mca_data()
            if self.scanparser.spec_scan_npts > 1:
                data = np.average(data, axis=1)
            else:
                data = data[0]
        else:
            data = super().mca_data(scan_step_index=self.scan_step_index)
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

    def material(self):
        """Get CeO2 as a `CHAP.common.utils.material.Material` object.

        :return: CeO2 material
        :rtype: CHAP.common.utils.material.Material
        """
        # local modules
        from CHAP.common.utils.material import Material

        material = Material(
            material_name=self.hexrd_h5_material_name,
            material_file=self.hexrd_h5_material_file,
            lattice_parameters_angstroms=self.lattice_parameter_angstrom)
        # The following kwargs will be needed if we allow the material to be
        # built using xrayutilities (for now, we only allow hexrd to make the
        # material):
        #   sgnum=225,
        #   atoms=['Ce4p', 'O2mdot'],
        #   pos=[(0.,0.,0.), (0.25,0.75,0.75)],
        #   enrgy=50000.)
        # Why do we need to specify an energy to get HKLs when using
        # xrayutilities?
        return material

    def unique_ds(self):
        """Get a list of unique HKLs and their lattice spacings

        :return: unique HKLs and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """

        unique_hkls, unique_ds = self.material().get_ds_unique(
            tth_tol=self.hkl_tth_tol, tth_max=self.tth_max)

        return unique_hkls, unique_ds

    def fit_ds(self):
        """
        Get a list of HKLs and their lattice spacings that will be fit in the
        calibration routine

        :return: HKLs to fit and their lattice spacings in angstroms
        :rtype: np.ndarray, np.ndarray
        """

        unique_hkls, unique_ds = self.unique_ds()

        fit_hkls = np.array([unique_hkls[i] for i in self.fit_hkls])
        fit_ds = np.array([unique_ds[i] for i in self.fit_hkls])

        return fit_hkls, fit_ds
