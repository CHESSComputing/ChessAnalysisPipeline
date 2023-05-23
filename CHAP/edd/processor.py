#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Processors used only by EDD experiments
"""

# system modules
from json import dumps

# third party modules
import numpy as np

# local modules
from CHAP.processor import Processor


class DiffractionVolumeLengthProcessor(Processor):
    """A Processor using a steel foil raster scan to calculate the
    length of the diffraction volume for an EDD setup.
    """

    def process(self, data, interactive=False):
        """Return calculated value of the DV length.

        :param data: input configuration for the raw scan data & DVL
            calculation procedure.
        :type data: list[PipelineData]
        :param dvl_model: method to use for calculating DVL. Choices:
            one of three acceptable scalars which will be multiplied
            by the standard deviation of a gauusian fit to the raster
            scan data, or "manual" (in which case the user is
            presented with a plot of the fit and unfit data, and they
            select the accepatble DVL by eye).
        :type dvl_model: Literal[1.0, 1.75, 2.0, "manual"]
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: complete DVL configuraiton dictionary
        :rtype: dict
        """

        dvl_config = self.get_config(data)
        dvl = self.measure_dvl(dvl_config, interactive=interactive)
        dvl_config.dvl_measured = dvl

        return dvl_config.dict()

    def get_config(self, data):
        """Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'MCACeriaCalibrationConfig'` for the
            `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be
            constructed from `data`.
        :return: a valid instance of a configuration object with field
            values taken from `data`.
        :rtype: MCACeriaCalibrationConfig
        """
        # local modules
        from CHAP.edd.models import DiffractionVolumeLengthConfig

        dvl_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'DiffractionVolumeLengthConfig':
                        dvl_config = item.get('data')
                        break

        if not dvl_config:
            raise ValueError(
                'No DVL calculation configuration found in input data')

        return DiffractionVolumeLengthConfig(**dvl_config)

    def measure_dvl(self, dvl_config, interactive=False):
        """Return a measured value for the length of the diffraction
        volume. Use the iron foil raster scan data provided in
        `dvl_config` and fit a gaussian to the sum of all MCA channel
        counts vs scanned motor position in the raster scan. The
        computed diffraction volume length is approximately equal to
        the standard deviation of the fitted peak.

        :param dvl_config: configuration for the DVL calculation
            procedure
        :type dvl_config: DiffractionVolumeLengthConfig
        :param dvl_model: method to use for calculating DVL. Choices:
            one of three acceptable scalars which will be multiplied
            by the standard deviation of a gauusian fit to the raster
            scan data, or "manual" (in which case the user is
            presented with a plot of the fit and unfit data, and they
            select the accepatble DVL by eye).
        :type dvl_model: Literal[1.0, 1.75, 2.0, "manual"]
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: calculated diffraction volume length
        :rtype: float
        """

        from CHAP.common.utils.fit import Fit
        from CHAP.common.utils.general import draw_mask_1d

        # Get raw MCA data from raster scan
        mca_data = dvl_config.mca_data()

        # Interactively set mask, if needed & possible.
        if dvl_config.include_bin_ranges is None:
            if interactive:
                mask, include_bin_ranges = draw_mask_1d(
                    np.sum(mca_data, axis=0),
                    xdata = np.arange(dvl_config.num_bins),
                    label='sum of MCA spectra over all scan points',
                    title='Click and drag to select ranges of MCA data to\n'
                    + 'include when measuring the diffraction volume length.',
                    xlabel='MCA channel (index)',
                    ylabel='MCA intensity (counts)'
                )
                dvl_config.include_bin_ranges = include_bin_ranges
            else:
                raise ValueError(
                    'No value provided for include_bin_ranges. '
                    + 'Provide them in the Diffraction Volume Length '
                    + 'Measurement Configuration, or re-run the pipeline '
                    + 'with the --interactive flag.')

        # Reduce the raw MCA data in 3 ways:
        # 1) sum of intensities in all detector bins
        # 2) max of intensities in detector bins after mask is applied
        # 3) sum of intensities in detector bins after mask is applied
        unmasked_sum = np.sum(mca_data, axis=1)
        mask = dvl_config.mca_mask()
        masked_mca_data = np.empty(
            (mca_data.shape[0], *mca_data[0][mask].shape))
        for i in range(mca_data.shape[0]):
            masked_mca_data[i] = mca_data[i][mask]
        masked_max = np.amax(masked_mca_data, axis=1)
        masked_sum = np.sum(masked_mca_data, axis=1)

        # Find the motor position corresponding roughly to the center
        # of the diffraction volume
        motor_vals = dvl_config.motor_vals
        scan_center = np.sum(motor_vals * masked_sum) / np.sum(masked_sum)
        x = motor_vals - scan_center

        # "Normalize" the masked summed data and fit a gaussian to it
        y = (masked_sum - min(masked_sum)) / max(masked_sum)
        fit = Fit.fit_data(y, 'gaussian', x=x, normalize=False)

        # Calculate / manually select diffraction volume length
        dvl = fit.best_values['sigma'] * dvl_config.sigma_to_dvl_factor
        if interactive:
            from CHAP.common.utils.general import input_yesno
            manual_dvl = input_yesno(
                'Indicate the diffraction volume length manually? (y/n)',
                default='y')
            if manual_dvl:
                dvl_config.measurement_mode = 'manual'
                mask, dvl_bounds = draw_mask_1d(
                    y, xdata=x,
                    label='total (masked, normalized)',
                    ref_data=[
                        ((x, fit.best_fit),
                         {'label': 'gaussian fit'}),
                        ((x, masked_max / max(masked_max)),
                         {'label': 'maximum (masked, normalized)'}),
                        ((x, unmasked_sum / max(unmasked_sum)),
                         {'label': 'total (unmasked, normalized)'})
                    ],
                    num_index_ranges_max=1,
                    title=('Click and drag to indicate the\n'
                           + 'boundary of the diffraction volume'),
                    xlabel=(dvl_config.motor_mne
                            + ' (offset from scan "center")'),
                    ylabel='MCA intensity (counts)')
                dvl_bounds = dvl_bounds[0]
                dvl = abs(x[dvl_bounds[1]] - x[dvl_bounds[0]])

        return dvl

class MCACeriaCalibrationProcessor(Processor):
    """A Processor using a CeO2 scan to obtain tuned values for the
    bragg diffraction angle and linear correction parameters for MCA
    channel energies for an EDD experimental setup.
    """

    def process(self, data, interactive=False):
        """Return tuned values for 2&theta and linear correction
        parameters for the MCA channel energies.

        :param data: input configuration for the raw data & tuning
            procedure
        :type data: list[dict[str,object]]
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: original configuration dictionary with tuned values
            added
        :rtype: dict[str,float]
        """

        calibration_config = self.get_config(data)

        tth, slope, intercept = self.calibrate(calibration_config,
                                               interactive=interactive)

        calibration_config.tth_calibrated = tth
        calibration_config.slope_calibrated = slope
        calibration_config.intercept_calibrated = intercept

        return calibration_config.dict()

    def get_config(self, data):
        """Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'MCACeriaCalibrationConfig'` for the
            `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be
            constructed from `data`.
        :return: a valid instance of a configuration object with field
            values taken from `data`.
        :rtype: MCACeriaCalibrationConfig
        """
        # local modules
        from CHAP.edd.models import MCACeriaCalibrationConfig

        calibration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'MCACeriaCalibrationConfig':
                        calibration_config = item.get('data')
                        break

        if not calibration_config:
            raise ValueError(
                'No MCA ceria calibration configuration found in input data')

        return MCACeriaCalibrationConfig(**calibration_config)

    def calibrate(self, calibration_config, interactive=False):
        """Iteratively calibrate 2&theta by fitting selected peaks of
        an MCA spectrum until the computed strain is sufficiently
        small. Use the fitted peak locations to determine linear
        correction parameters for the MCA's channel energies.

        :param calibration_config: object configuring the CeO2
            calibration procedure
        :type calibration_config: MCACeriaCalibrationConfig
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: calibrated values of 2&theta and linear correction
            parameters for MCA channel energies : tth, slope,
            intercept
        :rtype: float, float, float
        """
        # third party modules
        from scipy.constants import physical_constants

        # local modules
        from CHAP.common.utils.fit import Fit, FitMultipeak

        # We'll work in keV and A, not eV and m.
        hc = 1e7 * physical_constants['Planck constant in eV/Hz'][0] \
            * physical_constants['speed of light in vacuum'][0]

        # Collect raw MCA data of interest
        mca_data = calibration_config.mca_data()
        mca_bin_energies = np.arange(0, calibration_config.num_bins) \
            * (calibration_config.max_energy_kev/calibration_config.num_bins)

        # Mask out the corrected MCA data for fitting
        if calibration_config.include_bin_ranges is None:
            if interactive:
                from CHAP.common.utils.general import draw_mask_1d
                mask, include_bin_ranges = draw_mask_1d(
                    mca_data,
                    xdata=mca_bin_energies,
                    title='Click and drag to select ranges of Ceria'
                           +' calibration data to include',
                    xlabel='MCA channel energy (keV)',
                    ylabel='MCA intensity (counts)')
                calibration_config.include_bin_ranges = include_bin_ranges
            else:
                raise ValueError(
                    'No value provided for include_bin_ranges. '
                    'Provide them in the MCA Ceria Calibration Configuration, '
                    'or re-run the pipeline with the --interactive flag.')
        mca_mask = calibration_config.mca_mask()
        fit_mca_energies = mca_bin_energies[mca_mask]
        fit_mca_intensities = mca_data[mca_mask]

        # Correct raw MCA data for variable flux at different energies
        flux_correct = \
            calibration_config.flux_correction_interpolation_function()
        mca_intensity_weights = flux_correct(fit_mca_energies)
        fit_mca_intensities = fit_mca_intensities/mca_intensity_weights

        # Get the HKLs and lattice spacings that will be used for
        # fitting
        tth = calibration_config.tth_initial_guess
        if calibration_config.fit_hkls is None:
            if interactive:
                from CHAP.common.utils.general import select_peaks
                hkls, ds = calibration_config.unique_ds()
                peak_locations = hc / (2. * ds * np.sin(0.5 * np.radians(tth)))
                selected_peaks = select_peaks(
                    mca_data, mca_bin_energies, peak_locations,
                    mask=mca_mask)
                fit_hkls = [np.where(peak_locations == peak)[0][0]
                            for peak in selected_peaks]
                calibration_config.fit_hkls = fit_hkls
            else:
                raise ValueError(
                    'No value provided for fit_hkls. Provide them in '
                    'the MCA Ceria Calibration Configuration, or re-run '
                    'the pipeline with the --interactive flag.')
        fit_hkls, fit_ds = calibration_config.fit_ds()
        c_1 = fit_hkls[:,0]**2 + fit_hkls[:,1]**2 + fit_hkls[:,2]**2

        for iter_i in range(calibration_config.max_iter):

            # Perform the uniform fit first

            # Get expected peak energy locations for this iteration's
            # starting value of tth
            fit_lambda = 2.0*fit_ds*np.sin(0.5*np.radians(tth))
            fit_E0 = hc / fit_lambda

            # Run the uniform fit
            best_fit, residual, best_values, best_errors, redchi, success = \
                FitMultipeak.fit_multipeak(
                    fit_mca_intensities,
                    fit_E0,
                    x=fit_mca_energies,
                    fit_type='uniform',
                    plot=False)

            # Extract values of interest from the best values for the
            # uniform fit parameters
            uniform_fit_centers = [
                best_values[f'peak{i+1}_center']
                for i in range(len(calibration_config.fit_hkls))]
            # uniform_a = best_values['scale_factor']
            # uniform_strain = np.log(
            #     (uniform_a
            #      / calibration_config.lattice_parameter_angstrom))
            # uniform_tth = tth * (1.0 + uniform_strain)
            # uniform_rel_rms_error = (np.linalg.norm(residual)
            #                          / np.linalg.norm(fit_mca_intensities))

            # Next, perform the unconstrained fit

            # Use the peak locations found in the uniform fit as the
            # initial guesses for peak locations in the unconstrained
            # fit
            best_fit, residual, best_values, best_errors, redchi, success = \
                FitMultipeak.fit_multipeak(
                    fit_mca_intensities,
                    uniform_fit_centers,
                    x=fit_mca_energies,
                    fit_type='unconstrained',
                    plot=False)

            # Extract values of interest from the best values for the
            # unconstrained fit parameters
            unconstrained_fit_centers = np.array(
                [best_values[f'peak{i+1}_center']
                 for i in range(len(calibration_config.fit_hkls))])
            unconstrained_a = 0.5*hc*np.sqrt(c_1) \
                / (unconstrained_fit_centers*abs(np.sin(0.5*np.radians(tth))))
            unconstrained_strains = np.log(
                unconstrained_a/calibration_config.lattice_parameter_angstrom)
            unconstrained_strain = np.mean(unconstrained_strains)
            unconstrained_tth = tth * (1.0+unconstrained_strain)
            unconstrained_rel_rms_error = (
                np.linalg.norm(residual)/np.linalg.norm(fit_mca_intensities))

            # Update tth for the next iteration of tuning
            prev_tth = tth
            tth = unconstrained_tth

            # Stop tuning tth at this iteration if differences are
            # small enough
            if abs(tth - prev_tth) < calibration_config.tune_tth_tol:
                break

        # Fit line to expected / computed peak locations from the last
        # unconstrained fit.
        fit = Fit.fit_data(
            fit_E0,
            'linear',
            x=unconstrained_fit_centers,
            nan_policy='omit')
        slope = fit.best_values['slope']
        intercept = fit.best_values['intercept']

        return float(tth), float(slope), float(intercept)


class MCADataProcessor(Processor):
    """A Processor to return data from an MCA, restuctured to
    incorporate the shape & metadata associated with a map
    configuration to which the MCA data belongs, and linearly
    transformed according to the results of a ceria calibration.
    """

    def process(self, data):
        """Process configurations for a map and MCA detector(s), and
        return the calibrated MCA data collected over the map.

        :param data: input map configuration and results of ceria
            calibration
        :type data: list[dict[str,object]]
        :return: calibrated and flux-corrected MCA data
        :rtype: nexusformat.nexus.NXentry
        """

        map_config, calibration_config = self.get_configs(data)
        nxroot = self.get_nxroot(map_config, calibration_config)

        return nxroot

    def get_configs(self, data):
        """Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'MapConfig'` for the `'schema'` key, and at
            least one item has the value `'MCACeriaCalibrationConfig'`
            for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be
            constructed from `data`.
        :return: valid instances of the configuration objects with
            field values taken from `data`.
        :rtype: tuple[MapConfig, MCACeriaCalibrationConfig]
        """
        # local modules
        from CHAP.common.models.map import MapConfig
        from CHAP.edd.models import MCACeriaCalibrationConfig

        map_config = False
        calibration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if schema == 'MapConfig':
                        map_config = item.get('data')
                    elif schema == 'MCACeriaCalibrationConfig':
                        calibration_config = item.get('data')

        if not map_config:
            raise ValueError('No map configuration found in input data')
        if not calibration_config:
            raise ValueError('No MCA ceria calibration configuration found in '
                             'input data')

        return (MapConfig(**map_config),
                MCACeriaCalibrationConfig(**calibration_config))

    def get_nxroot(self, map_config, calibration_config):
        """Get a map of the MCA data collected by the scans in
        `map_config`. The MCA data will be calibrated and
        flux-corrected according to the parameters included in
        `calibration_config`. The data will be returned along with
        relevant metadata in the form of a NeXus structure.

        :param map_config: the map configuration
        :type map_config: MapConfig
        :param calibration_config: the calibration configuration
        :type calibration_config: MCACeriaCalibrationConfig
        :return: a map of the calibrated and flux-corrected MCA data
        :rtype: nexusformat.nexus.NXroot
        """
        # third party modules
        from nexusformat.nexus import (NXdata,
                                       NXdetector,
                                       NXinstrument,
                                       NXroot)

        # local modules
        from CHAP.common import MapProcessor

        nxroot = NXroot()

        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]

        nxentry.instrument = NXinstrument()
        nxentry.instrument.detector = NXdetector()
        nxentry.instrument.detector.calibration_configuration = dumps(
            calibration_config.dict())

        nxentry.instrument.detector.data = NXdata()
        nxdata = nxentry.instrument.detector.data
        nxdata.raw = np.empty((*map_config.shape, calibration_config.num_bins))
        nxdata.raw.attrs['units'] = 'counts'
        nxdata.channel_energy = calibration_config.slope_calibrated \
            * np.arange(0, calibration_config.num_bins) \
            * (calibration_config.max_energy_kev/calibration_config.num_bins) \
            + calibration_config.intercept_calibrated
        nxdata.channel_energy.attrs['units'] = 'keV'

        for scans in map_config.spec_scans:
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                for scan_step_index in range(scanparser.spec_scan_npts):
                    map_index = scans.get_index(
                        scan_number,
                        scan_step_index,
                        map_config)
                    nxdata.raw[map_index] = scanparser.get_detector_data(
                        calibration_config.detector_name,
                        scan_step_index)

        nxentry.data.makelink(
            nxdata.raw,
            name=calibration_config.detector_name)
        nxentry.data.makelink(
            nxdata.channel_energy,
            name=f'{calibration_config.detector_name}_channel_energy')
        if isinstance(nxentry.data.attrs['axes'], str):
            nxentry.data.attrs['axes'] = [
                nxentry.data.attrs['axes'],
                f'{calibration_config.detector_name}_channel_energy']
        else:
            nxentry.data.attrs['axes'] += [
                f'{calibration_config.detector_name}_channel_energy']
        nxentry.data.attrs['signal'] = calibration_config.detector_name

        return nxroot


if __name__ == '__main__':
    # local modules
    from CHAP.processor import main

    main()
