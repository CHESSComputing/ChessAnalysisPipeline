#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
'''
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Processors used only by EDD experiments
'''

# system modules
import argparse
import json
import logging
import sys

# local modules
from CHAP.processor import Processor
from CHAP.common import StrainAnalysisProcessor

class MCACeriaCalibrationProcessor(Processor):
    '''A Processor using a CeO2 scan to obtain tuned values for the bragg
    diffraction angle and linear correction parameters for MCA channel energies
    for an EDD experimental setup.
    '''

    def _process(self, data):
        '''Return tuned values for 2&theta and linear correction parameters for
        the MCA channel energies.

        :param data: input configuration for the raw data & tuning procedure
        :type data: list[dict[str,object]]
        :return: original configuration dictionary with tuned values added
        :rtype: dict[str,float]
        '''

        calibration_config = self.get_config(data)

        tth, slope, intercept = self.calibrate(calibration_config)

        calibration_config.tth_calibrated = tth
        calibration_config.slope_calibrated = slope
        calibration_config.intercept_calibrated = intercept

        return(calibration_config.dict())

    def get_config(self, data):
        '''Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'MCACeriaCalibrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be constructed from
            `data`.
        :return: a valid instance of a configuration object with field values
            taken from `data`.
        :rtype: MCACeriaCalibrationConfig
        '''

        from CHAP.edd.models import MCACeriaCalibrationConfig

        calibration_config = False
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'MCACeriaCalibrationConfig':
                        calibration_config = item.get('data')
                        break

        if not calibration_config:
            raise(ValueError('No MCA ceria calibration configuration found in input data'))

        return(MCACeriaCalibrationConfig(**calibration_config))

    def calibrate(self, calibration_config):
        '''Iteratively calibrate 2&theta by fitting selected peaks of an MCA
        spectrum until the computed strain is sufficiently small. Use the fitted
        peak locations to determine linear correction parameters for the MCA's
        channel energies.

        :param calibration_config: object configuring the CeO2 calibration
            procedure
        :type calibration_config: MCACeriaCalibrationConfig
        :return: calibrated values of 2&theta and linear correction parameters
            for MCA channel energies : tth, slope, intercept
        :rtype: float, float, float
        '''

        from msnctools.fit import Fit, FitMultipeak
        import numpy as np
        from scipy.constants import physical_constants

        hc = (physical_constants['Planck constant in eV/Hz'][0]
              * physical_constants['speed of light in vacuum'][0]
              * 1e7) # We'll work in keV and A, not eV and m.

        # Collect raw MCA data of interest
        mca_data = calibration_config.mca_data()
        mca_bin_energies = (np.arange(0, calibration_config.num_bins)
                            * (calibration_config.max_energy_kev
                               / calibration_config.num_bins))

        # Mask out the corrected MCA data for fitting
        mca_mask = calibration_config.mca_mask()
        fit_mca_energies = mca_bin_energies[mca_mask]
        fit_mca_intensities = mca_data[mca_mask]

        # Correct raw MCA data for variable flux at different energies
        flux_correct = calibration_config.flux_correction_interpolation_function()
        mca_intensity_weights = flux_correct(fit_mca_energies)
        fit_mca_intensities = fit_mca_intensities / mca_intensity_weights

        # Get the HKLs and lattice spacings that will be used for fitting
        tth = calibration_config.tth_initial_guess
        fit_hkls, fit_ds = calibration_config.fit_ds()
        c_1 = fit_hkls[:,0]**2 + fit_hkls[:,1]**2 + fit_hkls[:,2]**2

        for iter_i in range(calibration_config.max_iter):

            ### Perform the uniform fit first ###

            # Get expected peak energy locations for this iteration's starting
            # value of tth
            fit_lambda = 2.0 * fit_ds * np.sin(0.5*np.radians(tth))
            fit_E0 = hc / fit_lambda

            # Run the uniform fit
            best_fit, residual, best_values, best_errors, redchi, success = \
                FitMultipeak.fit_multipeak(
                    fit_mca_intensities,
                    fit_E0,
                    x=fit_mca_energies,
                    fit_type='uniform')

            # Extract values of interest from the best values for the uniform fit
            # parameters
            uniform_fit_centers = [best_values[f'peak{i+1}_center'] for i in range(len(calibration_config.fit_hkls))]
            # uniform_a = best_values['scale_factor']
            # uniform_strain = np.log(
            #     (uniform_a 
            #      / calibration_config.lattice_parameter_angstrom))
            # uniform_tth = tth * (1.0 + uniform_strain)
            # uniform_rel_rms_error = (np.linalg.norm(residual)
            #                          / np.linalg.norm(fit_mca_intensities))

            ### Next, perform the unconstrained fit ###

            # Use the peak locations found in the uniform fit as the initial
            # guesses for peak locations in the unconstrained fit
            best_fit, residual, best_values, best_errors, redchi, success = \
                FitMultipeak.fit_multipeak(
                    fit_mca_intensities,
                    uniform_fit_centers,
                    x=fit_mca_energies,
                    fit_type='unconstrained')

            # Extract values of interest from the best values for the
            # unconstrained fit parameters
            unconstrained_fit_centers = np.array(
                [best_values[f'peak{i+1}_center'] for i in range(len(calibration_config.fit_hkls))])
            unconstrained_a = (0.5 * hc * np.sqrt(c_1)
                               / (unconstrained_fit_centers
                                  * abs(np.sin(0.5*np.radians(tth)))))
            unconstrained_strains = np.log(
                (unconstrained_a
                 / calibration_config.lattice_parameter_angstrom))
            unconstrained_strain = np.mean(unconstrained_strains)
            unconstrained_tth = tth * (1.0 + unconstrained_strain)
            unconstrained_rel_rms_error = (np.linalg.norm(residual)
                                           / np.linalg.norm(fit_mca_intensities))


            # Update tth for the next iteration of tuning
            prev_tth = tth
            tth = unconstrained_tth

            # Stop tuning tth at this iteration if differences are small enough
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

        return(float(tth), float(slope), float(intercept))

class MCADataProcessor(Processor):
    '''A Processor to return data from an MCA, restuctured to incorporate the
    shape & metadata associated with a map configuration to which the MCA data
    belongs, and linearly transformed according to the results of a ceria
    calibration.
    '''

    def _process(self, data):
        '''Process configurations for a map and MCA detector(s), and return the
        calibrated MCA data collected over the map.

        :param data: input map configuration and results of ceria calibration
        :type data: list[dict[str,object]]
        :return: calibrated and flux-corrected MCA data
        :rtype: nexusformat.nexus.NXentry
        '''

        map_config, calibration_config = self.get_configs(data)
        nxroot = self.get_nxroot(map_config, calibration_config)

        return(nxroot)

    def get_configs(self, data):
        '''Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item has the
            value `'MapConfig'` for the `'schema'` key, and at least one item has
            the value `'MCACeriaCalibrationConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be constructed from
            `data`.
        :return: valid instances of the configuration objects with field values
            taken from `data`.
        :rtype: tuple[MapConfig, MCACeriaCalibrationConfig]
        '''

        from CHAP.common.models import MapConfig
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
            raise(ValueError('No map configuration found in input data'))
        if not calibration_config:
            raise(ValueError('No MCA ceria calibration configuration found in input data'))

        return(MapConfig(**map_config), MCACeriaCalibrationConfig(**calibration_config))

    def get_nxroot(self, map_config, calibration_config):
        '''Get a map of the MCA data collected by the scans in `map_config`. The
        MCA data will be calibrated and flux-corrected according to the
        parameters included in `calibration_config`. The data will be returned
        along with relevant metadata in the form of a NeXus structure.

        :param map_config: the map configuration
        :type map_config: MapConfig
        :param calibration_config: the calibration configuration
        :type calibration_config: MCACeriaCalibrationConfig
        :return: a map of the calibrated and flux-corrected MCA data
        :rtype: nexusformat.nexus.NXroot
        '''

        from CHAP.common import MapProcessor

        from nexusformat.nexus import (NXdata,
                                       NXdetector,
                                       NXentry,
                                       NXinstrument,
                                       NXroot)
        import numpy as np

        nxroot = NXroot()

        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]

        nxentry.instrument = NXinstrument()
        nxentry.instrument.detector = NXdetector()
        nxentry.instrument.detector.calibration_configuration = json.dumps(calibration_config.dict())

        nxentry.instrument.detector.data = NXdata()
        nxdata = nxentry.instrument.detector.data
        nxdata.raw = np.empty((*map_config.shape, calibration_config.num_bins))
        nxdata.raw.attrs['units'] = 'counts'
        nxdata.channel_energy = (calibration_config.slope_calibrated
                                * np.arange(0, calibration_config.num_bins)
                                * (calibration_config.max_energy_kev
                                   / calibration_config.num_bins)
                                + calibration_config.intercept_calibrated)
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
            nxentry.data.attrs['axes'] += [f'{calibration_config.detector_name}_channel_energy']
        nxentry.data.attrs['signal'] = calibration_config.detector_name

        return(nxroot)

if __name__ == '__main__':
    from CHAP.processor import main
    main()
