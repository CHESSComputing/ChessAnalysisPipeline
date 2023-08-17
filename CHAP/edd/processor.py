#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Keara Soloway, Rolf Verberg
Description: Module for Processors used only by EDD experiments
"""

# system modules
from json import dumps
import os

# third party modules
import numpy as np

# local modules
from CHAP.processor import Processor


class DiffractionVolumeLengthProcessor(Processor):
    """A Processor using a steel foil raster scan to calculate the
    length of the diffraction volume for an EDD setup.
    """

    def process(self,
                data,
                config=None,
                save_figures=False,
                outputdir='.',
                interactive=False):
        """Return calculated value of the DV length.

        :param data: input configuration for the raw scan data & DVL
            calculation procedure.
        :type data: list[PipelineData]
        :param config: initialization parameters for an instance of
            CHAP.edd.models.DiffractionVolumeLengthConfig, defaults to
            None
        :type config: dict, optional
        :param save_figures: save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False
        :type save_figures: bool, optional
        :param outputdir: directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: complete DVL configuraiton dictionary
        :rtype: dict
        """

        try:
            dvl_config = self.get_config(
                data, 'edd.models.DiffractionVolumeLengthConfig')
        except Exception as data_exc:
            self.logger.info('No valid DVL config in input pipeline data, '
                             + 'using config parameter instead.')
            try:
                from CHAP.edd.models import DiffractionVolumeLengthConfig
                dvl_config = DiffractionVolumeLengthConfig(**config)
            except Exception as dict_exc:
                self.logger.error('Could not get a valid DVL config')
                raise RuntimeError from dict_exc

        for detector in dvl_config.detectors:
            dvl = self.measure_dvl(dvl_config, detector,
                                   save_figures=save_figures,
                                   interactive=interactive,
                                   outputdir=outputdir)
            detector.dvl_measured = dvl

        return dvl_config.dict()

    def measure_dvl(self,
                    dvl_config,
                    detector,
                    save_figures=False,
                    outputdir='.',
                    interactive=False):
        """Return a measured value for the length of the diffraction
        volume. Use the iron foil raster scan data provided in
        `dvl_config` and fit a gaussian to the sum of all MCA channel
        counts vs scanned motor position in the raster scan. The
        computed diffraction volume length is approximately equal to
        the standard deviation of the fitted peak.

        :param dvl_config: configuration for the DVL calculation
            procedure
        :type dvl_config: DiffractionVolumeLengthConfig
        :param detector: A single MCA detector element configuration
        :type detector: CHAP.edd.models.MCAElementDiffractionVolumeLengthConfig
        :param save_figures: save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False
        :type save_figures: bool, optional
        :param outputdir: directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: calculated diffraction volume length
        :rtype: float
        """

        from CHAP.utils.fit import Fit
        from CHAP.utils.general import draw_mask_1d

        # Get raw MCA data from raster scan
        mca_data = dvl_config.mca_data(detector)

        # Interactively set mask, if needed & possible.
        if interactive or save_figures:
            self.logger.info(
                'Interactively select a mask in the matplotlib figure')
            mask, include_bin_ranges, figure = draw_mask_1d(
                np.sum(mca_data, axis=0),
                xdata = np.arange(detector.num_bins),
                current_index_ranges=detector.include_bin_ranges,
                label='sum of MCA spectra over all scan points',
                title='Click and drag to select ranges of MCA data to\n'
                + 'include when measuring the diffraction volume length.',
                xlabel='MCA channel (index)',
                ylabel='MCA intensity (counts)',
                test_mode=not interactive,
                return_figure=True
            )
            detector.include_bin_ranges = include_bin_ranges
            self.logger.debug('Mask selected. Including detector bin ranges: '
                              + str(detector.include_bin_ranges))
            if save_figures:
                figure.savefig(os.path.join(
                    outputdir, f'{detector.detector_name}_dvl_mask.png'))
            import matplotlib.pyplot as plt
            plt.close()
        if detector.include_bin_ranges is None:
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
        mask = detector.mca_mask()
        masked_mca_data = np.empty(
            (mca_data.shape[0], *mca_data[0][mask].shape))
        for i in range(mca_data.shape[0]):
            masked_mca_data[i] = mca_data[i][mask]
        masked_max = np.amax(masked_mca_data, axis=1)
        masked_sum = np.sum(masked_mca_data, axis=1)

        # Find the motor position corresponding roughly to the center
        # of the diffraction volume
        scanned_vals = dvl_config.scanned_vals
        scan_center = np.sum(scanned_vals * masked_sum) / np.sum(masked_sum)
        x = scanned_vals - scan_center

        # "Normalize" the masked summed data and fit a gaussian to it
        y = (masked_sum - min(masked_sum)) / max(masked_sum)
        fit = Fit.fit_data(y, 'gaussian', x=x, normalize=False)

        # Calculate / manually select diffraction volume length
        dvl = fit.best_values['sigma'] * detector.sigma_to_dvl_factor
        if detector.measurement_mode == 'manual':
            if interactive:
                mask, dvl_bounds = draw_mask_1d(
                    y, xdata=x,
                    label='total (masked & normalized)',
                    ref_data=[
                        ((x, fit.best_fit),
                         {'label': 'gaussian fit (to total)'}),
                        ((x, masked_max / max(masked_max)),
                         {'label': 'maximum (masked)'}),
                        ((x, unmasked_sum / max(unmasked_sum)),
                         {'label': 'total (unmasked)'})
                    ],
                    num_index_ranges_max=1,
                    title=('Click and drag to indicate the\n'
                           + 'boundary of the diffraction volume'),
                    xlabel=(dvl_config.scanned_dim_lbl
                            + ' (offset from scan "center")'),
                    ylabel='MCA intensity (normalized)')
                dvl_bounds = dvl_bounds[0]
                dvl = abs(x[dvl_bounds[1]] - x[dvl_bounds[0]])
            else:
                self.logger.warning(
                    'Cannot manually indicate DVL when running CHAP '
                    + 'non-interactively. '
                    + 'Using default DVL calcluation instead.')

        if interactive or save_figures:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_title(f'Diffraction Volume ({detector.detector_name})')
            ax.set_xlabel(dvl_config.scanned_dim_lbl \
                          + ' (offset from scan "center")')
            ax.set_ylabel('MCA intensity (normalized)')
            ax.plot(x, y, label='total (masked & normalized)')
            ax.plot(x, fit.best_fit, label='gaussian fit (to total)')
            ax.plot(x, masked_max / max(masked_max),
                    label='maximum (masked)')
            ax.plot(x, unmasked_sum / max(unmasked_sum),
                    label='total (unmasked)')
            ax.axvspan(-dvl / 2., dvl / 2.,
                       color='gray', alpha=0.5,
                       label='diffraction volume'
                       + f' ({detector.measurement_mode})')
            ax.legend()

            if save_figures:
                figfile = os.path.join(outputdir,
                                       f'{detector.detector_name}_dvl.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return dvl

class MCACeriaCalibrationProcessor(Processor):
    """A Processor using a CeO2 scan to obtain tuned values for the
    bragg diffraction angle and linear correction parameters for MCA
    channel energies for an EDD experimental setup.
    """

    def process(self,
                data,
                config=None,
                save_figures=False,
                outputdir='.',
                interactive=False):
        """Return tuned values for 2&theta and linear correction
        parameters for the MCA channel energies.

        :param data: input configuration for the raw data & tuning
            procedure
        :type data: list[dict[str,object]]
        :param config: initialization parameters for an instance of
            CHAP.edd.models.MCACeriaCalibrationConfig, defaults to
            None
        :type config: dict, optional
        :param save_figures: save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False
        :type save_figures: bool, optional
        :param outputdir: directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: original configuration dictionary with tuned values
            added
        :rtype: dict[str,float]
        """

        try:
            calibration_config = self.get_config(
                data, 'edd.models.MCACeriaCalibrationConfig')
        except Exception as data_exc:
            self.logger.info('No valid calibration config in input pipeline '
                             + 'data, using config parameter instead.')
            try:
                from CHAP.edd.models import MCACeriaCalibrationConfig
                calibration_config = MCACeriaCalibrationConfig(**config)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        for detector in calibration_config.detectors:
            tth, slope, intercept = self.calibrate(
                calibration_config, detector,
                save_figures=save_figures,
                interactive=interactive, outputdir=outputdir)

            detector.tth_calibrated = tth
            detector.slope_calibrated = slope
            detector.intercept_calibrated = intercept

        return calibration_config.dict()

    def calibrate(self,
                  calibration_config,
                  detector,
                  save_figures=False,
                  outputdir='.',
                  interactive=False):
        """Iteratively calibrate 2&theta by fitting selected peaks of
        an MCA spectrum until the computed strain is sufficiently
        small. Use the fitted peak locations to determine linear
        correction parameters for the MCA's channel energies.

        :param calibration_config: object configuring the CeO2
            calibration procedure
        :type calibration_config: MCACeriaCalibrationConfig
        :param detector: a single MCA detector element configuration
        :type detector: CHAP.edd.models.MCAElementCalibrationConfig
        :param save_figures: save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False
        :type save_figures: bool, optional
        :param outputdir: directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: calibrated values of 2&theta and linear correction
            parameters for MCA channel energies : tth, slope,
            intercept
        :rtype: float, float, float
        """
        from CHAP.edd.utils import hc
        from CHAP.utils.fit import Fit, FitMultipeak

        # Collect raw MCA data of interest
        mca_data = calibration_config.mca_data(detector)
        mca_bin_energies = np.arange(0, detector.num_bins) \
            * (detector.max_energy_kev / detector.num_bins)

        if interactive:
            # Interactively adjust initial tth guess
            from CHAP.edd.utils import select_tth_initial_guess
            select_tth_initial_guess(detector, calibration_config.material,
                                     mca_data, mca_bin_energies)
        self.logger.debug(f'tth_initial_guess = {detector.tth_initial_guess}')

        # Mask out the corrected MCA data for fitting
        if interactive:
            from CHAP.utils.general import draw_mask_1d
            self.logger.info(
                'Interactively select a mask in the matplotlib figure')
            mask, include_bin_ranges = draw_mask_1d(
                mca_data,
                xdata=mca_bin_energies,
                current_index_ranges=detector.include_bin_ranges,
                title='Click and drag to select ranges of Ceria'
                +' calibration data to include',
                xlabel='MCA channel energy (keV)',
                ylabel='MCA intensity (counts)')
            detector.include_bin_ranges = include_bin_ranges
            self.logger.debug('Mask selected. Including detector bin ranges: '
                              + str(detector.include_bin_ranges))
        if detector.include_bin_ranges is None:
            raise ValueError(
                'No value provided for include_bin_ranges. '
                'Provide them in the MCA Ceria Calibration Configuration, '
                'or re-run the pipeline with the --interactive flag.')
        mca_mask = detector.mca_mask()
        fit_mca_energies = mca_bin_energies[mca_mask]
        fit_mca_intensities = mca_data[mca_mask]

        # Correct raw MCA data for variable flux at different energies
        flux_correct = \
            calibration_config.flux_correction_interpolation_function()
        mca_intensity_weights = flux_correct(fit_mca_energies)
        fit_mca_intensities = fit_mca_intensities / mca_intensity_weights

        # Get the HKLs and lattice spacings that will be used for
        # fitting
        tth = detector.tth_initial_guess
        if interactive or save_figures:
            import matplotlib.pyplot as plt
            from CHAP.edd.utils import select_hkls
            fig = select_hkls(detector, [calibration_config.material], tth,
                              mca_data, mca_bin_energies, interactive)
            if save_figures:
                fig.savefig(os.path.join(
                    outputdir,
                    f'{detector.detector_name}_calibration_hkls.png'))
            plt.close()
        self.logger.debug(f'HKLs selected: {detector.fit_hkls}')
        if detector.fit_hkls is None:
            raise ValueError(
                'No value provided for fit_hkls. Provide them in '
                'the detector\'s MCA Ceria Calibration Configuration, or'
                ' re-run the pipeline with the --interactive flag.')
        fit_hkls, fit_ds = detector.fit_ds(calibration_config.material)
        c_1 = fit_hkls[:,0]**2 + fit_hkls[:,1]**2 + fit_hkls[:,2]**2

        for iter_i in range(calibration_config.max_iter):
            self.logger.debug(f'Tuning tth: iteration no. {iter_i}, '
                              + f'starting tth value = {tth} ')

            # Perform the uniform fit first

            # Get expected peak energy locations for this iteration's
            # starting value of tth
            fit_lambda = 2.0*fit_ds*np.sin(0.5*np.radians(tth))
            fit_E0 = hc / fit_lambda

            # Run the uniform fit
            uniform_best_fit, uniform_residual, best_values, \
                best_errors, redchi, success = \
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
                for i in range(len(detector.fit_hkls))]
            uniform_a = best_values['scale_factor']
            uniform_strain = np.log(
                (uniform_a
                 / calibration_config.material.lattice_parameters)) # CeO2 is cubic, so this is fine here.
            # uniform_tth = tth * (1.0 + uniform_strain)
            # uniform_rel_rms_error = (np.linalg.norm(residual)
            #                          / np.linalg.norm(fit_mca_intensities))

            # Next, perform the unconstrained fit

            # Use the peak locations found in the uniform fit as the
            # initial guesses for peak locations in the unconstrained
            # fit
            unconstrained_best_fit, unconstrained_residual, best_values, \
                best_errors, redchi, success = \
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
                 for i in range(len(detector.fit_hkls))])
            unconstrained_a = 0.5*hc*np.sqrt(c_1) \
                / (unconstrained_fit_centers*abs(np.sin(0.5*np.radians(tth))))
            unconstrained_strains = np.log(
                (unconstrained_a
                 / calibration_config.material.lattice_parameters))
            unconstrained_strain = np.mean(unconstrained_strains)
            unconstrained_tth = tth * (1.0 + unconstrained_strain)
            unconstrained_rel_rms_error = (
                np.linalg.norm(unconstrained_residual) \
                / np.linalg.norm(fit_mca_intensities))

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

        if interactive or save_figures:
            fig, axs = plt.subplots(2, 2, sharex='all', figsize=(11, 8.5))

            # Upper left axes: Input data & best fits
            axs[0,0].set_title('Ceria Calibration Fits')
            axs[0,0].set_xlabel('Energy (keV)')
            axs[0,0].set_ylabel('Intensity (a.u)')
            for i, hkl_E in enumerate(fit_E0):
                # KLS: annotate indicated HKLs w millier indices
                axs[0,0].axvline(hkl_E, color='k', linestyle='--')
                axs[0,0].text(hkl_E, 1, str(fit_hkls[i])[1:-1],
                              ha='right', va='top', rotation=90,
                              transform=axs[0,0].get_xaxis_transform())
            axs[0,0].plot(fit_mca_energies, uniform_best_fit,
                        label='Single Strain')
            axs[0,0].plot(fit_mca_energies, unconstrained_best_fit,
                        label='Unconstrained')
            #axs[0,0].plot(fit_mca_energies, MISSING?, label='least squares')
            axs[0,0].plot(fit_mca_energies, fit_mca_intensities,
                        label='Flux-Corrected & Masked MCA Data')
            axs[0,0].legend()

            # Lower left axes: fit residuals
            axs[1,0].set_title('Fit Residuals')
            axs[1,0].set_xlabel('Energy (keV)')
            axs[1,0].set_ylabel('Residual (a.u)')
            axs[1,0].plot(fit_mca_energies,
                          uniform_residual,
                          label='Single Strain')
            axs[1,0].plot(fit_mca_energies,
                          unconstrained_residual,
                          label='Unconstrained')
            axs[1,0].legend()

            # Upper right axes: E vs strain for each fit
            axs[0,1].set_title('HKL Energy vs. Microstrain')
            axs[0,1].set_xlabel('Energy (keV)')
            axs[0,1].set_ylabel('Strain (\u03BC\u03B5)')
            axs[0,1].axhline(uniform_strain * 1e6,
                             linestyle='--', label='Single Strain')
            axs[0,1].plot(fit_E0, unconstrained_strains * 1e6,
                          color='C1', marker='s', label='Unconstrained')
            axs[0,1].axhline(unconstrained_strain * 1e6,
                             color='C1', linestyle='--',
                             label='Unconstrained: Unweighted Mean')
            axs[0,1].legend()

            # Lower right axes: theoretical HKL E vs fit HKL E for
            # each fit
            axs[1,1].set_title('Theoretical vs. Fit HKL Energies')
            axs[1,1].set_xlabel('Energy (keV)')
            axs[1,1].set_ylabel('Energy (keV)')
            axs[1,1].plot(fit_E0, uniform_fit_centers,
                          marker='o', label='Single Strain')
            axs[1,1].plot(fit_E0, unconstrained_fit_centers,
                          linestyle='', marker='o', label='Unconstrained')
            axs[1,1].plot(slope * unconstrained_fit_centers + intercept,fit_E0,
                          color='C1', label='Unconstrained: Linear Fit')
            axs[1,1].legend()

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(outputdir, 'ceria_calibration_fits.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

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

        map_config = self.get_config(
            data, 'common.models.map.MapConfig')
        calibration_config = self.get_config(
            data, 'edd.models.MCACeriaCalibrationConfig')
        nxroot = self.get_nxroot(map_config, calibration_config)

        return nxroot

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

        for detector in calibration_config.detectors:
            nxentry.instrument = NXinstrument()
            nxentry.instrument[detector.detector_name] = NXdetector()
            nxentry.instrument[detector.detector_name].calibration = dumps(
                detector.dict())

            nxentry.instrument[detector.detector_name].data = NXdata()
            nxdata = nxentry.instrument[detector.detector_name].data
            nxdata.raw = np.empty((*map_config.shape,
                                   detector.num_bins))
            nxdata.raw.attrs['units'] = 'counts'
            nxdata.channel_energy = detector.slope_calibrated \
                * np.arange(0, detector.num_bins) \
                * (detector.max_energy_kev / detector.num_bins) \
                + detector.intercept_calibrated
            nxdata.channel_energy.attrs['units'] = 'keV'

            for map_index in np.ndindex(map_config.shape):
                scans, scan_number, scan_step_index = \
                    map_config.get_scan_step_index(map_index)
                scanparser = scans.get_scanparser(scan_number)
                nxdata.raw[map_index] = scanparser.get_detector_data(
                    calibration_config.detector_name,
                    scan_step_index)

            nxentry.data.makelink(nxdata.raw, name=detector.detector_name)
            nxentry.data.makelink(
                nxdata.channel_energy,
                name=f'{detector.detector_name}_channel_energy')
            if isinstance(nxentry.data.attrs['axes'], str):
                nxentry.data.attrs['axes'] = [
                    nxentry.data.attrs['axes'],
                    f'{detector.detector_name}_channel_energy']
            else:
                nxentry.data.attrs['axes'] += [
                    f'{detector.detector_name}_channel_energy']
            nxentry.data.attrs['signal'] = detector.detector_name

        return nxroot


class StrainAnalysisProcessor(Processor):
    """Processor that takes a map of MCA data and returns a map of
    sample strains
    """
    def process(self,
                data,
                config=None,
                save_figures=False,
                outputdir='.',
                interactive=False):
        """Return strain analysis maps & associated metadata in an NXprocess.

        :param data: input data containing configurations for a map,
            completed ceria calibration, and parameters for strain
            analysis
        :type data: list[PipelineData]
        :param config: initialization parameters for an instance of
            CHAP.edd.models.StrainAnalysisConfig, defaults to
            None
        :type config: dict, optional
        :param save_figures: save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False
        :type save_figures: bool, optional
        :param outputdir: directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: NXprocess containing metadata about strain analysis
            processing parameters and empty datasets for strain maps
            to be filled in later.
        :rtype: nexusformat.nexus.NXprocess

        """
        # Get required configuration models from input data
        # map_config = self.get_config(
        #     data, 'common.models.map.MapConfig')
        ceria_calibration_config = self.get_config(
            data, 'edd.models.MCACeriaCalibrationConfig')
        try:
            strain_analysis_config = self.get_config(
                data, 'edd.models.StrainAnalysisConfig')
        except Exception as data_exc:
            self.logger.info('No valid strain analysis config in input '
                             + 'pipeline data, using config parameter instead')
            from CHAP.edd.models import StrainAnalysisConfig
            try:
                strain_analysis_config = StrainAnalysisConfig(**config)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        nxroot = self.get_nxroot(
            #map_config,
            strain_analysis_config.map_config,
            ceria_calibration_config,
            strain_analysis_config,
            save_figures=save_figures,
            outputdir=outputdir,
            interactive=interactive)
        self.logger.debug(nxroot.tree)
        return nxroot


    def get_nxroot(self,
                   map_config,
                   ceria_calibration_config,
                   strain_analysis_config,
                   save_figures=False,
                   outputdir='.',
                   interactive=False):
        """Return NXroot containing strain maps.


        :param map_config: Input map configuration
        :type map_config: CHAP.common.models.map.MapConfig
        :param ceria_calibration_config: Results of ceria calibration
        :type ceria_calibration_config:
            'CHAP.edd.models.MCACeriaCalibrationConfig'
        :param strain_analysis_config: Strain analysis processing
            configuration
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :param save_figures: save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False
        :type save_figures: bool, optional
        :param outputdir: directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param interactive: allow for user interactions, defaults to
            False
        :type interactive: bool, optional
        :return: NXroot containing strain maps
        :rtype: nexusformat.nexus.NXroot
        """
        from nexusformat.nexus import (NXdata,
                                       NXdetector,
                                       NXfield,
                                       NXprocess,
                                       NXroot)
        import numpy as np
        from CHAP.common import MapProcessor
        from CHAP.edd.utils import hc
        from CHAP.utils.fit import FitMap

        for detector in strain_analysis_config.detectors:
            calibration = [
                d for d in ceria_calibration_config.detectors \
                if d.detector_name == detector.detector_name][0]
            detector.add_calibration(calibration)

        nxroot = NXroot()
        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]
        nxroot[f'{map_config.title}_strains'] = NXprocess()
        nxprocess = nxroot[f'{map_config.title}_strains']
        nxprocess.strain_analysis_config = dumps(strain_analysis_config.dict())

        # Setup plottable data group
        nxprocess.data = NXdata()
        nxprocess.default = 'data'
        nxdata = nxprocess.data
        nxdata.attrs['axes'] = map_config.dims
        for dim in map_config.dims:
            nxdata.makelink(nxentry.data[dim])
            nxdata.attrs[f'{dim}_indices'] = \
                nxentry.data.attrs[f'{dim}_indices']

        # Select interactive params / save figures
        if save_figures or interactive:
            import matplotlib.pyplot as plt
            from CHAP.edd.utils import select_hkls
            from CHAP.utils.general import draw_mask_1d
            for detector in strain_analysis_config.detectors:
                x = np.linspace(detector.intercept_calibrated,
                                detector.max_energy_kev \
                                * detector.slope_calibrated,
                                detector.num_bins)
                y = strain_analysis_config.mca_data(
                    detector,
                    (0,) * len(strain_analysis_config.map_config.shape))
                fig = select_hkls(detector,
                                  strain_analysis_config.materials,
                                  detector.tth_calibrated,
                                  y, x, interactive)
                if save_figures:
                    fig.savefig(os.path.join(
                        outputdir,
                        f'{detector.detector_name}_strainanalysis_hkls.png'))
                plt.close()

                self.logger.info(
                    'Interactively select a mask in the matplotlib figure')
                mask, include_bin_ranges, figure = draw_mask_1d(
                    y, xdata=x,
                    current_index_ranges=detector.include_bin_ranges,
                    label='reference spectrum',
                    title='Click and drag to select ranges of MCA data to\n'
                    + 'include when analyzing strain.',
                    xlabel='MCA channel (index)',
                    ylabel='MCA intensity (counts)',
                    test_mode=not interactive,
                    return_figure=True
                )
                detector.include_bin_ranges = include_bin_ranges
                if save_figures:
                    figure.savefig(os.path.join(
                        outputdir,
                        f'{detector.detector_name}_strainanalysis_mask.png'))
                plt.close()

            if interactive:
                from CHAP.edd.utils import select_material_params
                x = np.linspace(
                    strain_analysis_config.detectors[0].intercept_calibrated,
                    detector.max_energy_kev \
                    * detector.slope_calibrated,
                    detector.num_bins)
                y = strain_analysis_config.mca_data(
                    strain_analysis_config.detectors[0],
                    (0,) * len(strain_analysis_config.map_config.shape))
                tth = strain_analysis_config.detectors[0].tth_calibrated
                strain_analysis_config.materials = select_material_params(
                    x, y, tth, materials=strain_analysis_config.materials)

        for detector in strain_analysis_config.detectors:
            # Setup NXdata group
            self.logger.debug(
                f'Setting up NXdata group for {detector.detector_name}')
            nxprocess[detector.detector_name] = NXdetector()
            nxdetector = nxprocess[detector.detector_name]
            nxdetector.local_name = detector.detector_name
            # KLS: add calibration metadata here!
            nxdetector.data = NXdata()
            det_nxdata = nxdetector.data
            det_nxdata.attrs['axes'] = map_config.dims + ['energy']
            for dim in map_config.dims:
                det_nxdata.makelink(nxdata[dim].nxlink)
                det_nxdata.attrs[f'{dim}_indices'] = \
                    nxdata.attrs[f'{dim}_indices']
            all_energies = np.arange(0, detector.num_bins) \
                * (detector.max_energy_kev / detector.num_bins) \
                * detector.slope_calibrated \
                + detector.intercept_calibrated
            mask = detector.mca_mask()
            energies = all_energies[mask]
            det_nxdata.energy = NXfield(value=energies,
                                        attrs={'units': 'keV'})
            det_nxdata.attrs['energy_indices'] = len(map_config.dims)
            det_nxdata.intensity = NXfield(
                dtype='uint16',
                shape=(*map_config.shape, len(energies)),
                attrs={'units': 'counts'})
            det_nxdata.microstrain = NXfield(
                dtype='float64',
                shape=map_config.shape,
                attrs={'long_name': 'Strain (\u03BC\u03B5)'})

            # Gather detector daya
            self.logger.debug(
                f'Gathering detector data for {detector.detector_name}')
            fit_hkls, fit_ds = detector.fit_ds(
                strain_analysis_config.materials)
            peak_locations = hc / (
                2. * fit_ds * np.sin(0.5*np.radians(detector.tth_calibrated)))
            for map_index in np.ndindex(map_config.shape):
                try:
                    scans, scan_number, scan_step_index = \
                        map_config.get_scan_step_index(map_index)
                except:
                    continue
                scanparser = scans.get_scanparser(scan_number)
                intensity = scanparser.get_detector_data(
                    detector.detector_name, scan_step_index)\
                    .astype('uint16')[mask]
                det_nxdata.intensity[map_index] = intensity

            # Perform strain analysis
            self.logger.debug(
                f'Beginning strain analysis for {detector.detector_name}')

            # Perform initial fit: assume uniform strain for all HKLs
            uniform_fit = FitMap(det_nxdata.intensity.nxdata, x=energies)
            uniform_fit.create_multipeak_model(
                peak_locations,
                fit_type='uniform',
                peak_models=detector.peak_models,
                background=detector.background)
            uniform_fit.fit()
            uniform_fit_centers = [
                uniform_fit.best_values[
                    uniform_fit.best_parameters().index(f'peak{i+1}_center')]
                for i in range(len(peak_locations))]

            # Perform second fit: do not assume uniform strain for all
            # HKLs, and use the fit peak centers from the uniform fit
            # as inital guesses
            unconstrained_fit = FitMap(det_nxdata.intensity.nxdata, x=energies)
            unconstrained_fit.create_multipeak_model(
                np.mean(uniform_fit_centers, axis=1),
                fit_type='unconstrained',
                peak_models=detector.peak_models,
                background=detector.background)
            unconstrained_fit.fit()
            unconstrained_fit_centers = np.array(
                [unconstrained_fit.best_values[
                    unconstrained_fit.best_parameters()\
                    .index(f'peak{i+1}_center')]
                 for i in range(len(peak_locations))])
            unconstrained_strains = np.empty_like(unconstrained_fit_centers)
            for i, peak_loc in enumerate(peak_locations):
                unconstrained_strains[i] = np.log(
                    peak_loc / unconstrained_fit_centers[i])
            unconstrained_strain = np.mean(unconstrained_strains, axis=0)
            det_nxdata.microstrain.nxdata = unconstrained_strain * 1e6

        return nxroot


if __name__ == '__main__':
    # local modules
    from CHAP.processor import main

    main()
