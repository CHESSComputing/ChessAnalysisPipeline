#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Keara Soloway, Rolf Verberg
Description: Module for Processors used only by EDD experiments
"""

# System modules
from copy import deepcopy
from json import dumps
import os

# Third party modules
import numpy as np

# Local modules
from CHAP.processor import Processor


class DiffractionVolumeLengthProcessor(Processor):
    """A Processor using a steel foil raster scan to calculate the
    length of the diffraction volume for an EDD setup.
    """

    def process(self,
                data,
                config=None,
                save_figures=False,
                inputdir='.',
                outputdir='.',
                interactive=False):
        """Return the calculated value of the DV length.

        :param data: Input configuration for the raw scan data & DVL
            calculation procedure.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.DiffractionVolumeLengthConfig, defaults to
            `None`.
        :type config: dict, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'
        :type outputdir: str, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to '.'.
        :type inputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :raises RuntimeError: Unable to get a valid DVL configuration.
        :return: Complete DVL configuraiton dictionary.
        :rtype: dict
        """

        try:
            dvl_config = self.get_config(
                data, 'edd.models.DiffractionVolumeLengthConfig',
                inputdir=inputdir)
        except Exception as data_exc:
            self.logger.error(data_exc)
            self.logger.info('No valid DVL config in input pipeline data, '
                             + 'using config parameter instead.')
            try:
                # Local modules
                from CHAP.edd.models import DiffractionVolumeLengthConfig

                dvl_config = DiffractionVolumeLengthConfig(
                    **config, inputdir=inputdir)
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

        :param dvl_config: Configuration for the DVL calculation
            procedure.
        :type dvl_config: CHAP.edd.models.DiffractionVolumeLengthConfig
        :param detector: A single MCA detector element configuration.
        :type detector:
            CHAP.edd.models.MCAElementDiffractionVolumeLengthConfig
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :raises ValueError: No value provided for included bin ranges
            for the MCA detector element.
        :return: Calculated diffraction volume length.
        :rtype: float
        """
        # Local modules
        from CHAP.utils.fit import Fit
        from CHAP.utils.general import (
            index_nearest,
            select_mask_1d,
        )

        # Get raw MCA data from raster scan
        mca_data = dvl_config.mca_data(detector)

        # Interactively set or update mask, if needed & possible.
        if interactive or save_figures:
            if interactive:
                self.logger.info(
                    'Interactively select a mask in the matplotlib figure')
            if save_figures:
                filename = os.path.join(
                    outputdir, f'{detector.detector_name}_dvl_mask.png')
            else:
                filename = None
            _, include_bin_ranges = select_mask_1d(
                np.sum(mca_data, axis=0),
                x=np.linspace(0, detector.max_energy_kev, detector.num_bins),
                label='Sum of MCA spectra over all scan points',
                preselected_index_ranges=detector.include_bin_ranges,
                title='Click and drag to select data range to include when '
                      'measuring diffraction volume length',
                xlabel='Uncalibrated energy (keV)',
                ylabel='MCA intensity (counts)',
                min_num_index_ranges=1,
                interactive=interactive, filename=filename)
            detector.include_energy_ranges = \
                detector.get_include_energy_ranges(include_bin_ranges)
            self.logger.debug(
                'Mask selected. Including detector energy ranges: '
                + str(detector.include_energy_ranges))
        if not detector.include_energy_ranges:
            raise ValueError(
                'No value provided for include_energy_ranges. '
                'Provide them in the Diffraction Volume Length '
                'Measurement Configuration, or re-run the pipeline '
                'with the --interactive flag.')

        # Reduce the raw MCA data in 3 ways:
        # 1) sum of intensities in all detector bins
        # 2) max of intensities in detector bins after mask is applied
        # 3) sum of intensities in detector bins after mask is applied
        unmasked_sum = np.sum(mca_data, axis=1)
        mask = detector.mca_mask()
        masked_mca_data = mca_data[:,mask]
        masked_max = np.amax(masked_mca_data, axis=1)
        masked_sum = np.sum(masked_mca_data, axis=1)

        # Find the motor position corresponding roughly to the center
        # of the diffraction volume
        scanned_vals = dvl_config.scanned_vals
        scan_center = np.sum(scanned_vals * masked_sum) / np.sum(masked_sum)
        x = scanned_vals - scan_center

        # Normalize the data
        unmasked_sum = unmasked_sum / max(unmasked_sum)
        masked_max = masked_max / max(masked_max)
        masked_sum = masked_sum / max(masked_sum)

        # Fit the masked summed data with a gaussian
        fit = Fit.fit_data(masked_sum, ('constant', 'gaussian'), x=x)

        # Calculate / manually select diffraction volume length
        dvl = fit.best_values['sigma'] * detector.sigma_to_dvl_factor \
              - dvl_config.sample_thickness
        detector.fit_amplitude = fit.best_values['amplitude']
        detector.fit_center = scan_center + fit.best_values['center']
        detector.fit_sigma = fit.best_values['sigma']
        if detector.measurement_mode == 'manual':
            if interactive:
                _, dvl_bounds = select_mask_1d(
                    masked_sum, x=x,
                    label='Total (masked & normalized)',
                    preselected_index_ranges=[
                        (index_nearest(x, -dvl/2), index_nearest(x, dvl/2))],
                    title=('Click and drag to indicate the boundary '
                           'of the diffraction volume'),
                    xlabel=('Beam direction (offset from scan "center")'),
                    ylabel='MCA intensity (normalized)',
                    min_num_index_ranges=1,
                    max_num_index_ranges=1,
                    interactive=interactive)
                dvl_bounds = dvl_bounds[0]
                dvl = abs(x[dvl_bounds[1]] - x[dvl_bounds[0]])
            else:
                self.logger.warning(
                    'Cannot manually indicate DVL when running CHAP '
                    + 'non-interactively. '
                    + 'Using default DVL calcluation instead.')

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.set_title(f'Diffraction Volume ({detector.detector_name})')
            ax.set_xlabel('Beam direction (offset from scan "center")')
            ax.set_ylabel('MCA intensity (normalized)')
            ax.plot(x, masked_sum, label='total (masked & normalized)')
            ax.plot(x, fit.best_fit, label='gaussian fit (to total)')
            ax.plot(x, masked_max, label='maximum (masked)')
            ax.plot(x, unmasked_sum, label='total (unmasked)')
            ax.axvspan(
                fit.best_values['center']- dvl/2.,
                fit.best_values['center'] + dvl/2.,
                color='gray', alpha=0.5,
                label=f'diffraction volume ({detector.measurement_mode})')
            ax.legend()
            plt.figtext(
                0.5, 0.95,
                f'Diffraction volume length: {dvl:.2f}',
                fontsize='x-large',
                horizontalalignment='center',
                verticalalignment='bottom')
            if save_figures:
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                figfile = os.path.join(
                    outputdir, f'{detector.detector_name}_dvl.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return dvl


class LatticeParameterRefinementProcessor(Processor):
    """Processor to get a refined estimate for a sample's lattice
    parameters"""
    def process(self,
                data,
                config=None,
                save_figures=False,
                outputdir='.',
                inputdir='.',
                interactive=False):
        """Given a strain analysis configuration, return a copy
        contining refined values for the materials' lattice
        parameters."""
        ceria_calibration_config = self.get_config(
            data, 'edd.models.MCACeriaCalibrationConfig', inputdir=inputdir)
        try:
            strain_analysis_config = self.get_config(
                data, 'edd.models.StrainAnalysisConfig', inputdir=inputdir)
        except Exception as data_exc:
            # Local modules
            from CHAP.edd.models import StrainAnalysisConfig

            self.logger.info('No valid strain analysis config in input '
                             + 'pipeline data, using config parameter instead')
            try:
                strain_analysis_config = StrainAnalysisConfig(
                    **config, inputdir=inputdir)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        if len(strain_analysis_config.materials) > 1:
            msg = 'Not implemented for multiple materials'
            self.logger.error('Not implemented for multiple materials')
            raise NotImplementedError(msg)

        lattice_parameters = self.refine_lattice_parameters(
            strain_analysis_config, ceria_calibration_config, 0,
            interactive, save_figures, outputdir)
        self.logger.debug(f'Refined lattice parameters: {lattice_parameters}')

        strain_analysis_config.materials[0].lattice_parameters = \
            lattice_parameters
        return strain_analysis_config.dict()

    def refine_lattice_parameters(
            self, strain_analysis_config, ceria_calibration_config,
            detector_i, interactive, save_figures, outputdir):
        """Return refined values for the lattice parameters of the
        materials indicated in `strain_analysis_config`. Method: given
        a scan of a material, fit the peaks of each MCA
        spectrum. Based on those fitted peak locations, calculate the
        lattice parameters that would produce them. Return the
        avearaged value of the calculated lattice parameters across
        all spectra.

        :param strain_analysis_config: Strain analysis configuration
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :param detector_i: Index of the detector in
            `strain_analysis_config` whose data will be used for the
            refinement
        :type detector_i: int
        :param interactive: Boolean to indicate whether interactive
            matplotlib figures should be presented
        :type interactive: bool
        :param save_figures: Boolean to indicate whether figures
            indicating the selection should be saved
        :type save_figures: bool
        :param outputdir: Where to save figures (if `save_figures` is
            `True`)
        :type outputdir: str
        :returns: List of refined lattice parameters for materials in
            `strain_analysis_config`
        :rtype: list[numpy.ndarray]
        """
        # Third party modules
        from scipy.constants import physical_constants

        # Local modules
        from CHAP.edd.utils import (
            get_unique_hkls_ds,
            get_spectra_fits,
        )

        self.add_detector_calibrations(
            strain_analysis_config, ceria_calibration_config)

        detector = strain_analysis_config.detectors[detector_i]
        mca_bin_energies = self.get_mca_bin_energies(strain_analysis_config)
        mca_data = strain_analysis_config.mca_data()
        hkls, ds = get_unique_hkls_ds(
            strain_analysis_config.materials,
            tth_tol=detector.hkl_tth_tol,
            tth_max=detector.tth_max)

        self.select_material_params(
            strain_analysis_config, detector_i, mca_data, mca_bin_energies,
            interactive, save_figures, outputdir)
        self.logger.debug(
            'Starting lattice parameters: '
            + str(strain_analysis_config.materials[0].lattice_parameters))
        self.select_fit_mask_hkls(
            strain_analysis_config, detector_i, mca_data, mca_bin_energies,
            hkls, ds,
            interactive, save_figures, outputdir)

        (uniform_fit_centers, uniform_fit_centers_errors,
         uniform_fit_amplitudes, uniform_fit_amplitudes_errors,
         uniform_fit_sigmas, uniform_fit_sigmas_errors,
         uniform_best_fit, uniform_residuals,
         uniform_redchi, uniform_success,
         unconstrained_fit_centers, unconstrained_fit_centers_errors,
         unconstrained_fit_amplitudes, unconstrained_fit_amplitudes_errors,
         unconstrained_fit_sigmas, unconstrained_fit_sigmas_errors,
         unconstrained_best_fit, unconstrained_residuals,
         unconstrained_redchi, unconstrained_success) = \
            self.get_spectra_fits(
                strain_analysis_config, detector_i,
                mca_data, mca_bin_energies, hkls, ds)

        # Get the interplanar spacings measured for each fit HKL peak
        # at every point in the map.
        hc = 1e7 * physical_constants['Planck constant in eV/Hz'][0] \
             * physical_constants['speed of light in vacuum'][0]
        d_measured = hc / \
             (2.0 \
              * unconstrained_fit_centers \
              * np.sin(np.radians(detector.tth_calibrated / 2.0)))
        # Convert interplanar spacings to lattice parameters
        self.logger.warning('Implemented for cubic materials only!')
        fit_hkls  = np.asarray([hkls[i] for i in detector.hkl_indices])
        Rs = np.sqrt(np.sum(fit_hkls**2, 1))
        a_measured = Rs[:, None] * d_measured
        # Average all computed lattice parameters for every fit HKL
        # peak at every point in the map to get the refined estimate
        # for the material's lattice parameter
        a_refined = float(np.mean(a_measured))
        return [a_refined, a_refined, a_refined, 90.0, 90.0, 90.0]

    def get_mca_bin_energies(self, strain_analysis_config):
        """Return a list of the MCA bin energies for each detector.

        :param strain_analysis_config: Strain analysis configuration
            containing a list of detectors to return the bin energies
            for.
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :returns: List of MCA bin energies
        :rtype: list[numpy.ndarray]
        """
        raise RuntimeError(
            'Needs to be updated and tested for energy calibration.')
        mca_bin_energies = []
        for i, detector in enumerate(strain_analysis_config.detectors):
            mca_bin_energies.append(
                detector.slope_calibrated
                * np.linspace(0, detector.max_energy_kev, detector.num_bins)
                + detector.intercept_calibrated)
        return mca_bin_energies

    def add_detector_calibrations(
            self, strain_analysis_config, ceria_calibration_config):
        """Add calibrated quantities to the detectors configured in
        `strain_analysis_config`, modifying `strain_analysis_config`
        in place.

        :param strain_analysis_config: Strain analysisi configuration
            containing a list of detectors to add calibration values
            to.
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :param ceria_calibration_config: Configuration of a completed
            ceria calibration containing a list of detector swith the
            same names as those in `strain_analysis_config`
        :returns: None"""
        for detector in strain_analysis_config.detectors:
            calibration = [
                d for d in ceria_calibration_config.detectors \
                if d.detector_name == detector.detector_name][0]
            detector.add_calibration(calibration)

    def select_fit_mask_hkls(
            self, strain_analysis_config, detector_i,
            mca_data, mca_bin_energies, hkls, ds,
            interactive, save_figures, outputdir):
        """Select the maks & HKLs to use for fitting for each
        detector. Update `strain_analysis_config` with new values for
        `hkl_indices` and `include_bin_ranges` if needed.

        :param strain_analysis_config: Strain analysis configuration
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :param detector_i: Index of the detector in
            `strain_analysis_config` to select mask & HKLs for.
        :type detector_i: int
        :param mca_data: List of maps of MCA spectra for all detectors
            in `strain_analysis_config`
        :type mca_data: list[numpy.ndarray]
        :param mca_bin_energies: List of MCA bin energies for all
            detectors in `strain_analysis_config`
        :type mca_bin_energies: list[numpy.ndarray]
        :param hkls: Nominal HKL peak energy locations for the
            material in `strain_analysis_config`
        :type hkls: list[float]
        :param ds: Nominal d-spacing for the material in
            `strain_analysis_config`
        :type ds: list[float]
        :param interactive: Boolean to indicate whether interactive
            matplotlib figures should be presented
        :type interactive: bool
        :param save_figures: Boolean to indicate whether figures
            indicating the selection should be saved
        :type save_figures: bool
        :param outputdir: Where to save figures (if `save_figures` is
            `True`)
        :type outputdir: str
        :returns: None
        """
        if not interactive and not save_figures:
            return

        # Third party modules
        import matplotlib.pyplot as plt

        # Local modules
        from CHAP.edd.utils import select_mask_and_hkls

        detector = strain_analysis_config.detectors[detector_i]
        fig, include_bin_ranges, hkl_indices = \
            select_mask_and_hkls(
                mca_bin_energies[detector_i],
                np.sum(mca_data[detector_i], axis=0),
                hkls, ds,
                detector.tth_calibrated,
                detector.include_bin_ranges, detector.hkl_indices,
                detector.detector_name, mca_data[detector_i],
                calibration_bin_ranges=detector.calibration_bin_ranges,
                label='Sum of all spectra in the map',
                interactive=interactive)
        detector.include_energy_ranges = detector.get_include_energy_ranges(
            include_bin_ranges)
        detector.hkl_indices = hkl_indices
        if save_figures:
            fig.savefig(os.path.join(
                outputdir,
                f'{detector.detector_name}_strainanalysis_fit_mask_hkls.png'))
        plt.close()

    def select_material_params(self, strain_analysis_config, detector_i,
                               mca_data, mca_bin_energies,
                               interactive, save_figures, outputdir):
        """Select initial material parameters to use for determining
        nominal HKL peak locations. Modify `strain_analysis_config` in
        place if needed.

        :param strain_analysis_config: Strain analysis configuration
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :param detector_i: Index of the detector in
            `strain_analysis_config` to select mask & HKLs for.
        :type detector_i: int
        :param mca_data: List of maps of MCA spectra for all detectors
            in `strain_analysis_config`
        :type mca_data: list[numpy.ndarray]
        :param mca_bin_energies: List of MCA bin energies for all
            detectors in `strain_analysis_config`
        :type mca_bin_energies: list[numpy.ndarray]
        :param interactive: Boolean to indicate whether interactive
            matplotlib figures should be presented
        :type interactive: bool
        :param save_figures: Boolean to indicate whether figures
            indicating the selection should be saved
        :type save_figures: bool
        :param outputdir: Where to save figures (if `save_figures` is
            `True`)
        :type outputdir: str
        :returns: None
        """
        if not interactive and not save_figures:
            return

        # Third party modules
        import matplotlib.pyplot as plt

        # Local modules
        from CHAP.edd.utils import select_material_params

        fig, strain_analysis_config.materials = select_material_params(
            mca_bin_energies[detector_i], np.sum(mca_data[detector_i], axis=0),
            strain_analysis_config.detectors[detector_i].tth_calibrated,
            strain_analysis_config.materials,
            label='Sum of all spectra in the map', interactive=interactive)
        self.logger.debug(
                f'materials: {strain_analysis_config.materials}')
        if save_figures:
            detector_name = \
                strain_analysis_config.detectors[detector_i].detector_name
            fig.savefig(os.path.join(
                outputdir,
                f'{detector_name}_strainanalysis_material_config.png'))
        plt.close()

    def get_spectra_fits(
            self, strain_analysis_config, detector_i,
            mca_data, mca_bin_energies, hkls, ds):
        """Return uniform and unconstrained fit results for all
        spectra from a single detector.

        :param strain_analysis_config: Strain analysis configuration
        :type strain_analysis_config: CHAP.edd.models.StrainAnalysisConfig
        :param detector_i: Index of the detector in
            `strain_analysis_config` to select mask & HKLs for.
        :type detector_i: int
        :param mca_data: List of maps of MCA spectra for all detectors
            in `strain_analysis_config`
        :type mca_data: list[numpy.ndarray]
        :param mca_bin_energies: List of MCA bin energies for all
            detectors in `strain_analysis_config`
        :type mca_bin_energies: list[numpy.ndarray]
        :param hkls: Nominal HKL peak energy locations for the
            material in `strain_analysis_config`
        :type hkls: list[float]
        :param ds: Nominal d-spacing for the material in
            `strain_analysis_config`
        :type ds: list[float]
        :returns: Uniform and unconstrained centers, amplitdues,
            sigmas (and errors for all three), best fits, residuals
            between the best fits and the input spectra, reduced chi,
            and fit success statuses.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray, numpy.ndarray,
            numpy.ndarray, numpy.ndarray]
        """
        # Local modules
        from CHAP.edd.utils import (
            get_peak_locations,
            get_spectra_fits,
        )

        detector = strain_analysis_config.detectors[detector_i]
        self.logger.debug(
            f'Fitting spectra from detector {detector.detector_name}')
        mask = detector.mca_mask()
        energies = mca_bin_energies[detector_i][mask]
        intensities = np.empty(
            (*strain_analysis_config.map_config.shape, len(energies)),
            dtype='uint16')
        for j, map_index in \
            enumerate(np.ndindex(strain_analysis_config.map_config.shape)):
            intensities[map_index] = \
                mca_data[detector_i][j].astype('uint16')[mask]
        fit_hkls  = np.asarray([hkls[i] for i in detector.hkl_indices])
        fit_ds  = np.asarray([ds[i] for i in detector.hkl_indices])
        peak_locations = get_peak_locations(
            fit_ds, detector.tth_calibrated)
        return get_spectra_fits(
            intensities, energies, peak_locations, detector)


class MCACeriaCalibrationProcessor(Processor):
    """A Processor using a CeO2 scan to obtain tuned values for the
    bragg diffraction angle and linear correction parameters for MCA
    channel energies for an EDD experimental setup.
    """

    def process(self,
                data,
                config=None,
                quadratic_energy_calibration=False,
                save_figures=False,
                inputdir='.',
                outputdir='.',
                interactive=False):
        """Return tuned values for 2&theta and linear correction
        parameters for the MCA channel energies.

        :param data: Input configuration for the raw data & tuning
            procedure.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.MCACeriaCalibrationConfig, defaults to
            None.
        :type config: dict, optional
        :param quadratic_energy_calibration: Adds a quadratic term to
            the detector channel index to energy conversion, defaults
            to `False` (linear only).
        :type quadratic_energy_calibration: bool, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'.
        :type outputdir: str, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to '.'.
        :type inputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :raises RuntimeError: Invalid or missing input configuration.
        :return: Original configuration with the tuned values for
            2&theta and the linear correction parameters added.
        :rtype: dict[str,float]
        """
        try:
            calibration_config = self.get_config(
                data, 'edd.models.MCACeriaCalibrationConfig',
                inputdir=inputdir)
        except Exception as data_exc:
            self.logger.info('No valid calibration config in input pipeline '
                             'data, using config parameter instead.')
            try:
                # Local modules
                from CHAP.edd.models import MCACeriaCalibrationConfig

                calibration_config = MCACeriaCalibrationConfig(
                    **config, inputdir=inputdir)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        self.logger.debug(f'In process: save_figures = {save_figures}; '
                          f'interactive = {interactive}')

        for detector in calibration_config.detectors:
            tth, energy_calibration_coeffs = self.calibrate(
                calibration_config, detector, 
                quadratic_energy_calibration=quadratic_energy_calibration,
                save_figures=save_figures, interactive=interactive,
                outputdir=outputdir)
            detector.tth_calibrated = tth
            detector.energy_calibration_coeffs = energy_calibration_coeffs

        return calibration_config.dict()

    def calibrate(self,
                  calibration_config,
                  detector,
                  quadratic_energy_calibration=False,
                  save_figures=False,
                  outputdir='.',
                  interactive=False):
        """Iteratively calibrate 2&theta by fitting selected peaks of
        an MCA spectrum until the computed strain is sufficiently
        small. Use the fitted peak locations to determine linear
        correction parameters for the MCA channel energies.

        :param calibration_config: Object configuring the CeO2
            calibration procedure for an MCA detector.
        :type calibration_config:
            CHAP.edd.models.MCACeriaCalibrationConfig
        :param detector: A single MCA detector element configuration.
        :type detector: CHAP.edd.models.MCAElementCalibrationConfig
        :param quadratic_energy_calibration: Adds a quadratic term to
            the detector channel index to energy conversion, defaults
            to `False` (linear only).
        :type quadratic_energy_calibration: bool, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :raises ValueError: No value provided for included bin ranges
            or the fitted HKLs for the MCA detector element.
        :return: Calibrated values of 2&theta and the correction
            parameters for MCA channel energies: tth, and
            energy_calibration_coeffs.
        :rtype: float, [float, float, float]
        """
        # Local modules
        if interactive or save_figures:
            from CHAP.edd.utils import (
                select_tth_initial_guess,
                select_mask_and_hkls,
            )
        from CHAP.edd.utils import get_peak_locations
        from CHAP.utils.fit import Fit

        # Get the unique HKLs and lattice spacings for the calibration
        # material
        hkls, ds = calibration_config.material.unique_hkls_ds(
            tth_tol=detector.hkl_tth_tol, tth_max=detector.tth_max)

        # Collect raw MCA data of interest
        mca_bin_energies = detector.energies
        mca_data = calibration_config.mca_data(detector)

        # Blank out data below 25 keV as well as the last bin
        energy_mask = np.where(mca_bin_energies >= 25.0, 1, 0)
        energy_mask[-1] = 0
        mca_data = mca_data*energy_mask

        # Adjust initial tth guess
        if save_figures:
            filename = os.path.join(
               outputdir,
               f'{detector.detector_name}_calibration_tth_initial_guess.png')
        else:
            filename = None
        detector.tth_initial_guess = select_tth_initial_guess(
            mca_bin_energies, mca_data, hkls, ds,
            detector.tth_initial_guess, interactive, filename)
        self.logger.debug(f'tth_initial_guess = {detector.tth_initial_guess}')

        # Select mask & HKLs for fitting
        if save_figures:
            filename = os.path.join(
                outputdir,
                f'{detector.detector_name}_calibration_fit_mask_hkls.png')
        include_bin_ranges, hkl_indices = select_mask_and_hkls(
            mca_bin_energies, mca_data, hkls, ds,
            detector.tth_initial_guess, detector.include_bin_ranges,
            detector.hkl_indices, detector.detector_name,
            flux_energy_range=calibration_config.flux_file_energy_range,
            label='MCA data', interactive=interactive, filename=filename)
        detector.include_energy_ranges = detector.get_include_energy_ranges(
            include_bin_ranges)
        detector.hkl_indices = hkl_indices
        self.logger.debug(
            f'include_energy_ranges = {detector.include_energy_ranges}')
        if not detector.include_energy_ranges:
            raise ValueError(
                'No value provided for include_energy_ranges. '
                'Provide them in the MCA Ceria Calibration Configuration '
                'or re-run the pipeline with the --interactive flag.')
        if not detector.hkl_indices:
            raise ValueError(
                'No value provided for hkl_indices. Provide them in '
                'the detector\'s MCA Ceria Calibration Configuration or '
                're-run the pipeline with the --interactive flag.')
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
        # Restrict the range for the centers with some margin to 
        # prevent having centers near the edge of the fitting range
        delta = 0.1 * (fit_mca_energies[-1]-fit_mca_energies[0])
        centers_range = (
            max(0.0, fit_mca_energies[0]-delta), fit_mca_energies[-1]+delta)
        fit_hkls  = np.asarray([hkls[i] for i in detector.hkl_indices])
        fit_ds  = np.asarray([ds[i] for i in detector.hkl_indices])
        c_1 = fit_hkls[:,0]**2 + fit_hkls[:,1]**2 + fit_hkls[:,2]**2
        tth = float(detector.tth_initial_guess)
        fit_E0 = get_peak_locations(fit_ds, tth)
        for iter_i in range(calibration_config.max_iter):
            self.logger.debug(
                f'Tuning tth: iteration no. {iter_i}, starting value = {tth} ')

            # Perform the uniform fit first

            # Get expected peak energy locations for this iteration's
            # starting value of tth
            _fit_E0 = get_peak_locations(fit_ds, tth)

            # Run the uniform fit
            fit = Fit(fit_mca_intensities, x=fit_mca_energies)
            fit.create_multipeak_model(
                _fit_E0, fit_type='uniform', background=detector.background,
                centers_range=centers_range, fwhm_min=0.1, fwhm_max=1.0)
            fit.fit()

            # Extract values of interest from the best values for the
            # uniform fit parameters
            uniform_best_fit = fit.best_fit
            uniform_residual = fit.residual
            uniform_fit_centers = [
                fit.best_values[f'peak{i+1}_center']
                for i in range(len(fit_hkls))]
            uniform_a = fit.best_values['scale_factor']
            uniform_strain = np.log(
                (uniform_a
                 / calibration_config.material.lattice_parameters)) # CeO2 is cubic, so this is fine here.

            # Next, perform the unconstrained fit

            # Use the peak parameters from the uniform fit as the
            # initial guesses for peak locations in the unconstrained
            # fit
            fit.create_multipeak_model(fit_type='unconstrained')
            fit.fit()

            # Extract values of interest from the best values for the
            # unconstrained fit parameters
            unconstrained_best_fit = fit.best_fit
            unconstrained_residual = fit.residual
            unconstrained_fit_centers = np.array(
                [fit.best_values[f'peak{i+1}_center']
                 for i in range(len(fit_hkls))])
            unconstrained_a = np.sqrt(c_1)*abs(get_peak_locations(
                unconstrained_fit_centers, tth))
            unconstrained_strains = np.log(
                (unconstrained_a
                 / calibration_config.material.lattice_parameters))
            unconstrained_strain = np.mean(unconstrained_strains)
            unconstrained_tth = tth * (1.0 + unconstrained_strain)

            # Update tth for the next iteration of tuning
            prev_tth = tth
            tth = float(unconstrained_tth)

            # Stop tuning tth at this iteration if differences are
            # small enough
            if abs(tth - prev_tth) < calibration_config.tune_tth_tol:
                break

        # Fit line to expected / computed peak locations from the last
        # unconstrained fit.
        a_init, b_init, c_init = detector.energy_calibration_coeffs
        if quadratic_energy_calibration:
            fit = Fit.fit_data(
                _fit_E0, 'quadratic', x=unconstrained_fit_centers,
                nan_policy='omit')
            a_fit = fit.best_values['a']
            b_fit = fit.best_values['b']
            c_fit = fit.best_values['c']
        else:
            fit = Fit.fit_data(
                _fit_E0, 'linear', x=unconstrained_fit_centers,
                nan_policy='omit')
            a_fit = 0.0
            b_fit = fit.best_values['slope']
            c_fit = fit.best_values['intercept']
        a_final = float(b_init**2*a_fit)
        b_final = float(2*b_init*c_init*a_fit + b_init*b_fit)
        c_final = float(c_init**2*a_fit + c_init*b_fit + c_fit)

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 2, sharex='all', figsize=(11, 8.5))
            fig.suptitle(
                f'Detector {detector.detector_name} Ceria Calibration')

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
                          label='Single strain')
            axs[0,0].plot(fit_mca_energies, unconstrained_best_fit,
                          label='Unconstrained')
            #axs[0,0].plot(fit_mca_energies, MISSING?, label='least squares')
            axs[0,0].plot(fit_mca_energies, fit_mca_intensities,
                          label='Flux-corrected & wasked MCA data')
            axs[0,0].legend()

            # Lower left axes: fit residuals
            axs[1,0].set_title('Fit Residuals')
            axs[1,0].set_xlabel('Energy (keV)')
            axs[1,0].set_ylabel('Residual (a.u)')
            axs[1,0].plot(fit_mca_energies,
                          uniform_residual,
                          label='Single strain')
            axs[1,0].plot(fit_mca_energies,
                          unconstrained_residual,
                          label='Unconstrained')
            axs[1,0].legend()

            # Upper right axes: E vs strain for each fit
            axs[0,1].set_title('HKL Energy vs. Microstrain')
            axs[0,1].set_xlabel('Energy (keV)')
            axs[0,1].set_ylabel('Strain (\u03BC\u03B5)')
            axs[0,1].axhline(uniform_strain * 1e6,
                             linestyle='--', label='Single strain')
            axs[0,1].plot(fit_E0, unconstrained_strains * 1e6,
                          color='C1', marker='s', label='Unconstrained')
            axs[0,1].axhline(unconstrained_strain * 1e6,
                             color='C1', linestyle='--',
                             label='Unconstrained: unweighted mean')
            axs[0,1].legend()

            # Lower right axes: theoretical HKL E vs fit HKL E for
            # each fit
            axs[1,1].set_title('Theoretical vs. Fit HKL Energies')
            axs[1,1].set_xlabel('Energy (keV)')
            axs[1,1].set_ylabel('Energy (keV)')
            axs[1,1].plot(fit_E0, uniform_fit_centers,
                          c='b', marker='o', ms=6, mfc='none', ls='',
                          label='Single strain')
            axs[1,1].plot(fit_E0, unconstrained_fit_centers,
                          c='k', marker='+', ms=6, ls='',
                          label='Unconstrained')
            if quadratic_energy_calibration:
                axs[1,1].plot(fit_E0, (a_fit*_fit_E0 + b_fit)*_fit_E0 + c_fit,
                              color='C1', label='Unconstrained: quadratic fit')
            else:
                axs[1,1].plot(fit_E0, b_fit*_fit_E0 + c_fit, color='C1',
                              label='Unconstrained: linear fit')
            axs[1,1].legend()

            # Add a text box showing final calibrated values
            txt = 'Calibrated values:' \
                  f'\nTakeoff angle:\n    {tth:.5f}$^\circ$'
            if True or recalibrate_energy:
                if quadratic_energy_calibration:
                    txt += '\nQuadratic coefficient:' \
                           f'\n    {a_final:.5e} $keV$/channel$^2$'
                txt += '\nLinear coefficient:' \
                       f'\n    {b_final:.5f} $keV$/channel' \
                       f'\nConstant offset:\n    {c_final:.5f}'
            axs[1,1].text(
                0.98, 0.02, txt,
                ha='right', va='bottom', ma='left',
                transform=axs[1,1].transAxes,
                bbox=dict(boxstyle='round',
                          ec=(1., 0.5, 0.5),
                          fc=(1., 0.8, 0.8, 0.8)))

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(
                    outputdir,
                    f'{detector.detector_name}_ceria_calibration_fits.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        # Return the calibrated 2&theta value and the final energy
        # calibration coefficients
        return tth, [a_final, b_final, c_final]


class MCAEnergyCalibrationProcessor(Processor):
    """Processor to return parameters for linearly transforming MCA
    channel indices to energies (in keV). Procedure: provide a
    spectrum from the MCA element to be calibrated and the theoretical
    location of at least one peak present in that spectrum (peak
    locations must be given in keV). It is strongly recommended to use
    the location of fluorescence peaks whenever possible, _not_
    diffraction peaks, as this Processor does not account for
    2&theta."""
    def process(self,
                data,
                peak_energies,
                max_peak_index,
                config=None,
                fit_index_ranges=None,
                peak_index_fit_delta=1.0,
                max_energy_kev=200.0,
                save_figures=False,
                interactive=False,
                inputdir='.',
                outputdir='.'):
        """For each detector in the `MCACeriaCalibrationConfig`
        provided with `data`, fit the specified peaks in the MCA
        spectrum specified. Using the difference between the provided
        peak locations and the fit centers of those peaks, compute
        the correction coefficients to convert uncalibrated MCA
        channel energies to calibrated channel energies. Set the
        values in the calibration config provided for
        `energy_calibration_coeffs` to these values (for each detector)
        and return the updated configuration.

        :param data: A Ceria Calibration configuration.
        :type data: PipelineData
        :param peak_energies: Theoretical locations of peaks to use
            for calibrating the MCA channel energies. It is _strongly_
            recommended to use fluorescence peaks.
        :type peak_energies: list[float], optional
        :param max_peak_index: Index of the peak in `peak_energies`
            with the highest amplitude
        :type max_peak_index: int
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.MCACeriaCalibrationConfig, defaults to
            `None`.
        :type config: dict, optional
        :param fit_index_ranges: Explicit ranges of uncalibrated MCA
            channel index ranges to include when performing a fit of
            the given peaks to the provied MCA spectrum. Use this
            parameter or select it interactively by running a pipeline
            with `config.interactive: True`.
        :type fit_index_ranges: list[list[int]], optional
        :param peak_index_fit_delta: Set boundaries on the fit peak
            centers when performing the fit. The min/max possible
            values for the peak centers will be the initial values
            &pm; `peak_index_fit_delta`. Defaults to `20`.
        :type peak_index_fit_delta: int, optional
        :param max_energy_kev: Maximum channel energy of the MCA in
            keV, defaults to 200.0.
        :type max_energy_kev: float, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :returns: Dictionary representing the energy-calibrated
            version of the ceria calibration configuration.
        :rtype: dict
        """
        # Local modules
        from CHAP.utils.general import (
            is_int,
            is_int_pair,
            is_num_series,
        )

        # Validate arguments: fit_ranges & interactive
        if not is_num_series(peak_energies, gt=0):
            self.logger.exception(
                ValueError(
                    f'Invalid parameter `peak_energies`: {peak_energies}'),
                exc_info=False)
        if len(peak_energies) < 2:
            self.logger.exception(
                ValueError('Invalid parameter `peak_energies`: '
                           f'{peak_energies} (at least two values required)'),
                exc_info=False)
        if not is_int(max_peak_index, ge=0, lt=len(peak_energies)):
            self.logger.exception(
                ValueError(
                    f'Invalid parameter `max_peak_index`: {max_peak_index}'),
                exc_info=False)
        if fit_index_ranges is None and not interactive:
            self.logger.exception(
                RuntimeError(
                    'If `fit_index_ranges` is not explicitly provided, '
                    + self.__class__.__name__
                    + ' must be run with `interactive=True`.'),
                exc_info=False)
        if (fit_index_ranges is not None
                and (not isinstance(fit_index_ranges, list)
                     or any(not is_int_pair(v, ge=0)
                            for v in fit_index_ranges))):
            self.logger.exception(
                ValueError('Invalid parameter `fit_index_ranges`: '
                           f'{fit_index_ranges}'),
                exc_info=False)
        max_peak_energy = peak_energies[max_peak_index]
        peak_energies.sort()
        max_peak_index = peak_energies.index(max_peak_energy)

        # Validate arguments: load the calibration configuration
        try:
            calibration_config = self.get_config(
                data, 'edd.models.MCACeriaCalibrationConfig',
                inputdir=inputdir)
        except Exception as data_exc:
            self.logger.info('No valid calibration config in input pipeline '
                             'data, using config parameter instead.')
            try:
                # Local modules
                from CHAP.edd.models import MCACeriaCalibrationConfig

                calibration_config = MCACeriaCalibrationConfig(
                    **config, inputdir=inputdir)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        # Calibrate detector channel energies based on fluorescence peaks.
        for detector in calibration_config.detectors:
            energy_calibration_coeffs = self.calibrate(
                calibration_config, detector, fit_index_ranges,
                peak_energies, max_peak_index, peak_index_fit_delta,
                max_energy_kev, save_figures, interactive, outputdir)
            detector.energy_calibration_coeffs = energy_calibration_coeffs

        return calibration_config.dict()

    def calibrate(self, calibration_config, detector, fit_index_ranges,
            peak_energies, max_peak_index, peak_index_fit_delta,
            max_energy_kev, save_figures, interactive, outputdir):
        """Return energy_calibration_coeffs (a, b, and c) for
        quadratically converting the current detector's MCA channels
        to bin energies.

        :param calibration_config: Calibration configuration.
        :type calibration_config: MCACeriaCalibrationConfig
        :param detector: Configuration of the current detector.
        :type detector: MCAElementCalibrationConfig
        :param fit_index_ranges: Explicit ranges of uncalibrated MCA
            channel index ranges to include when performing a fit of
            the given peaks to the provied MCA spectrum. Use this
            parameter or select it interactively by running a pipeline
            with `config.interactive: True`.
        :type fit_ranges: list[list[float]]
        :param peak_energies: Theoretical locations of peaks to use
            for calibrating the MCA channel energies. It is _strongly_
            recommended to use fluorescence peaks.
        :type peak_energies: list[float]
        :param max_peak_index: Index of the peak in `peak_energies`
            with the highest amplitude.
        :type peak_energies: int
        :param peak_index_fit_delta: Set boundaries on the fit peak
            centers when performing the fit. The min/max possible
            values for the peak centers will be the initial values
            &pm; `peak_index_fit_delta`.
        :type peak_index_fit_delta: int
        :param max_energy_kev: Maximum channel energy of the MCA in
            keV, defaults to 200.0.
        :type max_energy_kev: float
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor.
        :type save_figures: bool
        :param interactive: Allows for user interactions.
        :type interactive: bool
        :param outputdir: Directory to which any output figures will
            be saved.
        :type outputdir: str
        :returns: Slope and intercept for linearly correcting the
            detector's MCA channels to bin energies.
        :rtype: tuple[float, float]
        """
        # Local modules
        from CHAP.utils.fit import Fit
        from CHAP.utils.general import (
            index_nearest,
            index_nearest_down,
            index_nearest_up,
            select_mask_1d,
        )

        self.logger.debug(f'Calibrating detector {detector.detector_name}')
        spectrum = calibration_config.mca_data(detector)
        uncalibrated_energies = np.linspace(
            0, max_energy_kev, detector.num_bins)
        bins = np.arange(detector.num_bins, dtype=np.int16)

        # Blank out data below bin 500 (~25keV) as well as the last bin
        energy_mask = np.ones(detector.num_bins, dtype=np.int16)
        energy_mask[:500] = 0
        energy_mask[-1] = 0
        spectrum = spectrum*energy_mask

        # Select the mask/detector channel ranges for fitting
        if save_figures:
            filename = os.path.join(
                outputdir,
                f'{detector.detector_name}_mca_energy_calibration_mask.png')
        else:
            filename = None
        mask, fit_index_ranges = select_mask_1d(
            spectrum, x=bins, preselected_index_ranges=fit_index_ranges,
            xlabel='Detector channel', ylabel='Intensity',
            min_num_index_ranges=1, interactive=False,#RV interactive,
            filename=filename)
        self.logger.debug(
            f'Selected index ranges to fit: {fit_index_ranges}')

        # Get the intial peak positions for fitting
        if save_figures:
            filename = os.path.join(
                outputdir,
                f'{detector.detector_name}'
                    '_mca_energy_calibration_initial_peak_positions.png')
        else:
            filename = None
        input_indices = [index_nearest(uncalibrated_energies, energy)
                         for energy in peak_energies]
        initial_peak_indices = self._get_initial_peak_positions(
            spectrum*mask.astype(np.int32), fit_index_ranges, input_indices,
            max_peak_index, interactive, filename, detector.detector_name)

        spectrum_fit = Fit(spectrum[mask], x=bins[mask])
        for i, index in enumerate(initial_peak_indices):
            spectrum_fit.add_model(
                'gaussian', prefix=f'peak{i+1}_', parameters=(
                    {'name': 'amplitude', 'min': 0.0},
                    {'name': 'center', 'value': index,
                     'min': index - peak_index_fit_delta,
                     'max': index + peak_index_fit_delta},
                    {'name': 'sigma', 'value': 1.0, 'min': 0.2, 'max': 8.0},
                ))
        self.logger.debug('Fitting spectrum')
        spectrum_fit.fit()
        
        fit_peak_indices = sorted([
            spectrum_fit.best_values[f'peak{i+1}_center']
            for i in range(len(initial_peak_indices))])
        self.logger.debug(f'Fit peak centers: {fit_peak_indices}')

        #RV for now stick with a linear energy correction
        energy_fit = Fit.fit_data(
            peak_energies, 'linear', x=fit_peak_indices, nan_policy='omit')
        a = 0.0
        b = float(energy_fit.best_values['slope'])
        c = float(energy_fit.best_values['intercept'])

        # Reference plot to see fit results:
        if interactive or save_figures:
            # Third part modules
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 2, figsize=(11, 4.25))
            fig.suptitle(
                f'Detector {detector.detector_name} Energy Calibration')
            # Left plot: raw MCA data & best fit of peaks
            axs[0].set_title(f'MCA Spectrum Peak Fit')
            axs[0].set_xlabel('Detector channel')
            axs[0].set_ylabel('Intensity (a.u)')
            axs[0].plot(bins[mask], spectrum[mask], 'b.', label='MCA data')
            axs[0].plot(
                bins[mask], spectrum_fit.best_fit, 'r', label='Best fit')
            axs[0].legend()
            # Right plot: linear fit of theoretical peak energies vs
            # fit peak locations
            axs[1].set_title(
                'Channel Energies vs. Channel Indices')
            axs[1].set_xlabel('Detector channel')
            axs[1].set_ylabel('Channel energy (keV)')
            axs[1].plot(fit_peak_indices, peak_energies,
                        c='b', marker='o', ms=6, mfc='none', ls='',
                        label='Initial peak positions')
            axs[1].plot(fit_peak_indices, energy_fit.best_fit,
                        c='k', marker='+', ms=6, ls='',
                        label='Fitted peak positions')
            axs[1].plot(bins[mask], b*bins[mask] + c, 'r',
                        label='Best linear fit')
            axs[1].legend()
            # Add text box showing computed values of linear E
            # correction parameters
            axs[1].text(
                0.98, 0.02,
                'Calibrated values:\n\n'
                    f'Linear coefficient:\n    {b:.5f} $keV$/channel\n\n'
                    f'Constant offset:\n    {c:.5f} $keV$',
                ha='right', va='bottom', ma='left',
                transform=axs[1].transAxes,
                bbox=dict(boxstyle='round',
                          ec=(1., 0.5, 0.5),
                          fc=(1., 0.8, 0.8, 0.8)))

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(
                    outputdir,
                    f'{detector.detector_name}_energy_calibration_fit.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return [a, b, c]

    def _get_initial_peak_positions(
            self, y, index_ranges, input_indices, input_max_peak_index,
            interactive, filename, detector_name, reset_flag=0):
        # Third party modules
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox, Button

        def change_fig_title(title):
            if fig_title:
                fig_title[0].remove()
                fig_title.pop()
            fig_title.append(plt.figtext(*title_pos, title, **title_props))

        def change_error_text(error=''):
            if error_texts:
                error_texts[0].remove()
                error_texts.pop()
            error_texts.append(plt.figtext(*error_pos, error, **error_props))

        def reset(event):
            """Callback function for the "Reset" button."""
            peak_indices.clear()
            plt.close()

        def confirm(event):
            """Callback function for the "Confirm" button."""
            if error_texts:
                error_texts[0].remove()
                error_texts.pop()
            plt.close()

        def find_peaks(min_height=0.05, min_width=5, tolerance=0.05):
            # Third party modules
            from scipy.signal import find_peaks as find_peaks_scipy

            # Find peaks
            peaks = find_peaks_scipy(y, height=min_height,
                prominence=0.05*y.max(), width=min_width)
            available_peak_indices = list(peaks[0])
            max_peak_index = np.asarray(peaks[1]).argmax()
            ratio = (available_peak_indices[max_peak_index]
                     / input_indices[input_max_peak_index])
            peak_indices = [-1]*num_peak
            peak_indices[input_max_peak_index] = \
                    available_peak_indices[max_peak_index]
            available_peak_indices.pop(max_peak_index)
            for i, input_index in enumerate(input_indices):
                if i != input_max_peak_index:
                    index_guess = int(input_index * ratio)
                    for index in available_peak_indices.copy():
                        if abs(index_guess-index) < tolerance*index:
                            index_guess = index
                            available_peak_indices.remove(index)
                            break
                    peak_indices[i] = index_guess
            return peak_indices

        def select_peaks():
            """Manually select initial peak indices."""
            peak_indices = []
            while len(set(peak_indices)) < num_peak:
                change_fig_title(f'Select {num_peak} peak positions')
                peak_indices = [
                    int(pt[0]) for pt in plt.ginput(num_peak, timeout=15)]
                if len(set(peak_indices)) < num_peak:
                    error_text = f'Choose {num_peak} unique position'
                    peak_indices.clear()
                outside_indices = []
                for index in peak_indices:
                    if not any(True if low <= index <= upp else False
                           for low, upp in index_ranges):
                        outside_indices.append(index)
                if len(outside_indices) == 1:
                    error_text = \
                        f'Index {outside_indices[0]} outside of selection ' \
                        f'region ({index_ranges}), try again'
                    peak_indices.clear()
                elif outside_indices:
                    error_text = \
                        f'Indices {outside_indices} outside of selection ' \
                        'region, try again'
                    peak_indices.clear()
                if not peak_indices:
                    plt.close()
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.set_xlabel('Detector channel', fontsize='x-large')
                    ax.set_ylabel('Intensity', fontsize='x-large')
                    ax.set_xlim(index_ranges[0][0], index_ranges[-1][1])
                    fig.subplots_adjust(bottom=0.0, top=0.85)
                    ax.plot(np.arange(y.size), y, color='k')
                    fig.subplots_adjust(bottom=0.2)
                    change_error_text(error_text)
                plt.draw()
            return peak_indices

        peak_indices = []
        fig_title = []
        error_texts = []

        y = np.asarray(y)
        if detector_name is None:
            detector_name = ''
        elif not isinstance(detector_name, str):
            raise ValueError(
                f'Invalid parameter `detector_name`: {detector_name}')
        elif not reset_flag:
            detector_name = f' on detector {detector_name}'
        num_peak = len(input_indices)

        # Setup the Matplotlib figure
        title_pos = (0.5, 0.95)
        title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        error_pos = (0.5, 0.90)
        error_props = {'fontsize': 'x-large', 'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        selected_peak_props = {
            'color': 'red', 'linestyle': '-', 'linewidth': 2,
            'marker': 10, 'markersize': 10, 'fillstyle': 'full'}

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.plot(np.arange(y.size), y, color='k')
        ax.set_xlabel('Detector channel', fontsize='x-large')
        ax.set_ylabel('Intensity', fontsize='x-large')
        ax.set_xlim(index_ranges[0][0], index_ranges[-1][1])
        fig.subplots_adjust(bottom=0.0, top=0.85)

        if not interactive:

            peak_indices += find_peaks()

            for index in peak_indices:
                ax.axvline(index, **selected_peak_props)
            change_fig_title('Initial peak positions from peak finding '
                             f'routine{detector_name}')

        else:

            fig.subplots_adjust(bottom=0.2)

            # Get initial peak indices
            if not reset_flag:
                peak_indices += find_peaks()
                change_fig_title('Initial peak positions from peak finding '
                                 f'routine{detector_name}')
            if peak_indices:
                for index in peak_indices:
                    if not any(True if low <= index <= upp else False
                           for low, upp in index_ranges):
                        peak_indices.clear
                        break
            if not peak_indices:
                peak_indices += select_peaks()
                change_fig_title(
                    'Selected initial peak positions{detector_name}')

            for index in peak_indices:
                ax.axvline(index, **selected_peak_props)

            # Setup "Reset" button
            if not reset_flag:
                reset_btn = Button(
                    plt.axes([0.1, 0.05, 0.15, 0.075]), 'Manually select')
            else:
                reset_btn = Button(
                    plt.axes([0.1, 0.05, 0.15, 0.075]), 'Reset')
            reset_cid = reset_btn.on_clicked(reset)

            # Setup "Confirm" button
            confirm_btn = Button(
                plt.axes([0.75, 0.05, 0.15, 0.075]), 'Confirm')
            confirm_cid = confirm_btn.on_clicked(confirm)

            plt.show()

            # Disconnect all widget callbacks when figure is closed
            reset_btn.disconnect(reset_cid)
            confirm_btn.disconnect(confirm_cid)

            # ... and remove the buttons before returning the figure
            reset_btn.ax.remove()
            confirm_btn.ax.remove()

        if filename is not None:
            fig_title[0].set_in_layout(True)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(filename)
        plt.close()

        if interactive and len(peak_indices) != num_peak:
            reset_flag += 1
            return self._get_initial_peak_positions(
                y, index_ranges, input_indices, max_peak_index,interactive,
                filename, detector_name, reset_flag=reset_flag)
        return peak_indices


class MCADataProcessor(Processor):
    """A Processor to return data from an MCA, restuctured to
    incorporate the shape & metadata associated with a map
    configuration to which the MCA data belongs, and linearly
    transformed according to the results of a ceria calibration.
    """

    def process(self,
                data,
                config=None,
                save_figures=False,
                inputdir='.',
                outputdir='.',
                interactive=False):
        """Process configurations for a map and MCA detector(s), and
        return the calibrated MCA data collected over the map.

        :param data: Input map configuration and results of ceria
            calibration.
        :type data: list[dict[str,object]]
        :return: Calibrated and flux-corrected MCA data.
        :rtype: nexusformat.nexus.NXentry
        """

        print(f'data:\n{data}')
        exit('Done Here')
        map_config = self.get_config(
            data, 'common.models.map.MapConfig', inputdir=inputdir)
        ceria_calibration_config = self.get_config(
            data, 'edd.models.MCACeriaCalibrationConfig', inputdir=inputdir)
        nxroot = self.get_nxroot(map_config, ceria_calibration_config)

        return nxroot

    def get_nxroot(self, map_config, calibration_config):
        """Get a map of the MCA data collected by the scans in
        `map_config`. The MCA data will be calibrated and
        flux-corrected according to the parameters included in
        `calibration_config`. The data will be returned along with
        relevant metadata in the form of a NeXus structure.

        :param map_config: The map configuration.
        :type map_config: CHAP.common.models.MapConfig.
        :param calibration_config: The calibration configuration.
        :type calibration_config:
            CHAP.edd.models.MCACeriaCalibrationConfig
        :return: A map of the calibrated and flux-corrected MCA data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXinstrument,
            NXroot,
        )

        # Local modules
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
                * np.linspace(0, detector.max_energy_kev, detector.num_bins) \
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


class MCACalibratedDataPlotter(Processor):
    """Convenience Processor for quickly visualizing calibrated MCA
       data from a single scan. Returns None!"""
    def process(self, data, spec_file, scan_number, scan_step_index=None,
                material=None, save_figures=False, interactive=False,
                outputdir='.'):
        """Show a maplotlib figure of the MCA data fom the scan
        provided on a calibrated energy axis. If `scan_step_index` is
        None, a plot of the sum of all spectra across the whole scan
        will be shown.

        :param data: PipelineData containing an MCA calibration.
        :type data: list[PipelineData]
        :param spec_file: SPEC file containing scan of interest.
        :type spec_file: str
        :param scan_number: Scan number of interest.
        :type scan_number: int
        :param scan_step_index: Scan step index of interest, defaults to None.
        :type scan_step_index: int, optional
        :param material: Material parameters to plot HKLs for.
        :type material: dict
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor.
        :type save_figures: bool
        :param interactive: Allows for user interactions.
        :type interactive: bool
        :param outputdir: Directory to which any output figures will
            be saved.
        :type outputdir: str
        :returns: None
        :rtype: None
        """
        # Third party modules
        import matplotlib.pyplot as plt

        # Local modules
        from CHAP.utils.scanparsers import SMBMCAScanParser as ScanParser

        if material is not None:
            self.logger.warning('Plotting HKL lines is not supported yet.')

        if scan_step_index is not None:
            if not isinstance(scan_step_index, int):
                try:
                    scan_step_index = int(scan_step_index)
                except:
                    msg = 'scan_step_index must be an int'
                    self.logger.error(msg)
                    raise TypeError(msg)

        calibration_config = self.get_config(
            data, 'edd.models.MCACeriaCalibrationConfig')
        scanparser = ScanParser(spec_file, scan_number)

        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        title = f'{scanparser.scan_title} MCA Data'
        if scan_step_index is None:
            title += ' (sum of all spectra in the scan)'
        ax.set_title(title)
        ax.set_xlabel('Calibrated energy (keV)')
        ax.set_ylabel('Intenstiy (a.u)')
        for detector in calibration_config.detectors:
            if scan_step_index is None:
                mca_data = np.sum(
                    scanparser.get_all_detector_data(detector.detector_name),
                    axis=0)
            else:
                mca_data = scanparser.get_detector_data(
                    detector.detector_name, scan_step_index=scan_step_index)
            ax.plot(detector.energies, mca_data,
                    label=f'Detector {detector.detector_name}')
        ax.legend()
        if interactive:
            plt.show()
        if save_figures:
            fig.savefig(os.path.join(
                outputdir, f'mca_data_{scanparser.scan_title}'))
        plt.close()
        return None


class StrainAnalysisProcessor(Processor):
    """Processor that takes a map of MCA data and returns a map of
    sample strains
    """
    def process(self,
                data,
                config=None,
                save_figures=False,
                inputdir='.',
                outputdir='.',
                interactive=False):
        """Return strain analysis maps & associated metadata in an NXprocess.

        :param data: Input data containing configurations for a map,
            completed ceria calibration, and parameters for strain
            analysis
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.StrainAnalysisConfig, defaults to
            None.
        :type config: dict, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'.
        :type outputdir: str, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to '.'.
        :type inputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :raises RuntimeError: Unable to get a valid strain analysis
            configuration.
        :return: NXprocess containing metadata about strain analysis
            processing parameters and empty datasets for strain maps
            to be filled in later.
        :rtype: nexusformat.nexus.NXprocess

        """
        # Get required configuration models from input data
        ceria_calibration_config = self.get_config(
            data, 'edd.models.MCACeriaCalibrationConfig', inputdir=inputdir)
        try:
            strain_analysis_config = self.get_config(
                data, 'edd.models.StrainAnalysisConfig', inputdir=inputdir)
        except Exception as data_exc:
            # Local modules
            from CHAP.edd.models import StrainAnalysisConfig

            self.logger.info('No valid strain analysis config in input '
                             + 'pipeline data, using config parameter instead')
            try:
                strain_analysis_config = StrainAnalysisConfig(
                    **config, inputdir=inputdir)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        nxroot = self.get_nxroot(
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


        :param map_config: The map configuration.
        :type map_config: CHAP.common.models.map.MapConfig
        :param ceria_calibration_config: The calibration configuration.
        :type ceria_calibration_config:
            'CHAP.edd.models.MCACeriaCalibrationConfig'
        :param strain_analysis_config: Strain analysis processing
            configuration.
        :type strain_analysis_config:
            CHAP.edd.models.StrainAnalysisConfig
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :return: NXroot containing strain maps.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXdetector,
            NXfield,
            NXparameters,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.common import MapProcessor
        from CHAP.edd.utils import (
            get_peak_locations,
            get_unique_hkls_ds,
            get_spectra_fits
        )
        if interactive or save_figures:
            from CHAP.edd.utils import (
                select_material_params,
                select_mask_and_hkls,
            )

        def linkdims(nxgroup, field_dims=[], oversampling_axis={}):
            if isinstance(field_dims, dict):
                field_dims = [field_dims]
            if map_config.map_type == 'structured':
                axes = deepcopy(map_config.dims)
                for dims in field_dims:
                    axes.append(dims['axes'])
            else:
                axes = ['map_index']
                for dims in field_dims:
                    axes.append(dims['axes'])
                nxgroup.attrs[f'map_index_indices'] = 0
            for dim in map_config.dims:
                if dim in oversampling_axis:
                    bin_name = dim.replace('fly_', 'bin_')
                    axes[axes.index(dim)] = bin_name
                    nxgroup[bin_name] = NXfield(
                        value=oversampling_axis[dim],
                        units=nxentry.data[dim].units,
                        attrs={
                            'long_name':
                                f'oversampled {nxentry.data[dim].long_name}',
                           'data_type': nxentry.data[dim].data_type,
                           'local_name':
                                f'oversampled {nxentry.data[dim].local_name}'})
                else:
                    nxgroup.makelink(nxentry.data[dim])
                    if f'{dim}_indices' in nxentry.data.attrs:
                        nxgroup.attrs[f'{dim}_indices'] = \
                            nxentry.data.attrs[f'{dim}_indices']
            nxgroup.attrs['axes'] = axes
            for dims in field_dims:
                nxgroup.attrs[f'{dims["axes"]}_indices'] = dims['index']

        if not interactive and not strain_analysis_config.materials:
            raise ValueError(
                'No material provided. Provide a material in the '
                'StrainAnalysis Configuration, or re-run the pipeline with '
                'the --interactive flag.')

        # Create the NXroot object
        nxroot = NXroot()
        nxroot[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxentry = nxroot[map_config.title]
        nxroot[f'{map_config.title}_strainanalysis'] = NXprocess()
        nxprocess = nxroot[f'{map_config.title}_strainanalysis']
        nxprocess.strain_analysis_config = dumps(strain_analysis_config.dict())

        # Setup plottable data group
        nxprocess.data = NXdata()
        nxprocess.default = 'data'
        nxdata = nxprocess.data
        linkdims(nxdata)

        # Collect the raw MCA data
        mca_data = strain_analysis_config.mca_data()
        if mca_data.ndim == 2:
            mca_data_summed = mca_data
        else:
            mca_data_summed = np.sum(
                mca_data, axis=tuple(np.arange(1, mca_data.ndim-1)))
        effective_map_shape = mca_data.shape[1:-1]
        self.logger.debug(f'mca_data.shape: {mca_data.shape}')
        self.logger.debug(f'mca_data_summed.shape: {mca_data_summed.shape}')
        self.logger.debug(f'effective_map_shape: {effective_map_shape}')

        # Check for oversampling axis and create the binned coordinates
        oversampling_axis = {}
        if (map_config.attrs.get('scan_type') == 4
                and strain_analysis_config.sum_fly_axes):
            # Local modules
            from CHAP.utils.general import rolling_average

            fly_axis = map_config.attrs.get('fly_axis_labels')[0]
            oversampling = strain_analysis_config.oversampling
            oversampling_axis[fly_axis] = rolling_average(
                    nxdata[fly_axis].nxdata,
                    start=oversampling.get('start', 0),
                    end=oversampling.get('end'),
                    width=oversampling.get('width'),
                    stride=oversampling.get('stride'),
                    num=oversampling.get('num'),
                    mode=oversampling.get('mode', 'valid'))

        # Loop over the detectors to perform the strain analysis
        for i, detector in enumerate(strain_analysis_config.detectors):

            # Get and add the calibration info to the detector
            calibration = [
                d for d in ceria_calibration_config.detectors \
                if d.detector_name == detector.detector_name][0]
            detector.add_calibration(calibration)

            # Get the MCA bin energies
            mca_bin_energies = detector.energies

            # Blank out data below 25 keV as well as the last bin
            energy_mask = np.where(mca_bin_energies >= 25.0, 1, 0)
            energy_mask[-1] = 0

            # Interactively adjust the material properties based on the
            # first detector calibration information and/or save figure
            # ASK: extend to multiple detectors?
            if not i and (interactive or save_figures):

                tth = detector.tth_calibrated
                if save_figures:
                    filename = os.path.join(
                        outputdir,
                        f'{detector.detector_name}_strainanalysis_'
                            'material_config.png')
                else:
                    filename = None
                strain_analysis_config.materials = select_material_params(
                    mca_bin_energies, mca_data_summed[i]*energy_mask, tth,
                    preselected_materials=strain_analysis_config.materials,
                    label='Sum of all spectra in the map',
                    interactive=interactive, filename=filename)
                self.logger.debug(
                    f'materials: {strain_analysis_config.materials}')

            # Mask during calibration
            calibration_bin_ranges = calibration.include_bin_ranges

            # Get the unique HKLs and lattice spacings for the strain
            # analysis materials
            hkls, ds = get_unique_hkls_ds(
                strain_analysis_config.materials,
                tth_tol=detector.hkl_tth_tol,
                tth_max=detector.tth_max)

            # Interactively adjust the mask and HKLs used in the
            # strain analysis
            if save_figures:
                filename = os.path.join(
                    outputdir,
                    f'{detector.detector_name}_strainanalysis_'
                        'fit_mask_hkls.png')
            else:
                filename = None
            include_bin_ranges, hkl_indices = \
                select_mask_and_hkls(
                    mca_bin_energies, mca_data_summed[i]*energy_mask,
                    hkls, ds, detector.tth_calibrated,
                    detector.include_bin_ranges, detector.hkl_indices,
                    detector.detector_name, mca_data[i]*energy_mask,
                    #calibration_mask=calibration_mask,
                    calibration_bin_ranges=calibration_bin_ranges,
                    label='Sum of all spectra in the map',
                    interactive=interactive, filename=filename)
            detector.include_energy_ranges = \
                detector.get_include_energy_ranges(include_bin_ranges)
            detector.hkl_indices = hkl_indices
            self.logger.debug(
                f'include_energy_ranges for detector {detector.detector_name}:'
                f' {detector.include_energy_ranges}')
            self.logger.debug(
                f'hkl_indices for detector {detector.detector_name}:'
                f' {detector.hkl_indices}')
            if not detector.include_energy_ranges:
                raise ValueError(
                    'No value provided for include_energy_ranges. '
                    'Provide them in the MCA Ceria Calibration Configuration, '
                    'or re-run the pipeline with the --interactive flag.')
            if not detector.hkl_indices:
                raise ValueError(
                    'No value provided for hkl_indices. Provide them in '
                    'the detector\'s MCA Ceria Calibration Configuration, or'
                    ' re-run the pipeline with the --interactive flag.')

            # Setup NXdata group
            self.logger.debug(
                f'Setting up NXdata group for {detector.detector_name}')
            nxprocess[detector.detector_name] = NXdetector()
            nxdetector = nxprocess[detector.detector_name]
            nxdetector.local_name = detector.detector_name
            nxdetector.detector_config = dumps(detector.dict())
            nxdetector.data = NXdata()
            det_nxdata = nxdetector.data
            linkdims(
                det_nxdata,
                {'axes': 'energy', 'index': len(effective_map_shape)},
                oversampling_axis=oversampling_axis)
            mask = detector.mca_mask()
            energies = mca_bin_energies[mask]
            det_nxdata.energy = NXfield(value=energies, attrs={'units': 'keV'})
            det_nxdata.intensity = NXfield(
                dtype='uint16',
                shape=(*effective_map_shape, len(energies)),
                attrs={'units': 'counts'})
            det_nxdata.tth = NXfield(
                dtype='float64',
                shape=effective_map_shape,
                attrs={'units':'degrees', 'long_name': '2\u03B8 (degrees)'}
            )
            det_nxdata.microstrain = NXfield(
                dtype='float64',
                shape=effective_map_shape,
                attrs={'long_name': 'Strain (\u03BC\u03B5)'})

            # Gather detector data
            self.logger.debug(
                f'Gathering detector data for {detector.detector_name}')
            for map_index in np.ndindex(effective_map_shape):
                det_nxdata.intensity[map_index] = \
                    mca_data[i][map_index].astype('uint16')[mask]
            det_nxdata.summed_intensity = det_nxdata.intensity.sum(axis=-1)

            # Perform strain analysis
            self.logger.debug(
                f'Beginning strain analysis for {detector.detector_name}')

            # Get the HKLs and lattice spacings that will be used for
            # fitting
            fit_hkls = np.asarray([hkls[i] for i in detector.hkl_indices])
            fit_ds = np.asarray([ds[i] for i in detector.hkl_indices])
            peak_locations = get_peak_locations(
                fit_ds, detector.tth_calibrated)

            (uniform_fit_centers, uniform_fit_centers_errors,
             uniform_fit_amplitudes, uniform_fit_amplitudes_errors,
             uniform_fit_sigmas, uniform_fit_sigmas_errors,
             uniform_best_fit, uniform_residuals,
             uniform_redchi, uniform_success,
             unconstrained_fit_centers, unconstrained_fit_centers_errors,
             unconstrained_fit_amplitudes, unconstrained_fit_amplitudes_errors,
             unconstrained_fit_sigmas, unconstrained_fit_sigmas_errors,
             unconstrained_best_fit, unconstrained_residuals,
             unconstrained_redchi, unconstrained_success) = \
                get_spectra_fits(
                    det_nxdata.intensity.nxdata, energies,
                    peak_locations, detector)

            # Add uniform fit results to the NeXus structure
            nxdetector.uniform_fit = NXcollection()
            fit_nxgroup = nxdetector.uniform_fit

            # Full map of results
            fit_nxgroup.results = NXdata()
            fit_nxdata = fit_nxgroup.results
            linkdims(
                fit_nxdata,
                {'axes': 'energy', 'index': len(map_config.shape)},
                oversampling_axis=oversampling_axis)
            fit_nxdata.makelink(det_nxdata.energy)
            fit_nxdata.best_fit= uniform_best_fit
            fit_nxdata.residuals = uniform_residuals
            fit_nxdata.redchi = uniform_redchi
            fit_nxdata.success = uniform_success

            # Peak-by-peak results
#            fit_nxgroup.fit_hkl_centers = NXdata()
#            fit_nxdata = fit_nxgroup.fit_hkl_centers
#            linkdims(fit_nxdata)
            for (hkl, center_guess, centers_fit, centers_error,
                    amplitudes_fit, amplitudes_error, sigmas_fit,
                    sigmas_error) in zip(
                        fit_hkls, peak_locations,
                        uniform_fit_centers, uniform_fit_centers_errors,
                        uniform_fit_amplitudes, uniform_fit_amplitudes_errors,
                        uniform_fit_sigmas, uniform_fit_sigmas_errors):
                hkl_name = '_'.join(str(hkl)[1:-1].split(' '))
                fit_nxgroup[hkl_name] = NXparameters()
                # Report initial HKL peak centers
                fit_nxgroup[hkl_name].center_initial_guess = center_guess
                fit_nxgroup[hkl_name].center_initial_guess.attrs['units'] = \
                    'keV'
                # Report HKL peak centers
                fit_nxgroup[hkl_name].centers = NXdata()
                linkdims(fit_nxgroup[hkl_name].centers)
                fit_nxgroup[hkl_name].centers.values = NXfield(
                    value=centers_fit, attrs={'units': 'keV'})
                fit_nxgroup[hkl_name].centers.errors = NXfield(
                    value=centers_error)
                fit_nxgroup[hkl_name].centers.attrs['signal'] = 'values'
#                fit_nxdata.makelink(
#                    fit_nxgroup[f'{hkl_name}/centers/values'], name=hkl_name)
                # Report HKL peak amplitudes
                fit_nxgroup[hkl_name].amplitudes = NXdata()
                linkdims(fit_nxgroup[hkl_name].amplitudes)
                fit_nxgroup[hkl_name].amplitudes.values = NXfield(
                    value=amplitudes_fit, attrs={'units': 'counts'})
                fit_nxgroup[hkl_name].amplitudes.errors = NXfield(
                    value=amplitudes_error)
                fit_nxgroup[hkl_name].amplitudes.attrs['signal'] = 'values'
                # Report HKL peak FWHM
                fit_nxgroup[hkl_name].sigmas = NXdata()
                linkdims(fit_nxgroup[hkl_name].sigmas)
                fit_nxgroup[hkl_name].sigmas.values = NXfield(
                    value=sigmas_fit, attrs={'units': 'keV'})
                fit_nxgroup[hkl_name].sigmas.errors = NXfield(
                    value=sigmas_error)
                fit_nxgroup[hkl_name].sigmas.attrs['signal'] = 'values'

            if interactive or save_figures:
                # Third party modules
                import matplotlib.animation as animation
                import matplotlib.pyplot as plt

                if save_figures:
                    path = os.path.join(
                        outputdir,
                        f'{detector.detector_name}_strainanalysis_'
                            'unconstrained_fits')
                    if not os.path.isdir(path):
                        os.mkdir(path)

                def animate(i):
                    map_index = np.unravel_index(i, effective_map_shape)
                    intensity.set_ydata(
                        det_nxdata.intensity.nxdata[map_index]
                        / det_nxdata.intensity.nxdata[map_index].max())
                    best_fit.set_ydata(
                        unconstrained_best_fit[map_index]
                        / unconstrained_best_fit[map_index].max())
                    # residual.set_ydata(unconstrained_residuals[map_index])
                    index.set_text('\n'.join(f'{k}[{i}] = {v}'
                        for k, v in map_config.get_coords(map_index).items()))
                    if save_figures:
                        plt.savefig(os.path.join(
                            path, f'frame_{str(i).zfill(num_digit)}.png'))
                    #return intensity, best_fit, residual, index
                    return intensity, best_fit, index

                fig, ax = plt.subplots()
                effective_map_shape
                map_index = np.unravel_index(0, effective_map_shape)
                data_normalized = (
                    det_nxdata.intensity.nxdata[map_index]
                    / det_nxdata.intensity.nxdata[map_index].max())
                intensity, = ax.plot(
                    energies, data_normalized, 'b.', label='data')
                fit_normalized = (unconstrained_best_fit[map_index]
                                  / unconstrained_best_fit[map_index].max())
                best_fit, = ax.plot(
                    energies, fit_normalized, 'k-', label='fit')
                # residual, = ax.plot(
                #     energies, unconstrained_residuals[map_index], 'r-',
                #     label='residual')
                ax.set(
                    title='Unconstrained fits',
                    xlabel='Energy (keV)',
                    ylabel='Normalized intensity (-)')
                ax.legend(loc='upper right')
                index = ax.text(
                    0.05, 0.95, '', transform=ax.transAxes, va='top')

                num_frame = int(det_nxdata.intensity.nxdata.size
                              / det_nxdata.intensity.nxdata.shape[-1])
                num_digit = len(str(num_frame))
                if not save_figures:
                    ani = animation.FuncAnimation(
                        fig, animate,
                        frames=int(det_nxdata.intensity.nxdata.size
                                   / det_nxdata.intensity.nxdata.shape[-1]),
                        interval=1000, blit=True, repeat=False)
                else:
                    for i in range(num_frame):
                        animate(i)

                    plt.close()
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

                    frames = []
                    for i in range(num_frame):
                        frame = plt.imread(
                            os.path.join(
                                path,
                                f'frame_{str(i).zfill(num_digit)}.png'))
                        im = plt.imshow(frame, animated=True)
                        if not i:
                            plt.imshow(frame)
                        frames.append([im])

                    ani = animation.ArtistAnimation(
                         plt.gcf(), frames, interval=1000, blit=True,
                         repeat=False)

                if interactive:
                    plt.show()

                if save_figures:
                    path = os.path.join(
                        outputdir,
                        f'{detector.detector_name}_strainanalysis_'
                            'unconstrained_fits.gif')
                    ani.save(path)
                plt.close()

            tth_map = detector.get_tth_map(effective_map_shape)
            det_nxdata.tth.nxdata = tth_map
            nominal_centers = np.asarray(
                [get_peak_locations(d0, tth_map) for d0 in fit_ds])
            unconstrained_strains = np.log(
                nominal_centers / unconstrained_fit_centers)
            unconstrained_strain = np.mean(unconstrained_strains, axis=0)
            det_nxdata.microstrain.nxdata = unconstrained_strain * 1e6

            # Add unconstrained fit results to the NeXus structure
            nxdetector.unconstrained_fit = NXcollection()
            fit_nxgroup = nxdetector.unconstrained_fit

            # Full map of results
            fit_nxgroup.results = NXdata()
            fit_nxdata = fit_nxgroup.results
            linkdims(
                fit_nxdata,
                {'axes': 'energy', 'index': len(map_config.shape)},
                oversampling_axis=oversampling_axis)
            fit_nxdata.makelink(det_nxdata.energy)
            fit_nxdata.best_fit= unconstrained_best_fit
            fit_nxdata.residuals = unconstrained_residuals
            fit_nxdata.redchi = unconstrained_redchi
            fit_nxdata.success = unconstrained_success

            # Peak-by-peak results
            fit_nxgroup.fit_hkl_centers = NXdata()
            fit_nxdata = fit_nxgroup.fit_hkl_centers
            linkdims(fit_nxdata)
            for (hkl, center_guesses, centers_fit, centers_error,
                amplitudes_fit, amplitudes_error, sigmas_fit,
                sigmas_error) in zip(
                    fit_hkls, uniform_fit_centers,
                    unconstrained_fit_centers,
                    unconstrained_fit_centers_errors,
                    unconstrained_fit_amplitudes,
                    unconstrained_fit_amplitudes_errors,
                    unconstrained_fit_sigmas, unconstrained_fit_sigmas_errors):
                hkl_name = '_'.join(str(hkl)[1:-1].split(' '))
                fit_nxgroup[hkl_name] = NXparameters()
                # Report initial guesses HKL peak centers
                fit_nxgroup[hkl_name].center_initial_guess = NXdata()
                linkdims(fit_nxgroup[hkl_name].center_initial_guess)
                fit_nxgroup[hkl_name].center_initial_guess.makelink(
                    nxdetector.uniform_fit[f'{hkl_name}/centers/values'],
                    name='values')
                fit_nxgroup[hkl_name].center_initial_guess.attrs['signal'] = \
                    'values'
                # Report HKL peak centers
                fit_nxgroup[hkl_name].centers = NXdata()
                linkdims(fit_nxgroup[hkl_name].centers)
                fit_nxgroup[hkl_name].centers.values = NXfield(
                    value=centers_fit, attrs={'units': 'keV'})
                fit_nxgroup[hkl_name].centers.errors = NXfield(
                    value=centers_error)
#                fit_nxdata.makelink(fit_nxgroup[f'{hkl_name}/centers/values'],
#                                    name=hkl_name)
                fit_nxgroup[hkl_name].centers.attrs['signal'] = 'values'
                # Report HKL peak amplitudes
                fit_nxgroup[hkl_name].amplitudes = NXdata()
                linkdims(fit_nxgroup[hkl_name].amplitudes)
                fit_nxgroup[hkl_name].amplitudes.values = NXfield(
                    value=amplitudes_fit, attrs={'units': 'counts'})
                fit_nxgroup[hkl_name].amplitudes.errors = NXfield(
                    value=amplitudes_error)
                fit_nxgroup[hkl_name].amplitudes.attrs['signal'] = 'values'
                # Report HKL peak sigmas
                fit_nxgroup[hkl_name].sigmas = NXdata()
                linkdims(fit_nxgroup[hkl_name].sigmas)
                fit_nxgroup[hkl_name].sigmas.values = NXfield(
                    value=sigmas_fit, attrs={'units': 'keV'})
                fit_nxgroup[hkl_name].sigmas.errors = NXfield(
                    value=sigmas_error)
                fit_nxgroup[hkl_name].sigmas.attrs['signal'] = 'values'

        return nxroot


class CreateStrainAnalysisConfigProcessor(Processor):
    """Processor that takes a basics stain analysis config file
    (the old style configuration without the map_config) and the output
    of EddMapReader and returns the old style stain analysis
    configuration.
    """
    def process(self, data, inputdir='.'):
        # Local modules
        from CHAP.common.models.map import MapConfig

        map_config = self.get_config(
            data, 'common.models.map.MapConfig', inputdir=inputdir)
        config = self.get_config(
            data, 'edd.models.StrainAnalysisConfig', inputdir=inputdir)
        config.map_config = map_config

        return config


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
