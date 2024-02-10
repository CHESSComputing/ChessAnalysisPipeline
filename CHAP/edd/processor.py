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
                from .models import DiffractionVolumeLengthConfig

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

        # Interactively set mask, if needed & possible.
        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            self.logger.info(
                'Interactively select a mask in the matplotlib figure')

            fig, mask, include_bin_ranges = select_mask_1d(
                np.sum(mca_data, axis=0),
                x=np.linspace(0, detector.max_energy_kev, detector.num_bins),
                label='Sum of MCA spectra over all scan points',
                preselected_index_ranges=detector.include_bin_ranges,
                title='Click and drag to select data range to include when '
                      'measuring diffraction volume length',
                xlabel='Uncalibrated Energy (keV)',
                ylabel='MCA intensity (counts)',
                min_num_index_ranges=1,
                interactive=interactive)
            detector.include_energy_ranges = detector.get_energy_ranges(
                include_bin_ranges)
            self.logger.debug(
                'Mask selected. Including detector energy ranges: '
                + str(detector.include_energy_ranges))
            if save_figures:
                fig.savefig(os.path.join(
                    outputdir, f'{detector.detector_name}_dvl_mask.png'))
            plt.close()
        if not detector.include_energy_ranges:
            raise ValueError(
                'No value provided for include_energy_ranges. '
                + 'Provide them in the Diffraction Volume Length '
                + 'Measurement Configuration, or re-run the pipeline '
                + 'with the --interactive flag.')

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
                _, _, dvl_bounds = select_mask_1d(
                    masked_sum, x=x,
                    label='Total (masked & normalized)',
                    preselected_index_ranges=[
                        (index_nearest(x, -dvl/2), index_nearest(x, dvl/2))],
                    title=('Click and drag to indicate the boundary '
                           'of the diffraction volume'),
                    xlabel=('Beam Direction (offset from scan "center")'),
                    ylabel='MCA intensity (normalized)',
                    min_num_index_ranges=1,
                    max_num_index_ranges=1)
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
            ax.set_xlabel('Beam Direction (offset from scan "center")')
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
            from .models import StrainAnalysisConfig

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
        import numpy as np
        from scipy.constants import physical_constants

        # Local modules
        from .utils import (
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
        # Third party modules
        import matplotlib.pyplot as plt
        import numpy as np

        # Local modules
        from .utils import select_mask_and_hkls

        if not interactive and not save_figures:
            return

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
        detector.include_energy_ranges = detector.get_energy_ranges(
            include_bin_ranges)
        detector.hkl_indices = hkl_indices
        if save_figures:
            fig.savefig(os.path.join(
                outputdir,
                f'{detector.detector_name}_strainanalysis_'
                'fit_mask_hkls.png'))
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
        # Third party modules
        import matplotlib.pyplot as plt
        import numpy as np

        # Local modules
        from .utils import select_material_params

        if not interactive and not save_figures:
            return
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
                f'{detector_name}_strainanalysis_'
                'material_config.png'))
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
        from .utils import (
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
                recalibrate_energy=True,
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
        :param recalibrate_energy: Get new linear energy correction
            parameters for detectors, defaults to True.
        :type recalibrate_energy: bool, optional
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
                from .models import MCACeriaCalibrationConfig

                calibration_config = MCACeriaCalibrationConfig(
                    **config, inputdir=inputdir)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        for detector in calibration_config.detectors:
            tth, slope, intercept = self.calibrate(
                calibration_config, detector, recalibrate_energy,
                save_figures=save_figures, interactive=interactive,
                outputdir=outputdir)
            detector.tth_calibrated = tth
            detector.slope_calibrated = slope
            detector.intercept_calibrated = intercept

        return calibration_config.dict()

    def calibrate(self,
                  calibration_config,
                  detector,
                  recalibrate_energy=True,
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
        :param recalibrate_energy: Get new linear energy correction
            parameters for the detector, defaults to True.
        :type recalibrate_energy: bool, optional
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
        :return: Calibrated values of 2&theta and the linear correction
            parameters for MCA channel energies: tth, slope, intercept.
        :rtype: float, float, float
        """
        # Local modules
        from .utils import get_peak_locations
        from CHAP.utils.fit import Fit

        # Get the unique HKLs and lattice spacings for the calibration
        # material
        hkls, ds = calibration_config.material.unique_hkls_ds(
            tth_tol=detector.hkl_tth_tol, tth_max=detector.tth_max)

        # Collect raw MCA data of interest
        mca_bin_energies = detector.energies
        mca_data = calibration_config.mca_data(detector)
        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            # Local modules
            from .utils import (
                select_tth_initial_guess,
                select_mask_and_hkls,
            )

            # Adjust initial tth guess
            fig, detector.tth_initial_guess = select_tth_initial_guess(
                mca_bin_energies, mca_data, hkls, ds,
                detector.tth_initial_guess, interactive)
            if save_figures:
                fig.savefig(os.path.join(
                   outputdir,
                   f'{detector.detector_name}_calibration_'
                   'tth_initial_guess.png'))
            plt.close()

            # Select mask & HKLs for fitting
            fig, include_bin_ranges, hkl_indices = select_mask_and_hkls(
                mca_bin_energies, mca_data, hkls, ds,
                detector.tth_initial_guess, detector.include_bin_ranges,
                detector.hkl_indices, detector.detector_name,
                flux_energy_range=calibration_config.flux_file_energy_range,
                label='MCA data',
                interactive=interactive)
            detector.include_energy_ranges = detector.get_energy_ranges(
                include_bin_ranges)
            detector.hkl_indices = hkl_indices
            if save_figures:
                fig.savefig(os.path.join(
                    outputdir,
                    f'{detector.detector_name}_calibration_fit_mask_hkls.png'))
            plt.close()
        self.logger.debug(f'tth_initial_guess = {detector.tth_initial_guess}')
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
        tth = detector.tth_initial_guess
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
                centers_range=centers_range)
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
            tth = unconstrained_tth

            # Stop tuning tth at this iteration if differences are
            # small enough
            if abs(tth - prev_tth) < calibration_config.tune_tth_tol:
                break

        # Fit line to expected / computed peak locations from the last
        # unconstrained fit.
        fit = Fit.fit_data(
            fit_E0, 'linear', x=unconstrained_fit_centers,
            nan_policy='omit')
        slope = fit.best_values['slope']
        intercept = fit.best_values['intercept']
        if recalibrate_energy:
            # Adjust final values for slope and intercept according to
            # their initial values
            final_slope = slope * detector.slope_initial_guess
            final_intercept = (
                slope * detector.intercept_initial_guess + intercept)
        else:
            # Keep initial values for slope and intercept as the final
            # values, no matter what the linear fit found.
            final_slope = detector.slope_initial_guess
            final_intercept = detector.intercept_initial_guess

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

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
            axs[1,1].plot(slope * unconstrained_fit_centers + intercept,
                          unconstrained_fit_centers,
                          color='C1', label='Unconstrained: Linear Fit')
            axs[1,1].legend()

            # Add a text box showing final calibrated values
            txt = 'Calibrated Values:\n\n' \
                  + f'Takeoff Angle:\n    {tth:.5f}$^\circ$'
            if recalibrate_energy:
                txt +=  f'\n\nSlope:\n    {final_slope:.5f}\n\n' \
                       + f'Intercept:\n    {final_intercept:.5f} $keV$'
            axs[1,1].text(
                0.98, 0.02, txt,
                ha='right', va='bottom', ma='left',
                transform=axs[1,1].transAxes,
                bbox=dict(boxstyle='round',
                          ec=(1., 0.5, 0.5),
                          fc=(1., 0.8, 0.8, 0.8)))

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(outputdir, 'ceria_calibration_fits.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return float(tth), float(final_slope), float(final_intercept)


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
                config=None,
                peak_initial_guesses=None,
                peak_center_fit_delta=2.0,
                fit_energy_ranges=None,
                save_figures=False,
                interactive=False,
                inputdir='.',
                outputdir='.'):
        """For each detector in the `MCACeriaCalibrationConfig`
        provided with `data`, fit the specified peaks in the MCA
        spectrum specified. Using the difference between the provided
        peak locations and the fit centers of those peaks, compute
        linear correction parameters to convert uncalibrated MCA
        channel energies to calibrated channel energies. Set the
        values in the calibration config provided for
        `slope_initial_guess` and `intercept_initial_guess` to these
        values (for each detector) and return the updated
        configuration.

        :param data: A Ceria Calibration configuration.
        :type data: PipelineData
        :param peak_energies: Theoretical locations of peaks to use
            for calibrating the MCA channel energies. It is _strongly_
            recommended to use fluorescence peaks.
        :type peak_energies: list[float]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.MCACeriaCalibrationConfig, defaults to
            None.
        :type config: dict, optional
        :param peak_initial_guesses: A list of values to use for the
            initial guesses for peak locations when performing the fit
            of the spectrum. Providing good values to this parameter
            can greatly improve the quality of the spectrum fit when
            the uncalibrated detector channel energies are too far off
            to use the values in `peak_energies` for the initial
            guesses for peak centers. Defaults to None.
        :type peak_inital_guesses: Optional[list[float]]
        :param peak_center_fit_delta: Set boundaries on the fit peak
            centers when performing the fit. The min/max possible
            values for the peak centers will be the values provided in
            `peak_energies` (or `peak_initial_guesses`, if used) &pm;
            `peak_center_fit_delta`. Defaults to 2.0.
        :type peak_center_fit_delta: float
        :param fit_energy_ranges: Explicit ranges of uncalibrated MCA
            channel energy ranges to include when performing a fit of
            the given peaks to the provied MCA spectrum. Use this
            parameter or select it interactively by running a pipeline
            with `config.interactive: True`. Defaults to []
        :type fit_energy_ranges: Optional[list[list[float]]]
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to False.
        :type save_figures: bool, optional
        :param interactive: Allows for user interactions, defaults to
            False.
        :type interactive: bool, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to '.'.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to '.'.
        :type outputdir: str, optional
        :returns: Dictionary representing the energy-calibrated
            version of the ceria calibration configuration.
        :rtype: dict
        """
        # Validate arguments: fit_ranges & interactive
        if not (fit_energy_ranges or interactive):
            self.logger.exception(
                RuntimeError(
                    'If `fit_energy_ranges` is not explicitly provided, '
                    + self.__class__.__name__
                    + ' must be run with `interactive=True`.'))
        # Validate arguments: peak_energies & peak_initial_guesses
        peak_energies.sort()
        if peak_initial_guesses is None:
            peak_initial_guesses = peak_energies
        else:
            # Local modules
            from CHAP.utils.general import is_num_series

            is_num_series(peak_initial_guesses, raise_error=True)
            if len(peak_initial_guesses) != len(peak_energies):
                self.logger.exception(
                    ValueError(
                        'peak_initial_guesses must have the same number of '
                        'values as peak_energies'))
        peak_initial_guesses.sort()
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
                from .models import MCACeriaCalibrationConfig

                calibration_config = MCACeriaCalibrationConfig(
                    **config, inputdir=inputdir)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        # Calibrate detector energy based on indicated peaks. Populate
        # the calibration configuration's "initial guesses" with these
        # values -- they may be re-calibrated later with
        # MCACeriaCalibrationProcessor.
        for detector in calibration_config.detectors:
            slope, intercept = self.calibrate(
                calibration_config, detector, peak_energies,
                peak_initial_guesses, peak_center_fit_delta, fit_energy_ranges,
                save_figures, interactive, outputdir)
            detector.slope_initial_guess = slope
            detector.intercept_initial_guess = intercept

        return calibration_config.dict()

    def calibrate(self, calibration_config, detector, peak_energies,
                  peak_initial_guesses, peak_center_fit_delta,
                  fit_energy_ranges, save_figures, interactive, outputdir):
        """Return calibrated slope & intercept for linearly correcting
        this detector's bin energies.

        :param peak_energies: Theoretical locations of peaks to use
            for calibrating the MCA channel energies. It is _strongly_
            recommended to use fluorescence peaks.
        :type peak_energies: list[float]
        :param peak_initial_guesses: A list of values to use for the
            initial guesses for peak locations when performing the fit
            of the spectrum. Providing good values to this parameter
            can greatly improve the quality of the spectrum fit when
            the uncalibrated detector channel energies are too far off
            to use the values in `peak_energies` for the initial
            guesses for peak centers.
        :type peak_inital_guesses: Optional[list[float]]
        :param peak_center_fit_delta: Set boundaries on the fit peak
            centers when performing the fit. The min/max possible
            values for the peak centers will be the values provided in
            `peak_energies` (or `peak_initial_guesses`, if used) &pm;
            `peak_center_fit_delta`.
        :type peak_center_fit_delta: float
        :param fit_energy_ranges: Explicit ranges of uncalibrated MCA
            channel energy ranges to include when performing a fit of
            the given peaks to the provied MCA spectrum. Use this
            parameter or select it interactively by running a pipeline
            with `config.interactive: True`.
        :type fit_ranges: list[list[float]]
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor.
        :type save_figures: bool
        :param interactive: Allows for user interactions.
        :type interactive: bool
        :param outputdir: Directory to which any output figures will
            be saved.
        :type outputdir: str
        :returns: Slope and intercept for linearly correcting the
            detector's bin energies.
        :rtype: tuple[float, float]
        """
        # Third party modules
        import numpy as np

        # Local modules
        from CHAP.utils.fit import Fit
        from CHAP.utils.general import select_mask_1d

        self.logger.debug(f'Calibrating detector {detector.detector_name}')
        spectrum = calibration_config.mca_data(detector)
        uncalibrated_energies = np.linspace(
            0, detector.max_energy_kev, detector.num_bins)

        mask = None
        if save_figures or interactive:
            # Third party modules
            import matplotlib.pyplot as plt

            # Local modules
            from CHAP.utils.general import (
                index_nearest_down,
                index_nearest_upp,
            )

            fit_index_ranges = []
            for e_min, e_max in fit_energy_ranges:
                fit_index_ranges.append(
                    [index_nearest_down(uncalibrated_energies, e_min),
                     index_nearest_upp(uncalibrated_energies, e_max)])

            fig, mask, fit_index_ranges = select_mask_1d(
                spectrum, x=uncalibrated_energies,
                preselected_index_ranges=fit_index_ranges,
                xlabel='Uncalibrated Energy', ylabel='Intensity',
                min_num_index_ranges=1, interactive=interactive)
            fit_energy_ranges = [[uncalibrated_energies[i] for i in range_]
                                 for range_ in fit_index_ranges]
            if save_figures:
                fig.savefig(os.path.join(
                    outputdir, 'mca_energy_calibration_mask.png'))
            plt.close()
        self.logger.debug(
            f'Selected energy ranges to fit: {fit_energy_ranges}')

        if mask is None:
            spectrum_fit = Fit(spectrum, x=uncalibrated_energies)
        else:
            spectrum_fit = Fit(spectrum[mask], x=uncalibrated_energies[mask])
        for i, (peak_energy, initial_guess) in enumerate(
                zip(peak_energies, peak_initial_guesses)):
            spectrum_fit.add_model(
                'gaussian', prefix=f'peak{i+1}_', parameters=(
                    {'name': 'amplitude', 'min': 0.0},
                    {'name': 'center', 'value': initial_guess,
                     'min': initial_guess - peak_center_fit_delta,
                     'max': initial_guess + peak_center_fit_delta}
                ))
        self.logger.debug('Fitting spectrum')
        spectrum_fit.fit()
        fit_peak_energies = sorted([
            spectrum_fit.best_values[f'peak{i+1}_center']
            for i in range(len(peak_energies))])
        self.logger.debug(f'Fit peak centers: {fit_peak_energies}')

        energy_fit = Fit.fit_data(
            peak_energies, 'linear', x=fit_peak_energies, nan_policy='omit')
        slope = energy_fit.best_values['slope']
        intercept = energy_fit.best_values['intercept']
        # If we want to rescale slope so results are a linear
        # correction from channel indices -> calibrated energies, not
        # uncalibrated energies -> calibrated energies, then uncooment
        # the following line.
        # slope = (detector.max_energy_kev / detector.num_bins) * slope

        # Reference plot to see fit results:
        if interactive or save_figures:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1,2, figsize=(11, 4.25))
            fig.suptitle(
                f'Detector {detector.detector_name} Energy Calibration')
            # Left plot: raw MCA data & best fit of fluorescence peaks
            axs[0].set_title(f'MCA Spectrum Peak Fit')
            axs[0].set_xlabel('Uncalibrated Energy (keV)')
            axs[0].set_ylabel('Intensity (a.u)')
            axs[0].plot(uncalibrated_energies[mask], spectrum[mask],
                    label='MCA data')
            axs[0].plot(uncalibrated_energies[mask], spectrum_fit.best_fit,
                    label='Best fit')
            axs[0].legend()
            # Right plot: linear fit of theoretical & fit peak locations
            axs[1].set_title('Theoretical vs. Fit Peak Energies')
            axs[1].set_xlabel('Theoretical Peak Energy (keV)')
            axs[1].set_ylabel('Energy (keV)')
            axs[1].plot(peak_energies, fit_peak_energies,
                        marker='o', ls='',
                        label='Fit peak energies (uncalibrated)')
            axs[1].plot(energy_fit.best_fit, fit_peak_energies,
                        label='Best linear fit')
            axs[1].plot(peak_energies,
                        slope * np.asarray(fit_peak_energies) + intercept,
                        marker='o', ls='',
                        label='Fit peak energies (calibrated)')
            axs[1].legend()
            # Add text box showing computed values of linear E
            # correction parameters
            axs[1].text(
                0.98, 0.02,
                'Calibrated Values:\n\n'
                + f'Slope:\n    {slope:.5f}\n\n'
                + f'Intercept:\n    {intercept:.5f} $keV$',
                ha='right', va='bottom', ma='left',
                transform=axs[1].transAxes,
                bbox=dict(boxstyle='round',
                          ec=(1., 0.5, 0.5),
                          fc=(1., 0.8, 0.8, 0.8)))

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(outputdir, f'energy_calibration_fit_{detector.detector_name}.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return float(slope), float(intercept)


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
            from .models import StrainAnalysisConfig

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
        from .utils import (
            get_peak_locations,
            get_unique_hkls_ds,
            get_spectra_fits
        )

        def linkdims(nxgroup, field_dims=[]):
            if isinstance(field_dims, dict):
                field_dims = [field_dims]
            if map_config.map_type == 'structured':
                axes = deepcopy(map_config.dims)
                for dims in field_dims:
                    axes.append(dims['axes'])
                nxgroup.attrs['axes'] = axes
            else:
                axes = ['map_index']
                for dims in field_dims:
                    axes.append(dims['axes'])
                nxgroup.attrs['axes'] = axes
                nxgroup.attrs[f'map_index_indices'] = 0
            for dim in map_config.dims:
                nxgroup.makelink(nxentry.data[dim])
                if f'{dim}_indices' in nxentry.data.attrs:
                    nxgroup.attrs[f'{dim}_indices'] = \
                        nxentry.data.attrs[f'{dim}_indices']
            for dims in field_dims:
                nxgroup.attrs[f'{dims["axes"]}_indices'] = dims['index']

        if len(strain_analysis_config.detectors) != 1:
            raise RuntimeError('Multiple detectors not tested')
        for detector in strain_analysis_config.detectors:
            calibration = [
                d for d in ceria_calibration_config.detectors \
                if d.detector_name == detector.detector_name][0]
            detector.add_calibration(calibration)

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

        # Collect raw MCA data of interest
        mca_bin_energies = []
        for i, detector in enumerate(strain_analysis_config.detectors):
            mca_bin_energies.append(
                detector.slope_calibrated
                * np.linspace(0, detector.max_energy_kev, detector.num_bins)
                + detector.intercept_calibrated)
        mca_data = strain_analysis_config.mca_data()

        # Select interactive params / save figures
        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            # Local modules
            from .utils import (
                select_material_params,
                select_mask_and_hkls,
            )

            # Mask during calibration
            if len(ceria_calibration_config.detectors) != 1:
                raise RuntimeError('Multiple detectors not implemented')
            for detector in ceria_calibration_config.detectors:
#                calibration_mask = detector.mca_mask()
                calibration_bin_ranges = detector.include_bin_ranges

            tth = strain_analysis_config.detectors[0].tth_calibrated
            fig, strain_analysis_config.materials = select_material_params(
                mca_bin_energies[0], np.sum(mca_data, axis=1)[0], tth,
                materials=strain_analysis_config.materials,
                label='Sum of all spectra in the map',
                interactive=interactive)
            self.logger.debug(
                f'materials: {strain_analysis_config.materials}')
            if save_figures:
                fig.savefig(os.path.join(
                    outputdir,
                    f'{detector.detector_name}_strainanalysis_'
                    'material_config.png'))
            plt.close()

            # ASK: can we assume same hkl_tth_tol and tth_max for
            # every detector in this part?
            hkls, ds = get_unique_hkls_ds(
                strain_analysis_config.materials,
                tth_tol=strain_analysis_config.detectors[0].hkl_tth_tol,
                tth_max=strain_analysis_config.detectors[0].tth_max)
            for i, detector in enumerate(strain_analysis_config.detectors):
                fig, include_bin_ranges, hkl_indices = \
                    select_mask_and_hkls(
                        mca_bin_energies[i],
                        np.sum(mca_data[i], axis=0),
                        hkls, ds,
                        detector.tth_calibrated,
                        detector.include_bin_ranges, detector.hkl_indices,
                        detector.detector_name, mca_data[i],
#                        calibration_mask=calibration_mask,
                        calibration_bin_ranges=calibration_bin_ranges,
                        label='Sum of all spectra in the map',
                        interactive=interactive)
                detector.include_energy_ranges = detector.get_energy_ranges(
                    include_bin_ranges)
                detector.hkl_indices = hkl_indices
                if save_figures:
                    fig.savefig(os.path.join(
                        outputdir,
                        f'{detector.detector_name}_strainanalysis_'
                        'fit_mask_hkls.png'))
                plt.close()
        else:
            # ASK: can we assume same hkl_tth_tol and tth_max for
            # every detector in this part?
            # Get the unique HKLs and lattice spacings for the strain
            # analysis materials (assume hkl_tth_tol and tth_max are the
            # same for each detector)
            hkls, ds = get_unique_hkls_ds(
                strain_analysis_config.materials,
                tth_tol=strain_analysis_config.detectors[0].hkl_tth_tol,
                tth_max=strain_analysis_config.detectors[0].tth_max)

        for i, detector in enumerate(strain_analysis_config.detectors):
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
                det_nxdata, {'axes': 'energy', 'index': len(map_config.shape)})
            mask = detector.mca_mask()
            energies = mca_bin_energies[i][mask]
            det_nxdata.energy = NXfield(value=energies,
                                        attrs={'units': 'keV'})
            det_nxdata.intensity = NXfield(
                dtype='uint16',
                shape=(*map_config.shape, len(energies)),
                attrs={'units': 'counts'})
            det_nxdata.tth = NXfield(
                dtype='float64',
                shape=map_config.shape,
                attrs={'units':'degrees', 'long_name': '2\u03B8 (degrees)'}
            )
            det_nxdata.microstrain = NXfield(
                dtype='float64',
                shape=map_config.shape,
                attrs={'long_name': 'Strain (\u03BC\u03B5)'})

            # Gather detector data
            self.logger.debug(
                f'Gathering detector data for {detector.detector_name}')
            for j, map_index in enumerate(np.ndindex(map_config.shape)):
                det_nxdata.intensity[map_index] = \
                    mca_data[i][j].astype('uint16')[mask]
            det_nxdata.summed_intensity = det_nxdata.intensity.sum(axis=-1)

            # Perform strain analysis
            self.logger.debug(
                f'Beginning strain analysis for {detector.detector_name}')

            # Get the HKLs and lattice spacings that will be used for
            # fitting
            fit_hkls  = np.asarray([hkls[i] for i in detector.hkl_indices])
            fit_ds  = np.asarray([ds[i] for i in detector.hkl_indices])
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
                fit_nxdata, {'axes': 'energy', 'index': len(map_config.shape)})
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

                if save_figures:
                    path = os.path.join(
                        outputdir, f'{detector.detector_name}_strainanalysis_'
                        'unconstrained_fits')
                    if not os.path.isdir(path):
                        os.mkdir(path)

                def animate(i):
                    map_index = np.unravel_index(i, map_config.shape)
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
                map_index = np.unravel_index(0, map_config.shape)
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
                    ylabel='Normalized Intensity (-)')
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

            tth_map = detector.get_tth_map(map_config)
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
                fit_nxdata, {'axes': 'energy', 'index': len(map_config.shape)})
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


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
