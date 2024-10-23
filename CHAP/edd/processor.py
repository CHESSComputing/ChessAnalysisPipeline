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

# Current good detector channels for the 23 channel EDD detector:
#    0, 2, 3, 5, 6, 7, 8, 10, 13, 14, 16, 17, 18, 19, 21, 22

def get_axes(nxdata, skip_axes=None):
    """Get the axes of an NXdata object."""
    if skip_axes is None:
        skip_axes = []
    if 'unstructured_axes' in nxdata.attrs:
        axes = nxdata.attrs['unstructured_axes']
    else:
        axes = nxdata.attrs['axes']
    if isinstance(axes, str):
        axes = [axes]
    return [str(a) for a in axes if a not in skip_axes]


class DiffractionVolumeLengthProcessor(Processor):
    """A Processor using a steel foil raster scan to calculate the
    length of the diffraction volume for an EDD setup.
    """
    def process(
            self, data, config=None, save_figures=False, inputdir='.',
            outputdir='.', interactive=False):
        """Return the calculated value of the DV length.

        :param data: Input configuration for the raw scan data & DVL
            calculation procedure.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.DiffractionVolumeLengthConfig.
        :type config: dict, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :raises RuntimeError: Unable to get a valid DVL configuration.
        :return: Complete DVL configuraiton dictionary.
        :rtype: dict
        """
        try:
            dvl_config = self.get_config(
                data, 'edd.models.DiffractionVolumeLengthConfig',
                inputdir=inputdir)
        except Exception as exc:
            self.logger.error(exc)
            self.logger.info('No valid DVL config in input pipeline data, '
                             'using config parameter instead.')
            try:
                # Local modules
                from CHAP.edd.models import DiffractionVolumeLengthConfig

                dvl_config = DiffractionVolumeLengthConfig(
                    **config, inputdir=inputdir)
            except Exception as exc:
                self.logger.error('Could not get a valid DVL config')
                raise RuntimeError from exc

        for detector in dvl_config.detectors:
            dvl = self.measure_dvl(
                dvl_config, detector, save_figures=save_figures,
                interactive=interactive, outputdir=outputdir)
            detector.dvl_measured = dvl

        return dvl_config.dict()

    def measure_dvl(
            self, dvl_config, detector, save_figures=False, outputdir='.',
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
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
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
        raise RuntimeError('DiffractionVolumeLengthProcessor not updated yet')
        mca_data = dvl_config.mca_data(detector)

        # Blank out data below bin 500 (~25keV) as well as the last bin
        # RV Not backward compatible with old detector
        energy_mask = np.ones(detector.num_bins, dtype=np.int16)
        energy_mask[:500] = 0
        energy_mask[-1] = 0
        mca_data = mca_data*energy_mask

        # Interactively set or update mask, if needed & possible.
        if interactive or save_figures:
            if interactive:
                self.logger.info(
                    'Interactively select a mask in the matplotlib figure')
            if save_figures:
                filename = os.path.join(
                    outputdir, f'{detector.id}_dvl_mask.png')
            else:
                filename = None
            _, detector.include_bin_ranges = select_mask_1d(
                np.sum(mca_data, axis=0),
                x=np.arange(detector.num_bins, dtype=np.int16),
                label='Sum of MCA spectra over all scan points',
                preselected_index_ranges=detector.include_bin_ranges,
                title='Click and drag to select data range to include when '
                      'measuring diffraction volume length',
                xlabel='Uncalibrated energy (keV)',
                ylabel='Intensity (counts)',
                min_num_index_ranges=1,
                interactive=interactive, filename=filename)
            self.logger.debug(
                'Mask selected. Including detector bin ranges: '
                + str(detector.include_bin_ranges))
        if not detector.include_bin_ranges:
            raise ValueError(
                'No value provided for include_bin_ranges. '
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
                    preselected_index_ranges=[
                        (index_nearest(x, -dvl/2), index_nearest(x, dvl/2))],
                    title=('Click and drag to indicate the boundary '
                           'of the diffraction volume'),
                    xlabel=('Beam direction (offset from scan "center")'),
                    ylabel='Normalized intensity (-)',
                    min_num_index_ranges=1,
                    max_num_index_ranges=1,
                    interactive=interactive)
                dvl_bounds = dvl_bounds[0]
                dvl = abs(x[dvl_bounds[1]] - x[dvl_bounds[0]])
            else:
                self.logger.warning(
                    'Cannot manually indicate DVL when running CHAP '
                    'non-interactively. Using default DVL calcluation '
                    'instead.')

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.set_title(f'Diffraction Volume ({detector.id})')
            ax.set_xlabel('Beam direction (offset from scan "center")')
            ax.set_ylabel('Normalized intensity (-)')
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
                    outputdir, f'{detector.id}_dvl.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return dvl


class LatticeParameterRefinementProcessor(Processor):
    """Processor to get a refined estimate for a sample's lattice
    parameters."""
    def process(
            self, data, config=None, save_figures=False, inputdir='.',
            outputdir='.', interactive=False):
        """Given a strain analysis configuration, return a copy
        contining refined values for the materials' lattice
        parameters.

        :param data: Input data for the lattice parameter refinement
            procedure.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.StrainAnalysisConfig.
        :type config: dict, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :raises RuntimeError: Unable to get a valid DVL configuration.
        :return: The strain analysis configuration with the refined
            lattice parameter configuration.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXsubentry,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.common import MapProcessor

        calibration_config = self.get_config(
            data, 'edd.models.MCATthCalibrationConfig', inputdir=inputdir)
        try:
            strain_analysis_config = self.get_config(
                data, 'edd.models.StrainAnalysisConfig', inputdir=inputdir)
        except:
            # Local modules
            from CHAP.edd.models import StrainAnalysisConfig

            self.logger.info('No valid strain analysis config in input '
                             'pipeline data, using config parameter instead')
            try:
                strain_analysis_config = StrainAnalysisConfig(
                    **config, inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        if len(strain_analysis_config.materials) > 1:
            error_msg = 'Not implemented for multiple materials'
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)

        # Collect the raw MCA data
        raise RuntimeError(
            'LatticeParameterRefinementProcessor not updated yet')
        self.logger.debug('Reading data ...')
        mca_data = strain_analysis_config.mca_data()
        self.logger.debug('... done')
        self.logger.debug(f'mca_data.shape: {mca_data.shape}')
        if mca_data.ndim == 2:
            mca_data_summed = mca_data
        else:
            mca_data_summed = np.mean(
                mca_data, axis=tuple(np.arange(1, mca_data.ndim-1)))
        self.logger.debug(f'mca_data_summed.shape: {mca_data_summed.shape}')

        # Create the NXroot object
        nxroot = NXroot()
        nxentry = NXentry()
        nxroot.entry = nxentry
        nxentry.set_default()
        nxsubentry = NXsubentry()
        nxentry.nexus_output = nxsubentry
        nxsubentry.attrs['schema'] = 'h5'
        nxsubentry.attrs['filename'] = 'lattice_parameter_map.nxs'
        map_config = strain_analysis_config.map_config
        nxsubentry[map_config.title] = MapProcessor.get_nxentry(map_config)
        nxsubentry[f'{map_config.title}_lat_par_refinement'] = NXprocess()
        nxprocess = nxsubentry[f'{map_config.title}_lat_par_refinement']
        nxprocess.strain_analysis_config = dumps(
            strain_analysis_config.dict())
        nxprocess.calibration_config = dumps(calibration_config.dict())

        lattice_parameters = []
        for i, detector in enumerate(strain_analysis_config.detectors):
            lattice_parameters.append(self.refine_lattice_parameters(
                strain_analysis_config, calibration_config, detector,
                mca_data[i], mca_data_summed[i], nxsubentry, interactive,
                save_figures, outputdir))
        lattice_parameters_mean = np.asarray(lattice_parameters).mean(axis=0)
        self.logger.debug(
            'Lattice parameters refinement averaged over all detectors: '
            f'{lattice_parameters_mean}')
        strain_analysis_config.materials[0].lattice_parameters = [
            float(v) for v in lattice_parameters_mean]
        nxprocess.lattice_parameters = lattice_parameters_mean

        nxentry.lat_par_output = NXsubentry()
        nxentry.lat_par_output.attrs['schema'] = 'yaml'
        nxentry.lat_par_output.attrs['filename'] = 'lattice_parameters.yaml'
        nxentry.lat_par_output.data = dumps(
            strain_analysis_config.dict())

        return nxroot

    def refine_lattice_parameters(
            self, strain_analysis_config, calibration_config, detector,
            mca_data, mca_data_summed, nxsubentry, interactive, save_figures,
            outputdir):
        """Return refined values for the lattice parameters of the
        materials indicated in `strain_analysis_config`. Method: given
        a scan of a material, fit the peaks of each MCA spectrum for a
        given detector. Based on those fitted peak locations,
        calculate the lattice parameters that would produce them.
        Return the averaged value of the calculated lattice parameters
        across all spectra.

        :param strain_analysis_config: Strain analysis configuration.
        :type strain_analysis_config:
            CHAP.edd.models.StrainAnalysisConfig
        :param calibration_config: Energy calibration configuration.
        :type calibration_config: edd.models.MCATthCalibrationConfig
        :param detector: A single MCA detector element configuration.
        :type detector: CHAP.edd.models.MCAElementStrainAnalysisConfig
        :param mca_data: Raw specta for the current MCA detector.
        :type mca_data: numpy.ndarray
        :param mca_data_summed: Raw specta for the current MCA detector
            summed over all data point.
        :type mca_data_summed: numpy.ndarray
        :param nxsubentry: NeXus subentry to store the detailed refined
            lattice parameters for each detector.
        :type nxsubentry: nexusformat.nexus.NXprocess
        :param interactive: Boolean to indicate whether interactive
            matplotlib figures should be presented.
        :type interactive: bool
        :param save_figures: Boolean to indicate whether figures
            indicating the selection should be saved.
        :type save_figures: bool
        :param outputdir: Where to save figures (if `save_figures` is
            `True`).
        :type outputdir: str
        :returns: List of refined lattice parameters for materials in
            `strain_analysis_config`
        :rtype: list[float]
        """
        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXdetector,
            NXfield,
        )

        # Local modules
        from CHAP.edd.utils import (
            get_peak_locations,
            get_spectra_fits,
            get_unique_hkls_ds,
        )

        def linkdims(nxgroup, field_dims=None):
            """Link the dimensions for a `nexusformat.nexus.NXgroup`
            object."""
            if field_dims is None:
                field_dims = []
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
                nxgroup.attrs['map_index_indices'] = 0
            for dim in map_config.dims:
                nxgroup.makelink(nxsubentry[map_config.title].data[dim])
                if f'{dim}_indices' in nxsubentry[map_config.title].data.attrs:
                    nxgroup.attrs[f'{dim}_indices'] = \
                        nxsubentry[map_config.title].data.attrs[
                            f'{dim}_indices']
            nxgroup.attrs['axes'] = axes
            for dims in field_dims:
                nxgroup.attrs[f'{dims["axes"]}_indices'] = dims['index']

        # Get and add the calibration info to the detector
        calibration = [
            d for d in calibration_config.detectors \
            if d.id == detector.id][0]
        detector.add_calibration(calibration)

        # Get the MCA bin energies
        mca_bin_energies = detector.energies

        # Blank out data below 25 keV as well as the last bin
        energy_mask = np.where(mca_bin_energies >= 25.0, 1, 0)
        energy_mask[-1] = 0

        # Subtract the baseline
        if detector.baseline:
            # Local modules
            from CHAP.edd.models import BaselineConfig
            from CHAP.common.processor import ConstructBaseline

            if isinstance(detector.baseline, bool):
                detector.baseline = BaselineConfig()
            if save_figures:
                filename = os.path.join(
                    outputdir,
                    f'{detector.id}_lat_param_refinement_baseline.png')
            else:
                filename = None
            baseline, baseline_config = \
                ConstructBaseline.construct_baseline(
                    mca_data_summed, mask=energy_mask,
                    tol=detector.baseline.tol, lam=detector.baseline.lam,
                    max_iter=detector.baseline.max_iter,
                    title=f'Baseline for detector {detector.id}',
                    xlabel='Energy (keV)', ylabel='Intensity (counts)',
                    interactive=interactive, filename=filename)

            mca_data_summed = np.maximum(mca_data_summed-baseline, 0)
            detector.baseline.lam = baseline_config['lambda']
            detector.baseline.attrs['num_iter'] = \
                baseline_config['num_iter']
            detector.baseline.attrs['error'] = baseline_config['error']

        # Get the unique HKLs and lattice spacings for the strain
        # analysis materials
        hkls, ds = get_unique_hkls_ds(
            strain_analysis_config.materials,
            tth_tol=detector.hkl_tth_tol,
            tth_max=detector.tth_max)

        if interactive or save_figures:
            # Local modules
            from CHAP.edd.utils import (
                select_material_params,
                select_mask_and_hkls,
            )

            # Interactively adjust the initial material parameters
            tth = detector.tth_calibrated
            if save_figures:
                filename = os.path.join(
                    outputdir,
                    f'{detector.id}_lat_param_refinement_'
                    'initial_material_config.png')
            else:
                filename = None
            strain_analysis_config.materials = select_material_params(
                mca_bin_energies, mca_data_summed*energy_mask, tth,
                preselected_materials=strain_analysis_config.materials,
                label='Sum of all spectra in the map',
                interactive=interactive, filename=filename)
            self.logger.debug(
                f'materials: {strain_analysis_config.materials}')

            # Interactively adjust the mask and HKLs used in the
            # lattice parameter refinement
            if save_figures:
                filename = os.path.join(
                    outputdir,
                    f'{detector.id}_lat_param_refinement_fit_mask_hkls.png')
            else:
                filename = None
            include_bin_ranges, hkl_indices = \
                select_mask_and_hkls(
                    mca_bin_energies, mca_data_summed*energy_mask,
                    hkls, ds, detector.tth_calibrated,
                    preselected_bin_ranges=detector.include_bin_ranges,
                    preselected_hkl_indices=detector.hkl_indices,
                    detector_id=detector.id,
                    ref_map=mca_data*energy_mask,
                    calibration_bin_ranges=detector.calibration_bin_ranges,
                    label='Sum of all spectra in the map',
                    interactive=interactive, filename=filename)
            detector.include_energy_ranges = \
                detector.get_include_energy_ranges(include_bin_ranges)
            detector.hkl_indices = hkl_indices
            self.logger.debug(
                f'include_energy_ranges for detector {detector.id}:'
                f' {detector.include_energy_ranges}')
            self.logger.debug(
                f'hkl_indices for detector {detector.id}:'
                f' {detector.hkl_indices}')
            if not detector.include_energy_ranges:
                raise ValueError(
                    'No value provided for include_energy_ranges. '
                    'Provide them in the MCA Tth Calibration Configuration, '
                    'or re-run the pipeline with the --interactive flag.')
            if not detector.hkl_indices:
                raise ValueError(
                    'No value provided for hkl_indices. Provide them in '
                    'the detector\'s MCA Tth Calibration Configuration, or'
                    ' re-run the pipeline with the --interactive flag.')

        effective_map_shape = mca_data.shape[:-1]
        mask = detector.mca_mask()
        energies = mca_bin_energies[mask]
        intensities = np.empty(
            (*effective_map_shape, len(energies)), dtype=np.float64)
        for map_index in np.ndindex(effective_map_shape):
            if detector.baseline:
                intensities[map_index] = \
                    np.maximum(mca_data[map_index]-baseline, 0).astype(
                        np.float64)[mask]
            else:
                intensities[map_index] = \
                    mca_data[map_index].astype(np.float64)[mask]
        mean_intensity = np.mean(
            intensities, axis=tuple(range(len(effective_map_shape))))
        hkls_fit = np.asarray([hkls[i] for i in detector.hkl_indices])
        ds_fit = np.asarray([ds[i] for i in detector.hkl_indices])
        rs = np.sqrt(np.sum(hkls_fit**2, 1))
        peak_locations = get_peak_locations(
            ds_fit, detector.tth_calibrated)

        map_config = strain_analysis_config.map_config
        nxprocess = nxsubentry[f'{map_config.title}_lat_par_refinement']
        nxprocess[detector.id] = NXdetector()
        nxdetector = nxprocess[detector.id]
        nxdetector.local_name = detector.id
        nxdetector.detector_config = dumps(detector.dict())
        nxdetector.data = NXdata()
        det_nxdata = nxdetector.data
        linkdims(
            det_nxdata,
            {'axes': 'energy', 'index': len(effective_map_shape)})
        det_nxdata.energy = NXfield(value=energies, attrs={'units': 'keV'})
        det_nxdata.intensity = NXfield(
            value=intensities,
            shape=(*effective_map_shape, len(energies)),
            dtype=np.float64,
            attrs={'units': 'counts'})
        det_nxdata.mean_intensity = mean_intensity

        # Get the interplanar spacings measured for each fit HKL peak
        # at every point in the map to get the refined estimate
        # for the material's lattice parameter
        (uniform_fit_centers, _, _, _, _, _, uniform_best_fit, _, _, _,
         unconstrained_fit_centers, _, _, _, _, _,
         unconstrained_best_fit, _, _, _) = get_spectra_fits(
            intensities, energies, peak_locations, detector)
        rs_map = rs.repeat(np.prod(effective_map_shape))
        d_uniform = get_peak_locations(
            np.asarray(uniform_fit_centers), detector.tth_calibrated)
        a_uniform = (rs_map * d_uniform.flatten()).reshape(d_uniform.shape)
        a_uniform = a_uniform.mean(axis=0)
        a_uniform_mean = float(a_uniform.mean())
        d_unconstrained = get_peak_locations(
            unconstrained_fit_centers, detector.tth_calibrated)
        a_unconstrained = (
            rs_map * d_unconstrained.flatten()).reshape(d_unconstrained.shape)
        a_unconstrained = np.moveaxis(a_unconstrained, 0, -1)
        a_unconstrained_mean = float(a_unconstrained.mean())
        self.logger.warning(
            'Lattice parameter refinement assumes cubic lattice')
        self.logger.info(
            f'Refined lattice parameter from uniform fit: {a_uniform_mean}')
        self.logger.info(
            'Refined lattice parameter from unconstrained fit: '
            f'{a_unconstrained_mean}')
        nxdetector.lat_pars = NXcollection()
        nxdetector.lat_pars.uniform = NXdata()
        nxdata = nxdetector.lat_pars.uniform
        nxdata.nxsignal = NXfield(
            value=a_uniform, name='a_uniform', attrs={'units': r'\AA'})
        linkdims(nxdata)
        nxdetector.lat_pars.a_uniform_mean = float(a_uniform.mean())
        nxdetector.lat_pars.unconstrained = NXdata()
        nxdata = nxdetector.lat_pars.unconstrained
        nxdata.nxsignal = NXfield(
            value=a_unconstrained, name='a_unconstrained',
            attrs={'units': r'\AA'})
        nxdata.hkl_index = detector.hkl_indices
        linkdims(
            nxdata,
            {'axes': 'hkl_index', 'index': len(effective_map_shape)})
        nxdetector.lat_pars.a_unconstrained_mean = a_unconstrained_mean

        # Get the interplanar spacings measured for each fit HKL peak
        # at the spectrum averaged over every point in the map to get
        # the refined estimate for the material's lattice parameter
        (uniform_fit_centers, _, _, _, _, _,
         uniform_best_fit, _, _, _, unconstrained_fit_centers, _, _, _,
         _, _, unconstrained_best_fit, _, _, _) = get_spectra_fits(
                mean_intensity, energies, peak_locations, detector)
        d_uniform = get_peak_locations(
            np.asarray(uniform_fit_centers), detector.tth_calibrated)
        d_unconstrained = get_peak_locations(
            np.asarray(unconstrained_fit_centers), detector.tth_calibrated)
        a_uniform = float((rs * d_uniform).mean())
        a_unconstrained = rs * d_unconstrained
        self.logger.warning(
            'Lattice parameter refinement assumes cubic lattice')
        self.logger.info(
            'Refined lattice parameter from uniform fit over averaged '
            f'spectrum: {a_uniform}')
        self.logger.info(
            'Refined lattice parameter from unconstrained fit over averaged '
            f'spectrum: {a_unconstrained}')
        nxdetector.lat_pars_mean_intensity = NXcollection()
        nxdetector.lat_pars_mean_intensity.a_uniform = a_uniform
        nxdetector.lat_pars_mean_intensity.a_unconstrained = a_unconstrained
        nxdetector.lat_pars_mean_intensity.a_unconstrained_mean = \
            float(a_unconstrained.mean())

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.set_title(
                f'Detector {detector.id}: Lattice Parameter Refinement')
            ax.set_xlabel('Detector energy (keV)')
            ax.set_ylabel('Mean intensity (counts)')
            ax.plot(energies, mean_intensity, 'k.', label='MCA data')
            ax.plot(energies, uniform_best_fit, 'r', label='Best uniform fit')
            ax.plot(
                energies, unconstrained_best_fit, 'b',
                label='Best unconstrained fit')
            ax.legend()
            for i, loc in enumerate(peak_locations):
                ax.axvline(loc, c='k', ls='--')
                ax.text(loc, 1, str(hkls_fit[i])[1:-1],
                              ha='right', va='top', rotation=90,
                              transform=ax.get_xaxis_transform())
            if save_figures:
                fig.tight_layout()#rect=(0, 0, 1, 0.95))
                figfile = os.path.join(
                    outputdir, f'{detector.id}_lat_param_fits.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return [
            a_uniform, a_uniform, a_uniform, 90., 90., 90.]


class MCAEnergyCalibrationProcessor(Processor):
    """Processor to return parameters for linearly transforming MCA
    channel indices to energies (in keV). Procedure: provide a
    spectrum from the MCA element to be calibrated and the theoretical
    location of at least one peak present in that spectrum (peak
    locations must be given in keV). It is strongly recommended to use
    the location of fluorescence peaks whenever possible, _not_
    diffraction peaks, as this Processor does not account for
    2&theta."""
    def process(
            self, data, config=None, centers_range=20, fwhm_min=None,
            fwhm_max=None, max_energy_kev=200.0, background=None,
            baseline=False, save_figures=False, interactive=False,
            inputdir='.', outputdir='.'):
        """For each detector in the `MCAEnergyCalibrationConfig`
        provided with `data`, fit the specified peaks in the MCA
        spectrum specified. Using the difference between the provided
        peak locations and the fit centers of those peaks, compute
        the correction coefficients to convert uncalibrated MCA
        channel energies to calibrated channel energies. For each
        detector, set `energy_calibration_coeffs` in the calibration
        config provided to these values and return the updated
        configuration.

        :param data: An energy Calibration configuration.
        :type data: PipelineData
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.MCAEnergyCalibrationConfig.
        :type config: dict, optional
        :param centers_range: Set boundaries on the peak centers in
            MCA channels when performing the fit. The min/max
            possible values for the peak centers will be the initial
            values &pm; `centers_range`. Defaults to `20`.
        :type centers_range: int, optional
        :param fwhm_min: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_min: float, optional
        :param fwhm_max: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_max: float, optional
        :param max_energy_kev: Maximum channel energy of the MCA in
            keV, defaults to `200.0`.
        :type max_energy_kev: float, optional
        :param background: Background model for peak fitting.
        :type background: str, list[str], optional
        :param baseline: Automated baseline subtraction configuration,
            defaults to `False`.
        :type baseline: bool, BaselineConfig, optional
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
            version of the calibrated configuration.
        :rtype: dict
        """
        # Third party modules
        from json import loads
        from nexusformat.nexus import NXroot

        # Local modules
        from CHAP.common.models.map import DetectorConfig
        from CHAP.edd.models import (
            BaselineConfig,
            MCAElementCalibrationConfig,
        )
        from CHAP.utils.general import (
            is_int,
            is_num,
            is_str_series,
        )

        # Load the detector data
        # FIX input a numpy and create/use NXobject to numpy proc
        # FIX right now spec info is lost in output yaml, add to it?
        nxroot = self.get_data(data)
        if not isinstance(nxroot, NXroot):
            raise RuntimeError('No valid NXroot data in input pipeline data')
        nxentry = nxroot[nxroot.default]

        # Load the validated energy calibration configuration
        try:
            calibration_config = self.get_config(
                data, 'edd.models.MCAEnergyCalibrationConfig',
                inputdir=inputdir)
        except:
            self.logger.info('No valid calibration config in input pipeline '
                             'data, using config parameter instead.')
            try:
                # Local modules
                from CHAP.edd.models import MCAEnergyCalibrationConfig

                calibration_config = MCAEnergyCalibrationConfig(
                    **config, inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        # Validate the detector configuration
        detector_config = DetectorConfig(**loads(str(nxentry.detectors)))
        available_detector_indices = [
            int(d.id) for d in detector_config.detectors]
        if calibration_config.detectors is None:
            calibration_config.detectors = [
                MCAElementCalibrationConfig(id=d)
                for d in available_detector_indices]
        else:
            for detector in deepcopy(calibration_config.detectors):
                index = int(detector.id)
                if index not in available_detector_indices:
                    self.logger.warning(
                        f'Skipping detector {index} (no raw data)')
                    calibration_config.detectors.remove(detector)

        # Validate the fit index range
        if calibration_config.fit_index_ranges is None and not interactive:
            raise RuntimeError(
                'If `fit_index_ranges` is not explicitly provided, '
                f'{self.__class__.__name__} must be run with '
                '`interactive=True`.')

        # Validate the optional inputs
        if not is_int(centers_range, gt=0, log=False):
            raise RuntimeError(
                f'Invalid centers_range parameter ({centers_range})')
        if fwhm_min is not None and not is_int(fwhm_min, gt=0, log=False):
            raise RuntimeError(f'Invalid fwhm_min parameter ({fwhm_min})')
        if fwhm_max is not None and not is_int(fwhm_max, gt=0, log=False):
            raise RuntimeError(f'Invalid fwhm_max parameter ({fwhm_max})')
        if not is_num(max_energy_kev, gt=0, log=False):
            raise RuntimeError(
                f'Invalid max_energy_kev parameter ({max_energy_kev})')
        if background is not None:
            if isinstance(background, str):
                background = [background]
            elif not is_str_series(background, log=False):
                raise RuntimeError(
                    f'Invalid background parameter ({background})')
        if isinstance(baseline, bool):
            if baseline:
                baseline = BaselineConfig()
        else:
            try:
                baseline = BaselineConfig(**baseline)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        # Collect and sum the detector data
        mca_data = []
        for scan_name in nxentry.spec_scans:
            for _, scan_data in nxentry.spec_scans[scan_name].items():
                mca_data.append(scan_data.data.data.nxdata)
        summed_detector_data = np.asarray(mca_data).sum(axis=(0,1))

        # Get the detectors' num_bins parameter
        for detector in calibration_config.detectors:
            if detector.num_bins is None:
                detector.num_bins = summed_detector_data.shape[-1]

        # Check each detector's include_energy_ranges field against the
        # flux file, if available.
        if calibration_config.flux_file is not None:
            flux = np.loadtxt(flux_file)
            flux_file_energies = flux[:,0]/1.e3
            flux_e_min = flux_file_energies.min()
            flux_e_max = flux_file_energies.max()
            for detector in calibration_config.detectors:
                for i, (det_e_min, det_e_max) in enumerate(
                        deepcopy(detector.include_energy_ranges)):
                    if det_e_min < flux_e_min or det_e_max > flux_e_max:
                        energy_range = [float(max(det_e_min, flux_e_min)),
                                        float(min(det_e_max, flux_e_max))]
                        print(
                            f'WARNING: include_energy_ranges[{i}] out of range'
                            f' ({detector.include_energy_ranges[i]}): adjusted'
                            f' to {energy_range}')
                        detector.include_energy_ranges[i] = energy_range

        # Calibrate detector channel energies based on fluorescence peaks
        for detector in calibration_config.detectors:
            index = available_detector_indices.index(int(detector.id))
            if background is not None:
                detector.background = background.copy()
            if baseline:
                detector.baseline = baseline.model_copy()
            detector.energy_calibration_coeffs = self.calibrate(
                calibration_config, detector, summed_detector_data[index],
                centers_range, fwhm_min, fwhm_max, max_energy_kev,
                save_figures, interactive, outputdir)

        return calibration_config.dict()

    def calibrate(
            self, calibration_config, detector, spectrum, centers_range,
            fwhm_min, fwhm_max, max_energy_kev, save_figures, interactive,
            outputdir):
        """Return energy_calibration_coeffs (a, b, and c) for
        quadratically converting the current detector's MCA channels
        to bin energies.

        :param calibration_config: Energy calibration configuration.
        :type calibration_config: MCAEnergyCalibrationConfig
        :param detector: Configuration of the current detector.
        :type detector: MCAElementCalibrationConfig
        :param spectrum: Summed MCA spectrum for the current detector.
        :type spectrum: numpy.ndarray
        :param centers_range: Set boundaries on the peak centers in
            MCA channels when performing the fit. The min/max
            possible values for the peak centers will be the initial
            values &pm; `centers_range`. Defaults to `20`.
        :type centers_range: int, optional
        :param fwhm_min: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_min: float, optional
        :param fwhm_max: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_max: float, optional
        :param max_energy_kev: Maximum channel energy of the MCA in
            keV, defaults to `200.0`.
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
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        # Local modules
        from CHAP.utils.fit import FitProcessor
        from CHAP.utils.general import (
            index_nearest,
            select_mask_1d,
        )

        self.logger.info(f'Calibrating detector {detector.id}')

        # Get the MCA bin energies
        uncalibrated_energies = np.linspace(
            0, max_energy_kev, detector.num_bins)
        bins = np.arange(detector.num_bins, dtype=np.int16)

        # Blank out data below 25keV as well as the last bin
        energy_mask = np.where(uncalibrated_energies >= 25.0, 1, 0)
        energy_mask[-1] = 0
        spectrum = spectrum*energy_mask

        # Subtract the baseline
        if detector.baseline:
            # Local modules
            from CHAP.common.processor import ConstructBaseline

            if save_figures:
                filename = os.path.join(outputdir,
                    f'{detector.id}_energy_calibration_baseline.png')
            else:
                filename = None
            baseline, baseline_config = ConstructBaseline.construct_baseline(
                spectrum, mask=energy_mask, tol=detector.baseline.tol,
                lam=detector.baseline.lam, max_iter=detector.baseline.max_iter,
                title=f'Baseline for detector {detector.id}',
                xlabel='Detector channel (-)', ylabel='Intensity (counts)',
                interactive=interactive, filename=filename)

            spectrum = np.maximum(spectrum-baseline, 0)
            detector.baseline.lam = baseline_config['lambda']
            detector.baseline.attrs['num_iter'] = baseline_config['num_iter']
            detector.baseline.attrs['error'] = baseline_config['error']

        # Select the mask/detector channel ranges for fitting
        if save_figures:
            filename = os.path.join(
                outputdir, f'{detector.id}_energy_calibration_mask.png')
        else:
            filename = None
        mask, fit_index_ranges = select_mask_1d(
            spectrum, x=bins,
            preselected_index_ranges=calibration_config.fit_index_ranges,
            xlabel='Detector channel (-)', ylabel='Intensity (counts)',
            min_num_index_ranges=1, interactive=interactive,
            filename=filename)
        self.logger.debug(
            f'Selected index ranges to fit: {fit_index_ranges}')

        # Get the intial peak positions for fitting
        max_peak_energy = calibration_config.peak_energies[
            calibration_config.max_peak_index]
        peak_energies = list(np.sort(calibration_config.peak_energies))
        max_peak_index = peak_energies.index(max_peak_energy)
        if save_figures:
            filename = os.path.join(
                outputdir,
                f'{detector.id}'
                    '_energy_calibration_initial_peak_positions.png')
        else:
            filename = None
        input_indices = [index_nearest(uncalibrated_energies, energy)
                         for energy in peak_energies]
        initial_peak_indices = self._get_initial_peak_positions(
            spectrum*np.asarray(mask).astype(np.int32), fit_index_ranges,
            input_indices, max_peak_index, interactive, filename,
            detector.id)

        # Construct the fit model
        models = []
        if detector.background is not None:
            if isinstance(detector.background, str):
                models.append(
                    {'model': detector.background, 'prefix': 'bkgd_'})
            else:
                for model in detector.background:
                    models.append({'model': model, 'prefix': f'{model}_'})
        models.append(
            {'model': 'multipeak', 'centers': initial_peak_indices,
             'centers_range': centers_range, 'fwhm_min': fwhm_min,
             'fwhm_max': fwhm_max})
        self.logger.debug('Fitting spectrum')
        fit = FitProcessor()
        spectrum_fit = fit.process(
                NXdata(NXfield(spectrum[mask], 'y'), NXfield(bins[mask], 'x')),
                {'models': models, 'method': 'trf'})

        fit_peak_amplitudes = sorted([
            spectrum_fit.best_values[f'peak{i+1}_amplitude']
            for i in range(len(initial_peak_indices))])
        self.logger.debug(f'Fit peak amplitudes: {fit_peak_amplitudes}')
        fit_peak_indices = sorted([
            spectrum_fit.best_values[f'peak{i+1}_center']
            for i in range(len(initial_peak_indices))])
        self.logger.debug(f'Fit peak center indices: {fit_peak_indices}')
        fit_peak_fwhms = sorted([
            2.35482*spectrum_fit.best_values[f'peak{i+1}_sigma']
            for i in range(len(initial_peak_indices))])
        self.logger.debug(f'Fit peak fwhms: {fit_peak_fwhms}')

        #RV for now stick with a linear energy correction
        fit = FitProcessor()
        energy_fit = fit.process(
                NXdata(
                    NXfield(peak_energies, 'y'),
                    NXfield(fit_peak_indices, 'x')),
                {'models': [{'model': 'linear'}]})
        a = 0.0
        b = float(energy_fit.best_values['slope'])
        c = float(energy_fit.best_values['intercept'])

        # Reference plot to see fit results:
        if interactive or save_figures:
            # Third part modules
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 2, figsize=(11, 4.25))
            fig.suptitle(f'Detector {detector.id} Energy Calibration')
            # Left plot: raw MCA data & best fit of peaks
            axs[0].set_title('MCA Spectrum Peak Fit')
            axs[0].set_xlabel('Detector channel (-)')
            axs[0].set_ylabel('Intensity (counts)')
            axs[0].plot(bins[mask], spectrum[mask], 'b.', label='MCA data')
            axs[0].plot(
                bins[mask], spectrum_fit.best_fit, 'r', label='Best fit')
            axs[0].legend()
            # Right plot: linear fit of theoretical peak energies vs
            # fit peak locations
            axs[1].set_title(
                'Channel Energies vs. Channel Indices')
            axs[1].set_xlabel('Detector channel (-)')
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
                bbox={'boxstyle': 'round', 'ec': (1., 0.5, 0.5),
                      'fc': (1., 0.8, 0.8, 0.8)},
            )

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(
                    outputdir, f'{detector.id}_energy_calibration_fit.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        return [a, b, c]

    def _get_initial_peak_positions(
            self, y, index_ranges, input_indices, input_max_peak_index,
            interactive, filename, detector_id, reset_flag=0):
        # Third party modules
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        def change_fig_title(title):
            """Change the figure title."""
            if fig_title:
                fig_title[0].remove()
                fig_title.pop()
            fig_title.append(plt.figtext(*title_pos, title, **title_props))

        def change_error_text(error=''):
            """Change the error text."""
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
            """Find the peaks.

            :param min_height: Minimum peak height in search, defaults
                to `0.05`.
            :type min_height: float, optional
            :param min_width: Minimum peak width in search, defaults
                to `5`.
            :type min_width: float, optional
            :param tolerance: Tolerance in peak index for finding
                matching peaks, defaults to `0.05`.
            :type tolerance: float, optional
            :return: The peak indices.
            :rtype: list[int]
            """
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
                    ax.set_xlabel('Detector channel (-)', fontsize='x-large')
                    ax.set_ylabel('Intensity (counts)', fontsize='x-large')
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
        if detector_id is None:
            detector_id = ''
        elif not isinstance(detector_id, str):
            raise ValueError(
                f'Invalid parameter `detector_id`: {detector_id}')
        elif not reset_flag:
            detector_id = f' on detector {detector_id}'
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
        ax.set_xlabel('Detector channel (-)', fontsize='x-large')
        ax.set_ylabel('Intensity (counts)', fontsize='x-large')
        ax.set_xlim(index_ranges[0][0], index_ranges[-1][1])
        fig.subplots_adjust(bottom=0.0, top=0.85)

        if not interactive:

            peak_indices += find_peaks()

            for index in peak_indices:
                ax.axvline(index, **selected_peak_props)
            change_fig_title('Initial peak positions from peak finding '
                             f'routine{detector_id}')

        else:

            fig.subplots_adjust(bottom=0.2)

            # Get initial peak indices
            if not reset_flag:
                peak_indices += find_peaks()
                change_fig_title('Initial peak positions from peak finding '
                                 f'routine{detector_id}')
            if peak_indices:
                for index in peak_indices:
                    if not any(True if low <= index <= upp else False
                           for low, upp in index_ranges):
                        peak_indices.clear()
                        break
            if not peak_indices:
                peak_indices += select_peaks()
                change_fig_title(
                    'Selected initial peak positions{detector_id}')

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
                y, index_ranges, input_indices, input_max_peak_index,
                interactive, filename, detector_id, reset_flag=reset_flag)
        return peak_indices


class MCATthCalibrationProcessor(Processor):
    """Processor to calibrate the 2&theta angle and fine tune the
    energy calibration coefficients for an EDD experimental setup.
    """
    def process(
            self, data, config=None, tth_initial_guess=None,
            include_energy_ranges=None,
            calibration_method='direct_fit_residual',
            quadratic_energy_calibration=False, centers_range=20,
            fwhm_min=None, fwhm_max=None, background='constant',
            baseline=False, save_figures=False, inputdir='.', outputdir='.',
            interactive=False):
        """Return the calibrated 2&theta value and the fine tuned
        energy calibration coefficients to convert MCA channel
        indices to MCA channel energies.

        :param data: Input configuration for the raw data & tuning
            procedure.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.MCATthCalibrationConfig.
        :type config: dict, optional
        :param tth_initial_guess: Initial guess for 2&theta to supercede
            the values from the energy calibration detector cofiguration
            on each of the detectors.
        :type tth_initial_guess: float, optional
        :param include_energy_ranges: List of MCA channel energy ranges
            in keV whose data should be included after applying a mask
            (bounds are inclusive). If specified, these supercede the
            values from the energy calibration detector cofiguration on
            each of the detectors.
        :type include_energy_ranges: list[[float, float]], optional
        :param calibration_method: Type of calibration method,
            defaults to `'direct_fit_residual'`.
        :type calibration_method:
            Union['direct_fit_residual', 'direct_fit_peak_energies',
                'direct_fit_combined', 'iterate_tth'], optional
        :param quadratic_energy_calibration: Adds a quadratic term to
            the detector channel index to energy conversion, defaults
            to `False` (linear only).
        :type quadratic_energy_calibration: bool, optional
        :param centers_range: Set boundaries on the peak centers in
            MCA channels when performing the fit. The min/max
            possible values for the peak centers will be the initial
            values &pm; `centers_range`. Defaults to `20`.
        :type centers_range: int, optional
        :param fwhm_min: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_min: float, optional
        :param fwhm_max: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_max: float, optional
        :param background: Background model for peak fitting.
        :type background: str, list[str], optional
        :param baseline: Automated baseline subtraction configuration,
            defaults to `False`.
        :type baseline: bool, BaselineConfig, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :param interactive: Allows for user interactions,
            defaults to `False`.
        :type interactive: bool, optional
        :raises RuntimeError: Invalid or missing input configuration.
        :return: Original configuration with the tuned values for
            2&theta and the linear correction parameters added.
        :rtype: dict[str,float]
        """
        # Third party modules
        from json import loads
        from nexusformat.nexus import NXroot

        # Local modules
        from CHAP.common.models.map import DetectorConfig
        from CHAP.edd.models import BaselineConfig
        from CHAP.utils.general import (
            is_int,
            is_str_series,
            list_to_string,
        )

        # Load the detector data
        # FIX input a numpy and create/use NXobject to numpy proc
        # FIX right now spec info is lost in output yaml, add to it?
        nxroot = self.get_data(data)
        if not isinstance(nxroot, NXroot):
            raise RuntimeError('No valid NXroot data in input pipeline data')
        nxentry = nxroot[nxroot.default]
        # Load the validated 2&theta calibration configuration
        try:
            calibration_config = self.get_config(
                data, 'edd.models.MCATthCalibrationConfig',
                calibration_method=calibration_method,
                inputdir=inputdir)
        except:
            self.logger.info('No valid calibration config in input pipeline '
                             'data, using config parameter instead.')
            try:
                # Local modules
                from CHAP.edd.models import MCATthCalibrationConfig

                calibration_config = MCATthCalibrationConfig(
                    **config, calibration_method=calibration_method,
                    inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        # Validate the detector configuration
        if calibration_config.detectors is None:
            raise RuntimeError('No available calibrated detectors')
        detector_config = DetectorConfig(**loads(str(nxentry.detectors)))
        available_detector_indices = [
            int(d.id) for d in detector_config.detectors]
        calibration_detector_indices = []
        for detector in deepcopy(calibration_config.detectors):
            index = int(detector.id)
            if index in available_detector_indices:
                calibration_detector_indices.append(index)
            else:
                self.logger.warning(f'Skipping detector {index} (no raw data)')
                calibration_config.detectors.remove(detector)
        detectors = calibration_config.detectors
        skipped_detector_indices = [
            index for index in available_detector_indices
            if index not in calibration_detector_indices]
        if skipped_detector_indices:
            self.logger.warning('Skipping detector(s) '
                                f'{list_to_string(skipped_detector_indices)} '
                                '(no calibration data)')

        # Validate the fit index range
        if calibration_config.fit_index_ranges is None and not interactive:
            raise RuntimeError(
                'If `fit_index_ranges` is not explicitly provided, '
                f'{self.__class__.__name__} must be run with '
                '`interactive=True`.')

        # Validate the optional inputs
        if not is_int(centers_range, gt=0, log=False):
            raise RuntimeError(
                f'Invalid centers_range parameter ({centers_range})')
        if fwhm_min is not None and not is_int(fwhm_min, gt=0, log=False):
            raise RuntimeError(f'Invalid fwhm_min parameter ({fwhm_min})')
        if fwhm_max is not None and not is_int(fwhm_max, gt=0, log=False):
            raise RuntimeError(f'Invalid fwhm_max parameter ({fwhm_max})')
        if background is not None:
            if isinstance(background, str):
                background = [background]
            elif not is_str_series(background, log=False):
                raise RuntimeError(
                    f'Invalid background parameter ({background})')
        if isinstance(baseline, bool):
            if baseline:
                baseline = BaselineConfig()
        else:
            try:
                baseline = BaselineConfig(**baseline)
            except Exception as dict_exc:
                raise RuntimeError from dict_exc

        # Collect and sum the detector data
        mca_data = []
        for scan_name in nxentry.spec_scans:
            for _, scan_data in nxentry.spec_scans[scan_name].items():
                mca_data.append(scan_data.data.data.nxdata)
        summed_detector_data = np.asarray(mca_data).sum(axis=(0,1))

        # Get the detectors' num_bins parameter
        for detector in detectors:
            if detector.num_bins is None:
                detector.num_bins = summed_detector_data.shape[-1]

        # Check each detector's include_energy_ranges field against the
        # flux file, if available.
        if calibration_config.flux_file is not None:
            flux = np.loadtxt(calibration_config.flux_file)
            flux_file_energies = flux[:,0]/1.e3
            flux_e_min = flux_file_energies.min()
            flux_e_max = flux_file_energies.max()
            for detector in detectors:
                for i, (det_e_min, det_e_max) in enumerate(
                        deepcopy(detector.include_energy_ranges)):
                    if det_e_min < flux_e_min or det_e_max > flux_e_max:
                        energy_range = [float(max(det_e_min, flux_e_min)),
                                        float(min(det_e_max, flux_e_max))]
                        print(
                            f'WARNING: include_energy_ranges[{i}] out of range'
                            f' ({detector.include_energy_ranges[i]}): adjusted'
                            f' to {energy_range}')
                        detector.include_energy_ranges[i] = energy_range

        # Calibrate detector channel energies
        for detector in detectors:
            index = available_detector_indices.index(int(detector.id))
            if tth_initial_guess is not None:
                detector.tth_initial_guess = tth_initial_guess
            if include_energy_ranges is not None:
                detector.include_energy_ranges = include_energy_ranges
            if background is not None:
                detector.background = background.copy()
            if baseline:
                detector.baseline = baseline
            self.calibrate(
                calibration_config, detector, summed_detector_data[index],
                quadratic_energy_calibration, centers_range, fwhm_min,
                fwhm_max, save_figures, interactive, outputdir)

        return calibration_config.dict()

    def calibrate(
            self, calibration_config, detector, spectrum,
            quadratic_energy_calibration=False, centers_range=20,
            fwhm_min=None, fwhm_max=None, save_figures=False,
            interactive=False, outputdir='.'):
        """Iteratively calibrate 2&theta by fitting selected peaks of
        an MCA spectrum until the computed strain is sufficiently
        small. Use the fitted peak locations to determine linear
        correction parameters for the MCA channel energies.

        :param calibration_config: Object configuring the CeO2
            calibration procedure for an MCA detector.
        :type calibration_config:
            CHAP.edd.models.MCATthCalibrationConfig
        :param detector: A single MCA detector element configuration.
        :type detector: CHAP.edd.models.MCAElementCalibrationConfig
        :param spectrum: Summed MCA spectrum for the current detector.
        :type spectrum: numpy.ndarray
        :param quadratic_energy_calibration: Adds a quadratic term to
            the detector channel index to energy conversion, defaults
            to `False` (linear only).
        :type quadratic_energy_calibration: bool, optional
        :param centers_range: Set boundaries on the peak centers in
            MCA channels when performing the fit. The min/max
            possible values for the peak centers will be the initial
            values &pm; `centers_range`. Defaults to `20`.
        :type centers_range: int, optional
        :param fwhm_min: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_min: float, optional
        :param fwhm_max: Lower bound on the peak FWHM in MCA channels
            when performing the fit.
        :type fwhm_max: float, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :raises ValueError: No value provided for included bin ranges
            or the fitted HKLs for the MCA detector element.
        """
        # System modules
        from sys import float_info

        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )
        from scipy.constants import physical_constants

        # Local modules
        if interactive or save_figures:
            from CHAP.edd.utils import (
                select_tth_initial_guess,
                select_mask_and_hkls,
            )
        from CHAP.edd.utils import get_peak_locations
        from CHAP.utils.fit import FitProcessor
        from CHAP.utils.general import index_nearest

        self.logger.info(f'Calibrating detector {detector.id}')

        calibration_method = calibration_config.calibration_method

        # Get the unique HKLs and lattice spacings for the calibration
        # material
        hkls, ds = calibration_config.material.unique_hkls_ds(
            tth_tol=detector.hkl_tth_tol, tth_max=detector.tth_max)

        # Get the MCA bin energies
        mca_bin_energies = detector.energies

        # Blank out data below 25 keV as well as the last bin
        energy_mask = np.where(mca_bin_energies >= 25.0, 1, 0)
        energy_mask[-1] = 0
        spectrum = spectrum*energy_mask

        # Subtract the baseline
        if detector.baseline:
            # Local modules
            from CHAP.common.processor import ConstructBaseline

            if save_figures:
                filename = os.path.join(
                    outputdir, f'{detector.id}_tth_calibration_baseline.png')
            else:
                filename = None
            baseline, baseline_config = ConstructBaseline.construct_baseline(
                spectrum, mask=energy_mask, tol=detector.baseline.tol,
                lam=detector.baseline.lam, max_iter=detector.baseline.max_iter,
                title=f'Baseline for detector {detector.id}',
                xlabel='Detector channel (-)', ylabel='Intensity (counts)',
                interactive=interactive, filename=filename)

            spectrum = np.maximum(spectrum-baseline, 0)
            detector.baseline.lam = baseline_config['lambda']
            detector.baseline.attrs['num_iter'] = baseline_config['num_iter']
            detector.baseline.attrs['error'] = baseline_config['error']

        # Adjust initial tth guess
        if save_figures:
            filename = os.path.join(
               outputdir, f'{detector.id}_tth_calibration_initial_guess.png')
        else:
            filename = None
        tth_init = select_tth_initial_guess(
            mca_bin_energies, spectrum, hkls, ds,
            detector.tth_initial_guess, interactive, filename)
        detector.tth_initial_guess = tth_init
        self.logger.debug(f'tth_initial_guess = {detector.tth_initial_guess}')

        # Select the mask and HKLs for the Bragg peaks
        if save_figures:
            filename = os.path.join(
                outputdir, f'{detector.id}_tth_calibration_mask_hkls.png')
        if calibration_method == 'iterate_tth':
            num_hkl_min = 2
        else:
            num_hkl_min = 1
        include_bin_ranges, hkl_indices = select_mask_and_hkls(
            mca_bin_energies, spectrum, hkls, ds,
            detector.tth_initial_guess,
            preselected_bin_ranges=detector.include_bin_ranges,
            num_hkl_min=num_hkl_min, detector_id=detector.id,
            flux_energy_range=calibration_config.flux_file_energy_range(),
            label='MCA data', interactive=interactive, filename=filename)

        # Add the mask for the fluorescence peaks
        if calibration_method != 'iterate_tth':
            include_bin_ranges = (
                calibration_config.fit_index_ranges + include_bin_ranges)

        # Apply the mask
        detector.include_energy_ranges = detector.get_include_energy_ranges(
            include_bin_ranges)
        detector.set_hkl_indices(hkl_indices)
        self.logger.debug(
            f'include_energy_ranges = {detector.include_energy_ranges}')
        self.logger.debug(
            f'hkl_indices = {detector.hkl_indices}')
        if not detector.include_energy_ranges:
            raise ValueError(
                'No value provided for include_energy_ranges. '
                'Provide them in the MCA Tth Calibration Configuration '
                'or re-run the pipeline with the --interactive flag.')
        if not detector.hkl_indices:
            raise ValueError(
                'Unable to get values for hkl_indices for the provided '
                'value of include_energy_ranges. Change its value in '
                'the detector\'s MCA Tth Calibration Configuration or '
                're-run the pipeline with the --interactive flag.')
        mca_mask = detector.mca_mask()
        spectrum_fit = spectrum[mca_mask]

        # Correct raw MCA data for variable flux at different energies
        flux_correct = \
            calibration_config.flux_correction_interpolation_function()
        if flux_correct is not None:
            mca_intensity_weights = flux_correct(
                 mca_bin_energies[mca_mask])
            spectrum_fit = spectrum_fit / mca_intensity_weights

        # Get the fluorescence peak info
        e_xrf = calibration_config.peak_energies
        num_xrf = len(e_xrf)

        # Get the Bragg peak HKLs, lattice spacings and energies
        hkls_fit  = np.asarray([hkls[i] for i in detector.hkl_indices])
        ds_fit  = np.asarray([ds[i] for i in detector.hkl_indices])
        c_1_fit = hkls_fit[:,0]**2 + hkls_fit[:,1]**2 + hkls_fit[:,2]**2
        e_bragg_init = get_peak_locations(ds_fit, tth_init)
        num_bragg = len(e_bragg_init)

        # Perform the fit
        if calibration_method == 'direct_fit_residual':

            # Get the initial free fit parameters
            tth_init = np.radians(tth_init)
            a_init, b_init, c_init = detector.energy_calibration_coeffs

            # For testing: hardwired limits:
            if False: #RV
                min_value = None
                tth_min = None
                tth_max = None
                b_min = None
                b_max = None
                sig_min = None
                sig_max = None
            else:
                min_value = float_info.min
                tth_min = 0.9*tth_init
                tth_max = 1.1*tth_init
                b_min = 0.1*b_init
                b_max = 10.0*b_init
                if isinstance(fwhm_min, (int,float)):
                    sig_min = fwhm_min/2.35482
                else:
                    sig_min = None
                if isinstance(fwhm_max, (int,float)):
                    sig_max = fwhm_max/2.35482
                else:
                    sig_max = None

            # Construct the free fit parameters
            parameters = [
                {'name': 'tth', 'value': tth_init, 'min': tth_min,
                 'max': tth_max}]
            if quadratic_energy_calibration:
                parameters.append({'name': 'a', 'value': a_init})
            parameters.append(
                {'name': 'b', 'value': b_init, 'min': b_min, 'max': b_max})
            parameters.append({'name': 'c', 'value': c_init})

            # Construct the fit model
            models = []

            # Add the background
            if detector.background is not None:
                if isinstance(detector.background, str):
                    models.append(
                        {'model': detector.background, 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})

            # Add the fluorescent peaks
            for i, e_peak in enumerate(e_xrf):
                expr = f'({e_peak}-c)/b'
                if quadratic_energy_calibration:
                    expr = '(' + expr + f')*(1.0-a*(({e_peak}-c)/(b*b)))'
                models.append(
                    {'model': 'gaussian', 'prefix': f'xrf{i+1}_',
                     'parameters': [
                        {'name': 'amplitude', 'min': min_value},
                        {'name': 'center', 'expr': expr},
                        {'name': 'sigma', 'min': sig_min, 'max': sig_max}]})

            # Add the Bragg peaks
            hc = 1.e7 * physical_constants['Planck constant in eV/Hz'][0] \
                 * physical_constants['speed of light in vacuum'][0]
            for i, (e_peak, ds) in enumerate(zip(e_bragg_init, ds_fit)):
                norm = 0.5*hc/ds
                expr = f'(({norm}/sin(0.5*tth))-c)/b'
                if quadratic_energy_calibration:
                    expr = '(' + expr \
                           + f')*(1.0-a*((({norm}/sin(0.5*tth))-c)/(b*b)))'
                models.append(
                    {'model': 'gaussian', 'prefix': f'peak{i+1}_',
                     'parameters': [
                        {'name': 'amplitude', 'min': min_value},
                        {'name': 'center', 'expr': expr},
                        {'name': 'sigma', 'min': sig_min, 'max': sig_max}]})

            # Perform the fit
            fit = FitProcessor()
            result = fit.process(
                NXdata(NXfield(spectrum_fit, 'y'),
                       NXfield(np.arange(detector.num_bins)[mca_mask], 'x')),
                {'parameters': parameters, 'models': models, 'method': 'trf'})

            # Extract values of interest from the best values
            best_fit_uniform = result.best_fit
            residual_uniform = result.residual
            tth_fit = np.degrees(result.best_values['tth'])
            if quadratic_energy_calibration:
                a_fit = result.best_values['a']
            else:
                a_fit = 0.0
            b_fit = result.best_values['b']
            c_fit = result.best_values['c']
            peak_indices_fit = np.asarray(
                [result.best_values[f'xrf{i+1}_center']
                 for i in range(num_xrf)] +
                [result.best_values[f'peak{i+1}_center']
                 for i in range(num_bragg)])
            peak_energies_fit = ((a_fit*peak_indices_fit + b_fit)
                                * peak_indices_fit + c_fit)
            e_bragg_uniform = peak_energies_fit[num_xrf:]
            a_uniform = np.sqrt(c_1_fit) * abs(
                get_peak_locations(e_bragg_uniform, tth_fit))
            strains_uniform = np.log(
                (a_uniform
                 / calibration_config.material.lattice_parameters))
            strain_uniform = np.mean(strains_uniform)

        elif calibration_method == 'direct_fit_peak_energies':
            # Third party modules
            from scipy.optimize import minimize

            def cost_function(
                    pars, quadratic_energy_calibration, ds_fit,
                    indices_unconstrained, e_xrf):
                tth = pars[0]
                b = pars[1]
                c = pars[2]
                if quadratic_energy_calibration:
                    a = pars[3]
                else:
                    a = 0.0
                energies_unconstrained = (
                    (a*indices_unconstrained + b) * indices_unconstrained + c)
                target_energies = np.concatenate(
                    (e_xrf, get_peak_locations(ds_fit, tth)))
                return np.sqrt(np.sum(
                    (energies_unconstrained-target_energies)**2))

            # Get the initial free fit parameters
            a_init, b_init, c_init = detector.energy_calibration_coeffs

            # Perform an unconstrained fit in terms of MCA bin index
            mca_bins_fit = np.arange(detector.num_bins)[mca_mask]
            centers = [index_nearest(mca_bin_energies, e_peak)
                       for e_peak in np.concatenate((e_xrf, e_bragg_init))]
            models = []
            if detector.background is not None:
                if isinstance(detector.background, str):
                    models.append(
                        {'model': detector.background, 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append(
                {'model': 'multipeak', 'centers': centers,
                 'centers_range': centers_range,
                 'fwhm_min': fwhm_min, 'fwhm_max': fwhm_max})
            fit = FitProcessor()
            result = fit.process(
                NXdata(NXfield(spectrum_fit, 'y'),
                       NXfield(mca_bins_fit, 'x')),
                {'models': models, 'method': 'trf'})

            # Extract the peak properties from the fit
            indices_unconstrained = np.asarray(
                [result.best_values[f'peak{i+1}_center']
                 for i in range(num_xrf+num_bragg)])

            # Perform a peak center fit using the theoretical values
            # for the fluorescense peaks and Bragg's law for the Bragg
            # peaks for given material properties and a freely
            # adjustable 2&theta angle and MCA energy axis calibration
            pars_init = [tth_init, b_init, c_init]
            if quadratic_energy_calibration:
                pars_init.append(a_init)
            # For testing: hardwired limits:
            if True:
                bounds = [
                    (0.9*tth_init, 1.1*tth_init),
                    (0.1*b_init, 10.*b_init),
                    (0.1*c_init, 10.*c_init)]
                if quadratic_energy_calibration:
                    if a_init:
                        bounds.append((0.1*a_init, 10.0*a_init))
                    else:
                        bounds.append((None, None))
            else:
                bounds = None
            result = minimize(
                cost_function, pars_init,
                args=(
                    quadratic_energy_calibration, ds_fit,
                    indices_unconstrained, e_xrf),
                method='Nelder-Mead', bounds=bounds)

            # Extract values of interest from the best values
            best_fit_uniform = None
            residual_uniform = None
            tth_fit = float(result['x'][0])
            b_fit = float(result['x'][1])
            c_fit = float(result['x'][2])
            if quadratic_energy_calibration:
                a_fit = float(result['x'][3])
            else:
                a_fit = 0.0
            e_bragg_fit = get_peak_locations(ds_fit, tth_fit)
            peak_energies_fit = [
                (a_fit*i + b_fit) * i + c_fit
                for i in indices_unconstrained[:num_xrf]] \
                + list(e_bragg_fit)
            e_bragg_uniform = e_bragg_fit
            strain_uniform = None

        elif calibration_method == 'direct_fit_combined':
            # Third party modules
            from scipy.optimize import minimize

            def gaussian(x, a, b, c, amp, sig, e_peak):
                sig2 = 2.*sig**2
                norm = sig*np.sqrt(2.0*np.pi)
                cen = (e_peak-c) * (1.0 - a * (e_peak-c) / b**2) / b
                return amp*np.exp(-(x-cen)**2/sig2)/norm

            def cost_function_combined(
                    pars, x, y, quadratic_energy_calibration, ds_fit,
                    indices_unconstrained, e_xrf):
                tth = pars[0]
                b = pars[1]
                c = pars[2]
                amplitudes = pars[3::2]
                sigmas = pars[4::2]
                if quadratic_energy_calibration:
                    a = pars[-1]
                else:
                    a = 0.0
                energies_unconstrained = (
                    (a*indices_unconstrained + b) * indices_unconstrained + c)
                target_energies = np.concatenate(
                    (e_xrf, get_peak_locations(ds_fit, tth)))
                y_fit = np.zeros((x.size))
                for i, e_peak in enumerate(target_energies):
                    y_fit += gaussian(
                        x, a, b, c, amplitudes[i], sigmas[i], e_peak)
                target_energies_error = np.sqrt(
                    np.sum(
                        (energies_unconstrained
                         - np.asarray(target_energies))**2)
                    / len(target_energies))
                residual_error = np.sqrt(
                    np.sum((y-y_fit)**2)
                    / (np.sum(y**2) * len(target_energies)))
                return target_energies_error+residual_error

            # Get the initial free fit parameters
            a_init, b_init, c_init = detector.energy_calibration_coeffs

            # Perform an unconstrained fit in terms of MCS bin index
            mca_bins_fit = np.arange(detector.num_bins)[mca_mask]
            centers = [index_nearest(mca_bin_energies, e_peak)
                       for e_peak in np.concatenate((e_xrf, e_bragg_init))]
            models = []
            if detector.background is not None:
                if isinstance(detector.background, str):
                    models.append(
                        {'model': detector.background, 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append(
                {'model': 'multipeak', 'centers': centers,
                 'centers_range': centers_range,
                 'fwhm_min': fwhm_min, 'fwhm_max': fwhm_max})
            fit = FitProcessor()
            result = fit.process(
                NXdata(NXfield(spectrum_fit, 'y'),
                       NXfield(mca_bins_fit, 'x')),
                {'models': models, 'method': 'trf'})

            # Extract the peak properties from the fit
            num_peak = num_xrf+num_bragg
            indices_unconstrained = np.asarray(
                [result.best_values[f'peak{i+1}_center']
                 for i in range(num_peak)])
            amplitudes_init = np.asarray(
                [result.best_values[f'peak{i+1}_amplitude']
                 for i in range(num_peak)])
            sigmas_init = np.asarray(
                [result.best_values[f'peak{i+1}_sigma']
                 for i in range(num_peak)])

            # Perform a peak center fit using the theoretical values
            # for the fluorescense peaks and Bragg's law for the Bragg
            # peaks for given material properties and a freely
            # adjustable 2&theta angle and MCA energy axis calibration
            norm = spectrum_fit.max()
            pars_init = [tth_init, b_init, c_init]
            for amp, sig in zip(amplitudes_init, sigmas_init):
                pars_init += [amp/norm, sig]
            if quadratic_energy_calibration:
                pars_init += [a_init]
            # For testing: hardwired limits:
            if True:
                bounds = [
                    (0.9*tth_init, 1.1*tth_init),
                    (0.1*b_init, 10.*b_init),
                    (0.1*c_init, 10.*c_init)]
                for amp, sig in zip(amplitudes_init, sigmas_init):
                    bounds += [
                        (0.9*amp/norm, 1.1*amp/norm), (0.9*sig, 1.1*sig)]
                if quadratic_energy_calibration:
                    if a_init:
                        bounds += [(0.1*a_init, 10.*a_init)]
                    else:
                        bounds += [(None, None)]
            else:
                bounds = None
            result = minimize(
                cost_function_combined, pars_init,
                args=(
                    mca_bins_fit, spectrum_fit/norm,
                    quadratic_energy_calibration, ds_fit,
                    indices_unconstrained, e_xrf),
                method='Nelder-Mead', bounds=bounds)

            # Extract values of interest from the best values
            tth_fit = float(result['x'][0])
            b_fit = float(result['x'][1])
            c_fit = float(result['x'][2])
            amplitudes_fit = norm * result['x'][3::2]
            sigmas_fit = result['x'][4::2]
            if quadratic_energy_calibration:
                a_fit = float(result['x'][-1])
            else:
                a_fit = 0.0
            e_bragg_fit = get_peak_locations(ds_fit, tth_fit)
            peak_energies_fit = [
                (a_fit*i + b_fit) * i + c_fit
                for i in indices_unconstrained[:num_xrf]] \
                + list(e_bragg_fit)

            best_fit_uniform = np.zeros((mca_bins_fit.size))
            for i, e_peak in enumerate(peak_energies_fit):
                best_fit_uniform += gaussian(
                    mca_bins_fit, a_fit, b_fit, c_fit, amplitudes_fit[i],
                    sigmas_fit[i], e_peak)
            residual_uniform = spectrum_fit - best_fit_uniform
            e_bragg_uniform = e_bragg_fit
            strain_uniform = 0.0

        elif calibration_method == 'iterate_tth':

            tth_fit = tth_init
            e_bragg_fit = e_bragg_init
            mca_bin_energies_fit = mca_bin_energies[mca_mask]
            a_init, b_init, c_init = detector.energy_calibration_coeffs
            if isinstance(fwhm_min, (int, float)):
                fwhm_min = fwhm_min*b_init
            else:
                fwhm_min = None
            if isinstance(fwhm_max, (int, float)):
                fwhm_max = fwhm_max*b_init
            else:
                fwhm_max = None
            for iter_i in range(calibration_config.max_iter):
                self.logger.debug(f'Tuning tth: iteration no. {iter_i}, '
                                  f'starting value = {tth_fit} ')

                # Construct the fit model
                models = []
                if detector.background is not None:
                    if isinstance(detector.background, str):
                        models.append(
                            {'model': detector.background, 'prefix': 'bkgd_'})
                    else:
                        for model in detector.background:
                            models.append(
                                {'model': model, 'prefix': f'{model}_'})
                models.append(
                    {'model': 'multipeak', 'centers': list(e_bragg_fit),
                     'fit_type': 'uniform',
                     'centers_range': centers_range*b_init,
                     'fwhm_min': fwhm_min, 'fwhm_max': fwhm_max})

                # Perform the uniform
                fit = FitProcessor()
                uniform_fit = fit.process(
                    NXdata(
                        NXfield(spectrum_fit, 'y'),
                        NXfield(mca_bin_energies_fit, 'x')),
                    {'models': models, 'method': 'trf'})

                # Extract values of interest from the best values for
                # the uniform fit parameters
                best_fit_uniform = uniform_fit.best_fit
                residual_uniform = uniform_fit.residual
                e_bragg_uniform = [
                    uniform_fit.best_values[f'peak{i+1}_center']
                    for i in range(num_bragg)]
                strain_uniform = -np.log(
                    uniform_fit.best_values['scale_factor'])

                # Next, perform the unconstrained fit

                # Use the peak parameters from the uniform fit as
                # the initial guesses for peak locations in the
                # unconstrained fit
                models[-1]['fit_type'] = 'unconstrained'
                unconstrained_fit = fit.process(
                    uniform_fit, {'models': models, 'method': 'trf'})

                # Extract values of interest from the best values for
                # the unconstrained fit parameters
                best_fit_unconstrained = unconstrained_fit.best_fit
                residual_unconstrained = unconstrained_fit.residual
                e_bragg_unconstrained = np.array(
                    [unconstrained_fit.best_values[f'peak{i+1}_center']
                     for i in range(num_bragg)])
                a_unconstrained = np.sqrt(c_1_fit)*abs(get_peak_locations(
                    e_bragg_unconstrained, tth_fit))
                strains_unconstrained = np.log(
                    (a_unconstrained
                     / calibration_config.material.lattice_parameters))
                strain_unconstrained = np.mean(strains_unconstrained)
                tth_unconstrained = tth_fit * (1.0 + strain_unconstrained)

                # Update tth for the next iteration of tuning
                prev_tth = tth_fit
                tth_fit = float(tth_unconstrained)

                # Update the peak energy locations for this iteration
                e_bragg_fit = get_peak_locations(ds_fit, tth_fit)

                # Stop tuning tth at this iteration if differences are
                # small enough
                if abs(tth_fit - prev_tth) < calibration_config.tune_tth_tol:
                    break

            # Fit line to expected / computed peak locations from the
            # last unconstrained fit.
            if quadratic_energy_calibration:
                fit = FitProcessor()
                result = fit.process(
                    NXdata(
                        NXfield(e_bragg_fit, 'y'),
                        NXfield(e_bragg_unconstrained, 'x')),
                    {'models': [{'model': 'quadratic'}]})
                a = result.best_values['a']
                b = result.best_values['b']
                c = result.best_values['c']
            else:
                fit = FitProcessor()
                result = fit.process(
                    NXdata(
                        NXfield(e_bragg_fit, 'y'),
                        NXfield(e_bragg_unconstrained, 'x')),
                    {'models': [{'model': 'linear'}]})
                a = 0.0
                b = result.best_values['slope']
                c = result.best_values['intercept']
            # The following assumes that a_init = 0
            if a_init:
                raise NotImplementedError(
                    'A linear energy calibration is required at this time')
            a_fit = float(a*b_init**2)
            b_fit = float(2*a*b_init*c_init + b*b_init)
            c_fit = float(a*c_init**2 + b*c_init + c)
            peak_energies_fit = ((a*e_bragg_unconstrained + b)
                                * e_bragg_unconstrained + c)

        # Store the results in the detector object
        detector.tth_calibrated = float(tth_fit)
        detector.energy_calibration_coeffs = [
            float(a_fit), float(b_fit), float(c_fit)]

        # Update the MCA channel energies with the newly calibrated
        # coefficients
        mca_bin_energies = detector.energies

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            # Update the peak energies and the MCA channel energies
            e_bragg_fit = get_peak_locations(ds_fit, tth_fit)
            mca_energies_fit = mca_bin_energies[mca_mask]

            # Get an unconstrained fit
            if calibration_method != 'iterate_tth':
                if isinstance(fwhm_min, (int, float)):
                    fwhm_min = fwhm_min*b_fit
                else:
                    fwhm_min = None
                if isinstance(fwhm_max, (int, float)):
                    fwhm_max = fwhm_max*b_fit
                else:
                    fwhm_max = None
                models = []
                if detector.background is not None:
                    if isinstance(detector.background, str):
                        models.append(
                            {'model': detector.background, 'prefix': 'bkgd_'})
                    else:
                        for model in detector.background:
                            models.append(
                                {'model': model, 'prefix': f'{model}_'})
                models.append(
                    {'model': 'multipeak', 'centers': list(peak_energies_fit),
                     'centers_range': centers_range*b_fit,
                     'fwhm_min': fwhm_min, 'fwhm_max': fwhm_max})
                fit = FitProcessor()
                result = fit.process(
                    NXdata(NXfield(spectrum_fit, 'y'),
                           NXfield(mca_energies_fit, 'x')),
                    {'models': models, 'method': 'trf'})
                best_fit_unconstrained = result.best_fit
                residual_unconstrained = result.residual
                e_bragg_unconstrained = np.sort(
                    [result.best_values[f'peak{i+1}_center']
                     for i in range(num_xrf, num_xrf+num_bragg)])
                a_unconstrained = np.sqrt(c_1_fit) * abs(
                    get_peak_locations(e_bragg_unconstrained, tth_fit))
                strains_unconstrained = np.log(
                    (a_unconstrained
                     / calibration_config.material.lattice_parameters))
                strain_unconstrained = np.mean(strains_unconstrained)

            # Create the figure
            fig, axs = plt.subplots(2, 2, sharex='all', figsize=(11, 8.5))
            fig.suptitle(
                f'Detector {detector.id} '
                r'2$\theta$ Calibration')

            # Upper left axes: best fit with calibrated peak centers
            axs[0,0].set_title(r'2$\theta$ Calibration Fits')
            axs[0,0].set_xlabel('Energy (keV)')
            axs[0,0].set_ylabel('Intensity (counts)')
            for i, e_peak in enumerate(e_bragg_fit):
                axs[0,0].axvline(e_peak, c='k', ls='--')
                axs[0,0].text(e_peak, 1, str(hkls_fit[i])[1:-1],
                              ha='right', va='top', rotation=90,
                              transform=axs[0,0].get_xaxis_transform())
            if flux_correct is None:
                axs[0,0].plot(
                    mca_energies_fit, spectrum_fit, marker='.', c='C2', ms=3,
                    ls='', label='MCA data')
            else:
                axs[0,0].plot(
                    mca_energies_fit, spectrum_fit, marker='.', c='C2', ms=3,
                    ls='', label='Flux-corrected MCA data')
            if calibration_method == 'iterate_tth':
                label_unconstrained = 'Unconstrained'
            else:
                if quadratic_energy_calibration:
                    label_unconstrained = \
                        'Unconstrained fit using calibrated a, b, and c'
                else:
                    label_unconstrained = \
                        'Unconstrained fit using calibrated b and c'
            if best_fit_uniform is None:
                axs[0,0].plot(
                    mca_energies_fit, best_fit_unconstrained, c='C1',
                    label=label_unconstrained)
            else:
                axs[0,0].plot(
                    mca_energies_fit, best_fit_uniform, c='C0',
                    label='Single strain')
                axs[0,0].plot(
                    mca_energies_fit, best_fit_unconstrained, c='C1', ls='--',
                    label=label_unconstrained)
            axs[0,0].legend()

            # Lower left axes: fit residual
            axs[1,0].set_title('Fit Residuals')
            axs[1,0].set_xlabel('Energy (keV)')
            axs[1,0].set_ylabel('Residual (counts)')
            if residual_uniform is None:
                axs[1,0].plot(
                    mca_energies_fit, residual_unconstrained, c='C1',
                    label=label_unconstrained)
            else:
                axs[1,0].plot(
                    mca_energies_fit, residual_uniform, c='C0',
                    label='Single strain')
                axs[1,0].plot(
                    mca_energies_fit, residual_unconstrained, c='C1', ls='--',
                    label=label_unconstrained)
            axs[1,0].legend()

            # Upper right axes: E vs strain for each fit
            axs[0,1].set_title('HKL Energy vs. Microstrain')
            axs[0,1].set_xlabel('Energy (keV)')
            axs[0,1].set_ylabel('Strain (\u03BC\u03B5)')
            if strain_uniform is not None:
                axs[0,1].axhline(strain_uniform * 1e6,
                                 ls='--', label='Single strain')
            axs[0,1].plot(e_bragg_fit, strains_unconstrained * 1e6,
                          marker='o', mfc='none', c='C1',
                          label='Unconstrained')
            axs[0,1].axhline(strain_unconstrained* 1e6,
                             ls='--', c='C1',
                             label='Unconstrained: unweighted mean')
            axs[0,1].legend()

            # Lower right axes: theoretical E vs fitted E for all peaks
            axs[1,1].set_title('Theoretical vs. Fitted Peak Energies')
            axs[1,1].set_xlabel('Energy (keV)')
            axs[1,1].set_ylabel('Energy (keV)')
            if calibration_method == 'iterate_tth':
                e_fit = e_bragg_fit
                e_unconstrained = e_bragg_unconstrained
                if quadratic_energy_calibration:
                    label = 'Unconstrained: quadratic fit'
                else:
                    label = 'Unconstrained: linear fit'
            else:
                e_fit = np.concatenate((e_xrf, e_bragg_fit))
                e_unconstrained = np.concatenate(
                    (e_xrf, e_bragg_unconstrained))
                if quadratic_energy_calibration:
                    label = 'Quadratic fit'
                else:
                    label = 'Linear fit'
            axs[1,1].plot(
                e_bragg_fit, e_bragg_uniform, marker='x', ls='',
                label='Single strain')
            axs[1,1].plot(
                e_fit, e_unconstrained, marker='o', mfc='none', ls='',
                label='Unconstrained')
            axs[1,1].plot(
                e_fit, peak_energies_fit, c='C1', label=label)
            axs[1,1].legend()
            txt = 'Calibrated values:' \
                  f'\nTakeoff angle:\n    {tth_fit:.5f}'r'$^\circ$'
            if quadratic_energy_calibration:
                txt += '\nQuadratic coefficient (a):' \
                       f'\n    {a_fit:.5e} $keV$/channel$^2$'
            txt += '\nLinear coefficient (b):' \
                   f'\n    {b_fit:.5f} $keV$/channel' \
                   f'\nConstant offset (c):\n    {c_fit:.5f}'
            axs[1,1].text(
                0.98, 0.02, txt,
                ha='right', va='bottom', ma='left',
                transform=axs[1,1].transAxes,
                bbox={'boxstyle': 'round', 'ec': (1., 0.5, 0.5),
                      'fc': (1., 0.8, 0.8, 0.8)},
            )

            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(
                    outputdir, f'{detector.id}_tth_calibration_fits.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()


class MCADataProcessor(Processor):
    """A Processor to return data from an MCA, restuctured to
    incorporate the shape & metadata associated with a map
    configuration to which the MCA data belongs, and linearly
    transformed according to the results of a energy/tth calibration.
    """
    def process(self, data, inputdir='.'):
        """Process configurations for a map and MCA detector(s), and
        return the calibrated MCA data collected over the map.

        :param data: Input map configuration and results of
            energy/tth calibration.
        :type data: list[dict[str,object]]
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :return: Calibrated and flux-corrected MCA data.
        :rtype: nexusformat.nexus.NXroot
        """
        print(f'data:\n{data}')
        exit('Done Here')
        map_config = self.get_config(
            data, 'common.models.map.MapConfig', inputdir=inputdir)
        calibration_config = self.get_config(
            data, 'edd.models.MCATthCalibrationConfig', inputdir=inputdir)
        nxroot = self.get_nxroot(map_config, calibration_config)

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
            CHAP.edd.models.MCATthCalibrationConfig
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
            nxentry.instrument[detector.id] = NXdetector()
            nxentry.instrument[detector.id].calibration = dumps(
                detector.dict())

            nxentry.instrument[detector.id].data = NXdata()
            nxdata = nxentry.instrument[detector.id].data
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
                    calibration_config.id, scan_step_index)

            nxentry.data.makelink(nxdata.raw, name=detector.id)
            nxentry.data.makelink(
                nxdata.channel_energy, name=f'{detector.id}_channel_energy')
            if isinstance(nxentry.data.attrs['axes'], str):
                nxentry.data.attrs['axes'] = [
                    nxentry.data.attrs['axes'],
                    f'{detector.id}_channel_energy']
            else:
                nxentry.data.attrs['axes'] += [f'{detector.id}_channel_energy']
            nxentry.data.attrs['signal'] = detector.id

        return nxroot


class MCACalibratedDataPlotter(Processor):
    """Convenience Processor for quickly visualizing calibrated MCA
       data from a single scan. Returns None!"""
    def process(
            self, data, spec_file, scan_number, scan_step_index=None,
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
        :param scan_step_index: Scan step index of interest.
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
        from chess_scanparsers import SMBMCAScanParser as ScanParser
        import matplotlib.pyplot as plt

        if material is not None:
            self.logger.warning('Plotting HKL lines is not supported yet.')

        if scan_step_index is not None:
            if not isinstance(scan_step_index, int):
                try:
                    scan_step_index = int(scan_step_index)
                except Exception as exc:
                    msg = 'scan_step_index must be an int'
                    self.logger.error(msg)
                    raise TypeError(msg) from exc

        calibration_config = self.get_config(
            data, 'edd.models.MCATthCalibrationConfig')
        scanparser = ScanParser(spec_file, scan_number)

        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
        title = f'{scanparser.scan_title} MCA Data'
        if scan_step_index is None:
            title += ' (sum of all spectra in the scan)'
        ax.set_title(title)
        ax.set_xlabel('Calibrated energy (keV)')
        ax.set_ylabel('Intenstiy (counts)')
        for detector in calibration_config.detectors:
            if scan_step_index is None:
                spectrum = np.sum(
                    scanparser.get_all_detector_data(detector._id), axis=0)
            else:
                spectrum = scanparser.get_detector_data(
                    detector._id, scan_step_index=scan_step_index)
            ax.plot(detector.energies, spectrum,
                    label=f'Detector {detector._id}')
        ax.legend()
        if interactive:
            plt.show()
        if save_figures:
            fig.savefig(os.path.join(
                outputdir, f'spectrum_{scanparser.scan_title}'))
        plt.close()


class StrainAnalysisProcessor(Processor):
    """Processor that takes a map of MCA data and returns a map of
    sample strains.
    """
    def __init__(self):
        super().__init__()
        self._find_peaks = False
        self._skip_animation = False
        self._save_figures = False
        self._inputdir = '.'
        self._outputdir = '.'
        self._interactive = False

        self._detectors = []
        self._energy_masks = []
        self._mean_data = []
        self._nxdata_detectors = []

    @staticmethod
    def add_points(nxroot, points):
        # Third party modules
        from nexusformat.nexus import (
            NXdetector,
            NXprocess,
        )

        nxprocess = None
        for nxobject in nxroot.values():
            if isinstance(nxobject, NXprocess):
                nxprocess = nxobject
                break
        if nxprocess is None:
            raise RuntimeError('Unable to find the strainanalysis object')

        nxdetector_ref = None
        for nxobject in nxprocess.values():
            if isinstance(nxobject, NXdetector):
                nxdetector_ref = nxobject
                break
        if nxdetector_ref is None:
            raise RuntimeError(
                'Unable to find a detector in strainanalysis object')
        axes = get_axes(nxdetector_ref.data, skip_axes=['energy'])
        coords = np.asarray([nxdetector_ref.data[a].nxdata for a in axes]).T

        for point in points:
            # Use pop?
            point_coords = [point[a] for a in axes]
            # FIX Can we someway use np.argwhere?
            for index, current_coords in enumerate(coords):
                if np.array_equal(point_coords, current_coords):
                    break
            else:
                raise RuntimeError(f'Unable to match point {point}')
            for k, v in point.items():
                if k not in axes:
                    nxprocess[k][index] = v

    def process(
            self, data, config=None, find_peaks=True, skip_animation=False,
            setup=True, update=True, save_figures=False, inputdir='.',
            outputdir='.', interactive=False):
        """Return the strain analysis results as a list of updated
        points or a `nexusformat.nexus.NXroot` object.

        :param data: Input data containing configurations for a map,
            completed energy/tth calibration, and parameters for strain
            analysis.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.StrainAnalysisConfig.
        :type config: dict, optional
        :param find_peaks: Exclude peaks where the average spectrum
            is below the `rel_height_cutoff` (in the detector
            configuration) cutoff relative to the maximum value of the
            average spectrum, defaults to `True`.
        :type find_peaks: bool, optional
        :param skip_animation: Skip the animation and plotting of
            the strain analysis fits, defaults to `False`.
        :type skip_animation: bool, optional
        :param setup: Setup the strain analysis `NXroot` object,
            defaults to `True`.
        :type setup: bool, optional
        :param update: Perform the strain analysis and return the
            results as a list of updated points or update the results
            in the `NXroot` object, defaults to `True`.
        :type update: bool, optional
        :param save_figures: Save .pngs of plots for checking inputs &
            outputs of this Processor, defaults to `False`.
        :type save_figures: bool, optional
        :param inputdir: Input directory, used only if files in the
            input configuration are not absolute paths,
            defaults to `'.'`.
        :type inputdir: str, optional
        :param outputdir: Directory to which any output figures will
            be saved, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions, defaults to
            `False`.
        :type interactive: bool, optional
        :raises RuntimeError: Unable to get a valid strain analysis
            configuration.
        :return: The strain analysis results.
        :rtype: Union[list[dict[str, object]],
                      nexusformat.nexus.NXroot]
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        # Local modules
        from CHAP.edd.models import (
            MCAElementStrainAnalysisConfig,
            StrainAnalysisConfig,
        )

        if setup:
            if interactive:
                self.logger.warning('Ineractive option disabled during setup')
                interactive = False
            if save_figures:
                self.logger.warning(
                    'Saving figures option disabled during setup')
                save_figures = False
        elif not update:
            raise RuntimeError('Illegal combination of setup and update')
        self._find_peaks = find_peaks
        self._skip_animation = skip_animation
        self._save_figures = save_figures
        self._outputdir = outputdir
        self._interactive = interactive

        # Load the detector data
        try:
            nxobject = self.get_data(data)
            if isinstance(nxobject, NXroot):
                nxroot = nxobject
            elif isinstance(nxobject, NXentry):
                nxroot = NXroot()
                nxroot[nxobject.nxname] = nxobject
                nxobject.set_default()
            else:
                raise RuntimeError
        except Exception as exc:
            raise RuntimeError(
                'No valid detector data in input pipeline data') from exc

        # Load the validated calibration configuration
        calibration_config = self.get_config(
            data, 'edd.models.MCATthCalibrationConfig', inputdir=inputdir)

        # Load the validated strain analysis configuration
        try:
            strain_analysis_config = self.get_config(
                data, 'edd.models.StrainAnalysisConfig', inputdir=inputdir)
        except:
            self.logger.info(
                'No valid strain analysis config in input '
                'pipeline data, using config parameter instead')
            try:
                strain_analysis_config = StrainAnalysisConfig(
                    **config, inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        # Validate the detector configuration and load, validate and
        # add the calibration info to the detectors
        if 'default' in nxroot.attrs:
            nxentry = nxroot[nxroot.default]
        else:
            nxentry = [v for v in nxroot.values() if isinstance(v, NXentry)][0]
        nxdata = nxentry[nxentry.default]
        calibration_detector_ids = [d.id for d in calibration_config.detectors]
        if strain_analysis_config.detectors is None:
            strain_analysis_config.detectors = [
                MCAElementStrainAnalysisConfig(**dict(d))
                for d in calibration_config.detectors if d.id in nxdata]
        for detector in deepcopy(strain_analysis_config.detectors):
            if detector.id not in nxdata:
                self.logger.warning(
                    f'Skipping detector {detector.id} (no raw data)')
                strain_analysis_config.detectors.remove(detector)
            elif detector.id in calibration_detector_ids:
                det_data = nxdata[detector.id].nxdata
                if det_data.ndim != 2:
                    self.logger.warning(
                        f'Skipping detector {detector.id} (Illegal data shape '
                        f'{det_data.shape})')
                elif setup or det_data.sum():
                    for k, v in nxdata[detector.id].attrs.items():
                        detector.attrs[k] = v
                    self._detectors.append(detector)
                    calibration = [
                        d for d in calibration_config.detectors
                        if d.id == detector.id][0]
                    detector.add_calibration(calibration)
                else:
                    self.logger.warning(
                        f'Skipping detector {detector.id} (zero intensity)')
            else:
                self.logger.warning(f'Skipping detector {detector.id} '
                                    '(no energy/tth calibration data)')
        if not self._detectors:
            raise ValueError('No valid data or unable to match an available '
                             'calibrated detector for the strain analysis')

        if setup and update:
            nxroot = self._get_nxroot(nxentry, strain_analysis_config, update)
            points = self._strain_analysis(
                nxentry.data, strain_analysis_config)
            self.add_points(nxroot, points)
            return nxroot
        elif setup:
            return self._get_nxroot(nxentry, strain_analysis_config, update)
        elif update:
            return self._strain_analysis(nxentry.data, strain_analysis_config)
        return None

    def _add_fit_nxcollection(self, nxdetector, fit_type, hkls):
        """Add the fit collection as a `nexusformat.nexus.NXcollection`
        object."""
        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXfield,
            NXparameters,
        )

        nxdetector[f'{fit_type}_fit'] = NXcollection()
        nxcollection = nxdetector[f'{fit_type}_fit']
        det_nxdata = nxdetector.data

        # Get data shape
        shape = det_nxdata.intensity.shape

        # Full map of results
        nxcollection.results = NXdata()
        nxdata = nxcollection.results
        self._linkdims(nxdata, det_nxdata)
        nxdata.best_fit = NXfield(shape=shape, dtype=np.float64)
        nxdata.residual = NXfield(shape=shape, dtype=np.float64)
        nxdata.redchi = NXfield(shape=[shape[0]], dtype=np.float64)
        nxdata.success = NXfield(shape=[shape[0]], dtype='bool')

        # Peak-by-peak results
        for hkl in hkls:
            hkl_name = '_'.join(str(hkl)[1:-1].split(' '))
            nxcollection[hkl_name] = NXparameters()
            # Create initial centers field
            if fit_type == 'uniform':
                nxcollection[hkl_name].center_initial_guess = 0.0
            else:
                nxcollection[hkl_name].center_initial_guess = NXdata()
                self._linkdims(
                    nxcollection[hkl_name].center_initial_guess, det_nxdata,
                    skip_field_dims=['energy'])
            # Report HKL peak centers
            nxcollection[hkl_name].centers = NXdata()
            self._linkdims(
                nxcollection[hkl_name].centers, det_nxdata,
                skip_field_dims=['energy'])
            nxcollection[hkl_name].centers.values = NXfield(
                shape=[shape[0]], dtype=np.float64, attrs={'units': 'keV'})
            nxcollection[hkl_name].centers.errors = NXfield(
                shape=[shape[0]], dtype=np.float64)
            nxcollection[hkl_name].centers.attrs['signal'] = 'values'
            # Report HKL peak amplitudes
            nxcollection[hkl_name].amplitudes = NXdata()
            self._linkdims(
                nxcollection[hkl_name].amplitudes, det_nxdata,
                skip_field_dims=['energy'])
            nxcollection[hkl_name].amplitudes.values = NXfield(
                shape=[shape[0]], dtype=np.float64, attrs={'units': 'counts'})
            nxcollection[hkl_name].amplitudes.errors = NXfield(
                shape=[shape[0]], dtype=np.float64)
            nxcollection[hkl_name].amplitudes.attrs['signal'] = 'values'
            # Report HKL peak FWHM
            nxcollection[hkl_name].sigmas = NXdata()
            self._linkdims(
                nxcollection[hkl_name].sigmas, det_nxdata,
                skip_field_dims=['energy'])
            nxcollection[hkl_name].sigmas.values = NXfield(
                shape=[shape[0]], dtype=np.float64, attrs={'units': 'keV'})
            nxcollection[hkl_name].sigmas.errors = NXfield(
                shape=[shape[0]], dtype=np.float64)
            nxcollection[hkl_name].sigmas.attrs['signal'] = 'values'

    def _adjust_material_props(self, materials):
        """Adjust the material properties."""
        # Local modules
        from CHAP.edd.utils import select_material_params

        # ASK: extend to multiple detectors?
        detector = self._detectors[0]
        if self._save_figures:
            filename = os.path.join(
                self._outputdir,
                f'{detector.id}_strainanalysis_material_config.png')
        else:
            filename = None
        return select_material_params(
            detector.energies, self._mean_data[0]*self._energy_masks[0],
            detector.tth_calibrated, label='Sum of all spectra in the map',
            preselected_materials=materials, interactive=self._interactive,
            filename=filename)

    def _create_animation(
            self, det_nxdata, energies, best_fits, detector_id):
        """Create an animation of the fit results."""
        # Third party modules
        from matplotlib import animation
        import matplotlib.pyplot as plt

        if (self._skip_animation
                or not (self._interactive or self._save_figures)):
            return

        def animate(i):
            data = det_nxdata.intensity.nxdata[i]
            max_ = data.max()
            norm = max(1.0, max_)
            intensity.set_ydata(data / norm)
            best_fit.set_ydata(best_fits[i] / norm)
            index.set_text('\n'.join(
                [f'norm = {int(max_)}'] +
                ['relative norm = {(max_ / norm_all_data):.5f}'] +
                [f'{dim}[{i}] = {det_nxdata[a][i]}' for a in axes]))
            if self._save_figures:
                plt.savefig(os.path.join(
                    path, f'frame_{str(i).zfill(num_digit)}.png'))
            return intensity, best_fit, index

        if self._save_figures:
            path = os.path.join(
                self._outputdir,
                f'{detector_id}_strainanalysis_unconstrained_fits')
            if not os.path.isdir(path):
                os.mkdir(path)

        axes = get_axes(det_nxdata)
        if 'energy' in axes:
            axes.remove('energy')
        norm_all_data = max(1.0, det_nxdata.intensity.max())

        fig, ax = plt.subplots()
        data = det_nxdata.intensity.nxdata[0]
        norm = max(1.0, data.max())
        intensity, = ax.plot(energies, data / norm, 'b.', label='data')
        best_fit, = ax.plot(energies, best_fits[0] / norm, 'k-', label='fit')
        ax.set(
            title='Unconstrained fits',
            xlabel='Detector energy (keV)',
            ylabel='Normalized intensity (-)')
        ax.legend(loc='upper right')
        ax.set_ylim(-0.05, 1.05)
        index = ax.text(
            0.05, 0.95, '', transform=ax.transAxes, va='top')

        num_frame = int(det_nxdata.intensity.nxdata.size
                      / det_nxdata.intensity.nxdata.shape[-1])
        num_digit = len(str(num_frame))
        if not self._save_figures:
            ani = animation.FuncAnimation(
                fig, animate,
                frames=int(det_nxdata.intensity.nxdata.size
                           / det_nxdata.intensity.nxdata.shape[-1]),
                interval=1000, blit=False, repeat=False)
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
                 plt.gcf(), frames, interval=1000, blit=False,
                 repeat=False)

        if self._interactive:
            plt.show()

        if self._save_figures:
            path = os.path.join(
                self._outputdir,
                f'{detector_id}_strainanalysis_unconstrained_fits.gif')
            ani.save(path)
        plt.close()

    def _get_energy_and_masks(self):
        """Get the energy mask by blanking out data below 25 keV as
        well as that in the last bin.
        """
        for detector in self._detectors:
            energy_mask = np.where(detector.energies >= 25.0, 1, 0)
            energy_mask[-1] = 0
            self._energy_masks.append(energy_mask)

    def _get_mask_hkls(self, materials):
        """Get the mask and HKLs used in the strain analysis."""
        # Local modules
        from CHAP.edd.utils import (
            get_unique_hkls_ds,
            select_mask_and_hkls,
        )

        for index, (nxdata, detector) in enumerate(
                zip(self._nxdata_detectors, self._detectors)):

            # Get the unique HKLs and lattice spacings for the strain
            # analysis materials
            hkls, ds = get_unique_hkls_ds(
                materials, tth_tol=detector.hkl_tth_tol,
                tth_max=detector.tth_max)

            # Interactively adjust the mask and HKLs used in the
            # strain analysis
            if self._save_figures:
                filename = os.path.join(
                    self._outputdir,
                    f'{detector.id}_strainanalysis_fit_mask_hkls.png')
            else:
                filename = None
            include_bin_ranges, hkl_indices = \
                select_mask_and_hkls(
                    detector.energies, self._mean_data, hkls, ds,
                    detector.tth_calibrated,
                    preselected_bin_ranges=detector.include_bin_ranges,
                    preselected_hkl_indices=detector.hkl_indices,
                    detector_id=detector.id,
                    ref_map=nxdata.nxsignal.nxdata*self._energy_masks[index],
                    calibration_bin_ranges=detector.calibration_bin_ranges,
                    label='Sum of all spectra in the map',
                    interactive=self._interactive, filename=filename)
            detector.include_energy_ranges = \
                detector.get_include_energy_ranges(include_bin_ranges)
            detector.hkl_indices = hkl_indices
            self.logger.debug(
                f'include_energy_ranges for detector {detector.id}:'
                f' {detector.include_energy_ranges}')
            self.logger.debug(
                f'hkl_indices for detector {detector.id}:'
                f' {detector.hkl_indices}')
            if not detector.include_energy_ranges:
                raise ValueError(
                    'No value provided for include_energy_ranges. '
                    'Provide them in the MCA Tth Calibration Configuration, '
                    'or re-run the pipeline with the --interactive flag.')
            if not detector.hkl_indices:
                raise ValueError(
                    'No value provided for hkl_indices. Provide them in '
                    'the detector\'s MCA Tth Calibration Configuration, or'
                    ' re-run the pipeline with the --interactive flag.')

    def _get_nxroot(self, nxentry, strain_analysis_config, update):
        """Return a `nexusformat.nexus.NXroot` object initialized for
        the stress analysis.

        :param nxentry: Strain analysis map, including the raw
            detector data.
        :type nxentry: nexusformat.nexus.NXentry
        :param strain_analysis_config: Strain analysis processing
            configuration.
        :type strain_analysis_config:
            CHAP.edd.models.StrainAnalysisConfig
        :param update: Perform the strain analysis and return the
            results as a list of updated points or update the results
            in the `NXroot` object, defaults to `True`.
        :type update: bool, optional
        :return: Strain analysis results & associated metadata..
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXfield,
            NXprocess,
            NXroot,
        )

        from CHAP.edd.utils import get_unique_hkls_ds
        from CHAP.utils.general import nxcopy

        if not self._interactive and not strain_analysis_config.materials:
            raise ValueError(
                'No material provided. Provide a material in the '
                'StrainAnalysis Configuration, or re-run the pipeline with '
                'the --interactive flag.')

        # Create the NXroot object
        nxroot = NXroot()
        nxroot[nxentry.nxname] = nxentry
        nxroot[f'{nxentry.nxname}_strainanalysis'] = NXprocess()
        nxprocess = nxroot[f'{nxentry.nxname}_strainanalysis']
        nxprocess.strain_analysis_config = dumps(
            strain_analysis_config.dict())
        nxdata_raw = nxentry[nxentry.default]

        have_det_nxdata = False
        oversampling_axis = {}
        if strain_analysis_config.sum_axes:
            scan_type = int(str(nxdata_raw.attrs.get('scan_type', 0)))

            if scan_type == 4:
                # Local modules
                from CHAP.utils.general import rolling_average

                # Check for oversampling axis and create the binned
                # coordinates
                raise RuntimeError('oversampling needs testing')
                fly_axis = nxdata_raw.attrs.get('fly_axis_labels').nxdata[0]
                oversampling = strain_analysis_config.oversampling
                oversampling_axis[fly_axis] = rolling_average(
                        nxdata_raw[fly_axis].nxdata,
                        start=oversampling.get('start', 0),
                        end=oversampling.get('end'),
                        width=oversampling.get('width'),
                        stride=oversampling.get('stride'),
                        num=oversampling.get('num'),
                        mode=oversampling.get('mode', 'valid'))
            elif (scan_type > 2
                    or isinstance(strain_analysis_config.sum_axes, list)):
                # Collect the raw MCA data averaged over sum_axes
                for index, detector in enumerate(self._detectors):
                    self._nxdata_detectors.append(
                        self._get_sum_axes_data(
                            nxdata_raw, detector.id,
                            strain_analysis_config.sum_axes))
                have_det_nxdata = True
        if not have_det_nxdata:
            # Collect the raw MCA data if not averaged over sum_axes
            axes = get_axes(nxdata_raw)
            for detector in self._detectors:
                nxdata_det = NXdata(
                    NXfield(nxdata_raw[detector.id].nxdata, 'detector_data'),
                    tuple([
                        NXfield(
                            nxdata_raw[a].nxdata, a, attrs=nxdata_raw[a].attrs)
                        for a in axes]))
                if len(axes) > 1:
                    nxdata_det.attrs['unstructured_axes'] = \
                        nxdata_det.attrs.pop('axes')
                self._nxdata_detectors.append(nxdata_det)
        if update:
            self._mean_data = [
                np.mean(
                    nxdata.nxsignal.nxdata[
                        [i for i in range(0, nxdata.nxsignal.shape[0])
                         if nxdata[i].nxsignal.nxdata.sum()]],
                    axis=tuple(i for i in range(0, nxdata.nxsignal.ndim-1)))
                for nxdata in self._nxdata_detectors]
        else:
            self._mean_data = len(self._nxdata_detectors)*[
                np.zeros((self._nxdata_detectors[0].nxsignal.shape[-1]))]
        self.logger.debug(
            f'mean_data shape: {np.asarray(self._mean_data).shape}')

        # Get the energy masks
        self._get_energy_and_masks()

        # Get the mask and HKLs used in the strain analysis
        self._get_mask_hkls(strain_analysis_config.materials)

        # Loop over the detectors to fill in the nxprocess
        for index, (nxdata, detector) in enumerate(
                zip(self._nxdata_detectors, self._detectors)):

            # Get the current data object
            data = nxdata.nxsignal

            # Get the MCA bin energies
            bin_energies = detector.energies

            # Setup the NXdetector object for the current detector
            self.logger.debug(
                f'Setting up NXdata group for {detector.id}')
            nxdetector = NXdetector()
            nxprocess[detector.id] = nxdetector
            nxdetector.local_name = detector.id
            nxdetector.detector_config = dumps(detector.dict())
            nxdetector.data = nxcopy(nxdata, exclude_nxpaths='detector_data')
            det_nxdata = nxdetector.data
            if 'axes' in det_nxdata.attrs:
                if isinstance(det_nxdata.attrs['axes'], str):
                    det_nxdata.attrs['axes'] = [
                        det_nxdata.attrs['axes'], 'energy']
                else:
                    det_nxdata.attrs['axes'].append('energy')
            else:
                det_nxdata.attrs['axes'] = ['energy']
            mask = detector.mca_mask()
            energies = bin_energies[mask]
            det_nxdata.energy = NXfield(value=energies, attrs={'units': 'keV'})
            det_nxdata.tth = NXfield(
                dtype=np.float64,
                shape=(nxdata.shape[0]),
                attrs={'units':'degrees', 'long_name': '2\u03B8 (degrees)'})
            det_nxdata.uniform_microstrain = NXfield(
                dtype=np.float64,
                shape=(nxdata.shape[0]),
                attrs={'long_name': 'Strain from uniform fit (\u03BC\u03B5)'})
            det_nxdata.unconstrained_microstrain = NXfield(
                dtype=np.float64,
                shape=(nxdata.shape[0]),
                attrs={'long_name':
                           'Strain from unconstrained fit (\u03BC\u03B5)'})

            # Add the detector data
            det_nxdata.intensity = NXfield(
                value=np.asarray([data[i].astype(np.float64)[mask]
                                  for i in range(data.shape[0])]),
                attrs={'units': 'counts'})
            det_nxdata.summed_intensity = det_nxdata.intensity.sum(axis=0)
            det_nxdata.attrs['signal'] = 'intensity'

            # Get the unique HKLs and lattice spacings for the strain
            # analysis materials
            hkls, _ = get_unique_hkls_ds(
                strain_analysis_config.materials, tth_tol=detector.hkl_tth_tol,
                tth_max=detector.tth_max)

            # Get the HKLs and lattice spacings that will be used for
            # fitting
            hkls_fit = np.asarray([hkls[i] for i in detector.hkl_indices])

            # Add the uniform fit nxcollection
            self._add_fit_nxcollection(nxdetector, 'uniform', hkls_fit)

            # Add the unconstrained fit nxcollection
            self._add_fit_nxcollection(nxdetector, 'unconstrained', hkls_fit)

            # Add the microstrain fields
            tth_map = detector.get_tth_map((nxdata.shape[0],))
            det_nxdata.tth.nxdata = tth_map

        return nxroot

    def _get_sum_axes_data(self, nxdata, index, sum_axes=True):
        """Get the raw MCA data collected by the scan averaged over the
        sum_axes.
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        data = nxdata[index].nxdata
        if not isinstance(sum_axes, list):
            if sum_axes and 'fly_axis_labels' in nxdata.attrs:
                sum_axes = nxdata.attrs['fly_axis_labels']
            else:
                sum_axes = []
        axes = [a for a in get_axes(nxdata) if a not in sum_axes]
        dims = np.asarray([nxdata[a].nxdata for a in axes], dtype=np.float64).T
        sum_indices = []
        unique_points = []
        for i in range(data.shape[0]):
            point = dims[i]
            found = False
            for index, unique_point in enumerate(unique_points):
                if all(point == unique_point):
                    sum_indices[index].append(i)
                    found = True
                    break
            if not found:
                unique_points.append(point)
                sum_indices.append([i])
        unique_points = np.asarray(unique_points).T
        mean_data = np.empty((unique_points.shape[1], data.shape[-1]))
        for i in range(unique_points.shape[1]):
            mean_data[i] = np.mean(data[sum_indices[i]], axis=0)
        nxdata_det = NXdata(
            NXfield(mean_data, 'detector_data'),
            tuple([
                NXfield(unique_points[i], a, attrs=nxdata[a].attrs)
                for i, a in enumerate(axes)]))
        if len(axes) > 1:
            nxdata_det.attrs['unstructured_axes'] = \
                nxdata_det.attrs.pop('axes')
        return nxdata_det

    def _linkdims(
            self, nxgroup, nxdata_source, add_field_dims=None,
            skip_field_dims=None, oversampling_axis=None):
        """Link the dimensions for a 'nexusformat.nexus.NXgroup`
        object."""
        # Third party modules
        from nexusformat.nexus import NXfield
        from nexusformat.nexus.tree import NXlinkfield

        if skip_field_dims is None:
            skip_field_dims = []
        if oversampling_axis is None:
            oversampling_axis = {}
        if 'axes' in nxdata_source.attrs:
            axes = nxdata_source.attrs['axes']
            if isinstance(axes, str):
                axes = [axes]
        else:
            axes = []
        axes = [a for a in axes if a not in skip_field_dims]
        if 'unstructured_axes' in nxdata_source.attrs:
            unstructured_axes = nxdata_source.attrs['unstructured_axes']
            if isinstance(unstructured_axes, str):
                unstructured_axes = [unstructured_axes]
        else:
            unstructured_axes = []
        link_axes = axes + unstructured_axes
        for dim in link_axes:
            if dim in oversampling_axis:
                bin_name = dim.replace('fly_', 'bin_')
                axes[axes.index(dim)] = bin_name
                exit('FIX need to replace in both axis and unstructured_axes if present')
                nxgroup[bin_name] = NXfield(
                    value=oversampling_axis[dim],
                    units=nxdata_source[dim].units,
                    attrs={
                        'long_name':
                            f'oversampled {nxdata_source[dim].long_name}',
                        'data_type': nxdata_source[dim].data_type,
                        'local_name': 'oversampled '
                                      f'{nxdata_source[dim].local_name}'})
            else:
                if isinstance(nxdata_source[dim], NXlinkfield):
                    nxgroup[dim] = nxdata_source[dim]
                else:
                    nxgroup.makelink(nxdata_source[dim])
                if f'{dim}_indices' in nxdata_source.attrs:
                    nxgroup.attrs[f'{dim}_indices'] = \
                        nxdata_source.attrs[f'{dim}_indices']
        if add_field_dims is None:
            if axes:
                nxgroup.attrs['axes'] = axes
            if unstructured_axes:
                nxgroup.attrs['unstructured_axes'] = unstructured_axes
        else:
            nxgroup.attrs['axes'] = axes + add_field_dims
        if unstructured_axes:
            nxgroup.attrs['unstructured_axes'] = unstructured_axes

    def _strain_analysis(self, nxdata, strain_analysis_config):
        """Perform the strain analysis on the full or partial map."""
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        # Local modules
        from CHAP.edd.utils import (
            get_peak_locations,
            get_spectra_fits,
            get_unique_hkls_ds,
        )

        # Collect the raw MCA data
        if not self._nxdata_detectors:
            scan_type = int(str(nxdata.attrs.get('scan_type', 0)))
            if (isinstance(strain_analysis_config.sum_axes, list)
                    or strain_analysis_config.sum_axes and scan_type > 2):
                # Collect the raw MCA data averaged over sum_axes
                for index, detector in enumerate(self._detectors):
                    self._nxdata_detectors.append(
                        self._get_sum_axes_data(
                            nxdata, detector.id,
                            strain_analysis_config.sum_axes))
            else:
                # Collect the raw MCA data if not averaged over sum_axes
                axes = get_axes(nxdata)
                for detector in self._detectors:
                    nxdata_det = NXdata(
                        NXfield(nxdata[detector.id].nxdata, 'detector_data'),
                        tuple([
                            NXfield(nxdata[a].nxdata, a, attrs=nxdata[a].attrs)
                            for a in axes]))
                    if len(axes) > 1:
                        nxdata_det.attrs['unstructured_axes'] = \
                            nxdata_det.attrs.pop('axes')
                    self._nxdata_detectors.append(nxdata_det)
                    axes = get_axes(nxdata)

        nxdata_ref = self._nxdata_detectors[0]

        # Setup the points list with the map axes values
        axes = get_axes(nxdata_ref)
        points = [
            {a: nxdata_ref[a].nxdata[i] for a in axes}
            for i in range(nxdata_ref[axes[0]].size)]

        # Get the energy masks
        self._get_energy_and_masks()

        # Get the mask and HKLs used in the strain analysis
        self._get_mask_hkls(strain_analysis_config.materials)

        # Loop over the detectors to fill in the nxprocess
        for index, (nxdata, detector) in enumerate(
                zip(self._nxdata_detectors, self._detectors)):

            self.logger.debug(
                f'Beginning strain analysis for {detector.id}')

            # Get the MCA bin energies
            bin_energies = detector.energies
            mask = detector.mca_mask()
            energies = bin_energies[mask]

            # Get the spectra for this detector
            intensities = nxdata.nxsignal.nxdata.T[mask].T

            # Get the unique HKLs and lattice spacings for the strain
            # analysis materials
            hkls, ds = get_unique_hkls_ds(
                strain_analysis_config.materials, tth_tol=detector.hkl_tth_tol,
                tth_max=detector.tth_max)

            # Get the HKLs and lattice spacings that will be used for
            # fitting
            hkls_fit = np.asarray([hkls[i] for i in detector.hkl_indices])
            ds_fit = np.asarray([ds[i] for i in detector.hkl_indices])
            peak_locations = get_peak_locations(ds_fit, detector.tth_calibrated)

            # Find initial peak estimates
            if not self._find_peaks or detector.rel_height_cutoff is None:
                use_peaks = np.ones((peak_locations.size)).astype(bool)
            else:
                # Third party modules
                from scipy.signal import find_peaks as find_peaks_scipy

                # FIX
                peaks = find_peaks_scipy(
                    self._mean_data[index],
                    height=(detector.rel_height_cutoff *
                        self._mean_data[index][mask].max()),
                    width=5)
                #heights = peaks[1]['peak_heights']
                widths = peaks[1]['widths']
                centers = [bin_energies[v] for v in peaks[0]]
                use_peaks = np.zeros((peak_locations.size)).astype(bool)
                # RV Potentially use peak_heights/widths as initial
                # values in fit?
                # peak_heights = np.zeros((peak_locations.size))
                # peak_widths = np.zeros((peak_locations.size))
                delta = bin_energies[1] - bin_energies[0]
                #for height, width, center in zip(heights, widths, centers):
                for width, center in zip(widths, centers):
                    for n, loc in enumerate(peak_locations):
                        # RV Hardwired range now, use detector.centers_range?
                        if center-width*delta < loc < center+width*delta:
                            use_peaks[n] = True
                            # peak_heights[n] = height
                            # peak_widths[n] = width*delta
                            break
            if any(use_peaks):
                self.logger.debug(
                    f'Using peaks with centers at {peak_locations[use_peaks]}')
            else:
                self.logger.warning(
                    'No matching peaks with heights above the threshold, '
                    f'skipping the fit for detector {detector.id}')
                return []
            hkls_fit = hkls_fit[use_peaks]

            # Perform the fit
            self.logger.debug(f'Fitting detector {detector.id} ...')
            uniform_results, unconstrained_results = get_spectra_fits(
                intensities, energies, peak_locations[use_peaks], detector)
            self.logger.debug('... done')

            # Add the fit results to the list of points
            tth_map = detector.get_tth_map((nxdata.shape[0],))
            nominal_centers = np.asarray(
                [get_peak_locations(d0, tth_map)
                 for d0, use_peak in zip(ds_fit, use_peaks) if use_peak])
            uniform_strains = np.log(
                nominal_centers / uniform_results['centers'])
            uniform_strain = np.mean(uniform_strains, axis=0)
            unconstrained_strains = np.log(
                nominal_centers / unconstrained_results['centers'])
            unconstrained_strain = np.mean(unconstrained_strains, axis=0)
            for i, point in enumerate(points):
                point.update({
                    f'{detector.id}/data/intensity': intensities[i],
                    f'{detector.id}/data/uniform_microstrain':
                        uniform_strain[i],
                    f'{detector.id}/data/unconstrained_microstrain':
                        unconstrained_strain[i],
                    f'{detector.id}/uniform_fit/results/best_fit':
                        uniform_results['best_fits'][i],
                    f'{detector.id}/uniform_fit/results/residual':
                        uniform_results['residuals'][i],
                    f'{detector.id}/uniform_fit/results/redchi':
                        uniform_results['redchis'][i],
                    f'{detector.id}/uniform_fit/results/success':
                        uniform_results['success'][i],
                    f'{detector.id}/unconstrained_fit/results/best_fit':
                        unconstrained_results['best_fits'][i],
                    f'{detector.id}/unconstrained_fit/results/residual':
                        unconstrained_results['residuals'][i],
                    f'{detector.id}/unconstrained_fit/results/redchi':
                        unconstrained_results['redchis'][i],
                    f'{detector.id}/unconstrained_fit/results/success':
                        unconstrained_results['success'][i],
                })
                for j, hkl in enumerate(hkls_fit):
                    hkl_name = '_'.join(str(hkl)[1:-1].split(' '))
                    uniform_fit_path = f'{detector.id}/uniform_fit/{hkl_name}'
                    unconstrained_fit_path = \
                        f'{detector.id}/unconstrained_fit/{hkl_name}'
                    centers = uniform_results['centers']
                    point.update({
                        f'{uniform_fit_path}/centers/values':
                            uniform_results['centers'][j][i],
                        f'{uniform_fit_path}/centers/errors':
                            uniform_results['centers_errors'][j][i],
                        f'{uniform_fit_path}/amplitudes/values':
                            uniform_results['amplitudes'][j][i],
                        f'{uniform_fit_path}/amplitudes/errors':
                            uniform_results['amplitudes_errors'][j][i],
                        f'{uniform_fit_path}/sigmas/values':
                            uniform_results['sigmas'][j][i],
                        f'{uniform_fit_path}/sigmas/errors':
                            uniform_results['sigmas_errors'][j][i],
                        f'{unconstrained_fit_path}/centers/values':
                            unconstrained_results['centers'][j][i],
                        f'{unconstrained_fit_path}/centers/errors':
                            unconstrained_results['centers_errors'][j][i],
                        f'{unconstrained_fit_path}/amplitudes/values':
                            unconstrained_results['amplitudes'][j][i],
                        f'{unconstrained_fit_path}/amplitudes/errors':
                            unconstrained_results['amplitudes_errors'][j][i],
                        f'{unconstrained_fit_path}/sigmas/values':
                            unconstrained_results['sigmas'][j][i],
                        f'{unconstrained_fit_path}/sigmas/errors':
                            unconstrained_results['sigmas_errors'][j][i],
                    })

            # Create an animation of the fit points
            #FIXself._create_animation(
            #    det_nxdata, energies, unconstrained_results['best_fits'],
            #    detector.id)

        return points

    def _subtract_baselines(self):
        """Get and subtract the detector baselines."""
        # Local modules
        from CHAP.edd.models import BaselineConfig
        from CHAP.common.processor import ConstructBaseline

        baselines = []
        for index, detector in enumerate(self._detectors):
            if detector.baseline:
                if isinstance(detector.baseline, bool):
                    detector.baseline = BaselineConfig()
                if self._save_figures:
                    filename = os.path.join(
                        self._outputdir,
                        f'{detector.id}_strainanalysis_baseline.png')
                else:
                    filename = None
                baseline, baseline_config = \
                    ConstructBaseline.construct_baseline(
                        self._mean_data[index], mask=self._energy_masks[index],
                        tol=detector.baseline.tol, lam=detector.baseline.lam,
                        max_iter=detector.baseline.max_iter,
                        title=f'Baseline for detector {detector.id}',
                        xlabel='Energy (keV)', ylabel='Intensity (counts)',
                        interactive=self._interactive, filename=filename)

                baselines.append(baseline)
                detector.baseline.lam = baseline_config['lambda']
                detector.baseline.attrs['num_iter'] = \
                    baseline_config['num_iter']
                detector.baseline.attrs['error'] = baseline_config['error']
        if baselines:
            for nxdata, baseline in zip(self._nxdata_detectors, baselines):
                nxdata.nxsignal = np.maximum(0, nxdata.nxsignal-baseline)
            self._mean_data = np.maximum(
                0, self._mean_data-np.asarray(baselines))


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
