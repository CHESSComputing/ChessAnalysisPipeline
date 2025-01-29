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
    """Get the axes of an NXdata object used in EDD."""
    if skip_axes is None:
        skip_axes = []
    if 'unstructured_axes' in nxdata.attrs:
        axes = nxdata.attrs['unstructured_axes']
    elif 'axes' in nxdata.attrs:
        axes = nxdata.attrs['axes']
    else:
        return []
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
        # FIX Not backward compatible with old detector
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
    def __init__(self):
        super().__init__()
        self._save_figures = False
        self._outputdir = '.'
        self._interactive = False

        self._detectors = []
        self._energies = []
        self._masks = []
        self._mean_data = []
        self._nxdata_detectors = []

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
            NXroot,
        )

        # Local modules
        from CHAP.edd.models import (
            MCAElementStrainAnalysisConfig,
            StrainAnalysisConfig,
        )

        self._save_figures = save_figures
        self._outputdir = outputdir
        self._interactive = interactive

        # Load the pipeline input data
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
                'No valid input in the pipeline data') from exc

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
                elif det_data.sum():
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
                self._energies.append(detector.energies)
            else:
                self.logger.warning(f'Skipping detector {detector.id} '
                                    '(no energy/tth calibration data)')
        if not self._detectors:
            raise ValueError('No valid data or unable to match an available '
                             'calibrated detector for the strain analysis')

        # Load the raw MCA data and compute the mean spectra
        self._setup_detector_data(
            nxentry[nxentry.default], strain_analysis_config)

        # Apply the energy mask
        self._apply_energy_mask()

        # Get the mask and HKLs used in the strain analysis
        self._get_mask_hkls(strain_analysis_config.materials)

        # Apply the combined energy ranges mask
        self._apply_combined_mask()

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Return the lattice parameter refinement from visual inspection
        return  self._refine_lattice_parameters(strain_analysis_config)

    def _adjust_material_props(self, materials, index):
        """Adjust the material properties."""
        # Local modules
        from CHAP.edd.utils import select_material_params

        detector = self._detectors[index]
        if self._save_figures:
            filename = os.path.join(
                self._outputdir,
                f'{detector.id}_strainanalysis_material_config.png')
        else:
            filename = None
        return select_material_params(
            self._energies[index], self._mean_data[index],
            detector.tth_calibrated, label='Sum of all spectra in the map',
            preselected_materials=materials, interactive=self._interactive,
            filename=filename)

    def _apply_combined_mask(self):
        """Apply the combined mask over the combined included energy
        ranges.
        """
        for index, (energies, mean_data, nxdata, detector) in enumerate(
                zip(self._energies, self._mean_data, self._nxdata_detectors,
                    self._detectors)):
            mask = detector.mca_mask()
            low, upp = np.argmax(mask), mask.size - np.argmax(mask[::-1])
            self._energies[index] = energies[low:upp]
            self._masks.append(detector.mca_mask()[low:upp])
            self._mean_data[index] = mean_data[low:upp]
            self._nxdata_detectors[index].nxsignal = nxdata.nxsignal[:,low:upp]

    def _apply_energy_mask(self, lower_cutoff=25, upper_cutoff=200):
        """Apply an energy mask by blanking out data below and/or
        above a certain threshold.
        """
        dtype = self._nxdata_detectors[0].nxsignal.dtype
        for index, (energies, detector) in enumerate(
                zip(self._energies, self._detectors)):
            energy_mask = np.where(energies >= lower_cutoff, 1, 0)
            energy_mask = np.where(energies <= upper_cutoff, energy_mask, 0)
            # Also blank out the last channel, which has shown to be
            # troublesome
            energy_mask[-1] = 0
            self._mean_data[index] *= energy_mask
            self._nxdata_detectors[index].nxsignal.nxdata *= \
                energy_mask.astype(dtype)

    def _get_mask_hkls(self, materials):
        """Get the mask and HKLs used in the strain analysis."""
        # Local modules
        from CHAP.edd.utils import (
            get_unique_hkls_ds,
            select_mask_and_hkls,
        )

        for energies, mean_data, nxdata, detector in zip(
                self._energies, self._mean_data, self._nxdata_detectors,
                self._detectors):

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
                    energies, mean_data, hkls, ds, detector.tth_calibrated,
                    preselected_bin_ranges=detector.include_bin_ranges,
                    preselected_hkl_indices=detector.hkl_indices,
                    detector_id=detector.id, ref_map=nxdata.nxsignal.nxdata,
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

    def _refine_lattice_parameters(self, strain_analysis_config):
        """Return a strain analysis configuration with the refined
        values for the material properties.

        :param strain_analysis_config: Strain analysis configuration.
        :type strain_analysis_config:
            CHAP.edd.models.StrainAnalysisConfig
        :returns: Strain analysis configuration with the refined
            values for the material properties.
        :rtype: CHAP.edd.models.StrainAnalysisConfig
        """
        # Local modules
        from CHAP.edd.models import MaterialConfig

        names = []
        sgnums = []
        lattice_parameters = []
        for i, detector in enumerate(self._detectors):
            for m in self._adjust_material_props(
                    strain_analysis_config.materials, i):
                if m.material_name in names:
                    lattice_parameters[names.index(m.material_name)].append(
                        m.lattice_parameters)
                else:
                    names.append(m.material_name)
                    sgnums.append(m.sgnum)
                    lattice_parameters.append([m.lattice_parameters])
        refined_materials = []
        for name, sgnum, lat_params in zip(names, sgnums, lattice_parameters):
            if len(lat_params):
                refined_materials.append(MaterialConfig(
                    material_name=name, sgnum=sgnum,
                    lattice_parameters=np.asarray(lat_params).mean(axis=0)))
            else:
                refined_materials.append(MaterialConfig(
                    material_name=name, sgnum=sgnum,
                    lattice_parameters=lat_params))
        strain_analysis_config.materials = refined_materials
        return strain_analysis_config

#        """
#        Method: given
#        a scan of a material, fit the peaks of each MCA spectrum for a
#        given detector. Based on those fitted peak locations,
#        calculate the lattice parameters that would produce them.
#        Return the averaged value of the calculated lattice parameters
#        across all spectra.
#        """
#        # Get the interplanar spacings measured for each fit HKL peak
#        # at the spectrum averaged over every point in the map to get
#        # the refined estimate for the material's lattice parameter
#        uniform_fit_centers = uniform_results['centers']
#        uniform_best_fit = uniform_results['best_fits']
#        unconstrained_fit_centers = unconstrained_results['centers']
#        unconstrained_best_fit = unconstrained_results['best_fits']
#        d_uniform = get_peak_locations(
#            np.asarray(uniform_fit_centers), detector.tth_calibrated)
#        d_unconstrained = get_peak_locations(
#            np.asarray(unconstrained_fit_centers), detector.tth_calibrated)
#        a_uniform = float((rs * d_uniform).mean())
#        a_unconstrained = rs * d_unconstrained
#        self.logger.warning(
#            'Lattice parameter refinement assumes cubic lattice')
#        self.logger.info(
#            'Refined lattice parameter from uniform fit over averaged '
#            f'spectrum: {a_uniform}')
#        self.logger.info(
#            'Refined lattice parameter from unconstrained fit over averaged '
#            f'spectrum: {a_unconstrained}')
#
#        if interactive or save_figures:
#            # Third party modules
#            import matplotlib.pyplot as plt
#
#            fig, ax = plt.subplots(figsize=(11, 8.5))
#            ax.set_title(
#                f'Detector {detector.id}: Lattice Parameter Refinement')
#            ax.set_xlabel('Detector energy (keV)')
#            ax.set_ylabel('Mean intensity (counts)')
#            ax.plot(energies, mean_intensity, 'k.', label='MCA data')
#            ax.plot(energies, uniform_best_fit, 'r', label='Best uniform fit')
#            ax.plot(
#                energies, unconstrained_best_fit, 'b',
#                label='Best unconstrained fit')
#            ax.legend()
#            for i, loc in enumerate(peak_locations):
#                ax.axvline(loc, c='k', ls='--')
#                ax.text(loc, 1, str(hkls_fit[i])[1:-1],
#                              ha='right', va='top', rotation=90,
#                              transform=ax.get_xaxis_transform())
#            if save_figures:
#                fig.tight_layout()#rect=(0, 0, 1, 0.95))
#                figfile = os.path.join(
#                    outputdir, f'{detector.id}_lat_param_fits.png')
#                plt.savefig(figfile)
#                self.logger.info(f'Saved figure to {figfile}')
#            if interactive:
#                plt.show()
#
#        return [
#            a_uniform, a_uniform, a_uniform, 90., 90., 90.]

    def _setup_detector_data(self, nxdata_raw, strain_analysis_config):
        """Load the raw MCA data accounting for oversampling or axes
        summation if requested, compute the mean spectrum, and select the
        energy mask and the HKL to use in the strain analysis"""
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

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
                for detector in self._detectors:
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
        self._mean_data = [
            np.mean(
                nxdata.nxsignal.nxdata[
                    [i for i in range(0, nxdata.nxsignal.shape[0])
                     if nxdata[i].nxsignal.nxdata.sum()]],
                axis=tuple(i for i in range(0, nxdata.nxsignal.ndim-1)))
            for nxdata in self._nxdata_detectors]
        self.logger.debug(
            f'data shape: {nxdata_raw[self._detectors[0].id].nxdata.shape}')
        self.logger.debug(
            f'mean_data shape: {np.asarray(self._mean_data).shape}')

    def _subtract_baselines(self):
        """Get and subtract the detector baselines."""
        # Local modules
        from CHAP.edd.models import BaselineConfig
        from CHAP.common.processor import ConstructBaseline

        baselines = []
        for mean_data, nxdata, detector in zip(
                self._mean_data, self._nxdata_detectors, self._detectors):
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
                        mean_data, tol=detector.baseline.tol,
                        lam=detector.baseline.lam,
                        max_iter=detector.baseline.max_iter,
                        title=f'Baseline for detector {detector.id}',
                        xlabel='Energy (keV)', ylabel='Intensity (counts)',
                        interactive=self._interactive, filename=filename)

                baselines.append(baseline)
                detector.baseline.lam = baseline_config['lambda']
                detector.baseline.attrs['num_iter'] = \
                    baseline_config['num_iter']
                detector.baseline.attrs['error'] = baseline_config['error']

                nxdata.nxsignal -= baseline
                mean_data -= baseline


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
            self, data, config=None, max_energy_kev=200.0, save_figures=False,
            interactive=False, inputdir='.', outputdir='.'):
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
        :param max_energy_kev: Maximum channel energy of the MCA in
            keV, defaults to `200.0`.
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
            version of the calibrated configuration.
        :rtype: dict
        """
        # Third party modules
        from json import loads
        from nexusformat.nexus import NXroot

        # Local modules
        from CHAP.common.models.map import DetectorConfig
        from CHAP.edd.models import MCAElementCalibrationConfig
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
        if detector_config.detectors[0].id == 'mca1':
            if len(detector_config.detectors) != 1:
                raise ValueError(
                    'Multiple detectors not implemented for mca1 detector')
            available_detector_indices = ['mca1']
            if calibration_config.detectors is None:
                calibration_config.detectors = [
                    MCAElementCalibrationConfig(id='mca1')]
            elif len(calibration_config.detectors) == 1:
                id_ = calibration_config.detectors[0].id
                if id_ != 'mca1':
                    self.logger.warning(
                        f'Skipping detector {id_} (no raw data)')
                    calibration_config.detectors = []
            else:
                raise ValueError('Multiple detectors not implemented '
                                 'for mca1 detector')
        else:
            available_detector_indices = [
                int(d.id) for d in detector_config.detectors]
            if calibration_config.detectors is None:
                calibration_config.detectors = [
                    MCAElementCalibrationConfig(id=i)
                    for i in available_detector_indices]
            else:
                for detector in deepcopy(calibration_config.detectors):
                    id_ = int(detector.id)
                    if id_ not in available_detector_indices:
                        self.logger.warning(
                            f'Skipping detector {id_} (no raw data)')
                        calibration_config.detectors.remove(detector)
            for detector in calibration_config.detectors:
                detector.id = int(detector.id)
        if not calibration_config.detectors:
            self.logger.warning(
                f'No raw data for the requested calibration detectors)')
            exit('Code terminated')
        detectors = calibration_config.detectors

        # Validate the fit index range
        if calibration_config.fit_index_ranges is None and not interactive:
            raise RuntimeError(
                'If `fit_index_ranges` is not explicitly provided, '
                f'{self.__class__.__name__} must be run with '
                '`interactive=True`.')

        # Validate the optional inputs
        if not is_num(max_energy_kev, gt=0, log=False):
            raise RuntimeError(
                f'Invalid max_energy_kev parameter ({max_energy_kev})')

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

        # Copy any configurational parameters that supersede the
        # individual input detector values
        for detector in detectors:
            if calibration_config.background is not None:
                detector.background = calibration_config.background.copy()
            if calibration_config.baseline:
                detector.baseline = calibration_config.baseline.model_copy()

        # Check each detector's include_energy_ranges field against the
        # flux file, if available.
        if calibration_config.flux_file is not None:
            flux = np.loadtxt(flux_file)
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

        # Calibrate detector channel energies based on fluorescence peaks
        for detector in detectors:
            index = available_detector_indices.index(detector.id)
            detector.energy_calibration_coeffs = self.calibrate(
                calibration_config, detector, summed_detector_data[index],
                max_energy_kev, save_figures, interactive, outputdir)

        return calibration_config.dict()

    def calibrate(
            self, calibration_config, detector, spectrum, max_energy_kev,
            save_figures, interactive, outputdir):
        """Return energy_calibration_coeffs (a, b, and c) for
        quadratically converting the current detector's MCA channels
        to bin energies.

        :param calibration_config: Energy calibration configuration.
        :type calibration_config: MCAEnergyCalibrationConfig
        :param detector: Configuration of the current detector.
        :type detector: MCAElementCalibrationConfig
        :param spectrum: Summed MCA spectrum for the current detector.
        :type spectrum: numpy.ndarray
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

            spectrum -= baseline
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
            if len(detector.background) == 1:
                models.append(
                    {'model': detector.background[0], 'prefix': 'bkgd_'})
            else:
                for model in detector.background:
                    models.append({'model': model, 'prefix': f'{model}_'})
        if calibration_config.centers_range is None:
            calibration_config.centers_range = 20
        models.append(
            {'model': 'multipeak', 'centers': initial_peak_indices,
             'centers_range': calibration_config.centers_range,
             'fwhm_min': calibration_config.fwhm_min,
             'fwhm_max': calibration_config.fwhm_max})
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

        # FIX for now stick with a linear energy correction
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
            axs[0].plot(
                bins[mask], spectrum_fit.residual, 'g', label='Residual')
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
            axs[1].plot(bins[mask], b*bins[mask] + c, 'r',
                        label=f'Best linear fit:\nm = {b:.5f} $keV$/channel\n'
                              f'b = {c:.5f} $keV$')
            axs[1].set_ylim(
                (None, 1.2*axs[1].get_ylim()[1]-0.2*axs[1].get_ylim()[0]))
            axs[1].legend()
            ax2 = axs[1].twinx()
            ax2.set_ylabel('Residual (keV)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax2.plot(
                fit_peak_indices, peak_energies-energy_fit.best_fit,
                c='g', marker='o', ms=6, ls='', label='Residual')
            ax2.set_ylim((None, 2*ax2.get_ylim()[1]-ax2.get_ylim()[0]))
            ax2.legend()
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
            max_peak_index = np.asarray(peaks[1]["peak_heights"]).argmax()
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
                    int(pt[0]) for pt in plt.ginput(num_peak, timeout=30)]
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
            self, data, config=None, save_figures=False, inputdir='.',
            outputdir='.', interactive=False):
        """Return the calibrated 2&theta value and the fine tuned
        energy calibration coefficients to convert MCA channel
        indices to MCA channel energies.

        :param data: Input configuration for the raw data & tuning
            procedure.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.MCATthCalibrationConfig.
        :type config: dict, optional
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
        from CHAP.edd.models import MCATthCalibrationConfig
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
            try:
                calibration_config = self.get_config(
                    data, 'edd.models.MCAEnergyCalibrationConfig',
                    inputdir=inputdir).dict()
            except:
                calibration_config = self.get_config(
                    data, 'edd.models.MCATthCalibrationConfig',
                    inputdir=inputdir).dict()
            if config is not None:
                calibration_config.update(config)
            calibration_config = MCATthCalibrationConfig(**calibration_config)
        except:
            self.logger.info('No valid calibration config in input pipeline '
                             'data, using config parameter instead.')
            try:
                calibration_config = MCATthCalibrationConfig(
                    **config, inputdir=inputdir)
            except Exception as exc:
                raise RuntimeError from exc

        # Validate the detector configuration
        if calibration_config.detectors is None:
            raise RuntimeError('No available calibrated detectors')
        detector_config = DetectorConfig(**loads(str(nxentry.detectors)))
        calibration_detector_indices = []
        if detector_config.detectors[0].id == 'mca1':
            if len(detector_config.detectors) != 1:
                raise ValueError(
                    'Multiple detectors not implemented for mca1 detector')
            available_detector_indices = ['mca1']
            if len(calibration_config.detectors) == 1:
                id_ = calibration_config.detectors[0].id
                if id_ in available_detector_indices:
                    calibration_detector_indices.append(id_)
                else:
                    self.logger.warning(
                        f'Skipping detector {id_} (no raw data)')
                    calibration_config.detectors.remove(detector)
            else:
                raise ValueError('Multiple detectors not implemented '
                                 'for mca1 detector')
        else:
            available_detector_indices = [
                int(d.id) for d in detector_config.detectors]
            for detector in deepcopy(calibration_config.detectors):
                id_ = int(detector.id)
                if id_ in available_detector_indices:
                    calibration_detector_indices.append(id_)
                else:
                    self.logger.warning(
                        f'Skipping detector {id_} (no raw data)')
                    calibration_config.detectors.remove(detector)
            for detector in calibration_config.detectors:
                detector.id = int(detector.id)
        detectors = calibration_config.detectors
        skipped_detector_indices = [
            id_ for id_ in available_detector_indices
            if id_ not in calibration_detector_indices]
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

        # Copy any configurational parameters that supersede the
        # detector values during the energy calibration
        for detector in detectors:
            if calibration_config.tth_initial_guess is not None:
                detector.tth_initial_guess = \
                    calibration_config.tth_initial_guess
            if calibration_config.include_energy_ranges is not None:
                detector.include_energy_ranges = \
                    calibration_config.include_energy_ranges
            if calibration_config.background is not None:
                detector.background = calibration_config.background.copy()
            if calibration_config.baseline:
                detector.baseline = calibration_config.baseline.model_copy()

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
            index = available_detector_indices.index(detector.id)
            self.calibrate(
                calibration_config, detector, summed_detector_data[index],
                save_figures, interactive, outputdir)

        return calibration_config.dict()

    def calibrate(
            self, calibration_config, detector, spectrum, save_figures=False,
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
        centers_range = calibration_config.centers_range
        if centers_range is None:
            centers_range = 20
        quadratic_energy_calibration = \
            calibration_config.quadratic_energy_calibration

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

            spectrum -= baseline
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
        if calibration_method in ('fix_tth_to_tth_init', 'iterate_tth'):
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
        if calibration_method  not in ('fix_tth_to_tth_init', 'iterate_tth'):
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
        if calibration_method == 'fix_tth_to_tth_init':
            # Third party modules
            from scipy.signal import find_peaks as find_peaks_scipy

            mca_bins_fit = np.arange(detector.num_bins)[mca_mask]

            # Get initial peak centers
            peaks = find_peaks_scipy(
                spectrum_fit, width=5,
                height=(0.005 * spectrum[mca_mask].max()))
            centers = [mca_bins_fit[v] for v in peaks[0]]
            centers = [centers[index_nearest(centers, c)]
                       for c in [index_nearest(mca_bin_energies, e)
                                 for e in e_bragg_init]]

            # Perform an unconstrained fit in terms of MCA bin index
            models = []
            if detector.background is not None:
                if len(detector.background) == 1:
                    models.append(
                        {'model': detector.background[0], 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append(
                {'model': 'multipeak', 'centers': centers,
                 'centers_range': centers_range,
                 'fwhm_min': calibration_config.fwhm_min,
                 'fwhm_max': calibration_config.fwhm_max})
            fit = FitProcessor()
            result = fit.process(
                NXdata(NXfield(spectrum_fit, 'y'),
                       NXfield(mca_bins_fit, 'x')),
                {'models': models, 'method': 'trf'})
            best_fit_unconstrained = result.best_fit
            residual_unconstrained = result.residual

            # Extract the Bragg peak indices from the fit
            i_bragg_fit = np.asarray(
                [result.best_values[f'peak{i+1}_center']
                 for i in range(num_bragg)])

            # Fit line through zero strain peak energies vs detector
            # energy bins
            if quadratic_energy_calibration:
                model = 'quadratic'
            else:
                model = 'linear'
            fit = FitProcessor()
            result = fit.process(
                NXdata(
                    NXfield(e_bragg_init, 'y'),
                    NXfield(i_bragg_fit, 'x')),
                {'models': [{'model': model}]})

            # Extract values of interest from the best values
            best_fit_uniform = None
            residual_uniform = None
            strain_uniform = None
            tth_fit = tth_init
            if quadratic_energy_calibration:
                a_fit = result.best_values['a']
                b_fit = result.best_values['b']
                c_fit = result.best_values['c']
            else:
                a_fit = 0.0
                b_fit = result.best_values['slope']
                c_fit = result.best_values['intercept']
            e_bragg_unconstrained = (
                (a_fit*i_bragg_fit + b_fit) * i_bragg_fit + c_fit)
            strains_unconstrained = np.log(
                (e_bragg_init / e_bragg_unconstrained))
            strain_unconstrained = np.mean(strains_unconstrained)

            e_bragg_fit = e_bragg_init
            peak_energies_fit = e_bragg_unconstrained

        elif calibration_method == 'direct_fit_residual':

            # Get the initial free fit parameters
            tth_init = np.radians(tth_init)
            a_init, b_init, c_init = detector.energy_calibration_coeffs

            # For testing: hardwired limits:
            if False: # FIX
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
                if calibration_config.fwhm_min is not None:
                    sig_min = calibration_config.fwhm_min/2.35482
                else:
                    sig_min = None
                if calibration_config.fwhm_max is not None:
                    sig_max = calibration_config.fwhm_max/2.35482
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
                if len(detector.background) == 1:
                    models.append(
                        {'model': detector.background[0], 'prefix': 'bkgd_'})
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
            e_bragg_fit = get_peak_locations(ds_fit, tth_fit)
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
                if len(detector.background) == 1:
                    models.append(
                        {'model': detector.background[0], 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append(
                {'model': 'multipeak', 'centers': centers,
                 'centers_range': centers_range,
                 'fwhm_min': calibration_config.fwhm_min,
                 'fwhm_max': calibration_config.fwhm_max})
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
                if len(detector.background) == 1:
                    models.append(
                        {'model': detector.background[0], 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append(
                {'model': 'multipeak', 'centers': centers,
                 'centers_range': centers_range,
                 'fwhm_min': calibration_config.fwhm_min,
                 'fwhm_max': calibration_config.fwhm_max})
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
            if calibration_config.fwhm_min is not None:
                fwhm_min = calibration_config.fwhm_min*b_init
            else:
                fwhm_min = None
            if calibration_config.fwhm_max is not None:
                fwhm_max = calibration_config.fwhm_max*b_init
            else:
                fwhm_max = None
            for iter_i in range(calibration_config.max_iter):
                self.logger.debug(f'Tuning tth: iteration no. {iter_i}, '
                                  f'starting value = {tth_fit} ')

                # Construct the fit model
                models = []
                if detector.background is not None:
                    if len(detector.background) == 1:
                        models.append(
                            {'model': detector.background[0],
                             'prefix': 'bkgd_'})
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

            # Get the MCA channel energies
            mca_energies_fit = mca_bin_energies[mca_mask]

            # Get an unconstrained fit
            if (calibration_method
                    not in ('fix_tth_to_tth_init', 'iterate_tth')):
                if calibration_config.fwhm_min is not None:
                    fwhm_min = calibration_config.fwhm_min*b_fit
                else:
                    fwhm_min = None
                if calibration_config.fwhm_max is not None:
                    fwhm_max = calibration_config.fwhm_max*b_fit
                else:
                    fwhm_max = None
                models = []
                if detector.background is not None:
                    if len(detector.background) == 1:
                        models.append(
                            {'model': detector.background[0],
                             'prefix': 'bkgd_'})
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
            axs[0,1].set_title('Peak Energy vs. Microstrain')
            axs[0,1].set_xlabel('Energy (keV)')
            axs[0,1].set_ylabel('Strain (\u03BC\u03B5)')
            if strain_uniform is not None:
                axs[0,1].axhline(strain_uniform * 1e6,
                                 ls='--', label='Single strain')
            axs[0,1].plot(e_bragg_fit, strains_unconstrained * 1e6,
                          marker='o', mfc='none', c='C1',
                          label='Unconstrained')
            axs[0,1].axhline(strain_unconstrained * 1e6,
                             ls='--', c='C1',
                             label='Unconstrained: unweighted mean')
            axs[0,1].legend()

            # Lower right axes: theoretical E vs fitted E for all peaks
            axs[1,1].set_title('Theoretical vs. Fitted Peak Energies')
            axs[1,1].set_xlabel('Energy (keV)')
            axs[1,1].set_ylabel('Energy (keV)')
            if calibration_method in ('fix_tth_to_tth_init', 'iterate_tth'):
                e_fit = e_bragg_fit
                if quadratic_energy_calibration:
                    label = 'Unconstrained: quadratic fit'
                else:
                    label = 'Unconstrained: linear fit'
            else:
                e_fit = np.concatenate((e_xrf, e_bragg_fit))
                if quadratic_energy_calibration:
                    label = 'Quadratic fit'
                else:
                    label = 'Linear fit'
            label += f'\nTakeoff angle: {tth_fit:.5f}'r'$^\circ$'
            if quadratic_energy_calibration:
                label += f'\na = {a_fit:.5e} $keV$/channel$^2$'
                label += \
                    f'\nb = {b_fit:.5f} $keV$/channel\nc = {c_fit:.5f} $keV$'
            else:
                label += f'\nm = {b_fit:.5f} $keV$/channel' \
                         f'\nb = {c_fit:.5f} $keV$'
            if calibration_method != 'fix_tth_to_tth_init':
                axs[1,1].plot(
                    e_bragg_fit, e_bragg_uniform, marker='x', ls='',
                    label='Single strain')
            axs[1,1].plot(
                e_fit, e_fit, marker='o', mfc='none', ls='',
                label='Theoretical peak positions')
            axs[1,1].plot(e_fit, peak_energies_fit, c='C1', label=label)
            axs[1,1].set_ylim(
                (None, 1.2*axs[1,1].get_ylim()[1]-0.2*axs[1,1].get_ylim()[0]))
            axs[1,1].legend()
            ax2 = axs[1,1].twinx()
            ax2.set_ylabel('Residual (keV)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax2.plot(
                e_fit, e_fit-peak_energies_fit, c='g', marker='o', ms=6, ls='',
                label='Residual')
            ax2.set_ylim((None, 2*ax2.get_ylim()[1]-ax2.get_ylim()[0]))
            ax2.legend()
            fig.tight_layout()

            if save_figures:
                figfile = os.path.join(
                    outputdir, f'{detector.id}_tth_calibration_fit.png')
                plt.savefig(figfile)
                self.logger.info(f'Saved figure to {figfile}')
            if interactive:
                plt.show()

        if not num_xrf:
            return

        # Get the mask for the fluorescence peaks
        ddetector = deepcopy(detector)
        ddetector.include_energy_ranges = \
            ddetector.get_include_energy_ranges(
                calibration_config.fit_index_ranges)
        ddetector.set_hkl_indices(hkl_indices)
        self.logger.debug(
            f'include_energy_ranges = {ddetector.include_energy_ranges}')
        self.logger.debug(
            f'hkl_indices = {ddetector.hkl_indices}')
        if not ddetector.include_energy_ranges:
            raise ValueError(
                'No value provided for include_energy_ranges. '
                'Provide them in the MCA Tth Calibration Configuration '
                'or re-run the pipeline with the --interactive flag.')
        if not ddetector.hkl_indices:
            raise ValueError(
                'Unable to get values for hkl_indices for the provided '
                'value of include_energy_ranges. Change its value in '
                'the detector\'s MCA Tth Calibration Configuration or '
                're-run the pipeline with the --interactive flag.')
        mca_mask = ddetector.mca_mask()
        spectrum_fit = spectrum[mca_mask]
        mca_energies_fit = ddetector.energies[mca_mask]

        # Perform an unconstrained fit on the fluorescence peaks
        models = []
        if ddetector.background is not None:
            if len(ddetector.background) == 1:
                models.append(
                    {'model': ddetector.background[0],
                     'prefix': 'bkgd_'})
            else:
                for model in ddetector.background:
                    models.append({'model': model, 'prefix': f'{model}_'})
        if calibration_config.fwhm_min is not None:
            fwhm_min = calibration_config.fwhm_min*b_fit
        else:
            fwhm_min = None
        if calibration_config.fwhm_max is not None:
            fwhm_max = calibration_config.fwhm_max*b_fit
        else:
            fwhm_max = None
        models.append(
            {'model': 'multipeak', 'centers': e_xrf,
             'centers_range': centers_range*b_fit,
             'fwhm_min': fwhm_min, 'fwhm_max': fwhm_max})
        fit = FitProcessor()
        result = fit.process(
            NXdata(NXfield(spectrum_fit, 'y'),
                   NXfield(mca_energies_fit, 'x')),
            {'models': models, 'method': 'trf'})
        e_xrf_fit = [result.best_values[f'peak{i+1}_center']
                     for i in range(num_xrf)]
        self.logger.info(
            f'Theoretical fluorescence peak energies (keV): {e_xrf}')
        self.logger.info(
            'Fluorescence peak energies from unconstrained fit (keV): '
            f'{[round(e, 4) for e in e_xrf_fit]}')

        # Create the figure
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.set_title(f'Detector {ddetector.id} '+r'2$\theta$ Calibration')
        ax.set_xlabel('Calibrated MCA channel energy (keV)')
        ax.set_ylabel('MCA intensity (counts)')
        for e, e_fit in zip(e_xrf, e_xrf_fit):
            ax.axvline(e_fit, c='k', ls='--')
            ax.text(e_fit, 1, f'Theoretical: {e},  Fit: {round(e_fit, 4)}',
                    ha='right', va='top', rotation=90,
                    transform=ax.get_xaxis_transform())
        if flux_correct is None:
            ax.plot(mca_energies_fit, spectrum_fit, marker='.', c='C2', ms=5,
                ls='', label='MCA data')
        else:
            ax.plot(mca_energies_fit, spectrum_fit, marker='.', c='C2', ms=5,
                ls='', label='Flux-corrected MCA data')
        ax.plot(
            mca_energies_fit, result.best_fit, c='C1',
            label='Unconstrained fit')
        ax.plot(
            mca_energies_fit, result.residual, c='r',
            label='Residual')
        ax.legend()
        fig.tight_layout()
        if save_figures:
            figfile = os.path.join(
                outputdir, f'{ddetector.id}_xrf_peaks_calibration_fit.png')
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
        self._save_figures = False
        self._inputdir = '.'
        self._outputdir = '.'
        self._interactive = False

        self._detectors = []
        self._energies = []
        self._masks = []
        self._mean_data = []
        self._nxdata_detectors = []

    @staticmethod
    def add_points(nxroot, points, logger=None):
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

        nxdata_detectors = []
        for nxobject in nxprocess.values():
            if isinstance(nxobject, NXdetector):
                nxdata_detectors.append(nxobject.data)
        if not nxdata_detectors:
            raise RuntimeError(
                'Unable to find detector data in strainanalysis object')
        axes = get_axes(nxdata_detectors[0], skip_axes=['energy'])

        if len(axes):
            coords = np.asarray(
                [nxdata_detectors[0][a].nxdata for a in axes]).T

            def get_matching_indices(all_coords, point_coords, decimals=None):
                if isinstance(decimals, int):
                    all_coords = np.round(all_coords, decimals=decimals)
                    point_coords = np.round(point_coords, decimals=decimals)
                coords_match = np.all(all_coords == point_coords, axis=1)
                index = np.where(coords_match)[0]
                return index

            # FIX: can we round to 3 decimals right away in general?
            # FIX: assumes points contains a sorted and continous
            # slice of updates
            i_0 = get_matching_indices(
                coords,
                np.asarray([points[0][a] for a in axes]), decimals=3)[0]
            i_f = get_matching_indices(
                coords,
                np.asarray([points[-1][a] for a in axes]), decimals=3)[0]
            slices = {k: np.asarray([p[k] for p in points]) for k in points[0]}
            for k, v in slices.items():
                if k not in axes:
                    logger.debug(f'Updating field {k}')
                    nxprocess[k][i_0:i_f+1] = v
        else:
            for k, v in points[0].items():
                nxprocess[k].nxdata = v

        # Add the summed intensity for each detector
        for nxdata in nxdata_detectors:
            nxdata.summed_intensity = nxdata.intensity.sum(axis=0)

    def process(
            self, data, config=None, setup=True, update=True,
            save_figures=False, inputdir='.', outputdir='.',
            interactive=False):
        """Setup the strain analysis and/or return the strain analysis
        results as a list of updated points or a
        `nexusformat.nexus.NXroot` object.

        :param data: Input data containing configurations for a map,
            completed energy/tth calibration, and parameters for strain
            analysis.
        :type data: list[PipelineData]
        :param config: Initialization parameters for an instance of
            CHAP.edd.models.StrainAnalysisConfig.
        :type config: dict, optional
        :param setup: Setup the strain analysis
            `nexusformat.nexus.NXroot` object, defaults to `True`.
        :type setup: bool, optional
        :param update: Perform the strain analysis and return the
            results as a list of updated points or update the result
            from the `setup` stage, defaults to `True`.
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
        :return: The strain analysis setup or results.
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

        if not (setup or update):
            raise RuntimeError('Illegal combination of setup and update')
        if not update:
            if interactive:
                self.logger.warning('Ineractive option disabled during setup')
                interactive = False
            if save_figures:
                self.logger.warning(
                    'Saving figures option disabled during setup')
                save_figures = False
        self._save_figures = save_figures
        self._outputdir = outputdir
        self._interactive = interactive

        # Load the pipeline input data
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
                'No valid input in the pipeline data') from exc

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
                    self._energies.append(detector.energies)
                else:
                    self.logger.warning(
                        f'Skipping detector {detector.id} (zero intensity)')
            else:
                self.logger.warning(f'Skipping detector {detector.id} '
                                    '(no energy/tth calibration data)')
        if not self._detectors:
            raise ValueError('No valid data or unable to match an available '
                             'calibrated detector for the strain analysis')

        # Load the raw MCA data and compute the mean spectra
        self._setup_detector_data(
            nxentry[nxentry.default], strain_analysis_config, update)

        # Apply the energy mask
        self._apply_energy_mask()

        # Get the mask and HKLs used in the strain analysis
        self._get_mask_hkls(strain_analysis_config.materials)

        # Apply the combined energy ranges mask
        self._apply_combined_mask()

        # Setup and/or run the strain analysis
        if setup and update:
            nxroot = self._get_nxroot(nxentry, strain_analysis_config, update)
            points = self._strain_analysis(strain_analysis_config)
            if points:
                self.logger.info(f'Adding {len(points) points}')
                self.add_points(nxroot, points, logger=self.logger)
                self.logger.info(f'... done')
            else:
                self.logger.warning('Skip adding points')
            return nxroot
        elif setup:
            return self._get_nxroot(nxentry, strain_analysis_config, update)
        elif update:
            return self._strain_analysis(strain_analysis_config)
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

    def _adjust_material_props(self, materials, index=0):
        """Adjust the material properties."""
        # Local modules
        if self._interactive:
            from CHAP.edd.select_material_params_gui import select_material_params
        else:
            from CHAP.edd.utils import select_material_params

        detector = self._detectors[index]
        if self._save_figures:
            filename = os.path.join(
                self._outputdir,
                f'{detector.id}_strainanalysis_material_config.png')
        else:
            filename = None
        return select_material_params(
            self._energies[index], self._mean_data[index],
            detector.tth_calibrated, label='Sum of all spectra in the map',
            preselected_materials=materials, interactive=self._interactive,
            filename=filename)

    def _apply_energy_mask(self, lower_cutoff=25, upper_cutoff=200):
        """Apply an energy mask by blanking out data below and/or
        above a certain threshold.
        """
        dtype = self._nxdata_detectors[0].nxsignal.dtype
        for index, (energies, detector) in enumerate(
                zip(self._energies, self._detectors)):
            energy_mask = np.where(energies >= lower_cutoff, 1, 0)
            energy_mask = np.where(energies <= upper_cutoff, energy_mask, 0)
            # Also blank out the last channel, which has shown to be
            # troublesome
            energy_mask[-1] = 0
            self._mean_data[index] *= energy_mask
            self._nxdata_detectors[index].nxsignal.nxdata *= \
                energy_mask.astype(dtype)

    def _apply_combined_mask(self):
        """Apply the combined mask over the combined included energy
        ranges.
        """
        for index, (energies, mean_data, nxdata, detector) in enumerate(
                zip(self._energies, self._mean_data, self._nxdata_detectors,
                    self._detectors)):
            mask = detector.mca_mask()
            low, upp = np.argmax(mask), mask.size - np.argmax(mask[::-1])
            self._energies[index] = energies[low:upp]
            self._masks.append(detector.mca_mask()[low:upp])
            self._mean_data[index] = mean_data[low:upp]
            self._nxdata_detectors[index].nxsignal = nxdata.nxsignal[:,low:upp]

    def _create_animation(
            self, nxdata, energies, intensities, best_fits, detector_id):
        """Create an animation of the fit results."""
        # Third party modules
        from matplotlib import animation
        import matplotlib.pyplot as plt

        def animate(i):
            data = intensities[i]
            max_ = data.max()
            norm = max(1.0, max_)
            intensity.set_ydata(data / norm)
            best_fit.set_ydata(best_fits[i] / norm)
            index.set_text('\n'.join(
                [f'norm = {int(max_)}'] +
                [f'relative norm = {(max_ / norm_all_data):.5f}'] +
                [f'{a}[{i}] = {nxdata[a][i]}' for a in axes]))
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

        axes = get_axes(nxdata)
        if 'energy' in axes:
            axes.remove('energy')
        norm_all_data = max(1.0, intensities.max())

        fig, ax = plt.subplots()
        data = intensities[0]
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

        num_frame = int(intensities.size / intensities.shape[-1])
        num_digit = len(str(num_frame))
        if not self._save_figures:
            ani = animation.FuncAnimation(
                fig, animate, frames=num_frame, interval=1000, blit=False,
                repeat=False)
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

    def _get_mask_hkls(self, materials):
        """Get the mask and HKLs used in the strain analysis."""
        # Local modules
        from CHAP.edd.utils import (
            get_unique_hkls_ds,
            select_mask_and_hkls,
        )

        for energies, mean_data, nxdata, detector in zip(
                self._energies, self._mean_data, self._nxdata_detectors,
                self._detectors):

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
                    energies, mean_data, hkls, ds, detector.tth_calibrated,
                    preselected_bin_ranges=detector.include_bin_ranges,
                    preselected_hkl_indices=detector.hkl_indices,
                    detector_id=detector.id, ref_map=nxdata.nxsignal.nxdata,
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
            MCA data.
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

        # Loop over the detectors to fill in the nxprocess
        for energies, mask, nxdata, detector in zip(
                self._energies, self._masks, self._nxdata_detectors,
                self._detectors):

            # Get the current data object
            data = nxdata.nxsignal
            num_points = data.shape[0]

            # Setup the NXdetector object for the current detector
            self.logger.debug(
                f'Setting up NXdetector group for {detector.id}')
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
            det_nxdata.energy = NXfield(
                value=energies[mask], attrs={'units': 'keV'})
            det_nxdata.tth = NXfield(
                dtype=np.float64,
                shape=(num_points,),
                attrs={'units':'degrees', 'long_name': '2\u03B8 (degrees)'})
            det_nxdata.uniform_microstrain = NXfield(
                dtype=np.float64,
                shape=(num_points,),
                attrs={'long_name': 'Strain from uniform fit (\u03BC\u03B5)'})
            det_nxdata.unconstrained_microstrain = NXfield(
                dtype=np.float64,
                shape=(num_points,),
                attrs={'long_name':
                           'Strain from unconstrained fit (\u03BC\u03B5)'})

            # Add the detector data
            det_nxdata.intensity = NXfield(
                value=np.asarray([data[i].astype(np.float64)[mask]
                                  for i in range(num_points)]),
                attrs={'units': 'counts'})
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
            tth_map = detector.get_tth_map((num_points,))
            det_nxdata.tth.nxdata = tth_map

        return nxroot

    def _get_sum_axes_data(self, nxdata, detector_id, sum_axes=True):
        """Get the raw MCA data collected by the scan averaged over the
        sum_axes.
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        data = nxdata[detector_id].nxdata
        if not isinstance(sum_axes, list):
            if sum_axes and 'fly_axis_labels' in nxdata.attrs:
                sum_axes = nxdata.attrs['fly_axis_labels']
                if isinstance(sum_axes, str):
                    sum_axes = [sum_axes]
            else:
                sum_axes = []
        axes = get_axes(nxdata, skip_axes=sum_axes)
        if not len(axes):
            return NXdata(NXfield([np.mean(data, axis=0)], 'detector_data'))
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

    def _setup_detector_data(self, nxdata_raw, strain_analysis_config, update):
        """Load the raw MCA data accounting for oversampling or axes
        summation if requested, compute the mean spectrum, and select the
        energy mask and the HKL to use in the strain analysis"""
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

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
                for detector in self._detectors:
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
            f'data shape: {nxdata_raw[self._detectors[0].id].nxdata.shape}')
        self.logger.debug(
            f'mean_data shape: {np.asarray(self._mean_data).shape}')

    def _strain_analysis(self, strain_analysis_config):
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

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Adjust the material properties
        self._adjust_material_props(strain_analysis_config.materials)

        # Setup the points list with the map axes values
        nxdata_ref = self._nxdata_detectors[0]
        axes = get_axes(nxdata_ref)
        if len(axes):
            points = [
                {a: nxdata_ref[a].nxdata[i] for a in axes}
                for i in range(nxdata_ref[axes[0]].size)]
        else:
            points = [{}]

        # Loop over the detectors to fill in the nxprocess
        for energies, mask, mean_data, nxdata, detector in zip(
                self._energies, self._masks, self._mean_data,
                self._nxdata_detectors, self._detectors):

            self.logger.debug(
                f'Beginning strain analysis for {detector.id}')

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
            peak_locations = get_peak_locations(
                ds_fit, detector.tth_calibrated)

            # Find initial peak estimates
            if (not strain_analysis_config.find_peaks
                    or detector.rel_height_cutoff is None):
                use_peaks = np.ones((peak_locations.size)).astype(bool)
            else:
                # Third party modules
                from scipy.signal import find_peaks as find_peaks_scipy

                peaks = find_peaks_scipy(
                    mean_data, width=5,
                    height=(detector.rel_height_cutoff * mean_data.max()))
                #heights = peaks[1]['peak_heights']
                widths = peaks[1]['widths']
                centers = [energies[v] for v in peaks[0]]
                use_peaks = np.zeros((peak_locations.size)).astype(bool)
                # FIX Potentially use peak_heights/widths as initial
                # values in fit?
                # peak_heights = np.zeros((peak_locations.size))
                # peak_widths = np.zeros((peak_locations.size))
                delta = energies[1] - energies[0]
                #for height, width, center in zip(heights, widths, centers):
                for width, center in zip(widths, centers):
                    for n, loc in enumerate(peak_locations):
                        # FIX Hardwired range now, use detector.centers_range?
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
            self.logger.info(f'Fitting detector {detector.id} ...')
            uniform_results, unconstrained_results = get_spectra_fits(
                np.squeeze(intensities), energies[mask],
                peak_locations[use_peaks], detector)
            if intensities.shape[0] == 1:
                uniform_results = {k: [v] for k, v in uniform_results.items()}
                unconstrained_results = {k: [v] for k, v in unconstrained_results.items()}
                for field in ('centers', 'amplitudes', 'sigmas'):
                    uniform_results[field] = np.asarray(uniform_results[field]).T
                    uniform_results[f'{field}_errors'] = np.asarray(uniform_results[f'{field}_errors']).T
                    unconstrained_results[field] = np.asarray(unconstrained_results[field]).T
                    unconstrained_results[f'{field}_errors'] = np.asarray(unconstrained_results[f'{field}_errors']).T

            self.logger.info('... done')

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
            if (not strain_analysis_config.skip_animation
                    and (self._interactive or self._save_figures)):
                self._create_animation(
                    nxdata, energies[mask], intensities,
                    unconstrained_results['best_fits'], detector.id)

        return points

    def _subtract_baselines(self):
        """Get and subtract the detector baselines."""
        # Local modules
        from CHAP.edd.models import BaselineConfig
        from CHAP.common.processor import ConstructBaseline

        baselines = []
        for mean_data, nxdata, detector in zip(
                self._mean_data, self._nxdata_detectors, self._detectors):
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
                        mean_data, tol=detector.baseline.tol,
                        lam=detector.baseline.lam,
                        max_iter=detector.baseline.max_iter,
                        title=f'Baseline for detector {detector.id}',
                        xlabel='Energy (keV)', ylabel='Intensity (counts)',
                        interactive=self._interactive, filename=filename)

                baselines.append(baseline)
                detector.baseline.lam = baseline_config['lambda']
                detector.baseline.attrs['num_iter'] = \
                    baseline_config['num_iter']
                detector.baseline.attrs['error'] = baseline_config['error']

                nxdata.nxsignal -= baseline
                mean_data -= baseline


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
