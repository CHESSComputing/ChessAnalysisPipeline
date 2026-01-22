#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File       : processor.py
Author     : Keara Soloway, Rolf Verberg
Description: Module for Processors used only by EDD experiments
"""

# System modules
from copy import deepcopy
import os
from sys import float_info
from time import time
from typing import Optional

# Third party modules
import numpy as np
from pydantic import (
    Field,
    PrivateAttr,
    confloat,
    model_validator,
)

# Local modules
from CHAP.common.models.map import DetectorConfig
from CHAP.edd.models import (
    MCADetectorCalibration,
    MCADetectorDiffractionVolumeLength,
    MCADetectorStrainAnalysis,
    MCADetectorConfig,
    DiffractionVolumeLengthConfig,
    MCAEnergyCalibrationConfig,
    MCATthCalibrationConfig,
    StrainAnalysisConfig,
)
from CHAP.pipeline import PipelineData
from CHAP.processor import Processor
from CHAP.utils.general import fig_to_iobuf

FLOAT_MIN = float_info.min

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


class BaseEddProcessor(Processor):
    """Base processor for the EDD processors."""
    _energies: list = PrivateAttr(default=[])
    _figures: list = PrivateAttr(default=[])
    _masks: list = PrivateAttr(default=[])
    _mask_index_ranges: list = PrivateAttr(default=[])
    _mean_data: list = PrivateAttr(default=[])
    _nxdata_detectors: list = PrivateAttr(default=[])

    def _apply_combined_mask(self, calibration_method=None):
        """Apply the combined mask over the combined included energy
        ranges.
        """
        for index, (energies, mean_data, nxdata, detector) in enumerate(
                zip(self._energies, self._mean_data, self._nxdata_detectors,
                    self.detector_config.detectors)):
            # Add the mask for the fluorescence peaks from the
            # energy calibration for certain tth calibrations
            if calibration_method == 'direct_fit_tth_ecc':
                detector.convert_mask_ranges(
                    detector._energy_calibration_mask_ranges +
                    detector.get_mask_ranges())

            mask = detector.mca_mask()
            low, upp = np.argmax(mask), mask.size - np.argmax(mask[::-1])
            self._energies[index] = energies[low:upp]
            self._masks.append(mask[low:upp])
            self._mask_index_ranges.append((low, upp))
            self._mean_data[index] = mean_data[low:upp]
            self._nxdata_detectors[index].nxsignal = nxdata.nxsignal[:,low:upp]

    def _apply_energy_mask(self, lower_cutoff=25, upper_cutoff=200):
        """Apply an energy mask by blanking out data below and/or
        above a certain threshold.
        """
        dtype = self._nxdata_detectors[0].nxsignal.dtype
        for index, (energies, _) in enumerate(
                zip(self._energies, self.detector_config.detectors)):
            energy_mask = np.where(energies >= lower_cutoff, 1, 0)
            energy_mask = np.where(energies <= upper_cutoff, energy_mask, 0)
            # Also blank out the last channel, which has shown to be
            # troublesome
            energy_mask[-1] = 0
            self._mean_data[index] *= energy_mask
            self._nxdata_detectors[index].nxsignal.nxdata *= \
                energy_mask.astype(dtype)

    def _apply_flux_correction(self):
        """Apply the flux correction."""
        # Check each detector's include_energy_ranges field against the
        # flux file, if available.
        if self.config.flux_file is not None:
            raise RuntimeError('Flux correction not tested after updates')
#            flux = np.loadtxt(self.config.flux_file)
#            flux_file_energies = flux[:,0]/1.e3
#            flux_e_min = flux_file_energies.min()
#            flux_e_max = flux_file_energies.max()
#            for detector in self.detector_config.detectors:
#                for i, (det_e_min, det_e_max) in enumerate(
#                        deepcopy(detector.include_energy_ranges)):
#                    if det_e_min < flux_e_min or det_e_max > flux_e_max:
#                        energy_range = [float(max(det_e_min, flux_e_min)),
#                                        float(min(det_e_max, flux_e_max))]
#                        print(
#                            f'WARNING: include_energy_ranges[{i}] out of range'
#                            f' ({detector.include_energy_ranges[i]}): adjusted'
#                            f' to {energy_range}')
#                        detector.include_energy_ranges[i] = energy_range

    def _get_mask_hkls(self):
        """Get the mask and HKLs used in the current processor."""
        # Local modules
        from CHAP.edd.utils import (
            get_unique_hkls_ds,
            select_mask_and_hkls,
        )

        if self.save_figures:
            if self.__name__ == 'MCATthCalibrationProcessor':
                basename = 'tth_calibration_mask_hkls'
            elif self.__name__ == 'StrainAnalysisProcessor':
                basename = 'strainanalysis_mask_hkls'
            elif self.__name__ == 'LatticeParameterRefinementProcessor':
                basename = 'lp_refinement_mask_hkls'
            else:
                basename = f'{self.__name__}_mask_hkls'

        for energies, mean_data, nxdata, detector in zip(
                self._energies, self._mean_data, self._nxdata_detectors,
                self.detector_config.detectors):

            # Get the unique HKLs and lattice spacings used in the
            # curent proessor
            hkls, ds = get_unique_hkls_ds(
                self.config.materials, tth_max=detector.tth_max,
                tth_tol=detector.tth_tol)

            # Interactively adjust the mask and HKLs used in the
            # current processor
            #if isinstance(detector, MCADetectorStrainAnalysis):
            if detector.processor_type == 'strainanalysis':
                calibration_bin_ranges = detector.get_calibration_mask_ranges()
            else:
                calibration_bin_ranges = None
            if detector.tth_calibrated is None:
                tth = detector.tth_initial_guess
            else:
                tth = detector.tth_calibrated
            mask_ranges, hkl_indices, buf = \
                select_mask_and_hkls(
                    energies, mean_data, hkls, ds, tth,
                    preselected_bin_ranges=detector.get_mask_ranges(),
                    preselected_hkl_indices=detector.hkl_indices,
                    detector_id=detector.get_id(),
                    ref_map=nxdata.nxsignal.nxdata,
                    calibration_bin_ranges=calibration_bin_ranges,
                    label='Sum of the spectra in the map',
                    interactive=self.interactive,
                    return_buf=self.save_figures)
            if self.save_figures:
                self._figures.append((buf, f'{detector.get_id()}_{basename}'))
            detector.hkl_indices = hkl_indices
            detector.convert_mask_ranges(mask_ranges)
            self.logger.debug(
                f'energy mask_ranges for detector {detector.get_id()}:'
                f' {detector.energy_mask_ranges}')
            self.logger.debug(
                f'hkl_indices for detector {detector.get_id()}:'
                f' {detector.hkl_indices}')
            if not detector.energy_mask_ranges:
                raise ValueError(
                    'No value provided for energy_mask_ranges. Provide '
                    'them in the tth calibration configuration, or re-run the '
                    'pipeline with the interactive flag set.')
            if not detector.hkl_indices:
                raise ValueError(
                    'No value provided for hkl_indices. Provide them in '
                    'the tth calibration configuration, or re-run the '
                    'pipeline with the interactive flag set.')

    def _setup_detector_data(self, nxobject, **kwargs):
        """Load the raw MCA data from the SpecReader output and compute
        the detector bin energies and the mean spectra.
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        available_detector_ids = kwargs['available_detector_ids']
        max_energy_kev = kwargs.get('max_energy_kev')

        scans = []
        raw_data = []
        for scan_name in nxobject.spec_scans:
            spec_scan = nxobject.spec_scans[scan_name]
            for scan_number, scan_data in spec_scan.items():
                scans.append(f'{scan_name}_{scan_number}')
                data = scan_data.data.data.nxdata
                if data.ndim != 3:
                    raise ValueError(
                        f'Illegal raw detector data shape ({data.shape})')
                if self.__name__ == 'DiffractionVolumeLengthProcessor':
                    raw_data.append(data)
                else:
                    raw_data.append(data.sum(axis=0))
        if self.__name__ == 'DiffractionVolumeLengthProcessor':
            raw_data = np.sum(raw_data, axis=0)
        else:
            raw_data = np.asarray(raw_data)
        num_bins = raw_data.shape[-1]

        for detector in self.detector_config.detectors:
            if detector.num_bins is None:
                detector.num_bins = num_bins
            elif detector.num_bins != num_bins:
                raise ValueError(
                    'Inconsistent number of MCA detector channels between '
                    'the raw data and the detector configuration '
                    f'({num_bins} vs {detector.num_bins})')
            if detector.energy_calibration_coeffs is None:
                if max_energy_kev is None:
                    raise ValueError(
                            'Missing max_energy_kev parameter')
                detector.energy_calibration_coeffs = [
                    0.0, max_energy_kev/(num_bins-1.0), 0.0]
            self._energies.append(detector.energies)
            index = int(available_detector_ids.index(detector.get_id()))
            nxdata_det = NXdata(
                NXfield(raw_data[:,index,:], 'detector_data'),
                (NXfield(scans, 'scans')))
            self._nxdata_detectors.append(nxdata_det)
        self._mean_data = [
            np.mean(
                nxdata.nxsignal.nxdata[
                    [i for i in range(0, nxdata.nxsignal.shape[0])
                     if nxdata.nxsignal.nxdata[i].sum()]],
                axis=tuple(i for i in range(0, nxdata.nxsignal.ndim-1)))
            for nxdata in self._nxdata_detectors]
        self.logger.debug(
            f'data shape: {self._nxdata_detectors[0].nxsignal.shape}')
        self.logger.debug(
            f'mean_data shape: {np.asarray(self._mean_data).shape}')

    def _subtract_baselines(self):
        """Get and subtract the detector baselines."""
        # Local modules
        from CHAP.edd.models import BaselineConfig
        from CHAP.common.processor import ConstructBaseline

        if self.save_figures:
            if self.__name__ == 'LatticeParameterRefinementProcessor':
                basename = 'lp_refinement_baseline'
            elif self.__name__ == 'DiffractionVolumeLengthProcessor':
                basename = 'dvl_baseline'
            elif self.__name__ == 'MCAEnergyCalibrationProcessor':
                basename = 'energy_calibration_baseline'
            elif self.__name__ == 'MCATthCalibrationProcessor':
                basename = 'tth_calibration_baseline'
            elif self.__name__ == 'StrainAnalysisProcessor':
                basename = 'strainanalysis_baseline'
            else:
                basename = f'{self.__name__}_baseline'

        baselines = []
        for energies, mean_data, (low, _), nxdata, detector in zip(
                self._energies, self._mean_data, self._mask_index_ranges,
                self._nxdata_detectors, self.detector_config.detectors):
            if detector.baseline:
                if isinstance(detector.baseline, bool):
                    detector.baseline = BaselineConfig()
                if self.__name__ in ('DiffractionVolumeLengthProcessor',
                                     'MCAEnergyCalibrationProcessor'):
                    x = low + np.arange(mean_data.size)
                    xlabel = 'Detector Channel (-)'
                else:
                    x = energies
                    xlabel = 'Energy (keV)'

                baseline, baseline_config, buf = \
                    ConstructBaseline.construct_baseline(
                        mean_data, x=x, tol=detector.baseline.tol,
                        lam=detector.baseline.lam,
                        max_iter=detector.baseline.max_iter,
                        title=f'Baseline for detector {detector.get_id()}',
                        xlabel=xlabel, ylabel='Intensity (counts)',
                        interactive=self.interactive,
                        return_buf=self.save_figures)
                if self.save_figures:
                    self._figures.append(
                        (buf, f'{detector.get_id()}_{basename}'))

                baselines.append(baseline)
                detector.baseline.lam = baseline_config['lambda']
                detector.baseline.attrs['num_iter'] = \
                    baseline_config['num_iter']
                detector.baseline.attrs['error'] = \
                    baseline_config['error']

                nxdata.nxsignal -= baseline
                mean_data -= baseline


class BaseStrainProcessor(BaseEddProcessor):
    """Base processor for LatticeParameterRefinementProcessor and
    StrainAnalysisProcessor.
    """
    def _adjust_material_props(self, materials, index=0):
        """Adjust the material properties."""
        # Local modules
        if self.interactive:
            from CHAP.edd.select_material_params_gui import \
                select_material_params
        else:
            from CHAP.edd.utils import select_material_params

        detector = self.detector_config.detectors[index]
        return select_material_params(
            self._energies[index], self._mean_data[index],
            detector.tth_calibrated, label='Sum of the spectra in the map',
            preselected_materials=materials, interactive=self.interactive,
            return_buf=self.save_figures)

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
        if not axes:
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

    def _setup_detector_data(self, nxobject, **kwargs):
        """Load the raw MCA data map accounting for oversampling or
        axes summation if requested and compute the detector bin
        energies and the mean spectra.
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        strain_analysis_config = kwargs['strain_analysis_config']
        update = kwargs.get('update', True)

        have_raw_detector_data = False
        oversampling_axis = {}
        if strain_analysis_config.sum_axes:
            scan_type = int(str(nxobject.attrs.get('scan_type', 0)))
            if scan_type == 4:
                # Local modules
                from CHAP.utils.general import rolling_average

                # Check for oversampling axis and create the binned
                # coordinates
                raise RuntimeError('oversampling needs testing')
                fly_axis = nxobject.attrs.get('fly_axis_labels').nxdata[0]
                oversampling = strain_analysis_config.oversampling
                oversampling_axis[fly_axis] = rolling_average(
                        nxobject[fly_axis].nxdata,
                        start=oversampling.get('start', 0),
                        end=oversampling.get('end'),
                        width=oversampling.get('width'),
                        stride=oversampling.get('stride'),
                        num=oversampling.get('num'),
                        mode=oversampling.get('mode', 'valid'))
            elif (scan_type > 2
                    or isinstance(strain_analysis_config.sum_axes, list)):
                # Collect the raw MCA data averaged over sum_axes
                for detector in self.detector_config.detectors:
                    self._nxdata_detectors.append(
                        self._get_sum_axes_data(
                            nxobject, detector.get_id(),
                            sum_axes=strain_analysis_config.sum_axes))
                have_raw_detector_data = True
        if not have_raw_detector_data:
            # Collect the raw MCA data if not averaged over sum_axes
            axes = get_axes(nxobject)
            for detector in self.detector_config.detectors:
                nxdata_det = NXdata(
                    NXfield(
                        nxobject[detector.get_id()].nxdata, 'detector_data'),
                    tuple([
                        NXfield(
                            nxobject[a].nxdata, a, attrs=nxobject[a].attrs)
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
        for detector in self.detector_config.detectors:
            self._energies.append(detector.energies)
        self.logger.debug(
            'data shape: '
            f'{nxobject[self.detector_config.detectors[0].get_id()].nxdata.shape}')
        self.logger.debug(
            f'mean_data shape: {np.asarray(self._mean_data).shape}')


class DiffractionVolumeLengthProcessor(BaseEddProcessor):
    """A Processor using a steel foil raster scan to calculate the
    diffraction volume length for an EDD setup.

    :ivar config: Initialization parameters for an instance of
        CHAP.edd.models.DiffractionVolumeLengthConfig.
    :type config: dict, optional
    :ivar detector_config: Initialization parameters for an instance of
        CHAP.edd.models.MCADetectorConfig. Defaults to the detector
        configuration of the raw detector data.
    :type detector_config: dict, optional
    :ivar save_figures: Save .pngs of plots for checking inputs &
        outputs of this Processor, defaults to `False`.
    :type save_figures: bool, optional
    """
    pipeline_fields: dict = Field(
            default = {
            'config': 'edd.models.DiffractionVolumeLengthConfig',
            'detector_config': {
                'schema': ['edd.models.DiffractionVolumeLengthConfig',
                           'edd.models.MCADetectorConfig'],
                'merge_key_paths': {'key_path': 'detectors/id', 'type': int}},
        },
        init_var=True)

    config: Optional[
        DiffractionVolumeLengthConfig] = DiffractionVolumeLengthConfig()
    detector_config: MCADetectorConfig
    save_figures: Optional[bool] = False

    @model_validator(mode='before')
    @classmethod
    def validate_diffractionvolumeLengthprocessor_before(cls, data):
        if isinstance(data, dict):
            detector_config = data.pop('detector_config', {})
            detector_config['processor_type'] = 'diffractionvolumelength'
            data['detector_config'] = detector_config
        return data

    @model_validator(mode='after')
    def validate_diffractionvolumeLengthprocessor_after(self):
        if self.config.sample_thickness is None:
            raise ValueError('Missing parameter "sample_thickness"')
        return self

    def process(self, data):
        """Return the calculated value of the DVL.

        :param data: DVL calculation input configuration.
        :type data: list[PipelineData]
        :raises RuntimeError: Unable to get a valid DVL configuration.
        :return: DVL configuration.
        :rtype: dict, PipelineData
        """
        # Third party modules
        from json import loads

        # Load the detector data
        # FIX input a numpy and create/use NXobject to numpy proc
        # FIX right now spec info is lost in output yaml, add to it?
        nxentry = self.get_default_nxentry(self.get_data(data))

        # Validate the detector configuration
        raw_detector_config = DetectorConfig(**loads(str(nxentry.detectors)))
        raw_detector_ids = [d.get_id() for d in raw_detector_config.detectors]
        if not self.detector_config.detectors:
            self.detector_config.detectors = [
                MCADetectorDiffractionVolumeLength(
                    **d.model_dump(), processor_type='diffractionvolumelength')
                for d in raw_detector_config.detectors]
            self.detector_config.update_detectors()
        else:
            skipped_detectors = []
            detectors = []
            for detector in self.detector_config.detectors:
                if detector.get_id() in raw_detector_ids:
                    raw_detector = raw_detector_config.detectors[
                        int(raw_detector_ids.index(detector.get_id()))]
                    for k, v in raw_detector.attrs.items():
                        if k not in detector.attrs:
                            if isinstance(v, list):
                                detector.attrs[k] = np.asarray(v)
                            else:
                                detector.attrs[k] = v
                    detector.energy_mask_ranges = None
                    detectors.append(detector)
                else:
                    skipped_detectors.append(detector.get_id())
            if len(skipped_detectors) == 1:
                self.logger.warning(
                    f'Skipping detector {skipped_detectors[0]} '
                    '(no raw data)')
            elif skipped_detectors:
                # Local modules
                from CHAP.utils.general import list_to_string

                skipped_detectors = [int(d) for d in skipped_detectors]
                self.logger.warning(
                    'Skipping detectors '
                    f'{list_to_string(skipped_detectors)} (no raw data)')
            self.detector_config.detectors = detectors
        if not self.detector_config.detectors:
            raise ValueError(
                'No raw data for the requested DVL measurement detectors)')

        # Load the raw MCA data and compute the detector bin energies
        # and the mean spectra
        self._setup_detector_data(
            nxentry, available_detector_ids=raw_detector_ids,
            max_energy_kev=self.config.max_energy_kev)

        # Load the scanned motor position values
        scanned_vals = self._get_scanned_vals(nxentry)

        # Apply the flux correction
#        self._apply_flux_correction()

        # Apply the energy mask
        self._apply_energy_mask()

        # Get the mask used in the DVL measurement
        self._get_mask()

        # Apply the combined energy ranges mask
        self._apply_combined_mask()

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Calculate or manually select the diffraction volume lengths
        self._measure_dvl(scanned_vals)

        # Combine the adiffraction volume length and detector
        # configuration and move default detector fields to the
        # detector attrs
        for d in self.detector_config.detectors:
            d.attrs['default_fields'] = {
                k:v.default for k, v in d.model_fields.items()
                if (k != 'attrs' and (k not in d.model_fields_set
                                      or v.default == getattr(d, k)))}
        configs = {
            **self.config.model_dump(),
            'detectors': [d.model_dump(exclude_defaults=True)
                          for d in self.detector_config.detectors]}
        return configs, PipelineData(
            name=self.__name__, data=self._figures,
            schema='common.write.ImageWriter')

    def _get_mask(self):
        """Get the mask used in the DVL measurement."""
        # Local modules
        from CHAP.utils.general import select_mask_1d

        for mean_data, detector in zip(
                self._mean_data, self.detector_config.detectors):

            # Interactively adjust the mask used in the energy
            # calibration
            buf, _, detector.mask_ranges = select_mask_1d(
                mean_data, preselected_index_ranges=detector.mask_ranges,
                title=f'Mask for detector {detector.get_id()}',
                xlabel='Detector Channel (-)',
                ylabel='Intensity (counts)',
                min_num_index_ranges=1, interactive=self.interactive,
                return_buf=self.save_figures)
            if self.save_figures:
                self._figures.append((buf, f'{detector.get_id()}_dvl_mask'))
            self.logger.debug(
                f'mask_ranges for detector {detector.get_id()}:'
                f' {detector.mask_ranges}')
            if not detector.mask_ranges:
                raise ValueError(
                    'No value provided for mask_ranges. Provide it in '
                    'the DVL configuration, or re-run the pipeline '
                    'with the interactive flag set.')

    def _get_scanned_vals(self, nxentry):
        """Load the raw MCA data from the SpecReader output and get
        the scan columns.
        """
        # Third party modules
        from json import loads

        scanned_vals = None
        for scan_name in nxentry.spec_scans:
            for scan_data in nxentry.spec_scans[scan_name].values():
                motor_mnes = loads(str(scan_data.spec_scan_motor_mnes))
                if scanned_vals is None:
                    scanned_vals = np.asarray(
                        loads(str(scan_data.scan_columns))[motor_mnes[0]])
                else:
                    assert np.array_equal(scanned_vals, np.asarray(
                        loads(str(scan_data.scan_columns))[motor_mnes[0]]))
        return scanned_vals

    def _measure_dvl(self, scanned_vals):
        """Return a measured value for the length of the diffraction
        volume. Use the iron foil raster scan data provided in
        `dvl_config` and fit a gaussian to the sum of all MCA channel
        counts vs scanned motor position in the raster scan. The
        computed diffraction volume length is approximately equal to
        the standard deviation of the fitted peak.

        :param scanned_vals: The scanned motor position values.
        :type scanned_vals: numpy.ndarray
        :return: Updated energy DVL measurement configuration and a list of
            byte stream representions of Matplotlib figures.
        :rtype: dict, PipelineData
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

        for mask, nxdata, detector in zip(
                self._masks, self._nxdata_detectors,
                self.detector_config.detectors):

            self.logger.info(f'Measuring DVL for detector {detector.get_id()}')

            masked_data = nxdata.nxsignal.nxdata[:,mask]
            masked_max = np.max(masked_data, axis=1).astype(float)
            masked_sum = np.sum(masked_data, axis=1).astype(float)

            # Find the motor position corresponding roughly to the center
            # of the diffraction volume
            scan_center = np.sum(scanned_vals*masked_sum) / np.sum(masked_sum)
            x = scanned_vals - scan_center

            # Normalize the data
            masked_max /= masked_max.max()
            masked_sum /= masked_sum.max()

            # Construct the fit model and preform the fit
            models = []
            if detector.background is not None:
                if len(detector.background) == 1:
                    models.append(
                        {'model': detector.background[0], 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append({'model': 'gaussian'})
            self.logger.debug('Fitting mean spectrum')
            fit = FitProcessor(**self.run_config)
            result = fit.process(
                NXdata(
                    NXfield(masked_sum, 'y'), NXfield(x, 'x')),
                    {'models': models, 'method': 'trf'})

            # Calculate / manually select diffraction volume length
            detector.dvl = float(
               result.best_values['sigma'] * self.config.sigma_to_dvl_factor -
               self.config.sample_thickness)
            detector.fit_amplitude = float(result.best_values['amplitude'])
            detector.fit_center = float(
                scan_center + result.best_values['center'])
            detector.fit_sigma = float(result.best_values['sigma'])
            if self.config.measurement_mode == 'manual':
                if self.interactive:
                    _, _, dvl_bounds = select_mask_1d(
                        masked_sum, x=x,
                        preselected_index_ranges=[
                            (index_nearest(x, -0.5*detector.dvl),
                             index_nearest(x, 0.5*detector.dvl))],
                        title=('Diffraction volume length'),
                        xlabel=('Beam direction (offset from scan "center")'),
                        ylabel='Normalized intensity (-)',
                        min_num_index_ranges=1, max_num_index_ranges=1,
                        interactive=self.interactive)
                    dvl_bounds = dvl_bounds[0]
                    detector.dvl = abs(x[dvl_bounds[1]] - x[dvl_bounds[0]])
                else:
                    self.logger.warning(
                        'Cannot manually indicate DVL when running CHAP '
                        'non-interactively. Using default DVL calcluation '
                        'instead.')

            if self.interactive or self.save_figures:
                # Third party modules
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.set_title(f'Diffraction Volume ({detector.get_id()})')
                ax.set_xlabel('Beam direction (offset from scan "center")')
                ax.set_ylabel('Normalized intensity (-)')
                ax.plot(x, masked_sum, label='Sum of masked data')
                ax.plot(x, masked_max, label='Maximum of masked data')
                ax.plot(x, result.best_fit, label='Gaussian fit (to sum)')
                ax.axvspan(
                    result.best_values['center']- 0.5*detector.dvl,
                    result.best_values['center'] + 0.5*detector.dvl,
                    color='gray', alpha=0.5,
                    label=f'diffraction volume ({self.config.measurement_mode})')
                ax.legend()
                plt.figtext(
                    0.5, 0.95,
                    f'Diffraction volume length: {detector.dvl:.2f}',
                    fontsize='x-large',
                    horizontalalignment='center',
                    verticalalignment='bottom')
                if self.save_figures:
                    fig.tight_layout(rect=(0, 0, 1, 0.95))
                    self._figures.append((
                        fig_to_iobuf(fig), f'{detector.get_id()}_dvl'))
                if self.interactive:
                    plt.show()
                plt.close()

        return self.config.model_dump(), PipelineData(
            name=self.__name__, data=self._figures,
            schema='common.write.ImageWriter')


class LatticeParameterRefinementProcessor(BaseStrainProcessor):
    """Processor to get a refined estimate for a sample's lattice
    parameters.

    :ivar config: Initialization parameters for an instance of
        CHAP.edd.models.StrainAnalysisConfig.
    :type config: dict, optional
    :ivar detector_config: Initialization parameters for an instance of
        CHAP.edd.models.MCADetectorConfig. Defaults to the detector
        configuration of the raw detector data merged with that of the
        2&theta calibration step..
    :ivar save_figures: Save .pngs of plots for checking inputs &
        outputs of this Processor, defaults to `False`.
    :type save_figures: bool, optional
    """
    pipeline_fields: dict = Field(
        default = {
            'config': 'edd.models.StrainAnalysisConfig',
            'detector_config': {
                'schema': 'edd.models.MCADetectorConfig',
                'merge_key_paths': {'key_path': 'detectors/id', 'type': int}},
        },
        init_var=True)
    config: Optional[StrainAnalysisConfig] = StrainAnalysisConfig()
    detector_config: MCADetectorConfig
    save_figures: Optional[bool] = False

    @model_validator(mode='before')
    @classmethod
    def validate_latticeparameterrefinementprocessor_before(cls, data):
        if isinstance(data, dict):
            detector_config = data.pop('detector_config', {})
            detector_config['processor_type'] = 'strainanalysis'
            data['detector_config'] = detector_config
        return data

    def process(self, data):
        """Given a strain analysis configuration, return a copy
        contining refined values for the materials' lattice
        parameters.

        :param data: Input data for the lattice parameter refinement
            procedure.
        :type data: list[PipelineData]
        :raises RuntimeError: Unable to refine the lattice parameters.
        :return: The strain analysis configuration with the refined
            lattice parameter configuration and, optionally, a list of
            byte stream representions of Matplotlib figures.
        :rtype: dict, PipelineData
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        # Local modules
        from CHAP.utils.general import list_to_string

        # Load the pipeline input data
        try:
            nxobject = self.get_data(data)
            if isinstance(nxobject, NXroot):
                nxroot = nxobject
            elif isinstance(nxobject, NXentry):
                nxroot = NXroot()
                nxroot[nxobject.nxname] = nxobject
                nxobject.set_default()
        except Exception as exc:
            raise RuntimeError(
                'No valid input in the pipeline data') from exc

        # Load the detector data
        nxentry = self.get_default_nxentry(nxroot)
        nxdata = nxentry[nxentry.default]

        # Load the validated calibration configuration
        calibration_config = self.get_config(
            data, schema='edd.models.MCATthCalibrationConfig', remove=False)

        # Load the validated calibration detector configurations
        calibration_detector_config = self.get_data(
            data, schema='edd.models.MCATthCalibrationConfig')
        calibration_detectors = [
            MCADetectorCalibration(**d)
            for d in calibration_detector_config.get('detectors', [])]
        calibration_detector_ids = [d.get_id() for d in calibration_detectors]

        # Check for available raw detector data and for the available 
        # calibration data
        if not self.detector_config.detectors:
            self.detector_config.detectors = [
                MCADetectorStrainAnalysis(
                    id=id_, processor_type='strainanalysis')
                for id_ in nxentry.detector_ids]
            self.detector_config.update_detectors()
        skipped_detectors = []
        sskipped_detectors = []
        detectors = []
        for detector in self.detector_config.detectors:
            detector_id = detector.get_id()
            if detector_id not in nxdata:
                skipped_detectors.append(detector_id)
            elif detector_id not in calibration_detector_ids:
                sskipped_detectors.append(detector_id)
            else:
                raw_detector_data = nxdata[detector_id].nxdata
                if raw_detector_data.ndim != 2:
                    self.logger.warning(
                        f'Skipping detector {detector_id} (Illegal data shape '
                        f'{raw_detector_data.shape})')
                elif raw_detector_data.sum():
                    for k, v in nxdata[detector_id].attrs.items():
                        detector.attrs[k] = v.nxdata
                    if self.config.rel_height_cutoff is not None:
                        detector.rel_height_cutoff = \
                            self.config.rel_height_cutoff
                    detector.add_calibration(
                        calibration_detectors[
                            int(calibration_detector_ids.index(detector_id))])
                    detectors.append(detector)
                else:
                    self.logger.warning(
                        f'Skipping detector {detector_id} (zero intensity)')
        if len(skipped_detectors) == 1:
            self.logger.warning(
                f'Skipping detector {skipped_detectors[0]} '
                '(no raw data)')
        elif skipped_detectors:
            skipped_detectors = [int(d) for d in skipped_detectors]
            self.logger.warning(
                'Skipping detectors '
                f'{list_to_string(skipped_detectors)} (no raw data)')
        if len(sskipped_detectors) == 1:
            self.logger.warning(
                f'Skipping detector {sskipped_detectors[0]} '
                '(no raw data)')
        elif sskipped_detectors:
            skipped_detectors = [int(d) for d in sskipped_detectors]
            self.logger.warning(
                'Skipping detectors '
                f'{list_to_string(skipped_detectors)} (no calibration data)')
        self.detector_config.detectors = detectors
        if not self.detector_config.detectors:
            raise ValueError('No valid data or unable to match an available '
                             'calibrated detector for the strain analysis')

        # Load the raw MCA data and compute the detector bin energies
        # and the mean spectra
        self._setup_detector_data(
            nxentry[nxentry.default], strain_analysis_config=self.config)

        # Apply the energy mask
        self._apply_energy_mask()

        # Get the mask and HKLs used in the strain analysis
        self._get_mask_hkls()

        # Apply the combined energy ranges mask
        self._apply_combined_mask()

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Get the refined values for the material properties
        self._refine_lattice_parameters()

        # Return the lattice parameter refinement from visual inspection
        if self._figures:
            return (
                self.config.model_dump(),
                PipelineData(
                    name=self.__name__, data=self._figures,
                    schema='common.write.ImageWriter'))
        return self.config.model_dump()

    def _refine_lattice_parameters(self):
        """Update the strain analysis configuration with the refined
        values for the material properties.
        """
        # Local modules
        from CHAP.edd.models import MaterialConfig

        names = []
        sgnums = []
        lattice_parameters = []
        for i, detector in enumerate(self.detector_config.detectors):
            materials, buf =  self._adjust_material_props(
                self.config.materials, i)
            for m in materials:
                if m.material_name in names:
                    lattice_parameters[names.index(m.material_name)].append(
                        m.lattice_parameters)
                else:
                    names.append(m.material_name)
                    sgnums.append(m.sgnum)
                    lattice_parameters.append([m.lattice_parameters])
            if self.save_figures:
                self._figures.append((
                    buf, f'{detector.get_id()}_lp_refinement_material_config'))
        refined_materials = []
        for name, sgnum, lat_params in zip(names, sgnums, lattice_parameters):
            if lat_params:
                refined_materials.append(MaterialConfig(
                    material_name=name, sgnum=sgnum,
                    lattice_parameters=np.asarray(lat_params).mean(axis=0)))
            else:
                refined_materials.append(MaterialConfig(
                    material_name=name, sgnum=sgnum,
                    lattice_parameters=lat_params))
        self.config.materials = refined_materials

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
#                f'Detector {detector.get_id()}: Lattice Parameter Refinement')
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
#                    outputdir, f'{detector.get_id()}_lat_param_fits')
#                plt.savefig(figfile)
#                self.logger.info(f'Saved figure to {figfile}')
#            if interactive:
#                plt.show()
#
#        return [
#            a_uniform, a_uniform, a_uniform, 90., 90., 90.]


class MCAEnergyCalibrationProcessor(BaseEddProcessor):
    """Processor to return parameters for linearly transforming MCA
    channel indices to energies (in keV). Procedure: provide a
    spectrum from the MCA element to be calibrated and the theoretical
    location of at least one peak present in that spectrum (peak
    locations must be given in keV). It is strongly recommended to use
    the location of fluorescence peaks whenever possible, _not_
    diffraction peaks, as this Processor does not account for
    2&theta.

    :ivar config: Initialization parameters for an instance of
        CHAP.edd.models.MCAEnergyCalibrationConfig.
    :type config: dict, optional
    :ivar detector_config: Initialization parameters for an instance of
        CHAP.edd.models.MCADetectorConfig. Defaults to the detector
        configuration of the raw detector data.
    :type detector_config: dict, optional
    :ivar save_figures: Save .pngs of plots for checking inputs &
        outputs of this Processor, defaults to `False`.
    :type save_figures: bool, optional
    """
    pipeline_fields: dict = Field(
        default = {
            'config': 'edd.models.MCAEnergyCalibrationConfig',
            'detector_config': {
                'schema': ['edd.models.MCAEnergyCalibrationConfig',
                           'edd.models.MCADetectorConfig'],
                'merge_key_paths': {'key_path': 'detectors/id', 'type': int}},
        },
        init_var=True)
    config: Optional[MCAEnergyCalibrationConfig] = MCAEnergyCalibrationConfig()
    detector_config: MCADetectorConfig
    save_figures: Optional[bool] = False

    @model_validator(mode='before')
    @classmethod
    def validate_mcaenergycalibrationprocessor_before(cls, data):
        if isinstance(data, dict):
            detector_config = data.pop('detector_config', {})
            detector_config['processor_type'] = 'calibration'
            data['detector_config'] = detector_config
        return data

    def process(self, data):
        """For each detector in the `MCAEnergyCalibrationConfig`
        provided with `data`, fit the specified peaks in the MCA
        spectrum specified. Using the difference between the provided
        peak locations and the fit centers of those peaks, compute
        the correction coefficients to convert uncalibrated MCA
        channel energies to calibrated channel energies. For each
        detector, set `energy_calibration_coeffs` in the calibration
        config provided to these values and return the updated
        configuration.

        :param data: Energy calibration configuration.
        :type data: list[PipelineData]
        :returns: Dictionary representing the energy-calibrated
            version of the calibrated configuration and a list of
            byte stream representions of Matplotlib figures.
        :rtype: dict, PipelineData
        """
        # Third party modules
        from json import loads

        # Load the detector data
        # FIX input a numpy and create/use NXobject to numpy proc
        # FIX right now spec info is lost in output yaml, add to it?
        nxentry = self.get_default_nxentry(self.get_data(data))

        # Check for available detectors and validate the raw detector
        # configuration
        raw_detector_config = DetectorConfig(**loads(str(nxentry.detectors)))
        raw_detector_ids = [d.get_id() for d in raw_detector_config.detectors]
        if not self.detector_config.detectors:
            self.detector_config.detectors = [
                MCADetectorCalibration(
                    **d.model_dump(), processor_type='calibration')
                for d in raw_detector_config.detectors]
            self.detector_config.update_detectors()
        else:
            skipped_detectors = []
            detectors = []
            for detector in self.detector_config.detectors:
                for raw_detector in raw_detector_config.detectors:
                    if detector.get_id() == raw_detector.get_id():
                        for k, v in raw_detector.attrs.items():
                            if k not in detector.attrs:
                                if isinstance(v, list):
                                    detector.attrs[k] = np.asarray(v)
                                else:
                                    detector.attrs[k] = v
                        detector.energy_mask_ranges = None
                        detectors.append(detector)
                        break
                else:
                    skipped_detectors.append(detector.get_id())
            if len(skipped_detectors) == 1:
                self.logger.warning(
                    f'Skipping detector {skipped_detectors[0]} '
                    '(no raw data)')
            elif skipped_detectors:
                # Local modules
                from CHAP.utils.general import list_to_string

                skipped_detectors = [int(d) for d in skipped_detectors]
                self.logger.warning(
                    'Skipping detectors '
                    f'{list_to_string(skipped_detectors)} (no raw data)')
            self.detector_config.detectors = detectors
        if not self.detector_config.detectors:
            raise ValueError(
                'No raw data for the requested calibration detectors)')

        # Load the raw MCA data and compute the detector bin energies
        # and the mean spectra
        self._setup_detector_data(
            nxentry, available_detector_ids=raw_detector_ids,
            max_energy_kev=self.config.max_energy_kev)

        # Apply the flux correction
        self._apply_flux_correction()

        # Apply the energy mask
        self._apply_energy_mask()

        # Get the mask used in the energy calibration
        self._get_mask()

        # Apply the combined energy ranges mask
        self._apply_combined_mask()

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Calibrate detector channel energies based on fluorescence peaks
        self._calibrate()

        # Combine the calibration and detector configuration
        # and move default detector fields to the detector attrs
        for d in self.detector_config.detectors:
            d.attrs['default_fields'] = {
                k:v.default for k, v in d.model_fields.items()
                if (k != 'attrs' and (k not in d.model_fields_set
                                      or v.default == getattr(d, k)))}
        configs = {
            **self.config.model_dump(),
            'detectors': [d.model_dump(exclude_defaults=True)
                          for d in self.detector_config.detectors]}
        return configs, PipelineData(
            name=self.__name__, data=self._figures,
            schema='common.write.ImageWriter')

    def _get_mask(self):
        """Get the mask used in the energy calibration."""
        # Local modules
        from CHAP.utils.general import select_mask_1d

        for mean_data, detector in zip(
                self._mean_data, self.detector_config.detectors):
            # Interactively adjust the mask used in the energy
            # calibration
            buf, _, detector.mask_ranges = select_mask_1d(
                mean_data, preselected_index_ranges=detector.mask_ranges,
                title=f'Mask for detector {detector.get_id()}',
                xlabel='Detector Channel (-)',
                ylabel='Intensity (counts)',
                min_num_index_ranges=1, interactive=self.interactive,
                return_buf=self.save_figures)
            self.logger.debug(
                f'mask_ranges for detector {detector.get_id()}:'
                f' {detector.mask_ranges}')
            if self.save_figures:
                self._figures.append((
                    buf, f'{detector.get_id()}_energy_calibration_mask'))
            if not detector.mask_ranges:
                raise ValueError(
                    'No value provided for mask_ranges. Provide it in '
                    'the energy calibration configuration, or re-run '
                    'the pipeline with the interactive flag set.')

    def _calibrate(self):
        """Return the energy calibration configuration dictionary
        after calibrating the energy_calibration_coeffs (a, b, and c)
        for quadratically converting the current detector's MCA
        channels to bin energies.

        :returns: Updated energy calibration configuration.
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        # Local modules
        from CHAP.utils.fit import FitProcessor
        from CHAP.utils.general import index_nearest

        max_peak_energy = self.config.peak_energies[
            self.config.max_peak_index]
        peak_energies = list(np.sort(self.config.peak_energies))
        max_peak_index = peak_energies.index(max_peak_energy)

        for energies, mask, mean_data, (low, _), detector in zip(
                self._energies, self._masks, self._mean_data,
                self._mask_index_ranges, self.detector_config.detectors):

            self.logger.info(f'Calibrating detector {detector.get_id()}')

            bins = low + np.arange(energies.size, dtype=np.int16)

            # Get the intial peak positions for fitting
            input_indices = [low + index_nearest(energies, energy)
                             for energy in peak_energies]
            buf, initial_peak_indices = self._get_initial_peak_positions(
                mean_data*np.asarray(mask).astype(np.int32), low,
                detector.mask_ranges, input_indices, max_peak_index,
                detector.get_id(), return_buf=self.save_figures)
            if self.save_figures:
                self._figures.append(
                    (buf,
                     f'{detector.get_id()}'
                         '_energy_calibration_initial_peak_positions'))

            # Construct the fit model and perform the fit
            models = []
            if detector.background is not None:
                if len(detector.background) == 1:
                    models.append(
                        {'model': detector.background[0], 'prefix': 'bkgd_'})
                else:
                    for model in detector.background:
                        models.append({'model': model, 'prefix': f'{model}_'})
            models.append(
                {'model': 'multipeak', 'centers': initial_peak_indices,
                 'centers_range': detector.centers_range,
                 'fwhm_min': detector.fwhm_min,
                 'fwhm_max': detector.fwhm_max})
            self.logger.debug('Fitting spectrum')
            fit = FitProcessor(**self.run_config)
            mean_data_fit = fit.process(
                NXdata(
                    NXfield(mean_data[mask], 'y'), NXfield(bins[mask], 'x')),
                {'models': models, 'method': 'trf'})

            # Extract the fit results for the peaks
            fit_peak_amplitudes = sorted([
                mean_data_fit.best_values[f'peak{i+1}_amplitude']
                for i in range(len(initial_peak_indices))])
            self.logger.debug(f'Fit peak amplitudes: {fit_peak_amplitudes}')
            fit_peak_indices = sorted([
                mean_data_fit.best_values[f'peak{i+1}_center']
                for i in range(len(initial_peak_indices))])
            self.logger.debug(f'Fit peak center indices: {fit_peak_indices}')
            fit_peak_fwhms = sorted([
                2.35482*mean_data_fit.best_values[f'peak{i+1}_sigma']
                for i in range(len(initial_peak_indices))])
            self.logger.debug(f'Fit peak fwhms: {fit_peak_fwhms}')

            # FIX for now stick with a linear energy correction
            fit = FitProcessor(**self.run_config)
            energy_fit = fit.process(
                    NXdata(
                        NXfield(peak_energies, 'y'),
                        NXfield(fit_peak_indices, 'x')),
                    {'models': [{'model': 'linear'}]})
            a = 0.0
            b = float(energy_fit.best_values['slope'])
            c = float(energy_fit.best_values['intercept'])
            detector.energy_calibration_coeffs = [a, b, c]
            delta_energy = self.config.max_energy_kev/detector.num_bins
            if not 0.95*delta_energy*0.95 < b < 1.05*delta_energy:
                self.logger.warning(
                    f'Calibrated slope ({b}) is outside the 5% tolerance '
                    'of the detector input value (max_energy_kev/num_bins '
                    f'= {delta_energy}.)')

            # Reference plot to visualize the fit results:
            if self.interactive or self.save_figures:
                # Third part modules
                import matplotlib.pyplot as plt

                bins_masked = bins[mask]
                fig, axs = plt.subplots(1, 2, figsize=(11, 4.25))
                fig.suptitle(
                    f'Detector {detector.get_id()} energy calibration')
                # Left plot: raw MCA data & best fit of peaks
                axs[0].set_title('MCA spectrum peak fit')
                axs[0].set_xlabel('Detector Channel (-)')
                axs[0].set_ylabel('Intensity (counts)')
                axs[0].plot(
                    bins_masked, mean_data[mask], 'b.', label='MCA data')
                axs[0].plot(
                    bins_masked, mean_data_fit.best_fit, 'r', label='Best fit')
                axs[0].plot(
                    bins_masked, mean_data_fit.residual, 'g', label='Residual')
                axs[0].legend()
                # Right plot: linear fit of theoretical peak energies vs
                # fit peak locations
                axs[1].set_title(
                    'Detector energy vs. detector channel')
                axs[1].set_xlabel('Detector Channel (-)')
                axs[1].set_ylabel('Detector Energy (keV)')
                axs[1].plot(
                    fit_peak_indices, peak_energies, c='b', marker='o',
                    ms=6, mfc='none', ls='', label='Initial peak positions')
                axs[1].plot(
                    bins_masked, b*bins_masked + c, 'r',
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

                if self.save_figures:
                    self._figures.append(
                        (fig_to_iobuf(fig),
                         f'{detector.get_id()}_energy_calibration_fit'))
                if self.interactive:
                    plt.show()
                plt.close()

    def _get_initial_peak_positions(
            self, y, low, index_ranges, input_indices, input_max_peak_index,
            detector_id, reset_flag=0, return_buf=False):
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

        def find_peaks(min_height=0.05, min_width=5, tolerance=10):
            """Find the peaks.

            :param min_height: Minimum peak height in search, defaults
                to `0.05`.
            :type min_height: float, optional
            :param min_width: Minimum peak width in search, defaults
                to `5`.
            :type min_width: float, optional
            :param tolerance: Tolerance in peak index in channels for
                finding matching peaks, defaults to `10`.
            :type tolerance: int, optional
            :return: The peak indices.
            :rtype: list[int]
            """
            # Third party modules
            from scipy.signal import find_peaks as find_peaks_scipy

            # Find peaks
            peaks = find_peaks_scipy(y, height=min_height,
                prominence=0.05*y.max(), width=min_width)
            #available_peak_indices = list(peaks[0])
            available_peak_indices = [low + i for i in peaks[0]]
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
                    min_error = np.inf
                    best_index = -1
                    for index in available_peak_indices.copy():
                        error = abs(index_guess-index)
                        if error < tolerance and error < min_error:
                            best_index = index
                            min_error = error
                    if best_index < 0:
                        best_index = index_guess
                    else:
                        available_peak_indices.remove(best_index)
                    peak_indices[i] = best_index
            return peak_indices

        def select_peaks():
            """Manually select initial peak indices."""
            peak_indices = []
            while len(set(peak_indices)) < num_peak:
                error_text = ''
                change_fig_title(f'Select {num_peak} peak positions')
                peak_indices = [
                    int(pt[0]) for pt in plt.ginput(num_peak, timeout=30)]
                if len(set(peak_indices)) < num_peak:
                    error_text = f'Choose {num_peak} unique position'
                    peak_indices.clear()
                outside_indices = []
                for index in peak_indices:
                    if not any(l <= index <= u for l, u in index_ranges):
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
                    ax.set_xlabel('Detector Channel (-)', fontsize='x-large')
                    ax.set_ylabel('Intensity (counts)', fontsize='x-large')
                    ax.set_xlim(index_ranges[0][0], index_ranges[-1][1])
                    fig.subplots_adjust(bottom=0.0, top=0.85)
                    ax.plot(low + np.arange(y.size), y, color='k')
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
        ax.plot(low + np.arange(y.size), y, color='k')
        ax.set_xlabel('Detector Channel (-)', fontsize='x-large')
        ax.set_ylabel('Intensity (counts)', fontsize='x-large')
        ax.set_xlim(index_ranges[0][0], index_ranges[-1][1])
        fig.subplots_adjust(bottom=0.0, top=0.85)

        if not self.interactive:

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
                    if not any(l <= index <= u for l, u in index_ranges):
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

        if return_buf:
            fig_title[0].set_in_layout(True)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            buf = fig_to_iobuf(fig)
        else:
            buf = None
        plt.close()

        if self.interactive and len(peak_indices) != num_peak:
            reset_flag += 1
            return self._get_initial_peak_positions(
                y, low, index_ranges, input_indices, input_max_peak_index,
                detector_id, reset_flag=reset_flag, return_buf=return_buf)
        return buf, peak_indices


class MCATthCalibrationProcessor(BaseEddProcessor):
    """Processor to calibrate the 2&theta angle and fine tune the
    energy calibration coefficients for an EDD experimental setup.

    :ivar config: Initialization parameters for an instance of
        CHAP.edd.models.MCATthCalibrationConfig.
    :type config: dict, optional
    :ivar detector_config: Initialization parameters for an instance of
        CHAP.edd.models.MCADetectorConfig. Defaults to the detector
        configuration of the raw detector data merged with that of the
        energy calibration step..
    :ivar save_figures: Save .pngs of plots for checking inputs &
        outputs of this Processor, defaults to `False`.
    :type save_figures: bool, optional
    """
    pipeline_fields: dict = Field(
        default = {
            'config': ['edd.models.MCAEnergyCalibrationConfig',
                       'edd.models.MCATthCalibrationConfig'],
            'detector_config': {
                'schema': ['edd.models.MCAEnergyCalibrationConfig',
                           'edd.models.MCATthCalibrationConfig',
                           'edd.models.MCADetectorConfig'],
                'merge_key_paths': {'key_path': 'detectors/id', 'type': int}},
        },
        init_var=True)
    config: Optional[MCATthCalibrationConfig] = MCATthCalibrationConfig()
    detector_config: MCADetectorConfig
    save_figures: Optional[bool] = False

    @model_validator(mode='before')
    @classmethod
    def validate_mcatthcalibrationprocessor_before(cls, data):
        if isinstance(data, dict):
            detector_config = data.pop('detector_config', {})
            detector_config['processor_type'] = 'calibration'
            data['detector_config'] = detector_config
        return data

    def process(self, data):
        """Return the calibrated 2&theta value and the fine tuned
        energy calibration coefficients to convert MCA channel
        indices to MCA channel energies.

        :param data: 2&theta calibration configuration.
        :type data: list[PipelineData]
        :raises RuntimeError: Invalid or missing input configuration.
        :return: Original configuration with the tuned values for
            2&theta and the linear correction parameters added and a
            list of byte stream representions of Matplotlib figures.
        :rtype: dict, PipelineData
        """
        # Third party modules
        from json import loads

        # Local modules
        from CHAP.utils.general import list_to_string

        # Load the detector data
        # FIX input a numpy and create/use NXobject to numpy proc
        # FIX right now spec info is lost in output yaml, add to it?
        nxentry = self.get_default_nxentry(self.get_data(data))

        # Check for available detectors and validate the raw detector
        # configuration
        if not self.detector_config.detectors:
            raise RuntimeError('No calibrated detectors')
        raw_detector_config = DetectorConfig(**loads(str(nxentry.detectors)))
        raw_detector_ids = [d.get_id() for d in raw_detector_config.detectors]
        skipped_detectors = []
        detectors = []
        for detector in self.detector_config.detectors:
            if detector.get_id() in raw_detector_ids:
                raw_detector = raw_detector_config.detectors[
                    int(raw_detector_ids.index(detector.get_id()))]
                for k, v in raw_detector.attrs.items():
                    if k not in detector.attrs:
                        detector.attrs[k] = v
                detectors.append(detector)
            else:
                skipped_detectors.append(detector.get_id())
        if len(skipped_detectors) == 1:
            self.logger.warning(
                f'Skipping detector {skipped_detectors[0]} '
                '(no raw data)')
        elif skipped_detectors:
            skipped_detectors = [int(d) for d in skipped_detectors]
            self.logger.warning(
                'Skipping detectors '
                f'{list_to_string(skipped_detectors)} (no raw data)')
        skipped_detectors = []
        for i, detector in reversed(list(enumerate(detectors))):
            if detector.energy_calibration_coeffs is None:
                skipped_detectors.append(detector.get_id())
                detectors.pop(i)
            else:
                detector.set_energy_calibration_mask_ranges()
                detector.tth_initial_guess = self.config.tth_initial_guess
                if detector.energy_mask_ranges is None:
                    raise ValueError('energy_mask_ranges is required for '
                                     'all detectors')
                detector.mask_ranges = None
        if len(skipped_detectors) == 1:
            self.logger.warning(
                f'Skipping detector {skipped_detectors[0]} '
                '(no calibration data)')
        elif skipped_detectors:
            skipped_detectors = [int(d) for d in skipped_detectors]
            self.logger.warning(
                'Skipping detectors '
                f'{list_to_string(skipped_detectors)} (no calibration data)')
        self.detector_config.detectors = detectors
        if not self.detector_config.detectors:
            raise RuntimeError('No raw or calibrated detectors')

        # Load the raw MCA data and compute the detector bin energies
        # and the mean spectra
        self._setup_detector_data(
            nxentry, available_detector_ids=raw_detector_ids)

        # Apply the flux correction
        self._apply_flux_correction()

        # Apply the energy mask
        self._apply_energy_mask()

        # Select the initial tth value
        if self.interactive or self.save_figures:
            self._select_tth_init()

        # Get the mask used in the energy calibration
        self._get_mask_hkls()

        # Apply the combined energy ranges mask
        self._apply_combined_mask(self.config.calibration_method)

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Calibrate detector channel energies
        self._calibrate()

        # Combine the calibration and detector configuration
        # and move default detector fields to the detector attrs
        for d in self.detector_config.detectors:
            d.attrs['default_fields'] = {
                k:v.default for k, v in d.model_fields.items()
                if (k != 'attrs' and (k not in d.model_fields_set
                                      or v.default == getattr(d, k)))}
        configs = {
            **self.config.model_dump(),
            'detectors': [d.model_dump(exclude_defaults=True)
                          for d in self.detector_config.detectors]}
        return configs, PipelineData(
            name=self.__name__, data=self._figures,
            schema='common.write.ImageWriter')

    def _calibrate(self):
        """Calibrate 2&theta and linear and fine tune the energy
        calibration coefficients to convert MCA channel indices to MCA
        channel energies.

            CHAP.edd.models.MCATthCalibrationConfig
        :returns: 2&theta calibration configuration.
        :rtype: dict
        """
        # Local modules
        from CHAP.edd.utils import (
            get_peak_locations,
            get_unique_hkls_ds,
        )

        quadratic_energy_calibration = \
            self.config.quadratic_energy_calibration
        calibration_method = self.config.calibration_method

        for energies, mask, mean_data, (low, upp), detector in zip(
                self._energies, self._masks, self._mean_data,
                self._mask_index_ranges, self.detector_config.detectors):

            self.logger.info(f'Calibrating detector {detector.get_id()}')

            tth = detector.tth_initial_guess
            bins = low + np.arange(energies.size, dtype=np.int16)

            # Correct raw MCA data for variable flux at different energies
            flux_correct = \
                self.config.flux_correction_interpolation_function()
            if flux_correct is not None:
                mca_intensity_weights = flux_correct(energies)
                mean_data = mean_data / mca_intensity_weights

            # Get the Bragg peak HKLs, lattice spacings and energies
            hkls, ds = get_unique_hkls_ds(
                self.config.materials, tth_max=detector.tth_max,
                tth_tol=detector.tth_tol)
            hkls  = np.asarray([hkls[i] for i in detector.hkl_indices])
            ds  = np.asarray([ds[i] for i in detector.hkl_indices])
            e_bragg = get_peak_locations(ds, tth)

            # Perform the fit
            t0 = time()
            if calibration_method == 'direct_fit_bragg':
                results = self._direct_bragg_peak_fit(
                    energies, mean_data, bins, mask, detector,
                    e_bragg, tth, quadratic_energy_calibration)
            elif calibration_method == 'direct_fit_tth_ecc':
                results = self._direct_fit_tth_ecc(
                    energies, mean_data, bins, mask, detector, ds, e_bragg,
                    self.config.peak_energies, tth,
                    quadratic_energy_calibration)
            else:
                raise ValueError(
                    f'calibration_method {calibration_method} not implemented')
            self.logger.info(
                f'Fitting detector {detector.get_id()} took '
                f'{time()-t0:.3f} seconds')
            detector.tth_calibrated = results['tth']
            detector.energy_calibration_coeffs = \
                results['energy_calibration_coeffs']

            # Update the peak energies and the MCA channel energies
            # with the newly calibrated coefficients
            energies = detector.energies[low:upp]
            if calibration_method == 'direct_fit_tth_ecc':
                e_bragg = get_peak_locations(ds, detector.tth_calibrated)

            if self.interactive or self.save_figures:
                # Third party modules
                import matplotlib.pyplot as plt

                # Create the figure
                fig, axs = plt.subplots(2, 2, sharex='all', figsize=(11, 8.5))
                fig.suptitle(
                    f'Detector {detector.get_id()} 'r'2$\theta$ Calibration')

                # Upper left axes: best fit with calibrated peak centers
                axs[0,0].set_title(r'2$\theta$ Calibration Fits')
                axs[0,0].set_xlabel('Energy (keV)')
                axs[0,0].set_ylabel('Intensity (counts)')
                for i, e_peak in enumerate(e_bragg):
                    axs[0,0].axvline(e_peak, c='k', ls='--')
                    axs[0,0].text(
                        e_peak, 1, str(hkls[i])[1:-1], ha='right', va='top',
                        rotation=90, transform=axs[0,0].get_xaxis_transform())
                if flux_correct is None:
                    axs[0,0].plot(
                        energies[mask], mean_data[mask], marker='.', c='C2',
                        ms=3, ls='', label='MCA data')
                else:
                    axs[0,0].plot(
                        energies[mask], mean_data[mask], marker='.', c='C2',
                        ms=3, ls='', label='Flux-corrected MCA data')
                if quadratic_energy_calibration:
                    label = 'Unconstrained fit using calibrated a, b, and c'
                else:
                    label = 'Unconstrained fit using calibrated b and c'
                axs[0,0].plot(
                    energies[mask], results['best_fit_unconstrained'], c='C1',
                    label=label)
                axs[0,0].legend()

                # Lower left axes: fit residual
                axs[1,0].set_title('Fit Residuals')
                axs[1,0].set_xlabel('Energy (keV)')
                axs[1,0].set_ylabel('Residual (counts)')
                axs[1,0].plot(
                    energies[mask], results['residual_unconstrained'], c='C1',
                    label=label)
                axs[1,0].legend()

                # Upper right axes: E vs strain for each fit
                strains_unconstrained = 1.e6 * results['strains_unconstrained']
                strain_unconstrained = np.mean(strains_unconstrained)
                axs[0,1].set_title('Peak Energy vs. Microstrain')
                axs[0,1].set_xlabel('Energy (keV)')
                axs[0,1].set_ylabel('Strain (\u03BC\u03B5)')
                axs[0,1].plot(
                    e_bragg, strains_unconstrained, marker='o', mfc='none',
                    c='C1', label='Unconstrained')
                axs[0,1].axhline(
                    strain_unconstrained, ls='--', c='C1',
                    label='Unconstrained: unweighted mean')
                axs[0,1].legend()

                # Lower right axes: theoretical E vs fitted E for all peaks
                a_fit, b_fit, c_fit = detector.energy_calibration_coeffs
                e_bragg_unconstrained = results['e_bragg_unconstrained']
                axs[1,1].set_title('Theoretical vs. Fitted Peak Energies')
                axs[1,1].set_xlabel('Energy (keV)')
                axs[1,1].set_ylabel('Energy (keV)')
                if calibration_method == 'direct_fit_tth_ecc':
                    e_fit = np.concatenate(
                        (self.config.peak_energies, e_bragg))
                    e_fit_unconstrained = np.concatenate(
                        (results['e_xrf_unconstrained'],
                         e_bragg_unconstrained))
                else:
                    e_fit = e_bragg
                    e_fit_unconstrained = e_bragg_unconstrained
                if quadratic_energy_calibration:
                    label = 'Unconstrained: quadratic fit'
                else:
                    label = 'Unconstrained: linear fit'
                label += f'\nTakeoff angle: {tth:.5f}'r'$^\circ$'
                if quadratic_energy_calibration:
                    label += f'\na = {a_fit:.5e} $keV$/channel$^2$'
                    label += f'\nb = {b_fit:.5f} $keV$/channel$'
                    label += f'\nc = {c_fit:.5f} $keV$'
                else:
                    label += f'\nm = {b_fit:.5f} $keV$/channel'
                    label += f'\nb = {c_fit:.5f} $keV$'
                axs[1,1].plot(
                    e_fit, e_fit, marker='o', mfc='none', ls='',
                    label='Theoretical peak positions')
                axs[1,1].plot(
                    e_fit, e_fit_unconstrained, c='C1', label=label)
                axs[1,1].set_ylim(
                    (None,
                     1.2*axs[1,1].get_ylim()[1]-0.2*axs[1,1].get_ylim()[0]))
                axs[1,1].legend()
                ax2 = axs[1,1].twinx()
                ax2.set_ylabel('Residual (keV)', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax2.plot(
                    e_fit, abs(e_fit-e_fit_unconstrained), c='g', marker='o',
                    ms=6, ls='', label='Residual')
                ax2.set_ylim((None, 2*ax2.get_ylim()[1]-ax2.get_ylim()[0]))
                ax2.legend()
                fig.tight_layout()

                if self.save_figures:
                    self._figures.append((
                        fig_to_iobuf(fig),
                        f'{detector.get_id()}_tth_calibration_fit'))
                if self.interactive:
                    plt.show()
                plt.close()

    def _direct_bragg_peak_fit(
            self, energies, mean_data, bins, mask, detector, e_bragg, tth,
            quadratic_energy_calibration):
        """Perform an unconstrained fit minimizing the residual on the
        Bragg peaks only for a given 2&theta.
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )
        from scipy.signal import find_peaks as find_peaks_scipy

        # Local modules
        from CHAP.utils.fit import FitProcessor
        from CHAP.utils.general import index_nearest

        # Get initial peak centers
        peaks = find_peaks_scipy(
            mean_data, width=5, height=0.005*mean_data.max())
        centers = list(peaks[0])
        centers = [bins[0] + centers[index_nearest(centers, c)]
                   for c in [index_nearest(energies, e) for e in e_bragg]]

        # Construct the fit model
        models = []
        if detector.background is not None:
            if len(detector.background) == 1:
                models.append(
                    {'model': detector.background[0], 'prefix': 'bkgd_'})
            else:
                for model in detector.background:
                    models.append({'model': model, 'prefix': f'{model}_'})
        if detector.backgroundpeaks is not None:
            backgroundpeaks = deepcopy(detector.backgroundpeaks)
            delta_energy = energies[1]-energies[0]
            if backgroundpeaks.centers_range is not None:
                backgroundpeaks.centers_range /= delta_energy
            if backgroundpeaks.fwhm_min is not None:
                backgroundpeaks.fwhm_min /= delta_energy
            if backgroundpeaks.fwhm_max is not None:
                backgroundpeaks.fwhm_max /= delta_energy
            backgroundpeaks.centers = [
                c/delta_energy for c in backgroundpeaks.centers]
            _, backgroundpeaks = FitProcessor.create_multipeak_model(
                backgroundpeaks)
            for peak in backgroundpeaks:
                peak.prefix = f'bkgd_{peak.prefix}'
            models += backgroundpeaks
        models.append(
            {'model': 'multipeak', 'centers': centers,
             'centers_range': detector.centers_range,
             'fwhm_min': detector.fwhm_min,
             'fwhm_max': detector.fwhm_max})

        # Perform an unconstrained fit in terms of MCA bin index
        fit = FitProcessor(**self.run_config)
        result = fit.process(
            NXdata(NXfield(mean_data[mask], 'y'), NXfield(bins[mask], 'x')),
            {'models': models, 'method': 'trf'})
        best_fit = result.best_fit
        residual = result.residual

        # Extract the Bragg peak indices from the fit
        i_bragg_fit = np.asarray(
            [result.best_values[f'peak{i+1}_center']
             for i in range(len(e_bragg))])

        # Fit a line through zero strain peak energies vs detector
        # energy bins
        if quadratic_energy_calibration:
            model = 'quadratic'
        else:
            model = 'linear'
        fit = FitProcessor(**self.run_config)
        result = fit.process(
            NXdata(NXfield(e_bragg, 'y'), NXfield(i_bragg_fit, 'x')),
            {'models': [{'model': model}]})
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

        return {
            'best_fit_unconstrained': best_fit,
            'residual_unconstrained': residual,
            'e_bragg_unconstrained': e_bragg_unconstrained,
            'strains_unconstrained': np.log(e_bragg / e_bragg_unconstrained),
            'tth': float(tth),
            'energy_calibration_coeffs': [
                 float(a_fit), float(b_fit), float(c_fit)],
        }

    def _direct_fit_tth_ecc(
            self, energies, mean_data, bins, mask, detector, ds, e_bragg,
            e_xrf, tth, quadratic_energy_calibration):
        """Perform a fit minimizing the residual on the Bragg peaks
        only or on both the Bragg peaks and the fluorescence peaks
        for a given 2&theta in terms of the energy calibration
        coefficients.
        """
        #RV FIX Right now only implemented for a cubic lattice
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )
        from scipy.constants import physical_constants

        # Local modules
        from CHAP.edd.utils import get_peak_locations
        from CHAP.utils.fit import FitProcessor

        # Collect the free fit parameters
        # RV FIX Confine b to a limited range about its expect value?
        a, b, c = detector.energy_calibration_coeffs
        parameters = [{'name': 'tth', 'value': np.radians(tth)}]
        if quadratic_energy_calibration:
            parameters.append({'name': 'a', 'value': a})
        parameters.append({'name': 'b', 'value': b})
        parameters.append({'name': 'c', 'value': c})

        # Construct the fit model
        num_bragg = len(e_bragg)
        num_xrf = len(e_xrf)

        # Get the background
        bkgd_models = []
        if detector.background is not None:
            if isinstance(detector.background, str):
                bkgd_models.append(
                    {'model': detector.background, 'prefix': 'bkgd_'})
            else:
                for model in detector.background:
                    bkgd_models.append({'model': model, 'prefix': f'{model}_'})

        # Add the background peaks in MCA channels
        models = deepcopy(bkgd_models)
        if detector.backgroundpeaks is not None:
            backgroundpeaks = deepcopy(detector.backgroundpeaks)
            delta_energy = energies[1]-energies[0]
            if backgroundpeaks.centers_range is not None:
                backgroundpeaks.centers_range /= delta_energy
            if backgroundpeaks.fwhm_min is not None:
                backgroundpeaks.fwhm_min /= delta_energy
            if backgroundpeaks.fwhm_max is not None:
                backgroundpeaks.fwhm_max /= delta_energy
            backgroundpeaks.centers = [
                c/delta_energy for c in backgroundpeaks.centers]
            _, backgroundpeaks = FitProcessor.create_multipeak_model(
                backgroundpeaks)
            for peak in backgroundpeaks:
                peak.prefix = f'bkgd_{peak.prefix}'
            models += backgroundpeaks

        # Add the fluorescent peaks
        sig_min, sig_max = ((detector.fwhm_min, detector.fwhm_max) /
            (2.0*np.sqrt(2.0*np.log(2.0))))
        for i, e_peak in enumerate(e_xrf):
            expr = f'({e_peak}-c)/b'
            if quadratic_energy_calibration:
                expr = '(' + expr + f')*(1.0-a*(({e_peak}-c)/(b*b)))'
            models.append(
                {'model': 'gaussian', 'prefix': f'xrf{i+1}_',
                 'parameters': [
                    {'name': 'amplitude', 'min': FLOAT_MIN},
                    {'name': 'center', 'expr': expr},
                    {'name': 'sigma', 'min': sig_min, 'max': sig_max}]})

        # Add the Bragg peaks
        hc = 1.e7 * physical_constants['Planck constant in eV/Hz'][0] \
             * physical_constants['speed of light in vacuum'][0]
        for i, d in enumerate(ds):
            norm = 0.5*hc/d
            expr = f'(({norm}/sin(0.5*tth))-c)/b'
            if quadratic_energy_calibration:
                expr = '(' + expr \
                       + f')*(1.0-a*((({norm}/sin(0.5*tth))-c)/(b*b)))'
            models.append(
                {'model': 'gaussian', 'prefix': f'peak{i+1}_',
                 'parameters': [
                    {'name': 'amplitude', 'min': FLOAT_MIN},
                    {'name': 'center', 'expr': expr},
                    {'name': 'sigma', 'min': sig_min, 'max': sig_max}]})

        # Perform the fit
        fit = FitProcessor(**self.run_config)
        result = fit.process(
            NXdata(NXfield(mean_data[mask], 'y'), NXfield(bins[mask], 'x')),
            {'parameters': parameters, 'models': models, 'method': 'trf'})

        # Extract values of interest from the best values
        tth_fit = np.degrees(result.best_values['tth'])
        if quadratic_energy_calibration:
            a_fit = result.best_values['a']
        else:
            a_fit = 0.0
        b_fit = result.best_values['b']
        c_fit = result.best_values['c']
        i_peak_fit = np.asarray(
            [result.best_values[f'xrf{i+1}_center'] for i in range(num_xrf)]
            + [result.best_values[f'peak{i+1}_center']
               for i in range(num_bragg)])
        e_peak_fit = (a_fit*i_peak_fit + b_fit) * i_peak_fit+ c_fit

        # Add the background peaks in keV
        models = deepcopy(bkgd_models)
        if detector.backgroundpeaks is not None:
            _, backgroundpeaks = FitProcessor.create_multipeak_model(
                detector.backgroundpeaks)
            for peak in backgroundpeaks:
                peak.prefix = f'bkgd_{peak.prefix}'
            bkgd_models += backgroundpeaks

        # Get an unconstrained fit for the fitted energy calibration
        # coefficients
        models = bkgd_models + [{
            'model': 'multipeak', 'centers': list(e_peak_fit),
            'centers_range': b_fit * detector.centers_range,
            'fwhm_min': b_fit * detector.fwhm_min,
            'fwhm_max': b_fit * detector.fwhm_max}]
        result = fit.process(
            NXdata(NXfield(mean_data[mask], 'y'), NXfield(bins[mask], 'x')),
            {'parameters': parameters, 'models': models, 'method': 'trf'})
        fit = FitProcessor(**self.run_config)
        result = fit.process(
            NXdata(NXfield(mean_data[mask], 'y'),
            NXfield(energies[mask], 'x')),
            {'models': models, 'method': 'trf'})
        e_xrf_unconstrained = np.sort(
            [result.best_values[f'peak{i+1}_center']
             for i in range(num_xrf)])
        e_bragg_unconstrained = np.sort(
            [result.best_values[f'peak{i+1}_center']
             for i in range(num_xrf, num_xrf+num_bragg)])

        # Update the peak energies with the newly calibrated tth
        e_bragg = get_peak_locations(ds, tth_fit)

        return {
            'best_fit_unconstrained': result.best_fit,
            'residual_unconstrained': result.residual,
            'e_xrf_unconstrained': e_xrf_unconstrained,
            'e_bragg_unconstrained': e_bragg_unconstrained,
            'strains_unconstrained': np.log(e_bragg / e_bragg_unconstrained),
            'tth': tth_fit,
            'energy_calibration_coeffs': [
                 float(a_fit), float(b_fit), float(c_fit)],
        }

    def _select_tth_init(self):
        """Select the initial 2&theta guess from the mean MCA
        spectrum.
        """
        # Local modules
        from CHAP.edd.utils import (
            get_unique_hkls_ds,
            select_tth_initial_guess,
        )

        for energies, mean_data, detector in zip(
                self._energies, self._mean_data,
                self.detector_config.detectors):

            # Get the unique HKLs and lattice spacings for the tth
            # calibration
            hkls, ds = get_unique_hkls_ds(
                self.config.materials, tth_max=detector.tth_max,
                tth_tol=detector.tth_tol)

            detector.tth_initial_guess, buf = select_tth_initial_guess(
                energies, mean_data, hkls, ds, detector.tth_initial_guess,
                detector.get_id(), self.interactive, self.save_figures)
            if self.save_figures:
                self._figures.append((
                    buf, f'{detector.get_id()}_tth_calibration_initial_guess'))
            self.logger.debug(
                f'tth_initial_guess for detector {detector.get_id()}: '
                f'{detector.tth_initial_guess}')


class ReducedDataProcessor(BaseStrainProcessor):
    """Processor that takes a map of MCA data and returns a map of
    reduced data.
    """
    lower_cutoff: Optional[confloat(ge=0, allow_inf_nan=False)] = 25
    upper_cutoff: Optional[confloat(gt=0, allow_inf_nan=False)] = 200

    def process(self, data):
        """Map the raw data onto the tth/energy calibration axes.

        :param data: Input data containing configurations for a map and
            a completed energy/tth calibration.
        :type data: list[PipelineData]
        :return: The reduced data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXentry,
            NXfield,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.utils.general import nxcopy

        # Load the detector data
        nxentry = self.get_default_nxentry(self.get_data(data))

        # Load the validated calibration configuration
        calibration_config = self.get_data(
            data, schema='edd.models.MCATthCalibrationConfig')
        calibration_detectors = [
            MCADetectorCalibration(**d) 
            for d in calibration_config.get('detectors', [])]
        calibration_config = MCATthCalibrationConfig(**calibration_config)

        # Create the NXroot object and add a NXprocess to store the
        # reduced data
        nxroot = NXroot()
        nxroot[nxentry.nxname] = nxentry
        nxroot[f'{nxentry.nxname}_reduced_data'] = NXprocess()
        nxprocess = nxroot[f'{nxentry.nxname}_reduced_data']
        nxprocess.calibration_config = calibration_config.model_dump_json()
        nxdata = nxroot[nxentry.nxname][nxentry.default]

        # Check for available calibration data and compute the detector
        # bin energies
        for detector in calibration_detectors:
            detector_id = detector.get_id()
            if detector_id not in nxdata:
                self.logger.warning(
                    f'Skipping detector {detector_id} (no raw data)')
                continue
            num_bins = nxdata[detector_id].shape[-1]
            if detector.num_bins is None:
                detector.num_bins = num_bins
            elif detector.num_bins != num_bins:
                raise ValueError(
                    'Inconsistent number of MCA detector channels between '
                    'the raw data and the detector configuration for '
                    f'detector {detector_id} ({num_bins} vs '
                    f'{detector.num_bins})')
            if detector.energy_calibration_coeffs is None:
                raise ValueError('Missing energy calibration coefficients for '
                                 f'detector {detector_id}')

            # Setup the NXdetector object for the current detector
            self.logger.debug(
                f'Setting up NXdetector group for {detector_id}')
            nxdetector = NXdetector()
            nxprocess[detector_id] = nxdetector
            nxdetector.local_name = detector_id
            nxdetector.detector_config = detector.model_dump_json()

            # Get the energy mask
            energies = detector.energies
            energy_mask = np.where(energies >= self.lower_cutoff, 1, 0)
            energy_mask = np.where(
                energies <= self.upper_cutoff, energy_mask, 0)
            # Also blank out the last channel, which has shown to be
            # troublesome
            energy_mask[-1] = 0

            # Add the channel and detector data
            nxdetector.data = NXdata()
            det_nxdata = nxdetector.data
            det_nxdata.set_default()
            det_nxdata.intensity = NXfield(
                value=nxdata[detector_id].nxdata * energy_mask.astype(
                    nxdata[detector_id].dtype),
                attrs={'units': 'counts'})
            self._linkdims(det_nxdata, nxdata)
            if 'axes' in det_nxdata.attrs:
                if isinstance(det_nxdata.attrs['axes'], str):
                    det_nxdata.attrs['axes'] = [
                        det_nxdata.attrs['axes'], 'energy']
                else:
                    det_nxdata.attrs['axes'].append('energy')
            else:
                det_nxdata.attrs['axes'] = ['energy']
            det_nxdata.energy = NXfield(
                value=energies, attrs={'units': 'keV'})
            det_nxdata.attrs['signal'] = 'intensity'

        return nxroot

    def _linkdims(
            self, nxgroup, nxdata_source, add_field_dims=None,
            skip_field_dims=None, oversampling_axis=None):
        """Link the dimensions for a 'nexusformat.nexus.NXgroup`
        object.
        """
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
                exit('FIX replace in both axis and unstructured_axes')
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


class StrainAnalysisProcessor(BaseStrainProcessor):
    """Processor that takes a map of MCA data and returns a map of
    sample strains.

    :ivar config: Initialization parameters for an instance of
        CHAP.edd.models.StrainAnalysisConfig.
    :type config: dict, optional
    :ivar detector_config: Initialization parameters for an instance of
        CHAP.edd.models.MCADetectorConfig. Defaults to the detector
        configuration of the raw detector data merged with that of the
        2&theta calibration step..
    :ivar save_figures: Save .pngs of plots for checking inputs &
        outputs of this Processor, defaults to `False`.
    :type save_figures: bool, optional
    :ivar setup: Setup the strain analysis
        `nexusformat.nexus.NXroot` object, defaults to `True`.
    :type setup: bool, optional
    :ivar update: Perform the strain analysis and return the
        results as a list of updated points or update the result
        from the `setup` stage, defaults to `True`.
    :type update: bool, optional
    """
    pipeline_fields: dict = Field(
        default = {
            'config': 'edd.models.StrainAnalysisConfig',
            'detector_config': {
                'schema': 'edd.models.MCADetectorConfig',
                'merge_key_paths': {'key_path': 'detectors/id', 'type': int}},
        },
        init_var=True)
    config: Optional[StrainAnalysisConfig] = StrainAnalysisConfig()
    detector_config: MCADetectorConfig
    save_figures: Optional[bool] = False
    setup: Optional[bool] = True
    update: Optional[bool] = True

    @model_validator(mode='before')
    @classmethod
    def validate_strainanalysisprocessor_before(cls, data):
        if isinstance(data, dict):
            detector_config = data.pop('detector_config', {})
            detector_config['processor_type'] = 'strainanalysis'
            data['detector_config'] = detector_config
        return data

    @staticmethod
    def add_points(nxroot, points, logger=None):
        """Add or update the strain analysis for a set of map points 
        in a `nexusformat.nexus.NXroot` object.

        :param nxroot: The strain analysis object to add/update the
            points to.
        :type nxroot: nexusformat.nexus.NXroot
        :param points: The strain analysis results for a set of points.
        :type points: list[dict[str, object]
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXdetector,
            NXprocess,
        )
        # pylint: enable=no-name-in-module

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

        if axes:
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

    def process(self, data):
        """Setup the strain analysis and/or return the strain analysis
        results as a list of updated points or a
        `nexusformat.nexus.NXroot` object.

        :param data: Input data containing configurations for a map,
            completed energy/tth calibration, and (optionally)
            parameters for the strain analysis.
        :type data: list[PipelineData]
        :raises RuntimeError: Unable to get a valid strain analysis
            configuration.
        :return: The strain analysis setup or results, a list of
            byte stream representions of Matplotlib figures and an
            animation of the fit results.
        :rtype: Union[list[dict[str, object]],
            nexusformat.nexus.NXroot], PipelineData, PipelineData
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        # Local modules
        from CHAP.utils.general import list_to_string

        if not (self.setup or self.update):
            raise RuntimeError('Illegal combination of setup and update')
        if not self.update:
            if self.interactive:
                self.logger.warning('Interactive option disabled during setup')
                self.interactive = False
            if self.save_figures:
                self.logger.warning(
                    'Saving figures option disabled during setup')
                self.save_figures = False
        self._animation = []

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

        # Load the detector data
        # FIX set rel_height_cutoff
        nxentry = self.get_default_nxentry(nxroot)
        nxdata = nxentry[nxentry.default]

        # Load the validated calibration configuration
        calibration_config = self.get_config(
            data, schema='edd.models.MCATthCalibrationConfig', remove=False)

        # Load the validated calibration detector configurations
        calibration_detector_config = self.get_data(
            data, schema='edd.models.MCATthCalibrationConfig')
        calibration_detectors = [
            MCADetectorCalibration(**d) 
            for d in calibration_detector_config.get('detectors', [])]
        calibration_detector_ids = [d.get_id() for d in calibration_detectors]

        # Check for available raw detector data and for the available 
        # calibration data
        if not self.detector_config.detectors:
            self.detector_config.detectors = [
                MCADetectorStrainAnalysis(
                    id=id_, processor_type='strainanalysis')
                for id_ in nxentry.detector_ids]
            self.detector_config.update_detectors()
        skipped_detectors = []
        sskipped_detectors = []
        detectors = []
        for detector in self.detector_config.detectors:
            detector_id = detector.get_id()
            if detector_id not in nxdata:
                skipped_detectors.append(detector_id)
            elif detector_id not in calibration_detector_ids:
                sskipped_detectors.append(detector_id)
            else:
                raw_detector_data = nxdata[detector_id].nxdata
                if raw_detector_data.ndim != 2:
                    self.logger.warning(
                        f'Skipping detector {detector_id} (Illegal data shape '
                        f'{raw_detector_data.shape})')
                elif raw_detector_data.sum():
                    for k, v in nxdata[detector_id].attrs.items():
                        detector.attrs[k] = v.nxdata
                    if self.config.rel_height_cutoff is not None:
                        detector.rel_height_cutoff = \
                            self.config.rel_height_cutoff
                    detector.add_calibration(
                        calibration_detectors[
                            int(calibration_detector_ids.index(detector_id))])
                    detectors.append(detector)
                else:
                    self.logger.warning(
                        f'Skipping detector {detector_id} (zero intensity)')
        if len(skipped_detectors) == 1:
            self.logger.warning(
                f'Skipping detector {skipped_detectors[0]} '
                '(no raw data)')
        elif skipped_detectors:
            skipped_detectors = [int(d) for d in skipped_detectors]
            self.logger.warning(
                'Skipping detectors '
                f'{list_to_string(skipped_detectors)} (no raw data)')
        if len(sskipped_detectors) == 1:
            self.logger.warning(
                f'Skipping detector {sskipped_detectors[0]} '
                '(no raw data)')
        elif sskipped_detectors:
            skipped_detectors = [int(d) for d in sskipped_detectors]
            self.logger.warning(
                'Skipping detectors '
                f'{list_to_string(skipped_detectors)} (no calibration data)')
        self.detector_config.detectors = detectors
        if not self.detector_config.detectors:
            raise ValueError('No valid data or unable to match an available '
                             'calibrated detector for the strain analysis')

        # Load the raw MCA data and compute the detector bin energies
        # and the mean spectra
        self._setup_detector_data(
            nxentry[nxentry.default],
            strain_analysis_config=self.config, update=self.update)

        # Apply the energy mask
        self._apply_energy_mask()

        # Get the mask and HKLs used in the strain analysis
        self._get_mask_hkls()

        # Apply the combined energy ranges mask
        self._apply_combined_mask()

        # Setup and/or run the strain analysis
        points = []
        if self.update:
            points = self._strain_analysis()
        if self.setup:
            nxroot = self._get_nxroot(nxentry, calibration_config)
            if points:
                self.logger.info(f'Adding {len(points)} points')
                self.add_points(nxroot, points, logger=self.logger)
                self.logger.info(f'... done')
            else:
                self.logger.warning('Skip adding points')
            if not (self._figures or self._animation):
                return nxroot
            ret = [nxroot]
        else:
            if not (self._figures or self._animation):
                return points
            ret = [points]
        if self._figures:
            ret.append(
                PipelineData(
                    name=self.__name__, data=self._figures,
                    schema='common.write.ImageWriter'))
        if self._animation:
            ret.append(
                PipelineData(
                    name=self.__name__, data=self._animation,
                    schema='common.write.ImageWriter'))
        return tuple(ret)

    def _add_fit_nxcollection(self, nxdetector, fit_type, hkls):
        """Add the fit collection as a `nexusformat.nexus.NXcollection`
        object.
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXfield,
            NXparameters,
        )
        # pylint: enable=no-name-in-module

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
            if self.save_figures:
                self._figures.append((
                    fig_to_iobuf(fig),
                    os.path.join(path, f'frame_{str(i).zfill(num_digit)}')))
            return intensity, best_fit, index

        if self.save_figures:
            start_index = len(self._figures)
            path = f'{detector_id}_strainanalysis_unconstrained_fits'
        else:
            start_index = 0

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
            title='Unconstrained Fits',
            xlabel='Energy (keV)',
            ylabel='Normalized Intensity (-)')
        ax.legend(loc='upper right')
        ax.set_ylim(-0.05, 1.05)
        index = ax.text(
            0.05, 0.95, '', transform=ax.transAxes, va='top')

        num_frame = intensities.size // intensities.shape[-1]
        num_digit = len(str(num_frame))
        if not self.save_figures:
            ani = animation.FuncAnimation(
                fig, animate, frames=num_frame, interval=1000, blit=False,
                repeat=False)
        else:
            for i in range(num_frame):
                animate(i)

            plt.close()
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

            frames = []
            for (buf, _), _ in self._figures[start_index:]:
                buf.seek(0)
                frame = plt.imread(buf)
                im = plt.imshow(frame, animated=True)
                if not i:
                    plt.imshow(frame)
                frames.append([im])

            ani = animation.ArtistAnimation(
                 plt.gcf(), frames, interval=1000, blit=False,
                 repeat=False)

        if self.interactive:
            plt.show()

        if self.save_figures:
            self._animation.append((
                (ani, 'gif'),
                f'{detector_id}_strainanalysis_unconstrained_fits'))
        plt.close()

    def _get_nxroot(self, nxentry, calibration_config):
        """Return a `nexusformat.nexus.NXroot` object initialized for
        the stress analysis.

        :param nxentry: Strain analysis map, including the raw
            MCA data.
        :type nxentry: nexusformat.nexus.NXentry
        :param calibration_config: 2&theta calibration configuration.
        :type calibration_config:
            CHAP.edd.models.MCATthCalibrationConfig
        :return: Strain analysis results & associated metadata..
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXdetector,
            NXfield,
            NXprocess,
            NXroot,
        )
        # pylint: enable=no-name-in-module

        # Local modules
        from CHAP.edd.utils import get_unique_hkls_ds
        from CHAP.utils.general import nxcopy

        if not self.interactive and not self.config.materials:
            raise ValueError(
                'No material provided. Provide a material in the '
                'StrainAnalysis Configuration, or re-run the pipeline with '
                'the --interactive flag.')

        # Create the NXroot object
        nxroot = NXroot()
        nxroot[nxentry.nxname] = nxentry
        nxroot[f'{nxentry.nxname}_strainanalysis'] = NXprocess()
        nxprocess = nxroot[f'{nxentry.nxname}_strainanalysis']
        nxprocess.calibration_config = \
            calibration_config.model_dump_json()
        nxprocess.strain_analysis_config = \
            self.config.model_dump_json()

        # Loop over the detectors to fill in the nxprocess
        for energies, mask, nxdata, detector in zip(
                self._energies, self._masks, self._nxdata_detectors,
                self.detector_config.detectors):

            # Get the current data object
            data = nxdata.nxsignal
            num_points = data.shape[0]

            # Setup the NXdetector object for the current detector
            self.logger.debug(
                f'Setting up NXdetector group for {detector.get_id()}')
            nxdetector = NXdetector()
            nxprocess[detector.get_id()] = nxdetector
            nxdetector.local_name = detector.get_id()
            nxdetector.detector_config = detector.model_dump_json()
            nxdetector.data = nxcopy(nxdata, exclude_nxpaths='detector_data')
            det_nxdata = nxdetector.data
            det_nxdata.set_default()
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
            det_nxdata.uniform_strain = NXfield(
                dtype=np.float64,
                shape=(num_points,),
                attrs={'long_name': 'Strain from uniform fit (\u03BC\u03B5)'})
            det_nxdata.unconstrained_strain = NXfield(
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
                self.config.materials, tth_max=detector.tth_max,
                tth_tol=detector.tth_tol)

            # Get the HKLs and lattice spacings that will be used for
            # fitting
            hkls_fit = np.asarray([hkls[i] for i in detector.hkl_indices])

            # Add the uniform fit nxcollection
            self._add_fit_nxcollection(nxdetector, 'uniform', hkls_fit)

            # Add the unconstrained fit nxcollection
            self._add_fit_nxcollection(nxdetector, 'unconstrained', hkls_fit)

            # Add the strain fields
            tth_map = detector.get_tth_map((num_points,))
            det_nxdata.tth.nxdata = tth_map

        return nxroot

    def _linkdims(
            self, nxgroup, nxdata_source, add_field_dims=None,
            skip_field_dims=None, oversampling_axis=None):
        """Link the dimensions for a 'nexusformat.nexus.NXgroup`
        object.
        """
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
                exit('FIX replace in both axis and unstructured_axes')
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

    def _strain_analysis(self):
        """Perform the strain analysis on the full or partial map."""
        # Local modules
        from CHAP.edd.utils import (
            get_peak_locations,
            get_spectra_fits,
            get_unique_hkls_ds,
        )

        # Get and subtract the detector baselines
        self._subtract_baselines()

        # Adjust the material properties
        _, buf = self._adjust_material_props(self.config.materials)
        if self.save_figures:
            self._figures.append((
                buf,
                f'{self.detector_config.detectors[0].get_id()}_'
                    'strainanalysis_material_config'))

        # Setup the points list with the map axes values
        nxdata_ref = self._nxdata_detectors[0]
        axes = get_axes(nxdata_ref)
        if axes:
            points = [
                {a: nxdata_ref[a].nxdata[i] for a in axes}
                for i in range(nxdata_ref[axes[0]].size)]
        else:
            points = [{}]

        # Loop over the detectors to fill in the nxprocess
        for energies, mask, mean_data, nxdata, detector in zip(
                self._energies, self._masks, self._mean_data,
                self._nxdata_detectors, self.detector_config.detectors):

            self.logger.debug(
                f'Beginning strain analysis for {detector.get_id()}')

            # Get the spectra for this detector
            intensities = nxdata.nxsignal.nxdata.T[mask].T

            # Get the unique HKLs and lattice spacings for the strain
            # analysis materials
            hkls, ds = get_unique_hkls_ds(
                self.config.materials, tth_max=detector.tth_max,
                tth_tol=detector.tth_tol)

            # Get the HKLs and lattice spacings that will be used for
            # fitting
            hkls_fit = np.asarray([hkls[i] for i in detector.hkl_indices])
            ds_fit = np.asarray([ds[i] for i in detector.hkl_indices])
            peak_locations = get_peak_locations(
                ds_fit, detector.tth_calibrated)

            # Find initial peak estimates
            if (not self.config.find_peaks
                    or detector.rel_height_cutoff is None):
                use_peaks = np.ones((peak_locations.size)).astype(bool)
            else:
                # Third party modules
                from scipy.signal import find_peaks as find_peaks_scipy

                peaks = find_peaks_scipy(
                    mean_data, width=5,
                    height=detector.rel_height_cutoff*mean_data.max())
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
                for _ in range(4):
                    for width, center in zip(widths, centers):
                        for n, loc in enumerate(peak_locations):
                            # FIX Hardwired range now, use detector.centers_range?
                            if center-width*delta < loc < center+width*delta:
                                use_peaks[n] = True
                                # peak_heights[n] = height
                                # peak_widths[n] = width*delta
                                break
                    if any(use_peaks):
                        break
                    delta *= 2
            if any(use_peaks):
                self.logger.debug(
                    f'Using peaks with centers at {peak_locations[use_peaks]}')
            else:
                self.logger.warning(
                    'No matching peaks with heights above the threshold, '
                    f'skipping the fit for detector {detector.get_id()}')
                return []
            hkls_fit = hkls_fit[use_peaks]

            # Perform the fit
            self.logger.info(f'Fitting detector {detector.get_id()} ...')
            uniform_results, unconstrained_results = get_spectra_fits(
                np.squeeze(intensities), energies[mask],
                peak_locations[use_peaks], detector,
                num_proc=self.config.num_proc, **self.run_config)
            if intensities.shape[0] == 1:
                uniform_results = {k: [v] for k, v in uniform_results.items()}
                unconstrained_results = {
                    k: [v] for k, v in unconstrained_results.items()}
                for field in ('centers', 'amplitudes', 'sigmas'):
                    uniform_results[field] = np.asarray(
                        uniform_results[field]).T
                    uniform_results[f'{field}_errors'] = np.asarray(
                        uniform_results[f'{field}_errors']).T
                    unconstrained_results[field] = np.asarray(
                        unconstrained_results[field]).T
                    unconstrained_results[f'{field}_errors'] = np.asarray(
                        unconstrained_results[f'{field}_errors']).T

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
                    f'{detector.get_id()}/data/intensity': intensities[i],
                    f'{detector.get_id()}/data/uniform_strain':
                        uniform_strain[i],
                    f'{detector.get_id()}/data/unconstrained_strain':
                        unconstrained_strain[i],
                    f'{detector.get_id()}/uniform_fit/results/best_fit':
                        uniform_results['best_fits'][i],
                    f'{detector.get_id()}/uniform_fit/results/residual':
                        uniform_results['residuals'][i],
                    f'{detector.get_id()}/uniform_fit/results/redchi':
                        uniform_results['redchis'][i],
                    f'{detector.get_id()}/uniform_fit/results/success':
                        uniform_results['success'][i],
                    f'{detector.get_id()}/unconstrained_fit/results/best_fit':
                        unconstrained_results['best_fits'][i],
                    f'{detector.get_id()}/unconstrained_fit/results/residual':
                        unconstrained_results['residuals'][i],
                    f'{detector.get_id()}/unconstrained_fit/results/redchi':
                        unconstrained_results['redchis'][i],
                    f'{detector.get_id()}/unconstrained_fit/results/success':
                        unconstrained_results['success'][i],
                })
                for j, hkl in enumerate(hkls_fit):
                    hkl_name = '_'.join(str(hkl)[1:-1].split(' '))
                    uniform_fit_path = \
                        f'{detector.get_id()}/uniform_fit/{hkl_name}'
                    unconstrained_fit_path = \
                        f'{detector.get_id()}/unconstrained_fit/{hkl_name}'
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
            if (not self.config.skip_animation
                    and (self.interactive or self.save_figures)):
                self._create_animation(
                    nxdata, energies[mask], intensities,
                    unconstrained_results['best_fits'], detector.get_id())

        return points


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
