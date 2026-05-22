#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module for Processors unique to the SAXSWAXS workflow.

Add discription of SAXS/WAXS
"""

# System modules
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
from pydantic import (
    conint,
    conlist,
    Field,
    FilePath,
)
import numpy as np

# Local modules
from CHAP.common.models.common import IndexSliceConfig
from CHAP.common.models.map import (
    DetectorConfig,
    Detector,
    MapConfig,
)
from CHAP.common.models.integration import PyfaiIntegrationConfig
from CHAP.common.processor import ExpressionProcessor
from CHAP.processor import Processor
from CHAP.pipeline import PipelineData
from CHAP.saxswaxs.models import (
    CorrectionsConfig,
    FluxCorrectionConfig,
    FluxAbsorptionCorrectionConfig,
    FluxAbsorptionBackgroundCorrectionConfig,
)
from CHAP.saxswaxs.utils import dict_to_zarr


class CfProcessor(Processor):
    """Processor to calculate the correction factor Cf that, when
    multiplied by appropriately processed SAXS/WAXS data, converts data
    to absolute cross-section / intensity in inverse cm.
    """

    def process(
            self, data, interactive=False, save_figures=True, nxpath=None,
            radial_range=None, scan_step_indices=None, eps=1.e-5):
        """Compute the correction factor Cf and the configuration
        parameters.

        :param data: Input data list containing the reference data
            labelled with `'reference_data'` as well as the NeXus
            input data with the azimuthally integrated SAXS/WAXS data.
        :type data: list[PipelineData]
        :param interactive: Allows for user interactions,
            defaults to `False`.
        :type interactive: bool, optional
        :param save_figures: Create Matplotlib correction factor
            image that can be saved to file downstream in the workflow,
            defaults to `True`.
        :type save_figures: bool, optional
        :param nxpath: Path to a specific NeXus Style
            `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html#index-0>`__
            object in the input NeXus file tree to the measured data
            from.
        :type nxpath: str, optional
        :param radial_range: q-range used to compute Cf.
        :type radial_range: list[float, float] or tuple[float, float]),
            optional
        :param scan_step_indices: Optional scan step indices to use for
            the calculation. If not specified, the correction factor
            will be computed on the average of all data for the scan.
        :type scan_step_indices: int, str, list[int], optional
        :param eps: Minimum plotting value of the corrected azimuthally
            integrated SAXS/WAXS data, defaults to `1.e-5`.
        :type eps: float, optional
        :returns: Computed correction factor Cf and the configuration
            parameters plus the optional correction factor image as a
            :class:`~CHAP.pipeline.PipelineData object`.
        :rtype: dict or (dict, PipelineData)]
        """
        # Third party modules
        # pylint: disable=import-error
        from pandas import DataFrame
        # pylint: enable=import-error
        from scipy.interpolate import interp1d

        # Local modules
        from CHAP.pipeline import PipelineData
        from CHAP.utils.general import round_to_n

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            # Local modules
            from CHAP.utils.general import fig_to_iobuf

        # Validate the input parameters
        if scan_step_indices is not None:
            if isinstance(scan_step_indices, int):
                scan_step_indices = [scan_step_indices]
            elif isinstance(scan_step_indices, str):
                # Local modules
                from CHAP.utils.general import string_to_list

                scan_step_indices = string_to_list(scan_step_indices)

        # Load the measured data
        if nxpath is None:
            try:
                nxdata = self.get_default_nxdata(self.get_data(data))
            except Exception as exc:
                raise ValueError(
                    'No valid default NXdata object in pipeline data') from exc
        else:
            # Third party modules
            from nexusformat.nexus import NXdata

            try:
                nxdata = self.get_data(data)[nxpath]
                assert isinstance(nxdata, NXdata)
            except Exception as exc:
                raise ValueError('No valid default NXdata object in pipeline '
                                 'data') from exc
        if scan_step_indices is not None and nxdata.nxsignal.ndim != 2:
            self.logger.warning(
                'Input parameters scan_step_indices for map with a number of '
                'independent dimensions other than 1')
            scan_step_indices = None
        if nxdata.nxsignal.ndim == 1:
            data_meas = nxdata.nxsignal.nxdata
        elif nxdata.nxsignal.ndim == 2 and scan_step_indices is not None:
            data_meas = \
                nxdata.nxsignal.nxdata[scan_step_indices,:].mean(axis=0)
        else:
            data_meas = nxdata.nxsignal.nxdata.mean(
                axis=tuple(range(nxdata.nxsignal.ndim-1)))
        q_meas = nxdata[nxdata.attrs['axes'][1]].nxdata

        # Load the reference data
        ddata = self.get_data(data, name='reference_data')
        try:
            assert isinstance(ddata, DataFrame)
            assert len(ddata.columns) in (2, 3)
            q_ref = ddata.values[:,0]
            data_ref = ddata.values[:,1]
        except Exception as exc:
            raise ValueError(
                'Invalid reference data format {type(ddata)}') from exc

        # Interpolate measured reference data onto reference q values
        mask = np.where(
            q_ref <= radial_range[1],
            np.where(q_ref >= radial_range[0], 1, 0), 0).astype(bool)
        if not q_ref[mask].size:
            raise ValueError(
                'No reference values within specified radial range')
        func = interp1d(q_meas, data_meas, kind='cubic')
        data_meas_intpol = func(q_ref[mask])
        if not data_meas_intpol.size:
            raise ValueError(
                'No measured values within specified radial range')

        # Get the correction factor
        ratio = data_ref[mask]/data_meas_intpol
        cf = ratio.mean()
        cf_stv = ratio.std()
        self.logger.info(f'correction factor Cf: {round_to_n(cf, 6):e} \u00B1 '
                         f'{round_to_n(100*cf_stv/cf, 2)}%')

        # Assemble result
        result = {
            'cf': float(cf),
            'error': float(cf_stv),
            'radial_range': radial_range,
        }

        # Plot the results
        data_meas *= cf
        figures = None
        if interactive or save_figures:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.plot(q_ref, data_ref, label='APS Reference Data')
            ax.plot(q_meas, data_meas, label=r'Corrected FMB Data/C$_f$')
            for v in radial_range:
                plt.axvline(v, color='r', linestyle='--')
            ax.set_yscale('log')
            ax.set_title('Absolute Intensity Calculation', fontsize='xx-large')
            ax.set_xlabel(r'{q_A^-1}', fontsize='x-large')
            ax.set_ylabel('Normalized Intensity', fontsize='x-large')
            min_x = q_meas.min()
            max_x = q_meas.max()
            delta_x = 0.1*(q_meas.max() - q_meas.min())
            ax.set_xlim((min_x-delta_x, max_x+delta_x))
            ax.set_ylim(
                (10**np.floor(np.log10(data_meas[data_meas>eps].min())),
                 10**np.floor(1+np.log10(data_meas.max()))))
            ax.legend(fontsize='x-large', edgecolor='grey')
            plt.annotate(
                r'C$_f$ = 'f'{round_to_n(cf, 6):e}'r' $\pm$ '
                f'{round_to_n(100*cf_stv/cf, 2)}%\n'
                r'1/C$_f$ = 'f'{round_to_n(1/cf, 6):e}',
                (.65, .9), xycoords = 'axes fraction', fontsize='x-large',
                bbox={'facecolor': 'white', 'edgecolor': 'grey', 'pad': 10.0})
            if save_figures:
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                figures  = fig_to_iobuf(fig)
            if interactive:
                plt.show()
            plt.close()

        if figures is not None:
            return (
                result,
                PipelineData(name=self.__name__, data=figures,
                    schema='common.write.ImageWriter'))
        return result


class FluxCorrectionProcessor(ExpressionProcessor):
    """Processor for applying a flux correction to azimuthally
    integrated SAXS/WAXS intensity data.

    Normalises the measured intensity by the pre-sample beam monitor
    counts, referenced to a fixed counting rate configured via
    :attr:`config`.

    :ivar config: Correction configuration.
    :vartype config: FluxCorrectionConfig
    """
    config: FluxCorrectionConfig
    def process(self, data, nxprocess=False):
        """Compute the flux-corrected intensity.

        Requires input data items named ``'intensity'`` (the raw
        integrated signal) and ``'presample_intensity'`` (beam monitor
        counts).  When
        :attr:`~saxswaxs.models.FluxCorrectionConfig.presample_intensity_reference_rate`
        is not pre-configured in :attr:`config`, an additional item
        named ``'dwell_time_actual'`` is also required and the
        reference rate is computed on the fly as
        ``np.nanmean(presample_intensity / dwell_time_actual)``.

        :param data: Input data list.
        :type data: list[PipelineData]
        :param nxprocess: Return the result as a NeXus
            `NXobject <https://manual.nexusformat.org/classes/base_classes/NXobject.html#index-0>`__
            object.  Defaults to ``False``.
        :type nxprocess: bool, optional
        :returns: Flux-corrected intensity array (or NXobject when
            ``nxprocess`` is ``True``).
        :rtype: numpy.ndarray or nexusformat.nexus.NXobject
        """
        if self.config.presample_intensity_reference_rate is None:
            self.config.presample_intensity_reference_rate = super().process(
                data,
                'np.nanmean(presample_intensity / dwell_time_actual)'
            )
        presample_intensity = self.get_data(
            data, name='presample_intensity',
        )
        intensity = self.get_data(
            data, name=self.config.uncorrected_data_name,
        )
        # nxfieldtable = {
        #     'intensity': intensity,
        #     'presample_intensity': presample_intensity,
        #     'presample_intensity_reference_rate': NXfield(
        #         name='presample_intensity_reference_rate',
        #         values=presample_intensity_reference_rate
        #     )
        # }

        # Extend presample_intensity along last dim to have same shape
        # as intensity
        for dim in intensity.shape[presample_intensity.ndim:]:
            presample_intensity = np.repeat(
                np.expand_dims(presample_intensity, axis=-1), dim, axis=-1
            )
        symtable = {
            'presample_intensity_reference_rate':
                self.config.presample_intensity_reference_rate,
            'intensity': intensity,
            'presample_intensity': presample_intensity
        }
        expression = (
            'intensity *'
            '(presample_intensity_reference_rate / presample_intensity)'
        )
        return super().process(
            data, expression, symtable=symtable, nxprocess=nxprocess,
            nxfieldtable={}
        )


class FluxAbsorptionCorrectionProcessor(ExpressionProcessor):
    """Processor for applying combined flux and absorption correction
    to azimuthally integrated SAXS/WAXS intensity data.

    In addition to flux normalisation (see
    :class:`FluxCorrectionProcessor`), divides by the sample
    transmission, which is estimated from the ratio of post-sample to
    pre-sample intensities for both the sample and a background scan.

    :ivar config: Correction configuration.
    :vartype config: FluxAbsorptionCorrectionConfig
    """
    config: FluxAbsorptionCorrectionConfig
    def process(self, data, nxprocess=False):
        """Compute the flux- and absorption-corrected intensity.

        Requires input data items named ``'intensity'``,
        ``'presample_intensity'``, ``'postsample_intensity'``,
        ``'background_presample_intensity'``, and
        ``'background_postsample_intensity'``.  When
        :attr:`~saxswaxs.models.FluxAbsorptionCorrectionConfig.presample_intensity_reference_rate`
        is not pre-configured in :attr:`config`, an additional item
        named ``'dwell_time_actual'`` is also required and the
        reference rate is computed on the fly as
        ``np.nanmean(presample_intensity / dwell_time_actual)``.

        :param data: Input data list.
        :type data: list[PipelineData]
        :param nxprocess: Return the result as a NeXus
            `NXobject <https://manual.nexusformat.org/classes/base_classes/NXobject.html#index-0>`__
            object.  Defaults to ``False``.
        :type nxprocess: bool, optional
        :returns: Flux- and absorption-corrected intensity array (or
            NXobject when ``nxprocess`` is ``True``).
        :rtype: numpy.ndarray or nexusformat.nexus.NXobject
        """
        intensity = self.get_data(
            data, name=self.config.uncorrected_data_name, #'intensity',
        )

        if self.config.presample_intensity_reference_rate is None:
            self.config.presample_intensity_reference_rate = super().process(
                data,
                'np.nanmean(presample_intensity / dwell_time_actual)'
            )

        tt = super().process(
            data,
            ('np.divide(postsample_intensity, presample_intensity) '
             '/ np.average('
             'np.divide(background_postsample_intensity, background_presample_intensity))')
        )
        # Extend tt along last dim to have same shape as intensity
        for dim in intensity.shape[tt.ndim:]:
            tt = np.repeat(np.expand_dims(tt, axis=-1), dim, axis=-1)

        presample_intensity = self.get_data(
            data, name='presample_intensity',
        )
        # Extend presample_intensity along last dim to have same shape
        # as intensity
        for dim in intensity.shape[presample_intensity.ndim:]:
            presample_intensity = np.repeat(
                np.expand_dims(presample_intensity, axis=-1), dim, axis=-1
            )

        symtable = {
            'presample_intensity_reference_rate':
                self.config.presample_intensity_reference_rate,
            'intensity': intensity,
            'presample_intensity': presample_intensity,
            'tt': tt
        }
        expression = (
            '(1 / tt)'
            '* intensity'
            '* (presample_intensity_reference_rate / presample_intensity)'
        )
        return super().process(
            data, expression, symtable=symtable, nxprocess=nxprocess,
            nxfieldtable={}
        )


class FluxAbsorptionBackgroundCorrectionProcessor(ExpressionProcessor):
    """Processor for applying combined flux, absorption, and
    background-subtraction correction to azimuthally integrated
    SAXS/WAXS intensity data, with optional thickness normalisation.

    Extends :class:`FluxAbsorptionCorrectionProcessor` by subtracting
    an integrated background signal after flux and absorption
    correction.  Thickness normalisation is applied when
    :attr:`~saxswaxs.models.FluxAbsorptionBackgroundCorrectionConfig.sample_thickness_cm`
    or
    :attr:`~saxswaxs.models.FluxAbsorptionBackgroundCorrectionConfig.sample_mu_inv_cm`
    is set in :attr:`config`.

    :ivar config: Correction configuration.
    :vartype config: FluxAbsorptionBackgroundCorrectionConfig
    """
    config: FluxAbsorptionBackgroundCorrectionConfig
    def process(self, data, nxprocess=False):
        """Compute the flux-, absorption-, and background-corrected
        intensity, with optional thickness normalisation.

        Requires input data items named ``'intensity'``,
        ``'presample_intensity'``, ``'postsample_intensity'``,
        ``'background_presample_intensity'``,
        ``'background_postsample_intensity'``, and
        ``'background_intensity'``.  When
        :attr:`~saxswaxs.models.FluxAbsorptionBackgroundCorrectionConfig.presample_intensity_reference_rate`
        is not pre-configured in :attr:`config`, an additional item
        named ``'dwell_time_actual'`` is also required and the
        reference rate is computed on the fly as
        ``np.nanmean(presample_intensity / dwell_time_actual)``.

        Thickness normalisation is controlled by :attr:`config`: if
        ``config.sample_thickness_cm`` is set the corrected signal is
        divided by that value; if ``config.sample_mu_inv_cm`` is set
        the effective thickness is derived from the measured
        transmission; otherwise no thickness normalisation is applied.

        :param data: Input data list.
        :type data: list[PipelineData]
        :param nxprocess: Return the result as a NeXus
            `NXobject <https://manual.nexusformat.org/classes/base_classes/NXobject.html#index-0>`__
            object.  Defaults to ``False``.
        :type nxprocess: bool, optional
        :returns: Flux-, absorption-, and background-corrected
            intensity array (or NXobject when ``nxprocess`` is
            ``True``).
        :rtype: numpy.ndarray or nexusformat.nexus.NXobject
        """
        intensity = self.get_data(
            data, name=self.config.uncorrected_data_name,
        )

        if self.config.presample_intensity_reference_rate is None:
            self.config.presample_intensity_reference_rate = super().process(
                data,
                'np.nanmean(presample_intensity / dwell_time_actual)'
            )

        tt = super().process(
            data,
            ('np.divide(postsample_intensity, presample_intensity) '
             '/ np.average('
             'np.divide(background_postsample_intensity, background_presample_intensity))')
        )
        # Extend tt along last dim to have same shape as intensity
        for dim in intensity.shape[tt.ndim:]:
            tt = np.repeat(np.expand_dims(tt, axis=-1), dim, axis=-1)

        presample_intensity = self.get_data(
            data, name='presample_intensity',
        )
        # Extend presample_intensity along last dim to have same shape
        # as intensity
        for dim in intensity.shape[presample_intensity.ndim:]:
            presample_intensity = np.repeat(
                np.expand_dims(presample_intensity, axis=-1), dim, axis=-1
            )

        # Broadcast background intensity signal to shape of measured
        # intensity signal
        background_intensity = self.get_data(
            data, name='background_intensity',
        )
        background_intensity = np.broadcast_to(
            background_intensity, intensity.shape)

        if self.config.sample_thickness_cm is not None:
            t = self.config.sample_thickness_cm
        elif self.config.sample_mu_inv_cm is not None:
            t = -np.log(tt / self.config.sample_mu_inv_cm)
        else:
            t = 1

        symtable = {
            't': t,
            'presample_intensity_reference_rate':
                self.config.presample_intensity_reference_rate,
            'intensity': intensity,
            'presample_intensity': presample_intensity,
            'tt': tt,
            'background_intensity': background_intensity
        }
        expression = (
            '(1 / t)'
            '* ('
            '(1 / tt)'
            '* intensity'
            '* (presample_intensity_reference_rate / presample_intensity)'
            ') - ('
            'background_intensity'
            '* (presample_intensity_reference_rate / np.average(background_presample_intensity))'
            ')'
        )
        return super().process(
            data, expression, symtable=symtable, nxprocess=nxprocess,
            nxfieldtable={}
        )


class PyfaiIntegrationProcessor(Processor):
    """A processor for azimuthally integrating images.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.common.models.integration.PyfaiIntegrationConfig`.
    :vartype config: dict, optional
    """

    pipeline_fields: dict = Field(
        default={
            'config': 'common.models.integration.PyfaiIntegrationConfig'
        },
        init_var=True)
    config: PyfaiIntegrationConfig

    def process(self, data):
        """Perform a set of integrations on 2D detector data.

        :param data: Input data.
        :type data: list[PipelineData]
        :param idx_slices: List of dictionaries identifying the sliced
            index at which the output data should be written in a
            dataset, defaults to `[{'start':0, 'step': 1}]`.
        :type idx_slices: list[dict[str, int]], optional
        :return: Integrated detector data ready for writing with
            :class:`~CHAP.saxswaxs.ZarrResultsWriter` or
            :class:`~CHAP.saxswaxs.NexusResultsWriter`.
        :rtype: list[dict[str, Any]]
        """
        # System modules
        import time

        # Organize input for integrations
        input_data = {d['name']: d['data']
            for d in [d for d in data if isinstance(d['data'], np.ndarray)]}
        ais = {ai.get_id(): ai for ai in self.config.azimuthal_integrators}

        # Perform integration(s), package results for ZarrResultsWriter
        results = []
        nframes = len(input_data[list(input_data.keys())[0]])
        for integration in self.config.integrations:
            t0 = time.time()
            self.logger.info(f'Integrating {integration.name}...')
            result = integration.integrate(ais, input_data)
            tf = time.time()
            self.logger.debug(
                f'Integrated {integration.name} '
                f'({nframes/(tf-t0):.3f} frames/sec)')
            results.extend(
                [
                    {
                        'path': f'{integration.name}/data/I',
                        'data': np.asarray(result['intensities']),
                    },
                ]
            )
        return results


class SetupProcessor(Processor):
    """Convenience Processor for setting up a container for SAXS/WAXS
    experiments.

    :ivar map_config: Map configuration.
    :vartype map_config: dict, optional
        :class:`~CHAP.common.models.integration.PyfaiIntegrationConfig`.
    :ivar pyfai_config: Initialization parameters for an instance of
        :class:`~CHAP.common.models.integration.PyfaiIntegrationConfig`.
    :vartype pyfai_config: dict, optional
    :ivar detector_config: Detector configurations.
    :vartype detector_config: DetectorConfig
    :ivar dataset_shape: Shape of the completed dataset that will be
        processed later on (shape of the measurement itself, _not_
        including the dimensions of any signals collected at each point
        in that measurement). Defaults to `[0]`.
    :vartype dataset_shape: int or list[int], optional
    :ivar dataset_chunks: Extent of chunks along each dimension of the
        completed dataset / measurement. Choose this according to how
        you will process your data -- for example, if your
        `dataset_shape` is `[m, n]`, and you are planning to process
        each of the `m` rows as chunks, `dataset_chunks` should be
        `[1, n]`. But if you plan to process each of the `n` columns as
        chunks, `dataset_chunks` should be `[m, 1]`,
        defaults to `"auto"`.
    :vartype dataset_chunks: list[int] or Literal["auto"], optional
    :ivar num_chunk: Used only if `dataset_chunks` is `"auto"`.
        Preferred number of chunks in the dataset, defaults to `1`.
    :vartype num_chunk: int, optional
    :ivar raw_data: Flag to indicate wether or not space for raw
        detector data should be included in the returned
        `Zarr group <https://zarr.readthedocs.io/en/latest/api/zarr/group/#zarr.Group>`__,
        defaults to `True`.
    :vartype raw_data: bool, optional
    """

    pipeline_fields: dict = Field(
        default={
            'map_config': 'common.models.map.MapConfig',
            'detector_config': 'common.models.map.DetectorConfig',
            'pyfai_config': 'common.models.integration.PyfaiIntegrationConfig',
            'correction_config': 'saxswaxs.models.CorrectionsConfig',
        },
        init_var=True)
    # map_config needs a default value because the map configuration
    # may be created by an earlier item in the Pipeline. If this is
    # the case, then map_config needs SOME value in order for the
    # Pipeline to pass validation.
    map_config: MapConfig = None
    pyfai_config: PyfaiIntegrationConfig
    detector_config: DetectorConfig = DetectorConfig(detectors=[])
    correction_config: CorrectionsConfig
    dataset_shape: Optional[
        conlist(item_type=conint(ge=0), min_length=1)] = [0]
    dataset_chunks: Optional[
        Union[
            Literal['auto'],
            conlist(item_type=conint(gt=0), min_length=1)
        ]] = 'auto'
    num_chunk: Optional[conint(gt=0)] = 1
    raw_data: Optional[bool] = True

    def process(self, data):
        """Set up a container for SAXS/WAXS experiments.

        :param data: Input data.
        :type data: list[PipelineData]
        :return:
            `Zarr group <https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group>`__
            object representing a Zarr tree of the container contents.
        :rtype: zarr.Group
        """
        # System modules
        import asyncio
        import logging

        # Third party modules
        # pylint: disable=import-error
        import zarr
        from zarr.core.buffer import default_buffer_prototype
        from zarr.storage import MemoryStore
        # pylint: enable=import-error

        # Local modules
        from CHAP.common.processor import (
            MapProcessor,
            NexusToZarrProcessor,
        )
        from CHAP.pipeline import PipelineData

        # Get NXroot container for raw data map
        map_processor_kwargs = {
            'config': self.map_config,
            'detector_config': self.detector_config,
            'remove_constant_dims': False,
        }
        if not self.raw_data:
            self.map_config.spec_scans[0].scan_numbers = []

        setup_map_processor = self.setup_pipelineitem(
            MapProcessor(**map_processor_kwargs)
        )
        ddata = [
            PipelineData(
                data=setup_map_processor.process(
                    data=data, fill_data=False),
            )
        ]

        if self.dataset_shape is None:
            try:
                nxroot = self.get_data(ddata, remove=False)
                nxentry = self.get_default_nxentry(nxroot)
                nxdata = nxentry[nxentry.default]
                for detector in self.detectors:
                    if self.dataset_shape is None:
                        self.dataset_shape = nxdata[
                            detector.get_id()].shape[:-2]
                    else:
                        assert (
                            self.dataset_shape == nxdata[
                                detector.get_id()].shape[:-2]
                        )
            except Exception as exc:
                raise ValueError(
                    'Unable to get consistent dataset shape from map') from exc
        if self.dataset_chunks == 'auto':
            self.dataset_chunks = self.dataset_shape[0]//self.num_chunk
            if self.num_chunk*self.dataset_chunks < self.dataset_shape[0]:
                self.dataset_chunks += 1
            self.dataset_chunks = [self.dataset_chunks]

        # Convert raw data map container to Zarr format
        ddata_converter = self.setup_pipelineitem(NexusToZarrProcessor())
        zarr_map = ddata_converter.process(ddata, chunks=self.dataset_chunks)

        # Get paths to independent_dimension arrays
        dim_paths = [
            f'{self.map_config.title}/independent_dimensions/{dim.label}'
            for dim in self.map_config.independent_dimensions
        ]

        # Get Zarr container for pyfai integrations
        zarr_pyfai_tree = self.pyfai_config.zarr_tree(
            self.dataset_shape, self.dataset_chunks,
            nxlinks=dim_paths
        )
        zarr_pyfai = dict_to_zarr(zarr_pyfai_tree, logger=self.logger)


        # Get zarr container for corrected datasets
        integration_shapes = {
            integration.name: integration.result_shape
            for integration in self.pyfai_config.integrations
        }
        intg_by_name = {
            intg.name: intg for intg in self.pyfai_config.integrations
        }
        corr_nxlinks = {
            corr.name: (
                dim_paths
                + [f'{corr.uncorrected_data_name}/data/I']
                + [
                    f'{corr.uncorrected_data_name}/data/{coord}'
                    for coord in intg_by_name[
                        corr.uncorrected_data_name].result_coords
                ]
            )
            for corr in self.correction_config.corrections
        }
        zarr_corr = dict_to_zarr(
            self.correction_config.zarr_tree(
                self.dataset_shape, self.dataset_chunks,
                integration_shapes,
                nxlinks=corr_nxlinks,
            ),
            logger=self.logger,
        )

        # For corrections that include a background, read and integrate
        # that background data and store it in zarr_corr
        for corr_cfg in self.correction_config.corrections:
            if corr_cfg.background is None:
                continue
            if not self.detector_config.detectors:
                self.logger.warning(
                    f'No detectors configured; skipping background '
                    f'processing for correction "{corr_cfg.name}"')
                continue
            # Read in background raw detector data
            self.logger.info(
                f'Reading background data for correction "{corr_cfg.name}"')
            idx_slice = corr_cfg.background.idx_slice
            bg_det_images = {
                d.get_id(): [] for d in self.detector_config.detectors}
            for scan_number in corr_cfg.background.scan_numbers:
                scanparser = corr_cfg.background.get_scanparser(scan_number)
                npts = int(scanparser.spec_scan_npts)
                slice_stop = (
                    min(idx_slice._slice.stop, npts)
                    if idx_slice._slice.stop > 0 else npts
                )
                scan_indices = range(npts)[
                    slice(idx_slice._slice.start, slice_stop,
                          idx_slice._slice.step)]
                for i in scan_indices:
                    for det in self.detector_config.detectors:
                        bg_det_images[det.get_id()].append(
                            scanparser.get_detector_data(det.get_id(), i))
            # Average all background data to a single frame before
            # processing with appropriate integration
            for det_id in bg_det_images:
                bg_det_images[det_id] = np.mean(
                    bg_det_images[det_id], axis=0, keepdims=True)
            self.logger.info(
                f'Integrating background data for correction '
                f'"{corr_cfg.name}"')
            bg_pyfai_input = [
                PipelineData(name=det_id, data=imgs)
                for det_id, imgs in bg_det_images.items()
            ]
            bg_pyfai_config = self.pyfai_config.model_copy(
                update={'integrations': [
                    intg for intg in self.pyfai_config.integrations
                    if intg.name == corr_cfg.uncorrected_data_name
                ]}
            )
            bg_integrated = self.setup_pipelineitem(
                PyfaiIntegrationProcessor(config=bg_pyfai_config)
            ).process(bg_pyfai_input)[0]
            # Read background scalar data
            bg_presample = self.map_config.presample_intensity.get_value(
                corr_cfg.background, scan_number, -1,
                self.map_config.scalar_data
            )[idx_slice._slice]
            bg_postsample = self.map_config.postsample_intensity.get_value(
                corr_cfg.background, scan_number, -1,
                self.map_config.scalar_data
            )[idx_slice._slice]
            # Fill in placeholder zarr arrays with real background
            # data
            data_group = zarr_corr[corr_cfg.name]['data']
            data_group['I_background'][:] = np.squeeze(
                bg_integrated['data'], axis=0
            )
            bg_presample_arr = data_group.create_array(
                'background_presample_intensity',
                shape=bg_presample.shape, dtype=bg_presample.dtype)
            bg_presample_arr[:] = bg_presample
            bg_postsample_arr = data_group.create_array(
                'background_postsample_intensity',
                shape=bg_postsample.shape, dtype=bg_postsample.dtype)
            bg_postsample_arr[:] = bg_postsample

        # Assemble containers for raw & processed data
        zarr_root = zarr.create_group(store=MemoryStore({}))
        async def copy_zarr_store(source_store, dest_store):
            """Copy a Zarr <https://zarr.readthedocs.io/en/stable/>`__
            store.

            :param source_store: Source Zarr group.
            :type: zarr.Group
            :param dest_store: Destination Zarr group.
            :type: zarr.Group
            """
            async for k in source_store.list():
                self.logger.info(f'Copying {k}')
                buf = await source_store.get(
                    k, prototype=default_buffer_prototype())
                await dest_store.set(k, buf)
        for zarr_group in (zarr_map, zarr_pyfai, zarr_corr):
            asyncio.run(copy_zarr_store(zarr_group.store, zarr_root.store))
        return zarr_root

    def setup_pipelineitem(self, pipeline_item):
        """Convenience method to use a nice logger when this
        ``Processor`` calls on another ``PipelineItem``. Set the
        logger and logging handler for given pipeline item.

        :param pipeline_item: Pipeline item.
        :type pipeline_item: PipelineItem
        :return: Input Pipeline item, with updated logger and
            logging handler.
        :rtype: PipelineItem
        """
        import logging

        pipeline_item.logger = logging.getLogger(
            pipeline_item.__class__.__name__,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{asctime}: {name:20} (from '+ self.__class__.__name__
            + ') (L{lineno}): {levelname}: {message}',
            datefmt='%Y-%m-%d %H:%M:%S', style='{'))
        pipeline_item.logger.handlers = [handler]
        return pipeline_item


class UnstructuredToStructuredProcessor(Processor):
    """Processor to aggregate "unstructured" data into a single NeXus
    style
    `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html#index-0>`__
    object with a "structured" representation.
    """

    def process(self, data, fields, name='data', attrs=None):
        """Create and return a Nexus style
        `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html#index-0>`__
        object containing a single structured dataset composed from
        multiple unstructured input datasets.

        This method validates the field configuration, validates and
        reshapes the input data, determines common axes across all
        signals, and constructs a `NXdata` group containing signal and
        axis fields.

        :param data: Input data objects containing unstructured
            datasets.
        :type data: list[PipelineData]
        :param fields: Configuration describing how to structure the
            input data.  This is a list of dictionaries. Each
            dictionary must contain the required keys:

            - ``"name"``: Name of the data item, which must correspond
              to the ``name`` field of an item in ``data``.
            - ``"type"``: Either ``"signal"`` or ``"axis"``.
            - ``"axes"``: Required only for items where ``"type"`` is
              ``"signal"``. List of the names of the fields containing
              coordinate axes data for each dimension of the signal.

            Optional keys include:
            - ``"attrs"``: Dictionary of NeXus attributes to attach to.

        :type fields: list[dict[str, Any]]
        :param name: Name of the resulting `NXdata` group,
            defaults to `"data"`.
        :type name: str, optional
        :param attrs: Attributes to attach to the resulting `NXdata`
            group. The common axes determined during processing will be
            added to this dictionary under the ``"axes"`` key.
        :type attrs: dict[str, Any], optional
        :returns: A structured `NXdata` object containing all signals
            and axes defined by the configuration.
        :rtype: nexusformat.nexus.NXdata
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
        )

        signals, axes = self.validate_config_fields(fields)
        signals, axes = self.validate_data(data, signals, axes)
        common_axes = self.get_common_axes(signals)

        signals, axes, common_axes = self.structure_signal_values(
            signals, axes, common_axes)
        if attrs is None:
            attrs = {}
        attrs.update({'axes': common_axes})

        return NXdata(
            name=name,
            attrs=attrs,
            **{
                signal['name']: NXfield(
                    name=signal['name'],
                    attrs=signal['attrs'],
                    value=signal['value_structured']
                )
                for signal in signals
            },
            **{
                axis['name']: NXfield(
                    name=axis['name'],
                    attrs=axis['attrs'],
                    value=axis['value_unique']
                )
                for axis in axes
            }
        )

    def validate_config_fields(self, fields):
        """Validate and normalize the field configuration.

        This method separates the input field configuration into signal
        and axis definitions, performs basic validation, and ensures
        that all axes referenced by signals are defined as axis fields.

        The returned signal and axis dictionaries are normalized into a
        consistent internal representation used by later processing
        stages.

        :param fields: Configuration describing how input data should
            be structured. Each item must define a ``"name"`` and
            ``"type"`` key, where ``"type"`` is either ``"signal"``
            or ``"axis"``. Signal entries must additionally define an
            ``"axes"`` list.
        :type fields: list[dict[str, Any]]
        :raises ValueError: If a signal references an axis that is not
            defined, or if a signal is defined before any axis exist.
        :returns: Validated signal and axis definitions.
        :rtype: tuple[list[dict], list[dict]]
        """
        self.logger.info('Validating fields parameter')

        axes = []
        signals = []

        for field in fields:
            field_type = field.get('type')
            name = field.get('name')
            attrs = field.get('attrs', {})

            if field_type == 'axis':
                axes.append({'name': name, 'value': None, 'attrs': attrs})
                self.logger.debug(f'Registered axis "{name}"')

            elif field_type == 'signal':
                _axes = field.get('axes', [])
                if not axes:
                    raise ValueError(f'Signal "{name}" has no axes defined')
                signals.append({'name': name, 'axes': _axes,
                                'value': None, 'attrs': attrs})
                self.logger.debug(
                    f'Registered signal "{name}" with axes {_axes}'
                )

        # Validate that all axes used by signals exist as type: axis
        axes_names = [a['name'] for a in axes]
        for signal in signals:
            for axis in signal['axes']:
                if axis not in axes_names:
                    raise ValueError(
                        f'Signal {signal["name"]} '
                        + f'references unknown axis "{axis}"'
                    )

        self.logger.info(
            'Validated configuration for '
            + f'{len(signals)} signals and {len(axes)} axes'
        )
        return signals, axes

    def get_common_axes(self, signals):
        """Determine the common leading axis shared by all signals.

        This method computes the longest common *prefix* of axis names
        across all signal definitions. Only axes that appear in the
        same order at the beginning of each signal's ``axes`` list are
        included in the result.

        This is used to identify the shared coordinate dimensions for a
        structured NeXus style
        `NXdata <https://manual.nexusformat.org/classes/base_classes/NXdata.html#index-0>`__`NXdata`
        object.

        :param signals: Validated signal definitions. Each signal must
            define an ``"axes"`` key containing an ordered list of axis
            names.
        :type signals: list[dict]
        :returns: Axis names that form the common leading axis for all
            signals. Returns an empty list if no common prefix exists.
        :rtype: list[str]
        """
        self.logger.info('Computing common dataset axes')

        if not signals:
            self.logger.warning('No signals provided; no common axes')
            return []

        # Start with the first signal's axes
        common_axes = list(signals[0]['axes'])

        for signal in signals[1:]:
            _axes = signal['axes']
            i = 0
            max_i = min(len(common_axes), len(_axes))
            while i < max_i and common_axes[i] == _axes[i]:
                i += 1
            common_axes = common_axes[:i]
            if not common_axes:
                break

        self.logger.info(f'Computed common axes: {common_axes}')
        return common_axes

    def validate_data(self, data, signals, axes):
        """Validate and normalize input data for axes and signals.

        This method retrieves raw input data for each axis and signal,
        propagates metadata attributes, computes unique axis values,
        and allocates structured arrays for signal data.

        For each axis:
          - Raw data is loaded.
          - Attributes are merged (without overwriting user-specified
            ones).
          - Unique axis values are computed.

        For each signal:
          - Raw data is loaded.
          - Attributes are merged (without overwriting user-specified
            ones).
          - A structured output array is allocated based on its axes.
          - Total signal size is validated against the expected shape.

        :param data: Input data.
        :type data: list[PipelineData]
        :param signals: Validated signal field definitions.
        :type signals: list[dict]
        :param axes: Validated axis field definitions.
        :type axes: list[dict]
        :raises ValueError: If a signal's data size does not match the
            expected size derived from its axes.
        :returns: Updated signal and axis definitions with populated
            values and derived metadata.
        :rtype: tuple[list[dict], list[dict]]
        """
        self.logger.info('Validating input data')
        self.logger.info('Validating axis data')
        for axis in axes:
            value = self.get_data(data, name=axis['name'])
            # Merge attributes, preserving explicitly defined ones
            axis['attrs'] = {
                **axis['attrs'],
                **{k: v for k, v in value.attrs.items()
                   if k not in axis['attrs'] and k != 'target'}
            }
            axis['value'] = value
            axis['value_unique'] = np.unique(value)
            self.logger.debug(
                f'Axis {axis["name"]}: {value.size} entries, '
                f'{axis["value_unique"].size} unique'
            )

        # Build a lookup table for faster axis access by name
        axes_by_name = {a['name']: a for a in axes}
        self.logger.info("Validating signal data")
        for signal in signals:
            name = signal['name']
            value = self.get_data(data, name=name)
            # Merge attributes, preserving explicitly defined ones
            signal['attrs'] = {
                **signal['attrs'],
                **{k: v for k, v in value.attrs.items()
                   if k not in signal['attrs'] and k != 'target'}
            }
            signal['value'] = value
            _axes = signal['axes']
            signal['attrs']['axes'] = _axes
            shape = tuple(
                [axes_by_name[a]['value_unique'].size for a in _axes]
            )
            signal['value_structured'] = np.empty(shape, dtype=value.dtype)
            size_expected = np.prod(shape)
            size_actual = signal['value'].size
            self.logger.debug(
                f'Signal "{name}": expected size {size_expected} (shape: {shape}), '
                f'actual size {size_actual} (shape: {value.shape})'
            )
            if size_actual != size_expected:
                raise(ValueError(
                    f'Signal {name} has size {size_actual}; '
                    + f'expected {size_expected}'
                ))
        self.logger.info('Validated input data')
        return signals, axes

    def structure_signal_values(self, signals, axes, common_axes):
        """Reshape and populate structured signal arrays using common
        axes.

        This method determines computes index mappings from raw axis
        values to their unique sorted representations, and inserts each
        signal's unstructured data into its preallocated structured
        array.

        Only the common axes are used for structuring; any trailing,
        signal-specific axes are assumed to have already been handled
        when allocating the structured signal arrays.

        :param signals: Signal definitions with raw and preallocated
            structured data arrays.
        :type signals: list[dict]
        :param axes: Axis definitions containing raw values and unique
            values.
        :type axes: list[dict]
        :param common_axes: Ordered list of the names of the dataset's
            common axes.
        :type common_axes: list[str]
        :returns:
            - Updated signal definitions with populated structured
              arrays.
            - Unmodified axis definitions.
            - Common axis names shared by all signals.
        :rtype: tuple[list[dict], list[dict], list[str]]
        """
        self.logger.info('Structuring dataset')
        axes_by_name = {a['name']: a for a in axes if a['name'] in common_axes}
        indices = {
            a: np.searchsorted(axes_by_name[a]['value_unique'],
                               axes_by_name[a]['value'])
            for a in common_axes
        }
        _indices = tuple(indices[a] for a in common_axes)
        for signal in signals:
            signal['value_structured'][_indices] = signal['value']

        return signals, axes, common_axes


class UpdateValuesProcessor(Processor):
    """Processes a slice of data for updating values in an existing
    container for a SAXS/WAXS experiment.

    :ivar map_config: Map Configuration.
    :vartype map_config: dict dict, optional
    :ivar pyfai_config: Initialization parameters for an instance of
        :class:`~CHAP.common.models.integration.PyfaiIntegrationConfig`.
    :vartype pyfai_config: dict, optional
    :ivar spec_file: SPEC file containing the scan from which to read
        and process a slice of raw data.
    :vartype spec_file: str
    :ivar scan_number: Scan number from which to read and process a
        slice of raw data.
    :vartype scan_number: int
    :ivar detector_config: Detector configurations.
    :vartype detector_config: :class:`~CHAP.common.models.map.DetectorConfig`
    :ivar raw_data: Flag to indicate wether or not space for raw
        detector data should be included in the values returned,
        defaults to `True`.
    :vartype raw_data: bool, optional
    """
    pipeline_fields: dict = Field(
        {
            'map_config': 'common.models.map.MapConfig',
            'detector_config': 'common.models.map.DetectorConfig',
            'pyfai_config': 'common.models.integration.PyfaiIntegrationConfig',
            'correction_config': 'saxswaxs.models.CorrectionsConfig',
        },
        init_var=True)
    # map_config needs a default value because the map configuration
    # may be created by an earlier item in the Pipeline. If this is
    # the case, then map_config needs SOME value in order for the
    # Pipeline to pass validation.
    map_config: MapConfig = None
    pyfai_config: PyfaiIntegrationConfig
    detector_config: DetectorConfig = DetectorConfig(detectors=[])
    correction_config: CorrectionsConfig
    spec_file: FilePath
    scan_number: conint(gt=0)
    filename: Optional[str] = None
    raw_data: Optional[bool] = True
    idx_slice: Optional[IndexSliceConfig] = IndexSliceConfig()

    def process(self, data):
        """Processes a slice of data for updating values in an existing
        container for a SAXS/WAXS experiment.

        :param data: Input data.
        :type data: list[PipelineData]
        :param idx_slice: Dictionaries identifying the sliced index at which
            the output data should be written in a dataset, defaults to
            `{'start':0, 'step': 1}`.
        :type idx_slice: dict[str, int], optional
        :return: Integrated detector data ready for writing with
            :class:`~CHAP.saxswaxs.ZarrResultsWriter` or
            :class:`~CHAP.saxswaxs.NexusResultsWriter`.
        :rtype: list[dict[str, Any]]
        :return: Integrated detector data for updating values in an
            existing container for a SAXS/WAXS experiment.
        :rtype: list[dict[str, Any]]
        """
        # Get updates with MapSliceProcessor
        # Pass detector data to PyfaiIntegration processor
        # Concatenate & return results
        # System modules
        from copy import deepcopy
        import logging
        import os

        # Local modules
        from CHAP.common.map_utils import MapSliceProcessor
        from CHAP.pipeline import PipelineData

        # Use a copy of input data so we can append to it inside this
        # Processor without modifying the actual Pipeline's data
        # unecessarily.
        _data = deepcopy(data)

        # Read in slice of raw data
        raw_values = self.setup_pipelineitem(
            MapSliceProcessor(
                map_config=self.map_config,
                detector_config=self.detector_config,
                spec_file=str(self.spec_file),
                scan_numbers=[self.scan_number],
                idx_slice=self.idx_slice
            )
        ).process(None)

        def _get_detector_data(values, name):
            """Get the detector data."""
            for v in values:
                if os.path.basename(v['path']) == name:
                    return v['data']
            return None

        # Use raw detector data as input to integration
        for d in self.detector_config.detectors:
            _data.append(
                PipelineData(
                    name=d.get_id(),
                    data=_get_detector_data(raw_values, d.get_id()),
                )
            )
        # Get integrated data
        processed_values = self.setup_pipelineitem(
            PyfaiIntegrationProcessor(config=self.pyfai_config)
        ).process(_data)

        # Get corrected data
        corrected_values = [
            {
                'data': self.setup_pipelineitem(
                    corr_cfg.processor
                ).process(
                    self.get_corrections_input_data(
                        raw_values, processed_values, corr_cfg
                    ),
                    nxprocess=False,
                ),
                'path': f'{corr_cfg.name}/data/I_corrected'
            }
            for corr_cfg in self.correction_config.corrections
        ]

        if self.raw_data:
            return raw_values + processed_values + corrected_values

        detector_ids = [d.get_id() for d in self.detector_config.detectors]
        scalar_values = [
            d for d in raw_values
            if not os.path.basename(d['path']) in detector_ids
        ]
        return scalar_values + processed_values + corrected_values

    def get_corrections_input_data(self, raw_values, processed_values,
                                   corr_cfg):
        corr_data = []
        for x in ('dwell_time_actual', 'presample_intensity',
                  'postsample_intensity'):
            for d in raw_values:
                if d['path'].endswith(x):
                    corr_data.append(
                        PipelineData(data=d['data'], name=x)
                    )
                    break
        for d in processed_values:
            if d['path'].startswith(f'{corr_cfg.uncorrected_data_name}/'):
                corr_data.append(
                    PipelineData(
                        data=d['data'],
                        name=corr_cfg.uncorrected_data_name,
                    )
                )
        if corr_cfg.background is not None:
            if self.filename is None:
                self.logger.warning(
                    f'No filename configured; cannot read background '
                    f'intensities for correction "{corr_cfg.name}"')
            else:
                # System modules
                import os
                pre_path = (f'{corr_cfg.name}/data/'
                            'background_presample_intensity')
                post_path = (f'{corr_cfg.name}/data/'
                             'background_postsample_intensity')
                if os.path.splitext(self.filename)[1] in ('.nxs', '.h5',
                                                          '.hdf5'):
                    import h5py
                    with h5py.File(self.filename, 'r') as f:
                        for path, name in (
                                (pre_path,
                                 'background_presample_intensity'),
                                (post_path,
                                 'background_postsample_intensity')):
                            if path in f:
                                corr_data.append(PipelineData(
                                    data=np.asarray(f[path]),
                                    name=name,
                                ))
                            else:
                                self.logger.warning(
                                    f'{path} not found in {self.filename}')
                else:
                    import zarr
                    zarrfile = zarr.open(self.filename, mode='r')
                    for path, name in (
                            (pre_path, 'background_presample_intensity'),
                            (post_path, 'background_postsample_intensity')):
                        try:
                            corr_data.append(PipelineData(
                                data=np.asarray(zarrfile[path]),
                                name=name,
                            ))
                        except KeyError:
                            self.logger.warning(
                                f'{path} not found in {self.filename}')
        return corr_data

    def setup_pipelineitem(self, pipeline_item):
        """Convenience method to use a nice logger when this
        ``Processor`` calls on another ``PipelineItem``. Set the
        logger and logging handler for given pipeline item.

        :param pipeline_item: Pipeline item.
        :type pipeline_item: PipelineItem
        :return: Input Pipeline item, with updated logger and
            logging handler.
        :rtype: PipelineItem
        """
        import logging

        pipeline_item.logger = logging.getLogger(
            pipeline_item.__class__.__name__,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{asctime}: {name:20} (from '+ self.__class__.__name__
            + ') (L{lineno}): {levelname}: {message}',
            datefmt='%Y-%m-%d %H:%M:%S', style='{'))
        pipeline_item.logger.handlers = [handler]
        return pipeline_item


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
