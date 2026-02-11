#!/usr/bin/env python
"""Processors used only by SAXSWAXS experiments."""

from typing import (
    Literal,
    Optional,
    Union,
)

from pydantic import (
    conint,
    conlist,
    Field,
    FilePath,
)
import numpy as np

from CHAP import Processor
from CHAP.common import ExpressionProcessor
from CHAP.common.models.map import (
    Detector,
    MapConfig,
)
from CHAP.common.models.integration import PyfaiIntegrationConfig


class CfProcessor(Processor):
    """Processor to calculate the correction factor Cf that, when
    multiplied by appropriately processed SAXSWAXS data obtained,
    converts data to absolute cross-section / intensity in inverse cm.
    """
    def process(
            self, data, interactive=False, save_figures=True, nxpath=None,
            radial_range=None, scan_step_indices=None, eps=1.e-5):
        """Return a dictionary with the computed correction factor Cf
        and the configuration parameters.

        :param data: Input data list containing the reference data
            labelled with `'reference_data'` as well as the NeXus
            input data with the azimuthally integrated SAXSWAXS data.
        :type data: list[PipelineData]
        :param interactive: Allows for user interactions,
            defaults to `False`.
        :type interactive: bool, optional
        :param save_figures: Create Matplotlib correction factor
            image that can be saved to file downstream in the workflow,
            defaults to `True`.
        :type save_figures: bool, optional
        :param nxpath: The path to a specific NeXus NXdata object in
            the input NeXus file tree to the measured data from.
        :type nxpath: str, optional
        :param radial_range: q-range used to compute Cf.
        :type radial_range: Union(
            list[float, float], tuple[float, float]), optional
        :ivar scan_step_indices: Optional scan step indices to use for
            the calculation. If not specified, the correction factor
            will be computed on the average of all data for the scan.
        :type scan_step_indices: int, str, list[int], optional
        :param eps: Minimum plotting value of the corrected azimuthally
            integrated SAXSWAXS data, default to `1.e-5`.
        :type eps: float
        :returns: Computed correction factor Cf and the configuration
            parameters plus the optional correction factor image as a
            CHAP.pipeline.PipelineData object.
        :rtype: Union[dict, (dict, PipelineData)]
        """
        # Third party modules
        from pandas import DataFrame
        from scipy.interpolate import interp1d

        if interactive or save_figures:
            # Third party modules
            import matplotlib.pyplot as plt

            # Local modules
            from CHAP.pipeline import PipelineData
            from CHAP.utils.general import (
                fig_to_iobuf,
                round_to_n,
            )

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
                f'No reference values within specified radial range')
        func = interp1d(q_meas, data_meas, kind='cubic')
        data_meas_intpol = func(q_ref[mask])
        if not data_meas_intpol.size:
            raise ValueError(
                f'No measured values within specified radial range')

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
            ax.plot(q_meas, data_meas, label=f'Corrected FMB Data/C$_f$')
            for v in radial_range:
                plt.axvline(v, color='r', linestyle='--')
            ax.set_yscale('log')
            ax.set_title(f'Absolute Intensity Calculation', fontsize='xx-large')
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
                f'C$_f$ = {round_to_n(cf, 6):e} $\pm$ '
                f'{round_to_n(100*cf_stv/cf, 2)}%\n'
                f'1/C$_f$ = {round_to_n(1/cf, 6):e}',
                (.65, .9), xycoords = 'axes fraction', fontsize='x-large',
                bbox=dict(facecolor='white', edgecolor='grey', pad=10.0))
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
    """Processor for flux correction."""
    def process(self, data, presample_intensity_reference_rate=None,
                nxprocess=False):
        """Given input data for `'intensity'`,
        `'presample_intensity'`, and `'dwell_time_actual'`, return
        flux corrected intensity signal.

        :param data: Input data list containing items with names
            `'intensity'`, `'presample_intensity'`, and (if
            `presample_intensity_reference_rate` is not specified)
            `'dwell_time_actual'`.
        :type data: list[PipelineData]
        :param presample_intensity_reference_rate: Reference counting
            rate for the `'presample_intensity'` signal. If not
            specified, it will be calculated with
            `'np.nanmean(presample_intensity /
            dwell_time_actual)'`. Defaults to `None`.
        :type presample_intensity_reference_rate: float, optional
        :param nxprocess: Flag to indicate the flux corrected data
            should be returned as an `NXprocess`. Defaults to `False`.
        :returns: Flux corrected version of input `'intensity'` data.
        :rtype: object
        """
        if presample_intensity_reference_rate is None:
            presample_intensity_reference_rate = self._process(
                data,
                'np.nanmean(presample_intensity / dwell_time_actual)'
            )
        presample_intensity = self.get_data(
            data, name='presample_intensity',
        )
        intensity = self.get_data(
            data, name='intensity',
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
                presample_intensity_reference_rate,
            'intensity': intensity,
            'presample_intensity': presample_intensity
        }
        expression = (
            'intensity *'
            '(presample_intensity_reference_rate / presample_intensity)'
        )
        return self._process(
            data, expression, symtable=symtable, nxprocess=nxprocess,
            nxfieldtable={}
        )


class FluxAbsorptionCorrectionProcessor(ExpressionProcessor):
    """Processor for flux and absorption correction."""
    def process(self, data,
                presample_intensity_reference_rate=None,
                nxprocess=False):
        """Given input data for `'intensity'`,
        `'presample_intensity'`, `'postsample_intensity'`,
        `'background_presample_intensity'`,
        `'background_postsample_intensity'`, and
        `'dwell_time_actual'`, return flux and absorption corrected
        intensity signal.

        :param data: Input data list containing all necessary data
            labelled with their proper names.
        :type data: list[PipelineData]
        :param presample_intensity_reference_rate: Reference counting
            rate for the `'presample_intensity'` signal. If not
            specified, it will be calculated with
            `'np.nanmean(presample_intensity /
            dwell_time_actual)'`. Defaults to `None`.
        :type presample_intensity_reference_rate: float, optional
        :param nxprocess: Flag to indicate the flux corrected data
            should be returned as an `NXprocess`. Defaults to `False`.
        :returns: Flux and absprption corrected version of input
            `'intensity'` data.
        :rtype: object
        """
        intensity = self.get_data(
            data, name='intensity',
        )

        if presample_intensity_reference_rate is None:
            presample_intensity_reference_rate = self._process(
                data,
                'np.nanmean(presample_intensity / dwell_time_actual)'
            )

        T = self._process(
            data,
            ('(postsample_intensity / presample_intensity) '
             '/ np.average('
             '(background_postsample_intensity / background_presample_intensity))')
        )
        # Extend T along last dim to have same shape as intensity
        for dim in intensity.shape[T.ndim:]:
            T = np.repeat(
                np.expand_dims(T, axis=-1), dim, axis=-1
            )

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
                presample_intensity_reference_rate,
            'intensity': intensity,
            'presample_intensity': presample_intensity,
            'T': T
        }
        expression = (
            '(1 / T)'
            '* intensity'
            '* (presample_intensity_reference_rate / presample_intensity)'
        )
        return self._process(
            data, expression, symtable=symtable, nxprocess=nxprocess,
            nxfieldtable={}
        )


class FluxAbsorptionBackgroundCorrectionProcessor(ExpressionProcessor):
    """Processor for flux, absorption, and background correction. May
    also perform thickness correction."""
    def process(self, data,
                presample_intensity_reference_rate=None,
                sample_thickness_cm=None,
                sample_mu_inv_cm=None,
                nxprocess=False):
        """Given input data for `'intensity'`,
        `'presample_intensity'`, `'postsample_intensity'`,
        `'background_presample_intensity'`,
        `'background_postsample_intensity'`, `'background_intensity'`,
        and `'dwell_time_actual'`, return flux and absorption
        corrected intensity signal.

        :param data: Input data list containing all necessary data
            labelled with their proper names.
        :type data: list[PipelineData]
        :param presample_intensity_reference_rate: Reference counting
            rate for the `'presample_intensity'` signal. If not
            specified, it will be calculated with
            `'np.nanmean(presample_intensity /
            dwell_time_actual)'`. Defaults to `None`.
        :type presample_intensity_reference_rate: float, optional
        :param sample_thickness_cm: Sample thickness in
            centimeters. If specified, this processor will
            additionally perform thickness correction. Use of this
            parameter is mutualy exclusive with
            use of `sample_mu_inv_cm`. Defaults to `None`.
        :type sample_thickness_cm: float, optional
        :param sample_mu_inv_cm: Sample linear attenuation coefficient
            in inverse centimeters. If specified, this processor will
            additionally perform thickness correction. Use of this
            parameter is mutualy exclusive with use of
            `sample_thickness_cm`. Defaults to `None`.
        :type sample_mu_inv_cm: float, optional
        :param nxprocess: Flag to indicate the flux corrected data
            should be returned as an `NXprocess`. Defaults to `False`.
        :returns: Flux and absprption corrected version of input
            `'intensity'` data.
        :rtype: object
        """
        if sample_thickness_cm is not None and sample_mu_inv_cm is not None:
            raise ValueError((
                'Cannot use sample_thickness_cm and sample_mu_inv_cm'
                ' at the same time'
            ))

        intensity = self.get_data(
            data, name='intensity',
        )

        if presample_intensity_reference_rate is None:
            presample_intensity_reference_rate = self._process(
                data,
                'np.nanmean(presample_intensity / dwell_time_actual)'
            )

        T = self._process(
            data,
            ('(postsample_intensity / presample_intensity) '
             '/ np.average('
             '(background_postsample_intensity / background_presample_intensity))')
        )
        # Extend T along last dim to have same shape as intensity
        for dim in intensity.shape[T.ndim:]:
            T = np.repeat(
                np.expand_dims(T, axis=-1), dim, axis=-1
            )

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

        if sample_thickness_cm is not None:
            t = sample_thickness_cm
        elif sample_mu_inv_cm is not None:
            t = -np.log(T / sample_mu_inv_cm)
        else:
            t = 1

        symtable = {
            't': t,
            'presample_intensity_reference_rate':
                presample_intensity_reference_rate,
            'intensity': intensity,
            'presample_intensity': presample_intensity,
            'T': T,
            'background_intensity': background_intensity
        }
        expression = (
            '(1 / t)'
            '* ('
            '(1 / T)'
            '* intensity'
            '* (presample_intensity_reference_rate / presample_intensity)'
            ') - ('
            'background_intensity'
            '* (presample_intensity_reference_rate / np.average(background_presample_intensity))'
            ')'
        )
        return self._process(
            data, expression, symtable=symtable, nxprocess=nxprocess,
            nxfieldtable={}
        )


class PyfaiIntegrationProcessor(Processor):
    """Processor for performing pyFAI integrations.

    :ivar config: PyfaiIntegrationConfig
    :type config: CHAP.common.models.integration.PyfaiIntegrationConfig
    """
    pipeline_fields: dict = Field(
        default={
            'config': 'common.models.integration.PyfaiIntegrationConfig'
        },
        init_var=True)
    config: PyfaiIntegrationConfig

    def process(self, data,
                idx_slices=[{'start':0, 'step': 1}]):
        """Perform a set of integrations on 2D detector data.

        :param data: input 2D detector data
        :type data: list[PipelineData]
        :param idx_slices: List of dicionaries identifying the sliced
            index at which the output data should be written in a
            dataset. Optional.
        :type idx_slices: list[dict[str, int]], defaults to
        `[{'start':0, 'step': 1}]`
        :return: List of dictionaries ready for use with
            `saxswaxs.ZarrResultsWriter` or
            `saxswaxs.NexusResultsWriter`.
        :rtype: list[dict[str, object]]
        """
        import time

        # Organize input for integrations
        input_data = {d['name']: d['data']
            for d in [d for d in data if isinstance(d['data'], np.ndarray)]}
        ais = {ai.get_id(): ai for ai in self.config.azimuthal_integrators}

        # Finalize idx slice for results
        idx = tuple(slice(idx_slice.get('start'),
                     idx_slice.get('stop'),
                     idx_slice.get('step')) for idx_slice in idx_slices)

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
                        'idx': idx,
                        'data': np.asarray(result['intensities']),
                    },
                ]
            )
        return results


class SetupResultsProcessor(Processor):
    """Processor for creating an intital zarr structure with empty datasets
    for filling in by `saxswaxs.PyfaiIntegrationProcessor` and
    `common.ZarrValuesWriter`.

    :ivar dataset_shape: Shape of the completed dataset that will
        be processed later on (shape of the measurement itself,
        _not_ including the dimensions of any signals collected at
        each point in that measurement).
    :type dataset_shape: Union[int, list[int]]
    :ivar dataset_chunks: Extent of chunks along each dimension
        of the completed dataset / measurement. Choose this
        according to how you will process your data -- for
        example, if your `dataset_shape` is `[m, n]`, and you are
        planning to process each of the `m` rows as chunks,
        `dataset_chunks` should be `[1, n]`. But if you plan to
        process each of the `n` columns as chunks,
        `dataset_chunks` should be `[m, 1]`.
    :type dataset_chunks: Union[list[int], Literal["auto"]]
    """
    pipeline_fields: dict = Field(
        default={
            'config': 'common.models.integration.PyfaiIntegrationConfig'
        },
        init_var=True)
    config: PyfaiIntegrationConfig
    dataset_shape: conlist(item_type=conint(gt=0), min_length=1)
    dataset_chunks: Optional[
        Union[
            Literal['auto'],
            conlist(item_type=conint(gt=0), min_length=1)
        ]] = 'auto'

    def process(self, data):
        """Return a `zarr.group` to hold processed SAXS/WAXS data
        processed by `saxswaxs.PyfaiIntegrationProcessor`.

        :param data: Input data (configurations).
        :type data: list[PipelineData]
        :return: Empty structure for filling in SAXS/WAXS data
        :rtype: zarr.group
        """

        # Get zarr tree as dict from the
        # PyfaiIntegrationConfig
        if isinstance(self.dataset_shape, int):
            self.dataset_shape = [self.dataset_shape]
        if isinstance(self.dataset_chunks, int):
            self.dataset_chunks = [self.dataset_chunks]
        tree = self.config.zarr_tree(self.dataset_shape, self.dataset_chunks)

        # Construct & return the root zarr.group
        return self.zarr_setup(tree)

    def zarr_setup(self, tree):
        """Return a `zarr.group` based on a
        dictionary representing a zarr tree of groups and arrays.

        :param tree: Nested dictionary representing a zarr tree of
            groups and arrays.
        :type tree: dict[str, object]
        :return: Zarr group corresponding to the contents of `tree`.
        :rtype: zarr.group
        """
        import zarr
        from zarr.storage import MemoryStore

        def create_group_or_dataset(node, zarr_parent, indent=0):
            # Set attributes if present
            if 'attributes' in node:
                for key, value in node['attributes'].items():
                    zarr_parent.attrs[key] = value
            # Create children (groups or datasets)
            if 'children' in node:
                for name, child in node['children'].items():
                    if 'shape' in child or 'data' in child:
                        # It's a dataset
                        self.logger.debug(f'Adding dset: {name}')
                        zarr_parent.create_dataset(
                            name,
                            **child,
                        )
                        # Set dataset attributes
                        if 'attributes' in child:
                            for key, value in child['attributes'].items():
                                zarr_parent[name].attrs[key] = value
                    else:
                        # It's a group
                        group = zarr_parent.create_group(name)
                        create_group_or_dataset(child, group, indent=indent+2)
        results = zarr.create_group(store=MemoryStore({}))
        create_group_or_dataset(tree['root'], results)
        return results


class SetupProcessor(Processor):
    """Convenience Processor for setting up a container for SAXS/WAXS
    experiments.

    :ivar detectors: List of basic detector configuration parameters.
    :type detectors: `CHAP.common.models.map.DetectorConfig`
    :ivar dataset_shape: Shape of the completed dataset that will
        be processed later on (shape of the measurement itself,
        _not_ including the dimensions of any signals collected at
        each point in that measurement).
    :type dataset_shape: Union[int, list[int]]
    :ivar dataset_chunks: Extent of chunks along each dimension
        of the completed dataset / measurement. Choose this
        according to how you will process your data -- for
        example, if your `dataset_shape` is `[m, n]`, and you are
        planning to process each of the `m` rows as chunks,
        `dataset_chunks` should be `[1, n]`. But if you plan to
        process each of the `n` columns as chunks,
        `dataset_chunks` should be `[m, 1]`.
    :type dataset_chunks: Union[list[int], Literal["auto"]]
    :ivar num_chunk: Used only if `dataset_chunks` is
        `"auto"`. Preferred number of chunks in the dataset. Defaults
        to `1`.
    :type num_chunk: int, optional
    """
    pipeline_fields: dict = Field(
        default={
            'map_config': 'common.models.map.MapConfig',
            'pyfai_config': 'common.models.integration.PyfaiIntegrationConfig'
        },
        init_var=True)
    map_config: MapConfig
    pyfai_config: PyfaiIntegrationConfig
    detectors: conlist(item_type=Detector, min_length=1)
    dataset_shape: conlist(item_type=conint(gt=0), min_length=1)
    dataset_chunks: Optional[
        Union[
            Literal['auto'],
            conlist(item_type=conint(gt=0), min_length=1)
        ]] = 'auto'
    num_chunk: Optional[conint(gt=0)] = 1

    def process(self, data):
        import asyncio
        import logging
        import zarr
        from zarr.core.buffer import default_buffer_prototype
        from zarr.storage import MemoryStore

        from CHAP.pipeline import PipelineData
        from CHAP.common import MapProcessor, NexusToZarrProcessor
        from CHAP.saxswaxs import SetupResultsProcessor

        def set_logger(pipeline_item):
            pipeline_item.logger = self.logger
            pipeline_item.logger.name = pipeline_item.__class__.__name__
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '{asctime}: {name:20} (from '+ self.__class__.__name__
                + '): {levelname}: {message}',
                datefmt='%Y-%m-%d %H:%M:%S', style='{'))
            pipeline_item.logger.removeHandler(
                pipeline_item.logger.handlers[0])
            pipeline_item.logger.addHandler(handler)
            return pipeline_item

        # Get NXroot container for raw data map
        setup_map_processor = set_logger(
            MapProcessor(
                config=self.map_config,
                detector_config={'detectors': self.detectors},
            )
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

        # Convert raw data map container to zarr format
        ddata_converter = set_logger(NexusToZarrProcessor())
        zarr_map = ddata_converter.process(ddata, chunks=self.dataset_chunks)

        # Get zarr container for integration results
        setup_results_processor = set_logger(
            SetupResultsProcessor(
                config=self.pyfai_config,
                dataset_shape=self.dataset_shape,
                dataset_chunks=self.dataset_chunks,
            )
        )
        zarr_results = setup_results_processor.process(data)

        # Assemble containers for raw & processed data
        zarr_root = zarr.create_group(store=MemoryStore({}))
        async def copy_zarr_store(source_store, dest_store):
            async for k in source_store.list():
                self.logger.info(f'Copying {k}')
                buf = await source_store.get(
                    k, prototype=default_buffer_prototype())
                await dest_store.set(k, buf)
        asyncio.run(copy_zarr_store(zarr_map.store, zarr_root.store))
        asyncio.run(copy_zarr_store(zarr_results.store, zarr_root.store))
        return zarr_root


class UnstructuredToStructuredProcessor(Processor):
    """Processor to aggregate "unstructured" data into a single NXdata
    with a "structured" representation.
    """
    def process(self, data, fields, name='data', attrs=None):
        """Return an `NXdata` object containing a single structured
        dataset composed from multiple unstructured input datasets.

        This method validates the field configuration, validates and
        reshapes the input data, determines common axes across all
        signals, and constructs a NeXus `NXdata` group containing
        signal and axis fields.

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
            - ``"attrs"``: Dictionary of NeXus attributes to attach to

        :type fields: list[dict[str, object]]
        :param name: Name of the resulting `NXdata` group.
        :type name: str
        :param attrs: Attributes to attach to the resulting `NXdata`
            group.  The common axes determined during processing will
            be added to this dictionary under the ``"axes"`` key.
        :type attrs: dict[str, object] or None
        :returns: A structured NeXus `NXdata` object containing all
            signals and axes defined by the configuration.
        :rtype: nexusformat.nexus.NXdata
        """
        from nexusformat.nexus import NXdata, NXfield


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

        This method separates the input field configuration into signal and
        axis definitions, performs basic validation, and ensures that all
        axes referenced by signals are defined as axis fields.

        The returned signal and axis dictionaries are normalized into a
        consistent internal representation used by later processing stages.

        :param fields: Configuration describing how input data should be
            structured. Each item must define a ``"name"`` and ``"type"``
            key, where ``"type"`` is either ``"signal"`` or ``"axis"``.
            Signal entries must additionally define an ``"axes"`` list.
        :type fields: list[dict[str, object]]
        :returns: Tuple of validated signal and axis definitions.
        :rtype: tuple[list[dict], list[dict]]
        :raises ValueError: If a signal references an axis that is not defined,
            or if a signal is defined before any axes exist.
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
        """Determine the common leading axes shared by all signals.

        This method computes the longest common *prefix* of axis names
        across all signal definitions. Only axes that appear in the
        same order at the beginning of each signal's ``axes`` list are
        included in the result.

        This is used to identify the shared coordinate dimensions
        for a structured `NXdata` group.

        :param signals: Validated signal definitions. Each signal must
            define an ``"axes"`` key containing an ordered list of
            axis names.
        :type signals: list[dict]

        :returns: List of axis names that form the common leading axes
            for all signals. Returns an empty list if no common prefix
            exists.
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
          - The raw data is loaded
          - Attributes are merged (without overwriting user-specified
            ones)
          - Unique axis values are computed

        For each signal:
          - The raw data is loaded
          - Attributes are merged (without overwriting user-specified
            ones)
          - A structured output array is allocated based on its axes
          - The total signal size is validated against the expected
            shape

        :param data: Input unstructured data items.
        :type data: list[PipelineData]
        :param signals: Validated signal field definitions.
        :type signals: list[dict]
        :param axes: Validated axis field definitions.
        :type axes: list[dict]
        :returns: Updated signal and axis definitions with populated
            values and derived metadata.
        :rtype: tuple[list[dict], list[dict]]
        :raises ValueError: If a signal's data size does not match the
            expected size derived from its axes.
        """
        import numpy as np

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
        """Reshape and populate structured signal arrays using common axes.

        This method determines computes index mappings from raw axis
        values to their unique sorted representations, and inserts
        each signal's unstructured data into its preallocated
        structured array.

        Only the common axes are used for structuring; any trailing,
        signal-specific axes are assumed to have already been handled when
        allocating the structured signal arrays.

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
            - Updated signal definitions with populated structured arrays
            - Unmodified axis definitions
            - List of common axis names shared by all signals
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
    :type map_config: CHAP.common.models.map.MapConfig
    :ivar pyfai_config: PyFAI integration configuration.
    :type pyfai_config: CHAP.common.models.integration.PyfaiIntegrationConfig
    :ivar spec_file: SPEC file containing scan from which to read and
        process a slice of raw data.
    :type spec_file: str
    :ivar scan_number: Number of scan from which to read and process a
        slice of raw data.
    :type scan_number: int
    :ivar detectors: List of detector configurations.
    :type detectors: list[Detector]
    """
    pipeline_fields: dict = Field(
        default={
            'map_config': 'common.models.map.MapConfig',
            'pyfai_config': 'common.models.integration.PyfaiIntegrationConfig'
        },
        init_var=True)
    map_config: MapConfig
    pyfai_config: PyfaiIntegrationConfig
    spec_file: FilePath
    scan_number: conint(gt=0)
    detectors: conlist(item_type=Detector, min_length=1)

    def process(self, data, idx_slice={'start': 0, 'step': 1}):
        # Get updates with MapSliceProcessor
        # Pass detector data to PyfaiIntegration processor
        # Concatenate & return results
        import logging
        import os

        from CHAP.common import MapSliceProcessor
        from CHAP.pipeline import PipelineData
        from CHAP.saxswaxs import PyfaiIntegrationProcessor

        def set_logger(pipeline_item):
            pipeline_item.logger = self.logger
            pipeline_item.logger.name = pipeline_item.__class__.__name__
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '{asctime}: {name:20} (from '+ self.__class__.__name__
                + '): {levelname}: {message}',
                datefmt='%Y-%m-%d %H:%M:%S', style='{'))
            pipeline_item.logger.removeHandler(
                pipeline_item.logger.handlers[0])
            pipeline_item.logger.addHandler(handler)
            return pipeline_item

        # Read in slice of raw data
        raw_values = set_logger(
            MapSliceProcessor(
                map_config=self.map_config,
                detectors=self.detectors,
                spec_file=str(self.spec_file),
                scan_number=self.scan_number,
            )
        ).process(None, idx_slice=idx_slice)

        def get_detector_data(values, name):
            for v in values:
                if os.path.basename(v['path']) == name:
                    return v['data']
            return None

        # Use raw detector data as input to integration
        for d in self.detectors:
            data.append(
                PipelineData(
                    name=d.get_id(),
                    data=get_detector_data(raw_values, d.get_id()),
                )
            )
        # Get integrated data
        processed_values = set_logger(
            PyfaiIntegrationProcessor(
                config=self.pyfai_config,
            )
        ).process(data, idx_slices=[idx_slice])

        return raw_values + processed_values


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
