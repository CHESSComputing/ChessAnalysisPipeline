#!/usr/bin/env python
"""Processors used only by SAXSWAXS experiments."""

import numpy as np

from CHAP import Processor
from CHAP.common import ExpressionProcessor


class CfProcessor(Processor):
    """Processor to calculate the correction factor Cf that, when
    multiplied by appropriately processed SAXSWAXS data obtained,
    converts data to absolute cross-section / intensity in inverse cm.
    """
    def process(
            self, data, interactive=False, save_figures=True, nxpath=None,
            radial_range=None, eps=1.e-5):
        """Return a dictionary with the computed correction factor Cf
        and the configuration parameters.

        :param data: Input data list containing the reference data
            labelled with `'reference_data'` as well as the NeXus
            input data with the azimuthally integrated SAXSWAXS data.
        :type data: list[PipelineData]
        :param interactive: Allows for user interactions,
            defaults to `False`.
        :type interactive: bool, optional
        :param save_figures: Create Matplotlib figures that can be
            saved to file downstream in the workflow,
            defaults to `True`.
        :type save_figures: bool, optional
        :param nxpath: The path to a specific NeXus NXdata object in
            the input NeXus file tree to the measured data from.
        :type nxpath: str, optional
        :param radial_range: q-range used to compute Cf.
        :type radial_range: Union(
            list[float, float], tuple[float, float]), optional
        :param eps: Minimum plotting value of the corrected azimuthally
            integrated SAXSWAXS data, default to `1.e-5`.
        :type eps: float
        :returns: Computed correction factor Cf and the configuration
            parameters
        :rtype: dict
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
        data_meas = nxdata.nxsignal.nxdata.mean(axis=0)
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

        # Interpolate reference data onto measured q values
        mask = np.where(
            q_meas <= radial_range[1],
            np.where(q_meas >= radial_range[0], 1, 0), 0).astype(bool)
        if not q_meas[mask].size:
            raise ValueError(
                f'No measured values within specified radial range')
        func = interp1d(q_ref, data_ref, kind='cubic')
        data_ref_intpol = func(q_meas[mask])
        if not data_ref_intpol.size:
            raise ValueError(
                f'No reference values within specified radial range')

        # Get the correction factor
        ratio = data_ref_intpol/data_meas[mask]
        cf = ratio.mean()
        cf_stv = ratio.std()

        # Assemble result
        result = {
            'cf': float(cf),
            'error': float(cf_stv),
            'radial_range': radial_range,
        }

        # Plot the results
        data_meas *= cf
        figures = []
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
                figures.append((
                    fig_to_iobuf(fig), f'correction_factor_cf'))
            if interactive:
                plt.show()
            plt.close()

        if figures:
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
            data, expression, symtable=symtable, nxprocess=nxprocess)


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
            data, expression, symtable=symtable, nxprocess=nxprocess)


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
            data, expression, symtable=symtable, nxprocess=nxprocess)


class PyfaiIntegrationProcessor(Processor):
    """Processor for performing pyFAI integrations."""
    def process(self, data, config=None, inputdir='.',
                idx_slices=[{'start':0, 'step': 1}]):
        """Perform a set of integrations on 2D detector data.

        :param data: input 2D detector data
        :type data: list[PipelineData]
        :param config: Configuration parameters for a
            `saxswaxs.models.PyfaiIntegrationProcessorConfig` object
            (_or_ the configuration may be supplied as an item in the
            input `data` list), optional
        :type config: dict, defaults to `None`.
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

        # Get config for PyfaiIntegrationProcessor from data or config
        config = self.get_config(
            data=data,
            config=config,
            inputdir=inputdir,
            schema='common.models.integration.PyfaiIntegrationConfig'
        )

        # Organize input for integrations
        input_data = {d['name']: d['data']
            for d in [d for d in data if isinstance(d['data'], np.ndarray)]}
        ais = {ai.id: ai for ai in config.azimuthal_integrators}

        # Finalize idx slice for results
        idx = tuple(slice(idx_slice.get('start'),
                     idx_slice.get('stop'),
                     idx_slice.get('step')) for idx_slice in idx_slices)

        # Perform integration(s), package results for ZarrResultsWriter
        results = []
        nframes = len(input_data[list(input_data.keys())[0]])
        for integration in config.integrations:
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
    """
    def process(self, data, dataset_shape, dataset_chunks,
                config=None, inputdir='.'):
        """Return a `zarr.group` to hold processed SAXS/WAXS data
        processed by `saxswaxs.PyfaiIntegrationProcessor`.

        :param data:
        `'saxswaxs.models.PyfaiIntegrationProcessorConfig`
        configuration which will be used to process the data later on.
        :type data: list[PipelineData]
        :param dataset_shape: Shape of the completed dataset that will
            be processed later on (shape of the measurement itself,
            _not_ including the dimensions of any signals collected at
            each point in that measurement).
        :type dataset_shape: Union[int, list[int]]
        :param dataset_chunks: Extent of chunks along each dimension
            of the completed dataset / measurement. Choose this
            according to how you will process your data -- for
            example, if your `dataset_shape` is `[m, n]`, and you are
            planning to process each of the `m` rows as chunks,
            `dataset_chunks` should be `[1, n]`. But if you plan to
            process each of the `n` columns as chunks,
            `dataset_chunks` should be `[m, 1]`.
        :type dataset_chunks: Union[int, list[int]]
        :return: Empty structure for filling in SAXS/WAXS data
        :rtype: zarr.group
        """
        # Get PyfaiIntegrationProcessorConfig
        config = self.get_config(
            data=data,
            config=config,
            inputdir=inputdir,
            schema='common.models.integration.PyfaiIntegrationConfig'
        )

        # Get zarr tree as dict from the
        # PyfaiIntegrationProcessorConfig
        if isinstance(dataset_shape, int):
            dataset_shape = [dataset_shape]
        if isinstance(dataset_chunks, int):
            dataset_chunks = [dataset_chunks]
        tree = config.zarr_tree(dataset_shape, dataset_chunks)

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
    """
    def process(self, data, dataset_shape, dataset_chunks, detectors,
                inputdir='.'):
        import asyncio
        import logging
        import zarr
        from zarr.core.buffer import default_buffer_prototype
        from zarr.storage import MemoryStore

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
        setup_map_processor = set_logger(MapProcessor())
        ddata = setup_map_processor.execute(
            data=data, detectors=detectors, fill_data=False, inputdir=inputdir)

        # Convert raw data map container to zarr format
        ddata_converter = set_logger(NexusToZarrProcessor())
        zarr_map = ddata_converter.process(ddata, chunks=dataset_chunks)

        # Get zarr container for integration results
        setup_results_processor = set_logger(SetupResultsProcessor())
        zarr_results = setup_results_processor.process(
            data, dataset_shape, dataset_chunks, inputdir=inputdir)

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


class UpdateValuesProcessor(Processor):
    """Processes a slice of data for updating values in an existing
    container for a SAXS/WAXS experiment.
    """
    def process(self, data, spec_file, scan_number,
                idx_slice={'start': 0, 'step': 1},
                detectors=None, config=None, inputdir='.'):
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
        raw_values = set_logger(MapSliceProcessor()).process(
            data, spec_file, scan_number, idx_slice=idx_slice,
            detectors=detectors, config=config, inputdir=inputdir)

        def get_detector_data(values, name):
            for v in values:
                if os.path.basename(v['path']) == name:
                    return v['data']
            return None

        # Use raw detector data as input to integration
        for d in detectors:
            data.append(
                PipelineData(
                    name=d['id'],
                    data=get_detector_data(raw_values, d['id']),
                )
            )
        # Get integrated data
        processed_values = set_logger(PyfaiIntegrationProcessor()).process(
            data, idx_slices=[idx_slice])

        return raw_values + processed_values


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
