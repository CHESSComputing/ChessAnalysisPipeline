#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Rolf Verberg <rolfverberg AT gmail dot com>
Description: Module for Processors used only by tomography experiments
"""

# System modules
from os import mkdir
from os import path as os_path
from sys import exit as sys_exit
from time import time

# Third party modules
import numpy as np

# Local modules
from CHAP.common.utils.general import (
    is_num,
    input_int,
    input_num,
    input_yesno,
    select_image_bounds,
    select_one_image_bound,
    draw_mask_1d,
    clear_plot,
    clear_imshow,
    quick_plot,
    quick_imshow,
)
from CHAP.common.utils.fit import Fit
from CHAP.processor import Processor
from CHAP.reader import main

NUM_CORE_TOMOPY_LIMIT = 24


class TomoDataProcessor(Processor):
    """
    A processor to reconstruct a set of tomographic images returning
    either a dictionary or a `nexusformat.nexus.NXroot` object
    containing the (meta) data after processing each individual step.
    """

    def process(
            self, data, interactive=False, reduce_data=False,
            find_center=False, reconstruct_data=False, combine_data=False,
            output_folder='.', save_figs=None, **kwargs):
        """
        Process the output of a `Reader` that contains a map or a
        `nexusformat.nexus.NXroot` object and one that contains the step
        specific instructions and return either a dictionary or a
        `nexusformat.nexus.NXroot` with the processed result.

        :param data: Input configuration for the individual steps in the
            tomographic image reduction.
        :type data: list[PipelineData]
        :param interactive: Allows for user interactions,
            defaults to False.
        :type bool, optional
        :param reduce_data: Generate reduced tomography images,
            defaults to False.
        :type bool, optional
        :param find_center: Find the calibrated center axis info,
            defaults to False.
        :type bool, optional
        :param reconstruct_data: Reconstruct the tomography data,
            defaults to False.
        :type bool, optional
        :param combine_data: Combine the reconstructed tomography
            stacks, defaults to False.
        :type bool, optional
        :param output_folder: Output folder name, defaults to '.'.
        :type str, optional
        :param save_figs: Display and/or save figures to file.
        :type str, optional
        :return: Processed (meta)data of the last step.
        :rtype: dict or nexusformat.nexus.NXroot
        """

        if not isinstance(reduce_data, bool):
            raise ValueError(f'Invalid parameter reduce_data ({reduce_data})')
        if not isinstance(find_center, bool):
            raise ValueError(f'Invalid parameter find_center ({find_center})')
        if not isinstance(reconstruct_data, bool):
            raise ValueError(
                f'Invalid parameter reconstruct_data ({reconstruct_data})')
        if not isinstance(combine_data, bool):
            raise ValueError(
                f'Invalid parameter combine_data ({combine_data})')

        tomo = Tomo(
            interactive=interactive, output_folder=output_folder,
            save_figs=save_figs)
        nxroot = None
        center_config = None

        # Get and validate the relevant configuration objects in data
        configs = self.get_configs(data)

        # Setup the pipeline for a tomography reconstruction
        if 'setup' in configs:
            configs.pop('nxroot', None)
            nxroot = self.get_nxroot(configs.pop('map'), configs.pop('setup'))
        else:
            nxroot = configs.pop('nxroot', None)

        # Reduce tomography images
        if reduce_data or 'reduce' in configs:
            args = {}
            if 'reduce' in configs:
                tool_config = configs.pop('reduce')
                args['img_x_bounds'] = tool_config.img_x_bounds
                args['delta_theta'] = tool_config.delta_theta
            else:
                if nxroot is None:
                    raise RuntimeError(
                        'Unable to reduce the data without providing a '
                        + 'reduced_data config file')
            if nxroot is None:
                map_config = configs.pop('map')
                nxroot = self.get_nxroot(map_config, tool_config)
            nxroot = tomo.gen_reduced_data(nxroot, **args)

        # Find rotation axis centers for the tomography stacks
        # RV pass tool_config directly to tomo.find_centers?
        if find_center or 'find_center' in configs:
            args = {}
            run_find_centers = False
            if 'find_center' in configs:
                tool_config = configs.pop('find_center')
                args['center_rows'] = (
                    tool_config.lower_row, tool_config.upper_row)
                args['center_stack_index'] = tool_config.center_stack_index
                lower_center_offset = tool_config.lower_center_offset
                upper_center_offset = tool_config.upper_center_offset
                if (None in args['center_rows'] or lower_center_offset is None
                        or upper_center_offset is None):
                    run_find_centers = True
            if run_find_centers or 'find_center' not in configs:
                center_config = tomo.find_centers(nxroot, **args)
            else:
                # RV make a convert to dict in basemodel?
                center_config = {
                    'lower_row': tool_config.lower_row,
                    'lower_center_offset': tool_config.lower_center_offset,
                    'upper_row': tool_config.upper_row,
                    'upper_center_offset': tool_config.upper_center_offset,
                    'center_stack_index': tool_config.center_stack_index,
                }

        # Reconstruct tomography stacks
        # RV pass tool_config and center_config directly to
        #     tomo.reconstruct_data?
        if reconstruct_data or 'reconstruct' in configs:
            args = {}
            if 'reconstruct' in configs:
                tool_config = configs.pop('reconstruct')
                args['x_bounds'] = tool_config.x_bounds
                args['y_bounds'] = tool_config.y_bounds
                args['z_bounds'] = tool_config.z_bounds
            nxroot = tomo.reconstruct_data(nxroot, center_config, **args)
            center_config = None

        # Combine reconstructed tomography stacks
        if combine_data or 'combine' in configs:
            args = {}
            if 'combine' in configs:
                tool_config = configs.pop('combine')
                args['x_bounds'] = tool_config.x_bounds
                args['y_bounds'] = tool_config.y_bounds
                args['z_bounds'] = tool_config.z_bounds
            nxroot = tomo.combine_data(nxroot, **args)

        if center_config is not None:
            return center_config
        return nxroot

    def get_configs(self, data):
        """
        Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            is of type `nexusformat.nexus.NXroot` or has the value
            `'MapConfig'` for the `'schema'` key, and at least one item
            has the value `'TomoSetupConfig'`, or `'TomoReduceConfig'`,
            or `'TomoFindCenterConfig'`, or `'TomoReconstructConfig'`,
            or `'TomoCombineConfig'` for the `'schema'` key.
        :type data: list[PipelineData]
        :return: valid instances of the configuration objects with field
            values taken from `data`.
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import NXroot

        # Local modules
        from CHAP.common.models.map import MapConfig
        from CHAP.tomo.models import (
            TomoSetupConfig,
            TomoReduceConfig,
            TomoFindCenterConfig,
            TomoReconstructConfig,
            TomoCombineConfig,
        )

        configs = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('data') is not None:
                    schema = item.get('schema')
                    if isinstance(item.get('data'), NXroot):
                        configs['nxroot'] = item.get('data')
                    if schema == 'MapConfig':
                        configs['map'] = MapConfig(**(item.get('data')))
                    if schema == 'TomoSetupConfig':
                        configs['setup'] = TomoSetupConfig(
                            **(item.get('data')))
                    if schema == 'TomoReduceConfig':
                        configs['reduce'] = TomoReduceConfig(
                            **(item.get('data')))
                    elif schema == 'TomoFindCenterConfig':
                        configs['find_center'] = TomoFindCenterConfig(
                            **(item.get('data')))
                    elif schema == 'TomoReconstructConfig':
                        configs['reconstruct'] = TomoReconstructConfig(
                            **(item.get('data')))
                    elif schema == 'TomoCombineConfig':
                        configs['combine'] = TomoCombineConfig(
                            **(item.get('data')))

        return configs

    def get_nxroot(self, map_config, tool_config):
        """
        Get a map of the collected tomography data from the scans in
        `map_config` and the detector info in `tool_config'.
        The map is returned as an `nexusformat.nexus.NXroot` object.

        :param map_config: Map configuration
        :type map_config: MapConfig
        :param tool_config: The tomography image reduction configuration
            containing at minimum the detector information.
        :type tool_config: TomoReduceConfig
        :return: a map of the collected tomography data along with the
        data reduction configuration
        :rtype: nexusformat.nexus.NXroot
        """
        # System modules
        from copy import deepcopy
        from logging import getLogger

        # Third party modules
        from nexusformat.nexus import (
            NXcollection,
            NXdata,
            NXdetector,
            NXinstrument,
            NXroot,
            NXsample,
            NXsource,
            NXsubentry,
        )

        # Local modules
        from CHAP.common import MapProcessor
        from CHAP.common.models.map import import_scanparser
        from CHAP.common.utils.general import index_nearest

        logger = getLogger(self.__name__)
        logger.propagate = False

        include_raw_data = getattr(tool_config, 'include_raw_data', False)

        # Construct NXroot
        nxroot = NXroot()

        # Construct base NXentry and add to NXroot
        nxentry = MapProcessor.get_nxentry(map_config)
        nxroot[map_config.title] = nxentry
        nxroot.attrs['default'] = map_config.title
        nxentry.definition = 'NXtomo'
        if 'data' in nxentry:
            del nxentry['data']

        # Add an NXinstrument to the NXentry
        nxinstrument = NXinstrument()
        nxentry.instrument = nxinstrument

        # Add an NXsource to the NXinstrument
        nxsource = NXsource()
        nxinstrument.source = nxsource
        nxsource.type = 'Synchrotron X-ray Source'
        nxsource.name = 'CHESS'
        nxsource.probe = 'x-ray'

        # Tag the NXsource with the runinfo (as an attribute)
#        nxsource.attrs['cycle'] = map_config.cycle
#        nxsource.attrs['btr'] = map_config.btr
        nxsource.attrs['station'] = map_config.station
        nxsource.attrs['experiment_type'] = map_config.experiment_type

        # Add an NXdetector to the NXinstrument
        # (do not fill in data fields yet)
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = tool_config.detector.prefix
        pixel_size = tool_config.detector.pixel_size
        if len(pixel_size) == 1:
            nxdetector.x_pixel_size = \
                pixel_size[0]/tool_config.detector.lens_magnification
            nxdetector.y_pixel_size = \
                pixel_size[0]/tool_config.detector.lens_magnification
        else:
            nxdetector.x_pixel_size = \
                pixel_size[0]/tool_config.detector.lens_magnification
            nxdetector.y_pixel_size = \
                pixel_size[1]/tool_config.detector.lens_magnification
        nxdetector.x_pixel_size.attrs['units'] = 'mm'
        nxdetector.y_pixel_size.attrs['units'] = 'mm'

        if include_raw_data:
            # Add an NXsample to NXentry
            # (do not fill in data fields yet)
            nxsample = NXsample()
            nxentry.sample = nxsample
            nxsample.name = map_config.sample.name
            nxsample.description = map_config.sample.description

        # Add NXcollection's to NXentry to hold metadata about the SPEC
        #     scans in the map
        # Also obtain the data fields in NXsample and NXdetector if
        #     requested
        import_scanparser(map_config.station, map_config.experiment_type)
        image_keys = []
        sequence_numbers = []
        image_stacks = []
        rotation_angles = []
        x_translations = []
        z_translations = []
        spec_scans = map_config.spec_scans
        if map_config.station == 'id3b':
            scan_types = len(spec_scans)*['ts1']
            if tool_config.dark_field is not None:
                spec_scans += tool_config.dark_field
                scan_types.append('df1')
            if tool_config.bright_field is not None:
                spec_scans += tool_config.bright_field
                scan_types.append('bf1')
        for n, scans in enumerate(spec_scans):
            for scan_number in scans.scan_numbers:
                scanparser = scans.get_scanparser(scan_number)
                if map_config.station in ('id1a3', 'id3a'):
                    scan_type = scanparser.scan_type
                    if scan_type == 'df1':
                        image_key = 2
                        field_name = 'dark_field'
                    elif scan_type == 'bf1':
                        image_key = 1
                        field_name = 'bright_field'
                    elif scan_type == 'ts1':
                        image_key = 0
                        field_name = 'tomo_fields'
                    else:
                        raise RuntimeError('Invalid scan type: {scan_type}')
                elif map_config.station == 'id3b':
                    if (scans.spec_file.endswith('_dark')
                            or scan_types[n] == 'df1'):
                        image_key = 2
                        field_name = 'dark_field'
                    elif (scans.spec_file.endswith('_flat')
                            or scan_types[n] == 'bf1'):
                        image_key = 1
                        field_name = 'bright_field'
                    else:
                        image_key = 0
                        field_name = 'tomo_fields'
                else:
                    raise RuntimeError(
                        f'Invalid station in map_config: {map_config.station}')

                # Create an NXcollection for each field type
                if field_name in nxentry.spec_scans:
                    nxcollection = nxentry.spec_scans[field_name]
                    if nxcollection.attrs['spec_file'] != str(scans.spec_file):
                        raise RuntimeError(
                            'Multiple SPEC files for a single field type not '
                            + f'yet implemented; field name: {field_name}, '
                            + f'SPEC file: {str(scans.spec_file)}')
                else:
                    nxcollection = NXcollection()
                    nxentry.spec_scans[field_name] = nxcollection
                    nxcollection.attrs['spec_file'] = str(scans.spec_file)
                    nxcollection.attrs['date'] = scanparser.spec_scan.file_date

                # Get thetas
                image_offset = scanparser.starting_image_offset
                if map_config.station in ('id1a3', 'id3a'):
                    thetas = scanparser.rotation_angles
                else:
                    try:
                        thetas = []
                        num_theta = scanparser.spec_scan_npts
                        for dim in map_config.independent_dimensions:
                            if dim.label != 'theta':
                                continue
                            for index in range(num_theta):
                                thetas.append(dim.get_value(
                                    scans, scan_number, index+image_offset))
                        thetas = np.asarray(thetas)
                        assert thetas == scanparser.rotation_angles
                    except:
                        thetas = scanparser.rotation_angles
                    if thetas[-1]-thetas[0] > 180.:
                        image_end = index_nearest(thetas, thetas[0]+180)
                        thetas = thetas[:image_end]

                # x and z translations
                x_translation = scanparser.horizontal_shift
                z_translation = scanparser.vertical_shift

                # Add an NXsubentry to the NXcollection for each scan
                entry_name = f'scan_{scan_number}'
                nxsubentry = NXsubentry()
                nxcollection[entry_name] = nxsubentry
                nxsubentry.start_time = scanparser.spec_scan.date
                nxsubentry.spec_command = scanparser.spec_command
                # Add an NXinstrument to the scan's NXsubentry
                nxsubentry.instrument = NXinstrument()
                # Add an NXdetector to the NXinstrument to the scan's
                #     NXsubentry
                nxsubentry.instrument.detector = deepcopy(nxdetector)
                nxsubentry.instrument.detector.frame_start_number = \
                    image_offset
                nxsubentry.instrument.detector.image_key = image_key
                # Add an NXsample to the scan's NXsubentry
                nxsubentry.sample = NXsample()
                nxsubentry.sample.rotation_angle = thetas
                nxsubentry.sample.rotation_angle.units = 'degrees'
                nxsubentry.sample.x_translation = x_translation
                nxsubentry.sample.x_translation.units = 'mm'
                nxsubentry.sample.z_translation = z_translation
                nxsubentry.sample.z_translation.units = 'mm'

                if include_raw_data:
                    num_image = len(thetas)
                    image_keys += num_image*[image_key]
                    sequence_numbers += list(range(num_image))
                    image_stacks.append(
                        scanparser.get_detector_data(
                            tool_config.detector.prefix,
                            scan_step_index=(image_offset,
                                             image_offset+num_image)))
                    rotation_angles += list(thetas)
                    x_translations += num_image*[x_translation]
                    z_translations += num_image*[z_translation]

        if include_raw_data:
            # Add image data to NXdetector
            nxinstrument.detector.image_key = image_keys
            nxinstrument.detector.sequence_number = sequence_numbers
            nxinstrument.detector.data = np.concatenate(image_stacks)

            # Add image data to NXsample
            nxsample.rotation_angle = rotation_angles
            nxsample.rotation_angle.attrs['units'] = 'degrees'
            nxsample.x_translation = x_translations
            nxsample.x_translation.attrs['units'] = 'mm'
            nxsample.z_translation = z_translations
            nxsample.z_translation.attrs['units'] = 'mm'

            # Add an NXdata to NXentry
            nxdata = NXdata()
            nxentry.data = nxdata
            nxdata.makelink(nxentry.instrument.detector.data, name='data')
            nxdata.makelink(nxentry.instrument.detector.image_key)
            nxdata.makelink(nxentry.sample.rotation_angle)
            nxdata.makelink(nxentry.sample.x_translation)
            nxdata.makelink(nxentry.sample.z_translation)
#            nxdata.attrs['axes'] = ['field', 'row', 'column']
#            nxdata.attrs['field_indices'] = 0
#            nxdata.attrs['row_indices'] = 1
#            nxdata.attrs['column_indices'] = 2

        return nxroot


def nxcopy(nxobject, exclude_nxpaths=None, nxpath_prefix=''):
    """
    Function that returns a copy of a nexus object, optionally exluding
    certain child items.

    :param nxobject: the original nexus object to return a "copy" of
    :type nxobject: nexusformat.nexus.NXobject
    :param exlude_nxpaths: a list of paths to child nexus objects that
        should be exluded from the returned "copy", defaults to `[]`
    :type exclude_nxpaths: list[str], optional
    :param nxpath_prefix: For use in recursive calls from inside this
        function only!
    :type nxpath_prefix: str
    :return: a copy of `nxobject` with some children optionally exluded.
    :rtype: NXobject
    """
    # Third party modules
    from nexusformat.nexus import NXgroup

    nxobject_copy = nxobject.__class__()
    if not nxpath_prefix:
        if 'default' in nxobject.attrs:
            nxobject_copy.attrs['default'] = nxobject.attrs['default']
    else:
        for k, v in nxobject.attrs.items():
            nxobject_copy.attrs[k] = v

    if exclude_nxpaths is None:
        exclude_nxpaths = []
    for k, v in nxobject.items():
        nxpath = os_path.join(nxpath_prefix, k)

        if nxpath in exclude_nxpaths:
            continue

        if isinstance(v, NXgroup):
            nxobject_copy[k] = nxcopy(
                v, exclude_nxpaths=exclude_nxpaths,
                nxpath_prefix=os_path.join(nxpath_prefix, k))
        else:
            nxobject_copy[k] = v

    return nxobject_copy


class SetNumexprThreads:
    """
    Class that sets and keeps track of the number of processors used by
    the code in general and by the num_expr package specifically.

    :ivar num_core: Number of processors used by the num_expr package
    :type num_core: int
    """

    def __init__(self, num_core):
        """Initialize SetNumexprThreads."""
        # System modules
        from multiprocessing import cpu_count

        if num_core is None or num_core < 1 or num_core > cpu_count():
            self._num_core = cpu_count()
        else:
            self._num_core = num_core
        self._num_core_org = self._num_core

    def __enter__(self):
        # Third party modules
        from numexpr import (
            MAX_THREADS,
            set_num_threads,
        )

        self._num_core_org = set_num_threads(
            min(self._num_core, MAX_THREADS))

    def __exit__(self, exc_type, exc_value, traceback):
        # Third party modules
        from numexpr import set_num_threads

        set_num_threads(self._num_core_org)


class Tomo:
    """Reconstruct a set of tomographic images."""

    def __init__(
            self, interactive=False, num_core=-1, output_folder='.',
            save_figs=None, test_mode=False):
        """Initialize Tomo."""
        # System modules
        from logging import getLogger
        from multiprocessing import cpu_count

        self.__name__ = self.__class__.__name__
        self._logger = getLogger(self.__name__)
        self._logger.propagate = False

        if not isinstance(interactive, bool):
            raise ValueError(f'Invalid parameter interactive ({interactive})')
        self._interactive = interactive
        self._num_core = num_core
        self._output_folder = os_path.abspath(output_folder)
        if not os_path.isdir(self._output_folder):
            mkdir(self._output_folder)
        if self._interactive:
            self._test_mode = False
        else:
            if not isinstance(test_mode, bool):
                raise ValueError(f'Invalid parameter test_mode ({test_mode})')
            self._test_mode = test_mode
            if save_figs is None:
                save_figs = 'no'
        self._test_config = {}
        if self._test_mode:
            if save_figs != 'only':
                self._logger.warning('Ignoring save_figs in test mode')
            save_figs = 'only'
        if save_figs == 'only':
            self._save_only = True
            self._save_figs = True
        elif save_figs == 'yes':
            self._save_only = False
            self._save_figs = True
        elif save_figs == 'no':
            self._save_only = False
            self._save_figs = False
        else:
            raise ValueError(f'Invalid parameter save_figs ({save_figs})')
        if self._save_only:
            self._block = False
        else:
            self._block = True
        if self._num_core == -1:
            self._num_core = cpu_count()
        if not isinstance(self._num_core, int) or self._num_core < 0:
            raise ValueError(f'Invalid parameter num_core ({num_core})')
        if self._num_core > cpu_count():
            self._logger.warning(
                f'num_core = {self._num_core} is larger than the number '
                + f'of available processors and reduced to {cpu_count()}')
            self._num_core = cpu_count()

    def gen_reduced_data(self, data, img_x_bounds=None, delta_theta=None):
        """
        Generate the reduced tomography images.

        :param data: Data object containing the raw data info and
            metadata required for a tomography data reduction
        :type data: nexusformat.nexus.NXroot
        :param img_x_bounds: Detector image bounds in the x-direction
        :type img_x_bounds: tuple(int, int), list[int], optional
        :return: Reduced tomography data
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXprocess,
            NXroot,
        )

        self._logger.info('Generate the reduced tomography images')
        if img_x_bounds is not None:
            if not isinstance(img_x_bounds, (tuple, list)):
                raise ValueError(
                    f'Invalid parameter img_x_bounds ({img_x_bounds})')
            img_x_bounds = tuple(img_x_bounds)

        if isinstance(data, NXroot):
            nxentry = data[data.attrs['default']]
        else:
            raise ValueError(f'Invalid parameter data ({data})')

        # Create an NXprocess to store data reduction (meta)data
        reduced_data = NXprocess()

        # Generate dark field
        if 'dark_field' in nxentry['spec_scans']:
            reduced_data = self._gen_dark(nxentry, reduced_data)

        # Generate bright field
        reduced_data = self._gen_bright(nxentry, reduced_data)

        # Get rotation angles for image stacks
        thetas = self._gen_thetas(nxentry)

        # Get the image stack mask to remove bad images from stack
        image_mask = None
        drop_fraction = 0 # fraction of images dropped as a percentage
        if drop_fraction:
            if delta_theta is not None:
                delta_theta = None
                self._logger.warning(
                    'Ignore delta_theta when an image mask is used')
            np.random.seed(0)
            image_mask = np.where(np.random.rand(
                len(thetas)) < drop_fraction/100, 0, 1).astype(bool)

        # Set zoom and/or rotation angle interval to reduce memory
        #     requirement
        if image_mask is None:
            zoom_perc, delta_theta = self._set_zoom_or_delta_theta(
                thetas, delta_theta)
            if delta_theta is not None:
                image_mask = np.asarray(
                    [0 if i%delta_theta else 1
                        for i in range(len(thetas))], dtype=bool)
            self._logger.debug(f'zoom_perc: {zoom_perc}')
            self._logger.debug(f'delta_theta: {delta_theta}')
            if zoom_perc is not None:
                reduced_data.attrs['zoom_perc'] = zoom_perc
        if image_mask is not None:
            self._logger.debug(f'image_mask = {image_mask}')
            reduced_data['image_mask'] = image_mask
            thetas = thetas[image_mask]
        self._logger.debug(f'thetas = {thetas}')
        reduced_data['rotation_angle'] = thetas

        # Set vertical detector bounds for image stack
        img_x_bounds = self._set_detector_bounds(
            nxentry, reduced_data, thetas[0], img_x_bounds=img_x_bounds)
        self._logger.info(f'img_x_bounds = {img_x_bounds}')
        reduced_data['img_x_bounds'] = img_x_bounds

        # Generate reduced tomography fields
        reduced_data = self._gen_tomo(nxentry, reduced_data)

        # Create a copy of the input Nexus object and remove raw and
        #     any existing reduced data
        if isinstance(data, NXroot):
            exclude_items = [
                f'{nxentry.nxname}/reduced_data/data',
                f'{nxentry.nxname}/instrument/detector/data',
                f'{nxentry.nxname}/instrument/detector/image_key',
                f'{nxentry.nxname}/instrument/detector/sequence_number',
                f'{nxentry.nxname}/sample/rotation_angle',
                f'{nxentry.nxname}/sample/x_translation',
                f'{nxentry.nxname}/sample/z_translation',
                f'{nxentry.nxname}/data/data',
                f'{nxentry.nxname}/data/image_key',
                f'{nxentry.nxname}/data/rotation_angle',
                f'{nxentry.nxname}/data/x_translation',
                f'{nxentry.nxname}/data/z_translation',
            ]
            nxroot = nxcopy(data, exclude_nxpaths=exclude_items)
            nxentry = nxroot[nxroot.attrs['default']]

        # Add the reduced data NXprocess
        nxentry.reduced_data = reduced_data

        if 'data' not in nxentry:
            nxentry.data = NXdata()
        nxentry.attrs['default'] = 'data'
        nxentry.data.makelink(
            nxentry.reduced_data.data.tomo_fields, name='reduced_data')
        nxentry.data.makelink(
            nxentry.reduced_data.rotation_angle, name='rotation_angle')
        nxentry.data.attrs['signal'] = 'reduced_data'

        return nxroot

    def find_centers(self, nxroot, center_rows=None, center_stack_index=None,
            gaussian_sigma=None, ring_width=None):
        """
        Find the calibrated center axis info

        :param nxroot: Data object containing the reduced data and
            metadata required to find the calibrated center axis info
        :type data: nexusformat.nexus.NXroot
        :param center_rows: Lower and upper row indices for center
            finding
        :type center_rows: tuple(int, int), list[int], optional
        :return: Calibrated center axis info
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )
        from yaml import safe_dump

        self._logger.info('Find the calibrated center axis info')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if (center_rows is not None
                and (not isinstance(center_rows, (tuple, list))
                     or len(center_rows) != 2)):
            raise ValueError(f'Invalid parameter center_rows ({center_rows})')
        if (not self._interactive
                and (center_rows is None
                     or (center_rows[0] is None and center_rows[1] is None))):
            self._logger.warning(
                'center_rows unspecified, find centers at reduced data bounds')
        if (center_stack_index is not None
                and (not isinstance(center_stack_index, int)
                     or center_stack_index < 0)):
            raise ValueError(
                'Invalid parameter center_stack_index '
                + f'({center_stack_index})')

        # Check if reduced data is available
        if ('reduced_data' not in nxentry
                or 'reduced_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reduced data in {nxentry}.')

        # Select the image stack to calibrate the center axis
        #     reduced data axes order: stack,theta,row,column
        # Note: Nexus can't follow a link if the data it points to is
        #     too big get the data from the actual place, not from
        #     nxentry.data
        tomo_fields_shape = nxentry.reduced_data.data.tomo_fields.shape
        if (len(tomo_fields_shape) != 4
                or any(True for dim in tomo_fields_shape if not dim)):
            raise KeyError(
                'Unable to load the required reduced tomography stack')
        num_tomo_stacks = tomo_fields_shape[0]
        if num_tomo_stacks == 1:
            center_stack_index = 0
            default = 'n'
        else:
            if self._test_mode:
                # Convert input value to offset 0
                center_stack_index = self._test_config['center_stack_index']
            elif self._interactive:
                if center_stack_index is None:
                    center_stack_index = input_int(
                        '\nEnter tomography stack index to calibrate the '
                        + 'center axis', ge=0, lt=num_tomo_stacks,
                        default=int(num_tomo_stacks/2))
            else:
                if center_stack_index is None:
                    center_stack_index = int(num_tomo_stacks/2)
                    self._logger.warning(
                        'center_stack_index unspecified, use stack '
                        + f'{center_stack_index} to find centers')
            default = 'y'

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)
        assert len(thetas) == tomo_fields_shape[1]

        # Get effective pixel_size
        if 'zoom_perc' in nxentry.reduced_data:
            eff_pixel_size = \
                100. * (nxentry.instrument.detector.x_pixel_size
                         / nxentry.reduced_data.attrs['zoom_perc'])
        else:
            eff_pixel_size = nxentry.instrument.detector.x_pixel_size

        # Get cross sectional diameter
        cross_sectional_dim = tomo_fields_shape[3]*eff_pixel_size
        self._logger.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        self._logger.info('Determine center offset at sample row boundaries')

        # Lower row center
        if self._test_mode:
            lower_row = self._test_config['lower_row']
        elif self._interactive:
            if center_rows is not None and center_rows[0] is not None:
                lower_row = center_rows[0]
                if lower_row == -1:
                    lower_row = 0
                if not 0 <= lower_row < tomo_fields_shape[2]-1:
                    raise ValueError(
                        f'Invalid parameter center_rows ({center_rows})')
            else:
                lower_row = select_one_image_bound(
                    nxentry.reduced_data.data.tomo_fields[
                        center_stack_index,0,:,:],
                    0, bound=0, title=f'theta={round(thetas[0], 2)+0}',
                    bound_name='row index to find lower center',
                    default=default, raise_error=True)
        else:
            if center_rows is None or center_rows[0] is None:
                lower_row = 0
            else:
                lower_row = center_rows[0]
                if lower_row == -1:
                    lower_row = 0
                if not 0 <= lower_row < tomo_fields_shape[2]-1:
                    raise ValueError(
                        f'Invalid parameter center_rows ({center_rows})')
        t0 = time()
        lower_center_offset = self._find_center_one_plane(
            nxentry.reduced_data.data.tomo_fields[
                center_stack_index,:,lower_row,:],
            lower_row, thetas, eff_pixel_size, cross_sectional_dim,
            path=self._output_folder, num_core=self._num_core,
            gaussian_sigma=gaussian_sigma, ring_width=ring_width)
        self._logger.info(f'Finding center took {time()-t0:.2f} seconds')
        self._logger.debug(f'lower_row = {lower_row:.2f}')
        self._logger.debug(f'lower_center_offset = {lower_center_offset:.2f}')

        # Upper row center
        if self._test_mode:
            upper_row = self._test_config['upper_row']
        elif self._interactive:
            if center_rows is not None and center_rows[1] is not None:
                upper_row = center_rows[1]
                if upper_row == -1:
                    upper_row = tomo_fields_shape[2]-1
                if not lower_row < upper_row < tomo_fields_shape[2]:
                    raise ValueError(
                        f'Invalid parameter center_rows ({center_rows})')
            else:
                upper_row = select_one_image_bound(
                    nxentry.reduced_data.data.tomo_fields[
                        center_stack_index,0,:,:],
                    0, bound=tomo_fields_shape[2]-1,
                    title=f'theta = {round(thetas[0], 2)+0}',
                    bound_name='row index to find upper center',
                    default=default, raise_error=True)
        else:
            if center_rows is None or center_rows[1] is None:
                upper_row = tomo_fields_shape[2]-1
            else:
                upper_row = center_rows[1]
                if upper_row == -1:
                    upper_row = tomo_fields_shape[2]-1
                if not lower_row < upper_row < tomo_fields_shape[2]:
                    raise ValueError(
                        f'Invalid parameter center_rows ({center_rows})')
        t0 = time()
        upper_center_offset = self._find_center_one_plane(
            nxentry.reduced_data.data.tomo_fields[
                center_stack_index,:,upper_row,:],
            upper_row, thetas, eff_pixel_size, cross_sectional_dim,
            path=self._output_folder, num_core=self._num_core,
            gaussian_sigma=gaussian_sigma, ring_width=ring_width)
        self._logger.info(f'Finding center took {time()-t0:.2f} seconds')
        self._logger.debug(f'upper_row = {upper_row:.2f}')
        self._logger.debug(f'upper_center_offset = {upper_center_offset:.2f}')

        center_config = {
            'lower_row': lower_row,
            'lower_center_offset': lower_center_offset,
            'upper_row': upper_row,
            'upper_center_offset': upper_center_offset,
        }
        if num_tomo_stacks > 1:
            center_config['center_stack_index'] = center_stack_index

        # Save test data to file
        if self._test_mode:
            with open(f'{self._output_folder}/center_config.yaml', 'w',
                      encoding='utf8') as f:
                safe_dump(center_config, f)

        return center_config

    def reconstruct_data(
            self, nxroot, center_info, x_bounds=None, y_bounds=None,
            z_bounds=None, secondary_iters=0, remove_stripe_sigma=None,
            ring_width=None):
        """
        Reconstruct the tomography data.

        :param nxroot: Reduced data
        :type data: nexusformat.nexus.NXroot
        :param center_info: Calibrated center axis info
        :type center_info: dict
        :param x_bounds: Reconstructed image bounds in the x-direction
        :type x_bounds: tuple(int, int), list[int], optional
        :param y_bounds: Reconstructed image bounds in the y-direction
        :type y_bounds: tuple(int, int), list[int], optional
        :param z_bounds: Reconstructed image bounds in the z-direction
        :type z_bounds: tuple(int, int), list[int], optional
        :return: Reconstructed tomography data
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXentry,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.common.utils.general import is_int_pair

        self._logger.info('Reconstruct the tomography data')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if not isinstance(center_info, dict):
            raise ValueError(f'Invalid parameter center_info ({center_info})')
        if x_bounds is not None:
            if not isinstance(x_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            x_bounds = tuple(x_bounds)
        if y_bounds is not None:
            if not isinstance(y_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            y_bounds = tuple(y_bounds)
        if z_bounds is not None:
            if not isinstance(z_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            z_bounds = tuple(z_bounds)

        # Check if reduced data is available
        if ('reduced_data' not in nxentry
                or 'reduced_data' not in nxentry.data):
            raise KeyError(f'Unable to find valid reduced data in {nxentry}.')

        # Create an NXprocess to store image reconstruction (meta)data
        nxprocess = NXprocess()

        # Get rotation axis rows and centers
        lower_row = center_info.get('lower_row')
        lower_center_offset = center_info.get('lower_center_offset')
        upper_row = center_info.get('upper_row')
        upper_center_offset = center_info.get('upper_center_offset')
        if (lower_row is None or lower_center_offset is None
                or upper_row is None or upper_center_offset is None):
            raise KeyError(
                'Unable to find valid calibrated center axis info in '
                + f'{center_info}.')
        center_slope = (upper_center_offset-lower_center_offset) \
            / (upper_row-lower_row)

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Reconstruct tomography data
        #     reduced data axes order: stack,theta,row,column
        #     reconstructed data order in each stack: row/z,x,y
        # Note: Nexus can't follow a link if the data it points to is
        #     too big get the data from the actual place, not from
        #     nxentry.data
        if 'zoom_perc' in nxentry.reduced_data:
            res_title = f'{nxentry.reduced_data.attrs["zoom_perc"]}p'
        else:
            res_title = 'fullres'
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        tomo_recon_stacks = num_tomo_stacks*[np.array([])]
        for i in range(num_tomo_stacks):
            # Convert reduced data stack from theta,row,column to
            #     row,theta,column
            t0 = time()
            tomo_stack = np.asarray(nxentry.reduced_data.data.tomo_fields[i])
            self._logger.info(
                f'Reading reduced data stack {i} took {time()-t0:.2f} '
                + 'seconds')
            if (len(tomo_stack.shape) != 3
                    or any(True for dim in tomo_stack.shape if not dim)):
                raise RuntimeError(
                    f'Unable to load tomography stack {i} for '
                    + 'reconstruction')
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)
            assert len(thetas) == tomo_stack.shape[1]
            assert 0 <= lower_row < upper_row < tomo_stack.shape[0]
            center_offsets = [
                lower_center_offset-lower_row*center_slope,
                upper_center_offset + center_slope * (
                    tomo_stack.shape[0]-1-upper_row),
            ]
            t0 = time()
            tomo_recon_stack = self._reconstruct_one_tomo_stack(
                tomo_stack, thetas, center_offsets=center_offsets,
                num_core=self._num_core, algorithm='gridrec',
                secondary_iters=secondary_iters,
                remove_stripe_sigma=remove_stripe_sigma, ring_width=ring_width)
            self._logger.info(
                f'Reconstruction of stack {i} took {time()-t0:.2f} seconds')

            # Combine stacks
            tomo_recon_stacks[i] = tomo_recon_stack

        # Resize the reconstructed tomography data
        #     reconstructed data order in each stack: row/z,x,y
        if self._test_mode:
            x_bounds = tuple(self._test_config.get('x_bounds'))
            y_bounds = tuple(self._test_config.get('y_bounds'))
            z_bounds = None
        elif self._interactive:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
                tomo_recon_stacks, x_bounds=x_bounds, y_bounds=y_bounds,
                z_bounds=z_bounds)
        else:
            if x_bounds == (-1, -1):
                x_bounds = None
            if x_bounds is None:
                self._logger.warning(
                    'x_bounds unspecified, reconstruct data for full x-range')
            elif not is_int_pair(x_bounds, ge=0,
                                 lt=tomo_recon_stacks[0].shape[1]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            if y_bounds == (-1, -1):
                y_bounds = None
            if y_bounds is None:
                self._logger.warning(
                    'y_bounds unspecified, reconstruct data for full y-range')
            elif not is_int_pair(
                    y_bounds, ge=0, lt=tomo_recon_stacks[0].shape[2]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = None
        if x_bounds is None:
            x_range = (0, tomo_recon_stacks[0].shape[1])
            x_slice = int(x_range[1]/2)
        else:
            x_range = (min(x_bounds), max(x_bounds))
            x_slice = int((x_bounds[0]+x_bounds[1]) / 2)
        if y_bounds is None:
            y_range = (0, tomo_recon_stacks[0].shape[2])
            y_slice = int(y_range[1] / 2)
        else:
            y_range = (min(y_bounds), max(y_bounds))
            y_slice = int((y_bounds[0]+y_bounds[1]) / 2)
        if z_bounds is None:
            z_range = (0, tomo_recon_stacks[0].shape[0])
            z_slice = int(z_range[1] / 2)
        else:
            z_range = (min(z_bounds), max(z_bounds))
            z_slice = int((z_bounds[0]+z_bounds[1]) / 2)

        # Plot a few reconstructed image slices
        if self._save_figs:
            for i, stack in enumerate(tomo_recon_stacks):
                if num_tomo_stacks == 1:
                    basetitle = 'recon'
                else:
                    basetitle = f'recon stack {i}'
                title = f'{basetitle} {res_title} xslice{x_slice}'
                quick_imshow(
                    stack[z_range[0]:z_range[1],x_slice,y_range[0]:y_range[1]],
                    title=title, path=self._output_folder, save_fig=True,
                    save_only=True)
                title = f'{basetitle} {res_title} yslice{y_slice}'
                quick_imshow(
                    stack[z_range[0]:z_range[1],x_range[0]:x_range[1],y_slice],
                    title=title, path=self._output_folder, save_fig=True,
                    save_only=True)
                title = f'{basetitle} {res_title} zslice{z_slice}'
                quick_imshow(
                    stack[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                    title=title, path=self._output_folder, save_fig=True,
                    save_only=True)

        # Save test data to file
        #     reconstructed data order in each stack: row/z,x,y
        if self._test_mode:
            for i, stack in enumerate(tomo_recon_stacks):
                np.savetxt(
                    f'{self._output_folder}/recon_stack_{i}.txt',
                    stack[z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                    fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #     reconstructed data order in each stack: row/z,x,y
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        for k, v in center_info.items():
            nxprocess[k] = v
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
        nxprocess.data['reconstructed_data'] = np.asarray(
            [stack[z_range[0]:z_range[1],x_range[0]:x_range[1],
             y_range[0]:y_range[1]] for stack in tomo_recon_stacks])
        nxprocess.data.attrs['signal'] = 'reconstructed_data'

        # Create a copy of the input Nexus object and remove reduced
        #     data
        exclude_items = [
            f'{nxentry.nxname}/reduced_data/data',
            f'{nxentry.nxname}/data/reduced_data',
        ]
        nxroot_copy = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the reconstructed data NXprocess to the new Nexus object
        nxentry_copy = nxroot_copy[nxroot_copy.attrs['default']]
        nxentry_copy.reconstructed_data = nxprocess
        if 'data' not in nxentry_copy:
            nxentry_copy.data = NXdata()
        nxentry_copy.attrs['default'] = 'data'
        nxentry_copy.data.makelink(
            nxprocess.data.reconstructed_data, name='reconstructed_data')
        nxentry_copy.data.attrs['signal'] = 'reconstructed_data'

        return nxroot_copy

    def combine_data(
            self, nxroot, x_bounds=None, y_bounds=None, z_bounds=None):
        """Combine the reconstructed tomography stacks.

        :param nxroot: A stack of reconstructed tomography datasets
        :type data: nexusformat.nexus.NXroot
        :param x_bounds: Combined image bounds in the x-direction
        :type x_bounds: tuple(int, int), list[int], optional
        :param y_bounds: Combined image bounds in the y-direction
        :type y_bounds: tuple(int, int), list[int], optional
        :param z_bounds: Combined image bounds in the z-direction
        :type z_bounds: tuple(int, int), list[int], optional
        :return: Combined reconstructed tomography data
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXentry,
            NXprocess,
            NXroot,
        )

        # Local modules
        from CHAP.common.utils.general import is_int_pair

        self._logger.info('Combine the reconstructed tomography stacks')

        if not isinstance(nxroot, NXroot):
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        nxentry = nxroot[nxroot.attrs['default']]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid nxentry ({nxentry})')
        if x_bounds is not None:
            if not isinstance(x_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            x_bounds = tuple(x_bounds)
        if y_bounds is not None:
            if not isinstance(y_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            y_bounds = tuple(y_bounds)
        if z_bounds is not None:
            if not isinstance(z_bounds, (tuple, list)):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            z_bounds = tuple(z_bounds)

        # Check if reconstructed image data is available
        if ('reconstructed_data' not in nxentry
                or 'reconstructed_data' not in nxentry.data):
            raise KeyError(
                f'Unable to find valid reconstructed image data in {nxentry}')

        # Create an NXprocess to store combined image reconstruction
        #     (meta)data
        nxprocess = NXprocess()

        # Get the reconstructed data
        #     reconstructed data order: stack,row(z),x,y
        # Note: Nexus can't follow a link if the data it points to is
        #     too big get the data from the actual place, not from
        #     nxentry.data
        num_tomo_stacks = \
            nxentry.reconstructed_data.data.reconstructed_data.shape[0]
        if num_tomo_stacks == 1:
            self._logger.info('Only one stack available: leaving combine_data')
            return None

        # Combine the reconstructed stacks
        # (load one stack at a time to reduce risk of hitting Nexus
        #     data access limit)
        t0 = time()
        tomo_recon_combined = np.asarray(
            nxentry.reconstructed_data.data.reconstructed_data[0])
        if num_tomo_stacks > 2:
            tomo_recon_combined = np.concatenate(
                [tomo_recon_combined]
                + [nxentry.reconstructed_data.data.reconstructed_data[i]
                   for i in range(1, num_tomo_stacks-1)])
        if num_tomo_stacks > 1:
            tomo_recon_combined = np.concatenate(
                [tomo_recon_combined]
                + [nxentry.reconstructed_data.data.reconstructed_data[
                   num_tomo_stacks-1]])
        self._logger.info(
            f'Combining the reconstructed stacks took {time()-t0:.2f} seconds')

        # Resize the combined tomography data stacks
        #     combined data order: row/z,x,y
        if self._test_mode:
            x_bounds = None
            y_bounds = None
            z_bounds = tuple(self._test_config.get('z_bounds'))
        elif self._interactive:
            if x_bounds is None and x_bounds in nxentry.reconstructed_data:
                x_bounds = (-1, -1)
            if y_bounds is None and y_bounds in nxentry.reconstructed_data:
                y_bounds = (-1, -1)
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
                tomo_recon_combined, z_only=True)
        else:
            if x_bounds is None:
                self._logger.warning(
                    'x_bounds unspecified, reconstruct data for full x-range')
            elif not is_int_pair(
                    x_bounds, ge=0, lt=tomo_recon_combined.shape[1]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            if y_bounds is None:
                self._logger.warning(
                    'y_bounds unspecified, reconstruct data for full y-range')
            elif not is_int_pair(
                    y_bounds, ge=0, lt=tomo_recon_combined.shape[2]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = None
        if x_bounds is None:
            x_range = (0, tomo_recon_combined.shape[1])
            x_slice = int(x_range[1]/2)
        else:
            x_range = x_bounds
            x_slice = int((x_bounds[0]+x_bounds[1]) / 2)
        if y_bounds is None:
            y_range = (0, tomo_recon_combined.shape[2])
            y_slice = int(y_range[1]/2)
        else:
            y_range = y_bounds
            y_slice = int((y_bounds[0]+y_bounds[1]) / 2)
        if z_bounds is None:
            z_range = (0, tomo_recon_combined.shape[0])
            z_slice = int(z_range[1]/2)
        else:
            z_range = z_bounds
            z_slice = int((z_bounds[0]+z_bounds[1]) / 2)

        # Plot a few combined image slices
        if self._save_figs:
            quick_imshow(
                tomo_recon_combined[
                    z_range[0]:z_range[1],x_slice,y_range[0]:y_range[1]],
                title=f'recon combined xslice{x_slice}',
                path=self._output_folder, save_fig=True, save_only=True)
            quick_imshow(
                tomo_recon_combined[
                    z_range[0]:z_range[1],x_range[0]:x_range[1],y_slice],
                title=f'recon combined yslice{y_slice}',
                path=self._output_folder, save_fig=True, save_only=True)
            quick_imshow(
                tomo_recon_combined[
                    z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                title=f'recon combined zslice{z_slice}',
                path=self._output_folder, save_fig=True, save_only=True)

        # Save test data to file
        #     combined data order: row/z,x,y
        if self._test_mode:
            np.savetxt(
                f'{self._output_folder}/recon_combined.txt',
                tomo_recon_combined[
                    z_slice,x_range[0]:x_range[1],y_range[0]:y_range[1]],
                fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #     combined data order: row/z,x,y
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
        nxprocess.data['combined_data'] = tomo_recon_combined[
            z_range[0]:z_range[1],x_range[0]:x_range[1],y_range[0]:y_range[1]]
        nxprocess.data.attrs['signal'] = 'combined_data'

        # Create a copy of the input Nexus object and remove
        #     reconstructed data
        exclude_items = [
            f'{nxentry.nxname}/reconstructed_data/data',
            f'{nxentry.nxname}/data/reconstructed_data',
        ]
        nxroot_copy = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the combined data NXprocess to the new Nexus object
        nxentry_copy = nxroot_copy[nxroot_copy.attrs['default']]
        nxentry_copy.combined_data = nxprocess
        if 'data' not in nxentry_copy:
            nxentry_copy.data = NXdata()
        nxentry_copy.attrs['default'] = 'data'
        nxentry_copy.data.makelink(
            nxprocess.data.combined_data, name='combined_data')
        nxentry_copy.data.attrs['signal'] = 'combined_data'

        return nxroot_copy

    def _gen_dark(self, nxentry, reduced_data):
        """Generate dark field."""
        # Third party modules
        from nexusformat.nexus import NXdata

        # Local modules
        from CHAP.common.models.map import (
            get_scanparser,
            import_scanparser,
        )

        # Get the dark field images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is not None and 'data' in nxentry.instrument.detector:
            field_indices = [
                index for index, key in enumerate(image_key) if key == 2]
            tdf_stack = nxentry.instrument.detector.data[field_indices,:,:]
            # RV the default NXtomo form does not accomodate dark field
            #     stacks
        else:
            import_scanparser(
                nxentry.instrument.source.attrs['station'],
                nxentry.instrument.source.attrs['experiment_type'])
            dark_field_scans = nxentry.spec_scans.dark_field
            detector_prefix = str(nxentry.instrument.detector.local_name)
            tdf_stack = []
            for nxsubentry_name, nxsubentry in dark_field_scans.items():
                scan_number = int(nxsubentry_name.split('_')[-1])
                scanparser = get_scanparser(
                    dark_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(
                    nxsubentry.instrument.detector.frame_start_number)
                num_image = len(nxsubentry.sample.rotation_angle)
                tdf_stack.append(
                    scanparser.get_detector_data(
                        detector_prefix,
                        (image_offset, image_offset+num_image)))
            if isinstance(tdf_stack, list):
                assert len(tdf_stack) == 1  # RV
                tdf_stack = tdf_stack[0]

        # Take median
        if tdf_stack.ndim == 2:
            tdf = tdf_stack
        elif tdf_stack.ndim == 3:
            tdf = np.median(tdf_stack, axis=0)
            del tdf_stack
        else:
            raise RuntimeError(f'Invalid tdf_stack shape ({tdf_stack.shape})')

        # Remove dark field intensities above the cutoff
#        tdf_cutoff = None
        tdf_cutoff = tdf.min() + 2 * (np.median(tdf)-tdf.min())
        self._logger.debug(f'tdf_cutoff = {tdf_cutoff}')
        if tdf_cutoff is not None:
            if not isinstance(tdf_cutoff, (int, float)) or tdf_cutoff < 0:
                self._logger.warning(
                    f'Ignoring illegal value of tdf_cutoff {tdf_cutoff}')
            else:
                tdf[tdf > tdf_cutoff] = np.nan
                self._logger.debug(f'tdf_cutoff = {tdf_cutoff}')

        # Remove nans
        tdf_mean = np.nanmean(tdf)
        self._logger.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(
            tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)

        # Plot dark field
        if self._save_figs:
            quick_imshow(
                tdf, title='dark field', path=self._output_folder,
                save_fig=True, save_only=True)

        # Add dark field to reduced data NXprocess
        reduced_data.data = NXdata()
        reduced_data.data['dark_field'] = tdf

        return reduced_data

    def _gen_bright(self, nxentry, reduced_data):
        """Generate bright field."""
        # Third party modules
        from nexusformat.nexus import NXdata

        # Local modules
        from CHAP.common.models.map import (
            get_scanparser,
            import_scanparser,
        )

        # Get the bright field images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is not None and 'data' in nxentry.instrument.detector:
            field_indices = [
                index for index, key in enumerate(image_key) if key == 1]
            tbf_stack = nxentry.instrument.detector.data[field_indices,:,:]
            # RV the default NXtomo form does not accomodate bright
            #     field stacks
        else:
            import_scanparser(
                nxentry.instrument.source.attrs['station'],
                nxentry.instrument.source.attrs['experiment_type'])
            bright_field_scans = nxentry.spec_scans.bright_field
            detector_prefix = str(nxentry.instrument.detector.local_name)
            tbf_stack = []
            for nxsubentry_name, nxsubentry in bright_field_scans.items():
                scan_number = int(nxsubentry_name.split('_')[-1])
                scanparser = get_scanparser(
                    bright_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(
                    nxsubentry.instrument.detector.frame_start_number)
                num_image = len(nxsubentry.sample.rotation_angle)
                tbf_stack.append(
                    scanparser.get_detector_data(
                        detector_prefix,
                        (image_offset, image_offset+num_image)))
            if isinstance(tbf_stack, list):
                assert len(tbf_stack) == 1  # RV
                tbf_stack = tbf_stack[0]

        # Take median if more than one image
        #
        # Median or mean: It may be best to try the median because of
        # some image artifacts that arise due to crinkles in the
        # upstream kapton tape windows causing some phase contrast
        # images to appear on the detector.
        #
        # One thing that also may be useful in a future implementation
        # is to do a brightfield adjustment on EACH frame of the tomo
        # based on a ROI in the corner of the frame where there is no
        # sample but there is the direct X-ray beam because there is
        # frame to frame fluctuations from the incoming beam. We don’t
        # typically account for them but potentially could.
        if tbf_stack.ndim == 2:
            tbf = tbf_stack
        elif tbf_stack.ndim == 3:
            tbf = np.median(tbf_stack, axis=0)
            del tbf_stack
        else:
            raise RuntimeError(f'Invalid tbf_stack shape ({tbf_stack.shape})')

        # Subtract dark field
        if 'data' in reduced_data and 'dark_field' in reduced_data.data:
            tbf -= reduced_data.data.dark_field
        else:
            self._logger.warning('Dark field unavailable')

        # Set any non-positive values to one
        # (avoid negative bright field values for spikes in dark field)
        tbf[tbf < 1] = 1

        # Plot bright field
        if self._save_figs:
            quick_imshow(
                tbf, title='bright field', path=self._output_folder,
                save_fig=True, save_only=True)

        # Add bright field to reduced data NXprocess
        if 'data' not in reduced_data:
            reduced_data.data = NXdata()
        reduced_data.data['bright_field'] = tbf

        return reduced_data

    def _set_detector_bounds(self, nxentry, reduced_data, theta,
            img_x_bounds=None):
        """
        Set vertical detector bounds for each image stack.Right now the
        range is the same for each set in the image stack.
        """
        # Local modules
        from CHAP.common.models.map import (
            get_scanparser,
            import_scanparser,
        )
        from CHAP.common.utils.general import is_index_range

        if self._test_mode:
            return tuple(self._test_config['img_x_bounds'])

        # Get the first tomography image and the reference heights
        image_mask = reduced_data.get('image_mask')
        if image_mask is None:
            first_image_index = 0
        else:
            image_mask = np.asarray(image_mask)
            first_image_index = int(np.argmax(image_mask))
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is not None and 'data' in nxentry.instrument.detector:
            field_indices = [
                index for index, key in enumerate(image_key) if key == 0]
            field_indices_masked = field_indices[image_mask]
            first_image = np.asarray(nxentry.instrument.detector.data[
                field_indices[first_image_index,:,:]])
            assert theta == float(nxentry.sample.rotation_angle[
                field_indices[first_image_index]])
            z_translation_all = \
                nxentry.sample.z_translation[field_indices_masked]
            vertical_shifts = sorted(list(set(z_translation_all)))
            num_tomo_stacks = len(vertical_shifts)
        else:
            import_scanparser(
                nxentry.instrument.source.attrs['station'],
                nxentry.instrument.source.attrs['experiment_type'])
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            num_tomo_stacks = len(tomo_field_scans.keys())
            center_stack_index = int(num_tomo_stacks/2)
            detector_prefix = str(nxentry.instrument.detector.local_name)
            vertical_shifts = []
            for i, nxsubentry in enumerate(tomo_field_scans.items()):
                scan_number = int(nxsubentry[0].split('_')[-1])
                scanparser = get_scanparser(
                    tomo_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(
                    nxsubentry[1].instrument.detector.frame_start_number)
                vertical_shifts.append(nxsubentry[1].sample.z_translation)
                if i == center_stack_index:
                    first_image = scanparser.get_detector_data(
                        detector_prefix, image_offset+first_image_index)
                    assert theta == float(
                        nxsubentry[1].sample.rotation_angle[first_image_index])

        # Select image bounds
        title = f'tomography image at theta={round(theta, 2)+0}'
        if img_x_bounds == (-1, -1):
            img_x_bounds = None
        if img_x_bounds is not None:
            if is_index_range(img_x_bounds, ge=0, le=first_image.shape[0]):
                return img_x_bounds
            if self._interactive:
                self._logger.warning(
                    f'Invalid parameter img_x_bounds ({img_x_bounds}), '
                    + 'ignoring img_x_bounds')
                img_x_bounds = None
            else:
                raise ValueError(
                    f'Invalid parameter img_x_bounds ({img_x_bounds})')
        if nxentry.instrument.source.attrs['station'] in ('id1a3', 'id3a'):
            pixel_size = nxentry.instrument.detector.x_pixel_size
            # Try to get a fit from the bright field
            tbf = np.asarray(reduced_data.data.bright_field)
            tbf_shape = tbf.shape
            x_sum = np.sum(tbf, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            fit = Fit.fit_data(
                x_sum, 'rectangle', x=np.array(range(len(x_sum))),
                form='atan', guess=True)
            parameters = fit.best_values
            x_low_fit = parameters.get('center1', None)
            x_upp_fit = parameters.get('center2', None)
            sig_low = parameters.get('sigma1', None)
            sig_upp = parameters.get('sigma2', None)
            have_fit = (fit.success and x_low_fit is not None
                        and x_upp_fit is not None and sig_low is not None
                        and sig_upp is not None
                        and 0 <= x_low_fit < x_upp_fit <= x_sum.size
                        and (sig_low+sig_upp) / (x_upp_fit-x_low_fit) < 0.1)
            if have_fit:
                # Set a 5% margin on each side
                margin = 0.05 * (x_upp_fit-x_low_fit)
                x_low_fit = max(0, x_low_fit-margin)
                x_upp_fit = min(tbf_shape[0], x_upp_fit+margin)
            if num_tomo_stacks == 1:
                if have_fit:
                    # Set the default range to enclose the full fitted
                    #     window
                    x_low = int(x_low_fit)
                    x_upp = int(x_upp_fit)
                else:
                    # Center a default range of 1 mm
                    # RV can we get this from the slits?
                    num_x_min = int((1. + 0.5*pixel_size) / pixel_size)
                    x_low = int((tbf_shape[0]-num_x_min) / 2)
                    x_upp = x_low+num_x_min
            else:
                # Get the default range from the reference heights
                delta_z = vertical_shifts[1]-vertical_shifts[0]
                for i in range(2, num_tomo_stacks):
                    delta_z = min(
                        delta_z, vertical_shifts[i]-vertical_shifts[i-1])
                self._logger.debug(f'delta_z = {delta_z}')
                num_x_min = int((delta_z + 0.5*pixel_size) / pixel_size)
                self._logger.debug(f'num_x_min = {num_x_min}')
                if num_x_min > tbf_shape[0]:
                    self._logger.warning(
                        'Image bounds and pixel size prevent seamless '
                        + 'stacking')
                if have_fit:
                    # Center the default range relative to the fitted
                    #     window
                    x_low = int((x_low_fit+x_upp_fit-num_x_min) / 2)
                    x_upp = x_low+num_x_min
                else:
                    # Center the default range
                    x_low = int((tbf_shape[0]-num_x_min) / 2)
                    x_upp = x_low+num_x_min
            if not self._interactive:
                img_x_bounds = (x_low, x_upp)
            else:
                tmp = np.copy(tbf)
                tmp_max = tmp.max()
                tmp[x_low,:] = tmp_max
                tmp[x_upp-1,:] = tmp_max
                quick_imshow(tmp, title='bright field')
                tmp = np.copy(first_image)
                tmp_max = tmp.max()
                tmp[x_low,:] = tmp_max
                tmp[x_upp-1,:] = tmp_max
                quick_imshow(tmp, title=title)
                del tmp
                quick_plot(
                    (range(x_sum.size), x_sum),
                    ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                    ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                    title='sum over theta and y')
                print(f'lower bound = {x_low} (inclusive)')
                print(f'upper bound = {x_upp} (exclusive)]')
                accept = input_yesno('Accept these bounds (y/n)?', 'y')
                clear_imshow('bright field')
                clear_imshow(title)
                clear_plot('sum over theta and y')
                if accept:
                    img_x_bounds = (x_low, x_upp)
                else:
                    while True:
                        _, img_x_bounds = draw_mask_1d(
                            x_sum, title='select x data range',
                            ylabel='sum over theta and y')
                        if len(img_x_bounds) == 1:
                            break
                        print('Choose a single connected data range')
                    img_x_bounds = tuple(img_x_bounds[0])
            if (num_tomo_stacks > 1
                    and (img_x_bounds[1]-img_x_bounds[0]+1)
                         < int((delta_z - 0.5*pixel_size) / pixel_size)):
                self._logger.warning(
                    'Image bounds and pixel size prevent seamless stacking')
        else:
            if num_tomo_stacks > 1:
                raise NotImplementedError(
                    'Selecting image bounds for multiple stacks on FMB')
            # For FMB: use the first tomography image to select range
            # RV revisit if they do tomography with multiple stacks
            x_sum = np.sum(first_image, 1)
            x_sum_min = x_sum.min()
            x_sum_max = x_sum.max()
            if self._interactive:
                print(
                    'Select vertical data reduction range from first '
                    + 'tomography image')
                img_x_bounds = select_image_bounds(first_image, 0, title=title)
                if img_x_bounds is None:
                    raise RuntimeError('Unable to select image bounds')
            else:
                if img_x_bounds is None:
                    self._logger.warning(
                        'img_x_bounds unspecified, reduce data for entire '
                        + 'detector range')
                    img_x_bounds = (0, first_image.shape[0])

        # Plot results
        if self._save_figs:
            x_low = img_x_bounds[0]
            x_upp = img_x_bounds[1]
            tmp = np.copy(first_image)
            tmp_max = tmp.max()
            tmp[x_low,:] = tmp_max
            tmp[x_upp-1,:] = tmp_max
            quick_imshow(
                tmp, title=title, path=self._output_folder, save_fig=True,
                save_only=True)
            quick_plot(
                (range(x_sum.size), x_sum),
                ([x_low, x_low], [x_sum_min, x_sum_max], 'r-'),
                ([x_upp, x_upp], [x_sum_min, x_sum_max], 'r-'),
                title='sum over theta and y', path=self._output_folder,
                save_fig=True, save_only=True)
            del tmp

        return img_x_bounds

    def _gen_thetas(self, nxentry):
        """Get the rotation angles for the image stacks."""

        # Get the rotation angles
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is not None and 'data' in nxentry.instrument.detector:
            field_indices_all = [
                index for index, key in enumerate(image_key) if key == 0]
            z_translation_all = nxentry.sample.z_translation[field_indices_all]
            z_translation_levels = sorted(list(set(z_translation_all)))
            thetas = None
            for i, z_translation in enumerate(z_translation_levels):
                field_indices = [
                    field_indices_all[index]
                    for index, z in enumerate(z_translation_all)
                    if z == z_translation]
                sequence_numbers = \
                    nxentry.instrument.detector.sequence_number[field_indices]
                assert (list(sequence_numbers)
                        == list(range((len(sequence_numbers)))))
                if thetas is None:
                    thetas = np.asarray(
                        nxentry.sample.rotation_angle[
                            field_indices])[sequence_numbers]
                else:
                    assert all(
                        thetas[i] == nxentry.sample.rotation_angle[
                            field_indices[index]]
                        for i, index in enumerate(sequence_numbers))
        else:
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            thetas = None
            for nxsubentry_name, nxsubentry in tomo_field_scans.items():
                if thetas is None:
                    thetas = np.asarray(nxsubentry.sample.rotation_angle)
                else:
                    assert all(
                        thetas[i] == theta for i, theta
                            in enumerate(nxsubentry.sample.rotation_angle))

        return thetas

    def _set_zoom_or_delta_theta(self, thetas, delta_theta=None):
        """
        Set zoom and/or delta theta to reduce memory the requirement
        for the analysis.
        """
        # Local modules
        from CHAP.common.utils.general import index_nearest

        if self._test_mode:
            return tuple(self._test_config['delta_theta'])

#        if input_yesno(
#                '\nDo you want to zoom in to reduce memory '
#                + 'requirement (y/n)?', 'n'):
#            zoom_perc = input_int(
#                '    Enter zoom percentage', ge=1, le=100)
#        else:
#            zoom_perc = None
        zoom_perc = None

        if delta_theta is not None and not is_num(delta_theta, gt=0):
            self._logger.warning(
                f'Invalid parameter delta_theta ({delta_theta}), '
                + 'ignoring delta_theta')
            delta_theta = None
        if self._interactive:
            if delta_theta is None:
                delta_theta = thetas[1]-thetas[0]
            print(f'Available theta range: [{thetas[0]}, {thetas[-1]}]')
            print(f'Current theta interval: {delta_theta}')
            if input_yesno(
                    'Do you want to change the theta interval to reduce the '
                    + 'memory requirement (y/n)?', 'n'):
                delta_theta = input_num(
                    '    Enter the desired theta interval',
                    ge=thetas[1]-thetas[0], lt=(thetas[-1]-thetas[0])/2)
        if delta_theta is not None:
            delta_theta = index_nearest(thetas, thetas[0]+delta_theta)
            if delta_theta <= 1:
                delta_theta = None

        return zoom_perc, delta_theta

    def _gen_tomo(self, nxentry, reduced_data):
        """Generate tomography fields."""
        # Third party modules
        from numexpr import evaluate
        from scipy.ndimage import zoom

        # Local modules
        from CHAP.common.models.map import (
            get_scanparser,
            import_scanparser,
        )

        # Get full bright field
        tbf = np.asarray(reduced_data.data.bright_field)
        tbf_shape = tbf.shape

        # Get image bounds
        img_x_bounds = tuple(
            reduced_data.get('img_x_bounds', (0, tbf_shape[0])))
        img_y_bounds = tuple(
            reduced_data.get('img_y_bounds', (0, tbf_shape[1])))

        # Get resized dark field
        if 'dark_field' in reduced_data.data:
            tdf = np.asarray(
                reduced_data.data.dark_field[
                    img_x_bounds[0]:img_x_bounds[1],
                    img_y_bounds[0]:img_y_bounds[1]])
        else:
            self._logger.warning('Dark field unavailable')
            tdf = None

        # Resize bright field
        if (img_x_bounds != (0, tbf.shape[0])
                or img_y_bounds != (0, tbf.shape[1])):
            tbf = tbf[
                img_x_bounds[0]:img_x_bounds[1],
                img_y_bounds[0]:img_y_bounds[1]]

        # Get thetas (in degrees)
        thetas = np.asarray(reduced_data.rotation_angle)

        # Get or create image mask
        image_mask = reduced_data.get('image_mask')
        if image_mask is None:
            image_mask = np.ones(len(thetas), dtype=bool)
        else:
            image_mask = np.asarray(image_mask)
#        exit(f'image_mask {type(image_mask)} {len(image_mask)}: {image_mask}')

        # Get the tomography images
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is not None and 'data' in nxentry.instrument.detector:
            field_indices_all = [
                index for index, key in enumerate(image_key) if key == 0]
            z_translation_all = nxentry.sample.z_translation[field_indices_all]
            z_translation_levels = sorted(list(set(z_translation_all)))
            num_tomo_stacks = len(z_translation_levels)
            tomo_stacks = num_tomo_stacks*[np.array([])]
            horizontal_shifts = []
            vertical_shifts = []
            tomo_stacks = []
            for i, z_translation in enumerate(z_translation_levels):
                field_indices = [
                    field_indices_all[index]
                    for index, z in enumerate(z_translation_all)
                    if z == z_translation]
                field_indices_masked = field_indices[image_mask]
                horizontal_shift = np.asarray(
                    set(nxentry.sample.x_translation[field_indices_masked]))
                assert len(horizontal_shift) == 1
                horizontal_shifts += list(horizontal_shift)
                vertical_shift = np.asarray(
                    set(nxentry.sample.z_translation[field_indices_masked]))
                assert len(vertical_shift) == 1
                vertical_shifts += vertical_shift
                sequence_numbers = nxentry.instrument.detector.sequence_number[
                    field_indices]
                if (list(sequence_numbers)
                        == list(range((len(sequence_numbers))))):
                    tomo_stack = np.asarray(
                        nxentry.instrument.detector.data[
                            field_indices_masked])
                else:
                    raise RuntimeError('Unable to load the tomography images')
                tomo_stacks.append(tomo_stack)
        else:
            import_scanparser(
                nxentry.instrument.source.attrs['station'],
                nxentry.instrument.source.attrs['experiment_type'])
            tomo_field_scans = nxentry.spec_scans.tomo_fields
            num_tomo_stacks = len(tomo_field_scans.keys())
            detector_prefix = str(nxentry.instrument.detector.local_name)
            tomo_stacks = []
            horizontal_shifts = []
            vertical_shifts = []
            for nxsubentry_name, nxsubentry in tomo_field_scans.items():
                scan_number = int(nxsubentry_name.split('_')[-1])
                scanparser = get_scanparser(
                    tomo_field_scans.attrs['spec_file'], scan_number)
                image_offset = int(
                    nxsubentry.instrument.detector.frame_start_number)
                tomo_stack = []
                n = 0
                for i in range(image_mask.size):
                    if image_mask[i]:
                        tomo_stack.append(
                            scanparser.get_detector_data(
                                detector_prefix, image_offset+n))
                        n += 1
                tomo_stacks.append(np.asarray(tomo_stack))
                horizontal_shift = float(nxsubentry.sample.x_translation)
                horizontal_shifts.append(horizontal_shift)
                vertical_shift = float(nxsubentry.sample.z_translation)
                vertical_shifts.append(vertical_shift)

        reduced_tomo_stacks = []
        for i, tomo_stack in enumerate(tomo_stacks):
            # Resize the tomography images
            # Right now the range is the same for each set in the stack
            assert len(thetas) == tomo_stack.shape[0]
            if (img_x_bounds == (0, tbf.shape[0])
                    and img_y_bounds == (0, tbf.shape[1])):
                tomo_stack = tomo_stack.astype('float64', copy=False)
            else:
                tomo_stack = tomo_stack[
                    :,img_x_bounds[0]:img_x_bounds[1],
                    img_y_bounds[0]:img_y_bounds[1]].astype(
                        'float64', copy=False)

            # Subtract dark field
            if tdf is not None:
                try:
                    with SetNumexprThreads(self._num_core):
                        evaluate('tomo_stack-tdf', out=tomo_stack)
                except TypeError as e:
                    sys_exit(
                        f'\nA {type(e).__name__} occured while subtracting '
                        + 'the dark field with num_expr.evaluate()'
                        + '\nTry reducing the detector range'
                        + f'\n(currently img_x_bounds = {img_x_bounds}, and '
                        + f'img_y_bounds = {img_y_bounds})\n')

            # Normalize
            try:
                with SetNumexprThreads(self._num_core):
                    evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            except TypeError as e:
                sys_exit(
                    f'\nA {type(e).__name__} occured while normalizing the '
                    + 'tomography data with num_expr.evaluate()'
                    + '\nTry reducing the detector range'
                    + f'\n(currently img_x_bounds = {img_x_bounds}, and '
                    + f'img_y_bounds = {img_y_bounds})\n')

            # Remove non-positive values and linearize data
            # RV make input argument? cutoff = 1.e-6
            with SetNumexprThreads(self._num_core):
                evaluate(
                    'where(tomo_stack < 1.e-6, 1.e-6, tomo_stack)',
                    out=tomo_stack)
            with SetNumexprThreads(self._num_core):
                evaluate('-log(tomo_stack)', out=tomo_stack)

            # Get rid of nans/infs that may be introduced by normalization
            np.where(np.isfinite(tomo_stack), tomo_stack, 0.)

            # Downsize tomography stack to smaller size
            tomo_stack = tomo_stack.astype('float32', copy=False)
            if not self._test_mode:
                if len(tomo_stacks) == 1:
                    title = f'red fullres theta {round(thetas[0], 2)+0}'
                else:
                    title = f'red stack {i} fullres theta ' \
                        + f'{round(thetas[0], 2)+0}'
                quick_imshow(
                    tomo_stack[0,:,:], title=title, path=self._output_folder,
                    save_fig=self._save_figs, save_only=self._save_only,
                    block=self._block)
#                if not self._block:
#                    clear_imshow(title)
            zoom_perc = 100
            if zoom_perc != 100:
                t0 = time()
                self._logger.debug('Zooming in ...')
                tomo_zoom_list = []
                for j in range(tomo_stack.shape[0]):
                    tomo_zoom = zoom(tomo_stack[j,:,:], 0.01*zoom_perc)
                    tomo_zoom_list.append(tomo_zoom)
                tomo_stack = np.stack(tomo_zoom_list)
                self._logger.info(f'Zooming in took {time()-t0:.2f} seconds')
                del tomo_zoom_list
                if not self._test_mode:
                    title = f'red stack {zoom_perc}p theta ' \
                        + f'{round(thetas[0], 2)+0}'
                    quick_imshow(
                        tomo_stack[0,:,:], title=title,
                        path=self._output_folder, save_fig=self._save_figs,
                        save_only=self._save_only, block=self._block)
#                    if not self._block:
#                        clear_imshow(title)

            # Save test data to file
            if self._test_mode:
                row_index = int(tomo_stack.shape[1]/2)
                np.savetxt(
                    f'{self._output_folder}/red_stack_{i}.txt',
                    tomo_stack[:,row_index,:], fmt='%.6e')

            # Combine resized stacks
            reduced_tomo_stacks.append(tomo_stack)

        # Add tomo field info to reduced data NXprocess
        reduced_data['x_translation'] = np.asarray(horizontal_shifts)
        reduced_data['z_translation'] = np.asarray(vertical_shifts)
        reduced_data.data['tomo_fields'] = np.asarray(reduced_tomo_stacks)

        if tdf is not None:
            del tdf
        del tbf

        return reduced_data

    def _find_center_one_plane(
            self, sinogram, row, thetas, eff_pixel_size, cross_sectional_dim,
            path=None, num_core=1, gaussian_sigma=None, ring_width=None):
        """Find center for a single tomography plane."""
        # Third party modules
        from tomopy import find_center_vo

        # Try automatic center finding routines for initial value
        # sinogram index order: theta,column
        # need column,theta for iradon, so take transpose
        sinogram = np.asarray(sinogram)
        sinogram_t = sinogram.T
        center = sinogram.shape[1]/2

#        quick_imshow(
#            sinogram_t, f'sinogram row{row}',
#            aspect='auto', path=self._output_folder,
#            save_fig=self._save_figs, save_only=self._save_only,
#            block=self._block)

        # Try using Nghia Vo’s method
        t0 = time()
        if num_core > NUM_CORE_TOMOPY_LIMIT:
            self._logger.debug(
                f'Running find_center_vo on {NUM_CORE_TOMOPY_LIMIT} cores ...')
            tomo_center = find_center_vo(
                sinogram, ncore=NUM_CORE_TOMOPY_LIMIT)
        else:
            tomo_center = find_center_vo(sinogram, ncore=num_core)
        self._logger.info(
            f'Finding center using Nghia Vo’s method took {time()-t0:.2f} '
            + 'seconds')
        center_offset_vo = tomo_center-center
        self._logger.info(
            f'Center at row {row} using Nghia Vo’s method = '
            + f'{center_offset_vo:.2f}')
        t0 = time()
        recon_plane = self._reconstruct_one_plane(
            sinogram_t, tomo_center, thetas, eff_pixel_size,
            cross_sectional_dim, False, num_core, gaussian_sigma, ring_width)
        self._logger.info(
            f'Reconstructing row {row} took {time()-t0:.2f} seconds')

        if self._save_figs:
            title = f'edges row{row} center offset{center_offset_vo:.2f} Vo'
            self._plot_edges_one_plane(recon_plane, title, path=path)

        # Try using phase correlation method
#        if input_yesno('
#                Try finding center using phase correlation (y/n)?',
#                'n'):
#            t0 = time()
#            tomo_center = find_center_pc(
#                sinogram, sinogram, tol=0.1, rotc_guess=tomo_center)
#            error = 1.
#            while error > tol:
#                prev = tomo_center
#                tomo_center = find_center_pc(
#                    sinogram, sinogram, tol=tol, rotc_guess=tomo_center)
#                error = np.abs(tomo_center-prev)
#            self._logger.info(
#                'Finding center using the phase correlation method '
#                + f'took {time()-t0:.2f} seconds')
#            center_offset = tomo_center-center
#            print(
#                f'Center at row {row} using phase correlation = '
#                + f'{center_offset:.2f}')
#            t0 = time()
#            recon_plane = self._reconstruct_one_plane(
#                sinogram_t, tomo_center, thetas, eff_pixel_size,
#                cross_sectional_dim, False, num_core, gaussian_sigma, ring_width)
#            self._logger.info(
#                f'Reconstructing row {row} took {time()-t0:.2f} seconds')
#
#            title = \
#                f'edges row{row} center_offset{center_offset:.2f} PC'
#            self._plot_edges_one_plane(recon_plane, title, path=path)

        # Select center location
#        if input_yesno(
#            'Accept a center location (y) or continue search (n)?',
#            'y'):
#            center_offset = input_num('    Enter chosen center offset',
#                    ge=-center, le=center, default=center_offset_vo)
#            return float(center_offset)

        # Perform center finding search
#        while True:
#            center_offset_low = input_int(
#                '\nEnter lower bound for center offset', ge=-center,le=center)
#            center_offset_upp = input_int(
#                'Enter upper bound for center offset', ge=center_offset_low,
#                le=center)
#            if center_offset_upp == center_offset_low:
#                center_offset_step = 1
#            else:
#                center_offset_step = input_int(
#                    'Enter step size for center offset search', ge=1,
#                        le=center_offset_upp-center_offset_low)
#            num_center_offset = 1 + int(
#                (center_offset_upp-center_offset_low) / center_offset_step)
#            center_offsets = np.linspace(
#                center_offset_low, center_offset_upp, num_center_offset)
#            for center_offset in center_offsets:
#                if center_offset == center_offset_vo:
#                    continue
#                t0 = time()
#                recon_plane = self._reconstruct_one_plane(
#                    sinogram_t, center_offset+center, thetas, eff_pixel_size,
#                    cross_sectional_dim, False, num_core, gaussian_sigma, ring_width)
#                self._logger.info(
#                    f'Reconstructing center_offset {center_offset} took '
#                    + 'f{time()-t0:.2f} seconds')
#                title = f'edges row{row} center_offset{center_offset:.2f}'
#                self._plot_edges_one_plane(recon_plane, title, path=path)
#            if input_int('\nContinue (0) or end the search (1)', ge=0, le=1):
#                break

        del sinogram_t
        del recon_plane
#        center_offset = input_num(
#            '    Enter chosen center offset', ge=-center, le=center)
        center_offset = center_offset_vo

        return float(center_offset)

    def _reconstruct_one_plane(
            self, tomo_plane_t, center, thetas, eff_pixel_size,
            cross_sectional_dim, plot_sinogram=True, num_core=1,
            gaussian_sigma=None, ring_width=None):
        """Invert the sinogram for a single tomography plane."""
        # Third party modules
        from scipy.ndimage import gaussian_filter
        from skimage.transform import iradon
        from tomopy import misc

        # tomo_plane_t index order: column,theta
        assert 0 <= center < tomo_plane_t.shape[0]
        center_offset = center-tomo_plane_t.shape[0]/2
        two_offset = 2 * int(np.round(center_offset))
        two_offset_abs = np.abs(two_offset)
        # Add 10% slack to max_rad to avoid edge effects
        max_rad = int(0.55 * (cross_sectional_dim/eff_pixel_size))
        if max_rad > 0.5*tomo_plane_t.shape[0]:
            max_rad = 0.5*tomo_plane_t.shape[0]
        dist_from_edge = max(1, int(np.floor(
            (tomo_plane_t.shape[0] - two_offset_abs) / 2.) - max_rad))
        if two_offset >= 0:
            self._logger.debug(
                f'sinogram range = [{two_offset+dist_from_edge}, '
                + f'{-dist_from_edge}]')
            sinogram = tomo_plane_t[
                two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            self._logger.debug(
                f'sinogram range = [{dist_from_edge}, '
                + f'{two_offset-dist_from_edge}]')
            sinogram = tomo_plane_t[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram:
            quick_imshow(
                sinogram.T, f'sinogram center offset{center_offset:.2f}',
                aspect='auto', path=self._output_folder,
                save_fig=self._save_figs, save_only=self._save_only,
                block=self._block)

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=thetas, circle=True)
        self._logger.info(f'Inverting sinogram took {time()-t0:.2f} seconds')
        del sinogram

        # Performing Gaussian filtering and removing ring artifacts
        if gaussian_sigma is not None:
            recon_sinogram = gaussian_filter(
                recon_sinogram, gaussian_sigma, mode='nearest')
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        if ring_width is not None:
            recon_clean = misc.corr.remove_ring(
                recon_clean, rwidth=ring_width, ncore=num_core)

        return recon_clean

    def _plot_edges_one_plane(self, recon_plane, title, path=None):
        """
        Create an "edges plot" for a singled reconstructed tomography
        data plane.
        """
        # Third party modules
        from skimage.restoration import denoise_tv_chambolle

        vis_parameters = None  # self._config.get('vis_parameters')
        if vis_parameters is None:
            weight = 0.1
        else:
            weight = vis_parameters.get('denoise_weight', 0.1)
            if not is_num(weight, ge=0.):
                self._logger.warning(
                    f'Invalid weight ({weight}) in _plot_edges_one_plane, '
                    + 'set to a default of 0.1')
                weight = 0.1
        edges = denoise_tv_chambolle(recon_plane, weight=weight)
        vmax = np.max(edges[0,:,:])
        vmin = -vmax
        if path is None:
            path = self._output_folder
        quick_imshow(
            edges[0,:,:], f'{title} coolwarm', path=path, cmap='coolwarm',
            save_fig=self._save_figs, save_only=self._save_only,
            block=self._block)
        quick_imshow(
            edges[0,:,:], f'{title} gray', path=path, cmap='gray', vmin=vmin,
            vmax=vmax, save_fig=self._save_figs, save_only=self._save_only,
            block=self._block)
        del edges

    def _reconstruct_one_tomo_stack(
            self, tomo_stack, thetas, center_offsets=None, num_core=1,
            algorithm='gridrec', secondary_iters=0, remove_stripe_sigma=None,
            ring_width=None):
        """Reconstruct a single tomography stack."""
        # Third party modules
        from tomopy import (
            astra,
            misc,
            prep,
            recon,
        )

        # tomo_stack order: row,theta,column
        # input thetas must be in degrees
        # centers_offset: tomography axis shift in pixels relative
        #     to column center
        # RV should we remove stripes?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.stripe.html
        # RV should we remove rings?
        # https://tomopy.readthedocs.io/en/latest/api/tomopy.misc.corr.html
        # RV add an option to do (extra) secondary iterations later or
        #     to do some sort of convergence test?
        if center_offsets is None:
            centers = np.zeros((tomo_stack.shape[0]))
        elif len(center_offsets) == 2:
            centers = np.linspace(
                center_offsets[0], center_offsets[1], tomo_stack.shape[0])
        else:
            if center_offsets.size != tomo_stack.shape[0]:
                raise RuntimeError(
                    'center_offsets dimension mismatch in '
                    + 'reconstruct_one_tomo_stack')
            centers = center_offsets
        centers += tomo_stack.shape[2]/2

        # Remove horizontal stripe
        if remove_stripe_sigma is not None:
            if num_core > NUM_CORE_TOMOPY_LIMIT:
                tomo_stack = prep.stripe.remove_stripe_fw(
                    tomo_stack, sigma=remove_stripe_sigma,
                    ncore=NUM_CORE_TOMOPY_LIMIT)
            else:
                tomo_stack = prep.stripe.remove_stripe_fw(
                    tomo_stack, sigma=remove_stripe_sigma, ncore=num_core)

        # Perform initial image reconstruction
        self._logger.debug('Performing initial image reconstruction')
        t0 = time()
        tomo_recon_stack = recon(
            tomo_stack, np.radians(thetas), centers, sinogram_order=True,
            algorithm=algorithm, ncore=num_core)
        self._logger.info(
            f'Performing initial image reconstruction took {time()-t0:.2f} '
            + 'seconds')

        # Run optional secondary iterations
        if secondary_iters > 0:
            self._logger.debug(
                'Running {secondary_iters} secondary iterations')
#            options = {
#                'method': 'SIRT_CUDA',
#                'proj_type': 'cuda',
#                'num_iter': secondary_iters
#            }
# RV doesn't work for me:
# "Error: CUDA error 803: system has unsupported display driver/cuda driver
#     combination."
#            options = {
#                'method': 'SIRT',
#                'proj_type': 'linear',
#                'MinConstraint': 0,
#                'num_iter':secondary_iters
#            }
# SIRT did not finish while running overnight
#            options = {
#                'method': 'SART',
#                'proj_type': 'linear',
#                'num_iter':secondary_iters
#            }
            options = {
                'method': 'SART',
                'proj_type': 'linear',
                'MinConstraint': 0,
                'num_iter': secondary_iters,
            }
            t0 = time()
            tomo_recon_stack = recon(
                tomo_stack, np.radians(thetas), centers,
                init_recon=tomo_recon_stack, options=options,
                sinogram_order=True, algorithm=astra, ncore=num_core)
            self._logger.info(
                f'Performing secondary iterations took {time()-t0:.2f} '
                + 'seconds')

        # Remove ring artifacts
        if ring_width is not None:
            misc.corr.remove_ring(
                tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_core)

        return tomo_recon_stack

    def _resize_reconstructed_data(
            self, data, x_bounds=None, y_bounds=None, z_bounds=None,
            z_only=False):
        """Resize the reconstructed tomography data."""
        # Data order: row(z),x,y or stack,row(z),x,y
        if isinstance(data, list):
            for stack in data:
                assert stack.ndim == 3
            num_tomo_stacks = len(data)
            tomo_recon_stacks = data
        else:
            assert data.ndim == 3
            num_tomo_stacks = 1
            tomo_recon_stacks = [data]

        if x_bounds == (-1, -1):
            x_bounds = None
        elif not z_only and x_bounds is None:
            # Selecting x bounds (in yz-plane)
            tomosum = 0
            for i in range(num_tomo_stacks):
                tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=(0,2))
            select_x_bounds = input_yesno(
                '\nDo you want to change the image x-bounds (y/n)?', 'y')
            if not select_x_bounds:
                x_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    _, x_bounds = draw_mask_1d(
                        tomosum, current_index_ranges=index_ranges,
                        title='select x data range',
                        ylabel='sum yz')
                    while len(x_bounds) != 1:
                        print('Please select exactly one continuous range')
                        _, x_bounds = draw_mask_1d(
                            tomosum, title='select x data range',
                            ylabel='sum yz')
                    x_bounds = x_bounds[0]
                    accept = True
            self._logger.debug(f'x_bounds = {x_bounds}')

        if y_bounds == (-1, -1):
            y_bounds = None
        elif not z_only and y_bounds is None:
            # Selecting y bounds (in xz-plane)
            tomosum = 0
            for i in range(num_tomo_stacks):
                tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=(0,1))
            select_y_bounds = input_yesno(
                '\nDo you want to change the image y-bounds (y/n)?', 'y')
            if not select_y_bounds:
                y_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    _, y_bounds = draw_mask_1d(
                        tomosum, current_index_ranges=index_ranges,
                        title='select x data range',
                        ylabel='sum xz')
                    while len(y_bounds) != 1:
                        print('Please select exactly one continuous range')
                        _, y_bounds = draw_mask_1d(
                            tomosum, title='select x data range',
                            ylabel='sum xz')
                    y_bounds = y_bounds[0]
                    accept = True
            self._logger.debug(f'y_bounds = {y_bounds}')

        # Selecting z bounds (in xy-plane)
        # (only valid for a single image stack)
        if z_bounds == (-1, -1):
            z_bounds = None
        elif z_bounds is None and num_tomo_stacks != 1:
            tomosum = 0
            for i in range(num_tomo_stacks):
                tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=(1,2))
            select_z_bounds = input_yesno(
                'Do you want to change the image z-bounds (y/n)?', 'n')
            if not select_z_bounds:
                z_bounds = None
            else:
                accept = False
                index_ranges = None
                while not accept:
                    _, z_bounds = draw_mask_1d(
                        tomosum, current_index_ranges=index_ranges,
                        title='select x data range',
                        ylabel='sum xy')
                    while len(z_bounds) != 1:
                        print('Please select exactly one continuous range')
                        _, z_bounds = draw_mask_1d(
                            tomosum, title='select x data range',
                            ylabel='sum xy')
                    z_bounds = z_bounds[0]
                    accept = True
            self._logger.debug(f'z_bounds = {z_bounds}')

        return x_bounds, y_bounds, z_bounds


class TomoSimFieldProcessor(Processor):
    """
    Class representing the process to create a simulated tomography
    data set returning a `nexusformat.nexus.NXroot` object containing
    the simulated tomography detector images.
    """

    def process(self, data, **kwargs):
        # Third party modules
        from nexusformat.nexus import (
            NXdetector,
            NXentry,
            NXinstrument,
            NXroot,
            NXsample,
            NXsource,
        )

        # Get and validate the relevant configuration object in data
        config = self.get_configs(data)

        station = config.station
        sample_type = config.sample_type
        sample_size = config.sample_size
        if len(sample_size) == 1:
            sample_size = (sample_size[0], sample_size[0])
        wall_thickness = config.wall_thickness
        mu = config.mu
        theta_step = config.theta_step
        beam_intensity = config.beam_intensity
        background_intensity = config.background_intensity
        slit_size = config.slit_size
        pixel_size = config.detector.pixel_size
        if len(pixel_size) == 1:
            pixel_size = (
                pixel_size[0]/config.detector.lens_magnification,
                pixel_size[0]/config.detector.lens_magnification,
            )
        else:
            pixel_size = (
                pixel_size[0]/config.detector.lens_magnification,
                pixel_size[1]/config.detector.lens_magnification,
            )
        detector_size = (config.detector.rows, config.detector.columns)
        if slit_size-0.5*pixel_size[0] > detector_size[0]*pixel_size[0]:
            raise ValueError(
                f'Slit size ({slit_size}) larger than detector height '
                f'({detector_size[0]*pixel_size[0]})')

        # Get the rotation angles (start at a arbitrarily choose angle
        #    and add thetas for a full 360 degrees rotation series)
        if station in ('id1a3', 'id3a'):
            theta_start = 0.
        else:
            theta_start = -17
        theta_end = theta_start + 360.
        thetas = list(
            np.arange(theta_start, theta_end+0.5*theta_step, theta_step))

        # Get the number of horizontal stacks bases on the diagonal
        #     of the square and for now don't allow more than one
        num_tomo_stack = 1 + int((sample_size[1]*np.sqrt(2)-pixel_size[1])
                                 / (detector_size[1]*pixel_size[1]))
        if num_tomo_stack > 1:
            raise ValueError('Sample is too wide for the detector')

        # Create the x-ray path length through a solid square
        #     crosssection for a set of rotation angles.
        path_lengths_solid = self._create_pathlength_solid_square(
                sample_size[1], thetas, pixel_size[1], detector_size[1])

        # Create the x-ray path length through a hollow square
        #     crosssection for a set of rotation angles.
        path_lengths_hollow = None
        if sample_type in ('square_pipe', 'hollow_cube', 'hollow_brick'):
            path_lengths_hollow = path_lengths_solid \
                - self._create_pathlength_solid_square(
                    sample_size[1] - 2*wall_thickness, thetas,
                    pixel_size[1], detector_size[1])

        # Get the number of stacks
        num_tomo_stack = 1 + int((sample_size[0]-pixel_size[0])/slit_size)
        if num_tomo_stack > 1 and station == 'id3b':
            raise ValueError('Sample is to tall for the detector')

        # Get the column coordinates
        img_x_offset = -0.5 * (detector_size[0]*pixel_size[0]
                               + slit_size * (num_tomo_stack-1))
        img_x_coords = (img_x_offset
            + pixel_size[0] * (0.5 + np.asarray(range(int(detector_size[0])))))

        # Get the transmitted intensities
        num_theta = len(thetas)
        vertical_shifts = []
        tomo_fields_stack = []
        img_dim = (len(img_x_coords), path_lengths_solid.shape[1])
        intensities_solid = None
        intensities_hollow = None
        for n in range(num_tomo_stack):
            vertical_shifts.append(img_x_offset + n*slit_size)
            tomo_field = beam_intensity * np.ones((num_theta, *img_dim))
            if sample_type == 'square_rod':
                intensities_solid = \
                    beam_intensity * np.exp(-mu*path_lengths_solid)
                for n in range(num_theta):
                    tomo_field[n,:,:] = intensities_solid[n]
            elif sample_type == 'square_pipe':
                intensities_hollow = \
                    beam_intensity * np.exp(-mu*path_lengths_hollow)
                for n in range(num_theta):
                    tomo_field[n,:,:] = intensities_hollow[n]
            else:
                intensities_solid = \
                    beam_intensity * np.exp(-mu*path_lengths_solid)
                intensities_hollow = \
                    beam_intensity * np.exp(-mu*path_lengths_hollow)
                outer_indices = \
                    np.where(abs(img_x_coords) <= sample_size[0]/2)[0]
                inner_indices = np.where(
                    abs(img_x_coords) < sample_size[0]/2 - wall_thickness)[0]
                wall_indices = list(set(outer_indices)-set(inner_indices))
                for i in wall_indices:
                    for n in range(num_theta):
                        tomo_field[n,i] = intensities_solid[n]
                for i in inner_indices:
                    for n in range(num_theta):
                        tomo_field[n,i] = intensities_hollow[n]
            tomo_field += background_intensity
            tomo_fields_stack.append(tomo_field.astype(np.int64))
            if num_tomo_stack > 1:
                img_x_coords += slit_size
            
        # Add dummy snapshots at each end to mimic FMB/SMB
        if station in ('id1a3', 'id3a'):
            num_theta_dummy_start = 5
            num_theta_dummy_end = 0
        else:
            num_theta_dummy_start = 1
            num_theta_dummy_end = 1
        thetas = [theta_start-n*theta_step
            for n in range(num_theta_dummy_start, 0, -1)] + thetas
        thetas += [theta_end+n*theta_step
            for n in range(1, num_theta_dummy_end+1)]
        if num_theta_dummy_start:
            dummy_fields = background_intensity * np.ones(
                (num_theta_dummy_start, *img_dim), dtype=np.int64)
            for n, tomo_field in enumerate(tomo_fields_stack):
                tomo_fields_stack[n] = np.concatenate(
                    (dummy_fields, tomo_field))
        if num_theta_dummy_end:
            dummy_fields = background_intensity * np.ones(
                (num_theta_dummy_end, *img_dim), dtype=np.int64)
            for n, tomo_field in enumerate(tomo_fields_stack):
                tomo_fields_stack[n] = np.concatenate(
                    (tomo_field, dummy_fields))
        if num_tomo_stack == 1:
            tomo_fields_stack = tomo_fields_stack[0]

        # Create Nexus object and write to file
        nxroot = NXroot()
        nxroot.entry = NXentry()
        nxroot.entry.sample = NXsample()
        nxroot.entry.sample.sample_type = sample_type
        nxroot.entry.sample.sample_size = sample_size
        if wall_thickness is not None:
            nxroot.entry.sample.wall_thickness = wall_thickness
        nxroot.entry.sample.mu = mu
        nxinstrument = NXinstrument()
        nxroot.entry.instrument = nxinstrument
        nxinstrument.source = NXsource()
        nxinstrument.source.attrs['station'] = station
        nxinstrument.source.type = 'Synchrotron X-ray Source'
        nxinstrument.source.name = 'Tomography Simulator'
        nxinstrument.source.probe = 'x-ray'
        nxinstrument.source.background_intensity = background_intensity
        nxinstrument.source.beam_intensity = beam_intensity
        nxinstrument.source.slit_size = slit_size
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = config.detector.prefix
        nxdetector.x_pixel_size = pixel_size[0]
        nxdetector.y_pixel_size = pixel_size[1]
        nxdetector.x_pixel_size.attrs['units'] = 'mm'
        nxdetector.y_pixel_size.attrs['units'] = 'mm'
        nxdetector.data = tomo_fields_stack
        nxdetector.thetas = thetas
        nxdetector.z_translation = vertical_shifts
        nxdetector.frame_start_number = num_theta_dummy_start
#        nxdetector.path_lengths_solid = path_lengths_solid
#        nxdetector.path_lengths_hollow = path_lengths_hollow
#        nxdetector.intensities_solid = intensities_solid
#        nxdetector.intensities_hollow = intensities_hollow

        return nxroot

    def get_configs(self, data):
        """
        Get instances of the configuration objects needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has `'TomoReduceConfig'` for the `'schema'` key, and at
            least one item has `'TomoSimConfig'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If valid config objects cannot be constructed
            from `data`.
        :return: valid instance of the configuration object with field
            values taken from `data`.
        :rtype: TomoSimConfig
        """
        # Local modules
        from CHAP.tomo.models import TomoSimConfig

        tomo_sim_data = None
        if isinstance(data, list):
            for item in data:
                if (isinstance(item, dict) and item.get('data') is not None
                        and item.get('schema') == 'TomoSimConfig'):
                    tomo_sim_data = item.get('data')
        if tomo_sim_data is None:
            raise ValueError(f'Invalid parameter data ({data})')

        return TomoSimConfig(**tomo_sim_data)

    def _create_pathlength_solid_square(self, dim, thetas, pixel_size,
            detector_size):
        """
        Create the x-ray path length through a solid square
        crosssection for a set of rotation angles.
        """

        # Get the column coordinates
        img_y_coords = pixel_size * (0.5 * (1 - detector_size%2)
            + np.asarray(range(int(0.5 * (detector_size+1)))))

        # Get the path lenghts for position column coordinates
        lengths = np.zeros((len(thetas), len(img_y_coords)), dtype=np.float64)
        for i, theta in enumerate(thetas):
            dummy = theta
            theta = theta - 90.*np.floor(theta/90.)
            if 45. < theta <= 90.:
                theta = 90.-theta
            theta_rad = theta*np.pi/180.
            len_ab = dim/np.cos(theta_rad)
            len_oc = dim*np.cos(theta_rad+0.25*np.pi)/np.sqrt(2.)
            len_ce = dim*np.sin(theta_rad)
            index1 = int(np.argmin(np.abs(img_y_coords-len_oc)))
            if len_oc < img_y_coords[index1] and index1 > 0:
                index1 -= 1
            index2 = int(np.argmin(np.abs(img_y_coords-len_oc-len_ce)))
            if len_oc+len_ce < img_y_coords[index2]:
                index2 -= 1
            index1 += 1
            index2 += 1
            for j in range(index1):
                lengths[i,j] = len_ab
            for j, column in enumerate(img_y_coords[index1:index2]):
                lengths[i,j+index1] = len_ab*(len_oc+len_ce-column)/len_ce

        # Add the mirror image for negative column coordinates
        if len(img_y_coords)%2:
            lengths = np.concatenate(
                (np.fliplr(lengths[:,1:]), lengths), axis=1)
        else:
            lengths = np.concatenate((np.fliplr(lengths), lengths), axis=1)

        return lengths


class TomoDarkFieldProcessor(Processor):
    """
    Class representing the process to create a dark field associated
    with a simulated tomography data set created by TomoSimProcessor
    returning a `nexusformat.nexus.NXroot` object containing the
    dark field tomography detector images.
    """

    def process(self, data, num_image=5, **kwargs):
        # Third party modules
        from nexusformat.nexus import (
            NXroot,
            NXentry,
            NXinstrument,
            NXdetector,
        )

        # Get the TomoSimField configuration object in data
        nxroot = self.get_config(data)
        source = nxroot.entry.instrument.source
        detector = nxroot.entry.instrument.detector
        background_intensity = source.background_intensity
        detector_size = detector.data.shape[-2:]

        # Add dummy snapshots at start to mimic SMB
        if source.station in ('id1a3', 'id3a'):
            num_theta_dummy_start = 5
        else:
            num_theta_dummy_start = 0
        num_image += num_theta_dummy_start

        # Create the dark field
        dark_field = int(background_intensity) * np.ones(
            (num_image, detector_size[0], detector_size[1]), dtype=np.int64)

        # Create Nexus object and write to file
        nxdark = NXroot()
        nxdark.entry = NXentry()
        nxdark.entry.sample = nxroot.entry.sample
        nxinstrument = NXinstrument()
        nxdark.entry.instrument = nxinstrument
        nxinstrument.source = nxroot.entry.instrument.source
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = detector.local_name
        nxdetector.x_pixel_size = detector.x_pixel_size
        nxdetector.y_pixel_size = detector.y_pixel_size
        nxdetector.data = dark_field
        nxdetector.thetas = np.asarray(num_image*[0])
        nxdetector.frame_start_number = num_theta_dummy_start

        return nxdark

    def get_config(self, data):
        """Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'TomoSimField'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be
            constructed from `data`.
        :return: a valid instance of a configuration object with field
            values taken from `data`.
        :rtype: nexusformat.nexus.NXroot
        """
        nxroot = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'TomoSimField':
                        nxroot = item.get('data')
                        break
        if nxroot is None:
            raise ValueError(
                'No tomography data set found in input data')

        return nxroot


class TomoBrightFieldProcessor(Processor):
    """
    Class representing the process to create a bright field associated
    with a simulated tomography data set created by TomoSimProcessor
    returning a `nexusformat.nexus.NXroot` object containing the
    bright field tomography detector images.
    """

    def process(self, data, num_image=5, **kwargs):
        # Third party modules
        from nexusformat.nexus import (
            NeXusError,
            NXroot,
            NXentry,
            NXinstrument,
            NXdetector,
        )

        # Get and validate the TomoSimField configuration object in data
        nxroot = self.get_config(data)
        source = nxroot.entry.instrument.source
        detector = nxroot.entry.instrument.detector
        beam_intensity = source.beam_intensity
        background_intensity = source.background_intensity
        detector_size = detector.data.shape[-2:]

        # Add dummy snapshots at start to mimic SMB
        if source.station in ('id1a3', 'id3a'):
            num_theta_dummy_start = 5
        else:
            num_theta_dummy_start = 0
        num_image += num_theta_dummy_start

        # Create the bright field
        bright_field = int(background_intensity+beam_intensity) * np.ones(
            (num_image, detector_size[0], detector_size[1]), dtype=np.int64)
        # Add 10% to slit size to make the bright beam slightly taller
        #     than the vertical displacements between stacks
        slit_size = 1.10*source.slit_size
        if slit_size < detector.x_pixel_size*detector_size[0]:
            img_x_coords = detector.x_pixel_size \
                * (0.5 + np.asarray(range(int(detector_size[0])))
                   - 0.5*detector_size[0])
            outer_indices = np.where(abs(img_x_coords) > slit_size/2)[0]
            bright_field[:,outer_indices,:] = 0

        # Create Nexus object and write to file
        nxbright = NXroot()
        nxbright.entry = NXentry()
        nxbright.entry.sample = nxroot.entry.sample
        nxinstrument = NXinstrument()
        nxbright.entry.instrument = nxinstrument
        nxinstrument.source = nxroot.entry.instrument.source
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = detector.local_name
        nxdetector.x_pixel_size = detector.x_pixel_size
        nxdetector.y_pixel_size = detector.y_pixel_size
        nxdetector.data = bright_field
        nxdetector.thetas = np.asarray(num_image*[0])
        nxdetector.frame_start_number = num_theta_dummy_start

        return nxbright

    def get_config(self, data):
        """Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'TomoSimField'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be
            constructed from `data`.
        :return: a valid instance of a configuration object with field
            values taken from `data`.
        :rtype: nexusformat.nexus.NXroot
        """
        nxroot = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('schema') == 'TomoSimField':
                        nxroot = item.get('data')
                        break
        if nxroot is None:
            raise ValueError(
                'No tomography data set found in input data')

        return nxroot


class TomoSpecProcessor(Processor):
    """
    Class representing the process to create a tomography SPEC file
    associated with a simulated tomography data set created by
    TomoSimProcessor returning a plain text object containing the
    tomography SPEC file.
    """

    def process(self, data, spec_folder='.', scan_numbers=[1], **kwargs):
        # System modules
        from json import dump
        from datetime import datetime

        # Third party modules
        from nexusformat.nexus import NeXusError

        # Get and validate the TomoSimField, TomoDarkField, or
        #     TomoBrightField configuration object in data
        scan_numbers = list(set(scan_numbers))
        configs = self.get_config(data)
        station = None
        sample_type = None
        num_scan = 0
        for schema, nxroot in configs.items():
            source = nxroot.entry.instrument.source
            if station is None:
                station = source.attrs.get('station')
            else:
                if station != source.attrs.get('station'):
                    raise ValueError('Inconsistent station among scans')
            if sample_type is None:
                sample_type = nxroot.entry.sample.sample_type
            else:
                if sample_type != nxroot.entry.sample.sample_type:
                    raise ValueError('Inconsistent sample_type among scans')
            detector = nxroot.entry.instrument.detector
            if 'z_translation' in detector:
                num_stack = np.asarray(detector.z_translation).size
            else:
                num_stack = 1
            data_shape = detector.data.shape
            if len(data_shape) == 3:
                if num_stack != 1:
                    raise ValueError(
                        'Inconsistent z_translation and data dimensions'
                        f'({num_stack} vs {1})')
            elif len(data_shape) == 4:
                if num_stack != data_shape[0]:
                    raise ValueError(
                        'Inconsistent z_translation dimension and data shape '
                        f'({num_stack} vs {data_shape[0]})')
            else:
                raise ValueError(f'Invalid data shape ({data_shape})')
            num_scan += num_stack
        if len(scan_numbers) != num_scan:
            raise ValueError(
                f'Inconsistent number of scans ({num_scan}), '
                f'len(scan_numbers) = {len(scan_numbers)})')

        # Create the SPEC output folder if needed
        spec_folder = os_path.abspath(spec_folder)
        if not os_path.isdir(spec_folder):
            mkdir(spec_folder)

        # Create the SPEC file header
        spec_file = [f'#F {sample_type}']
        spec_file.append('#E 0')
        spec_file.append(
            f'#D {datetime.now().strftime("%a %b %d %I:%M:%S %Y")}')
        spec_file.append(f'#C spec  User = chess_{station}\n')
        if station in ('id1a3', 'id3a'):
            spec_file.append('#O0 ramsx  ramsz')
        else:
            #RV Fix main code to use independent dim info
            spec_file.append('#O0 GI_samx  GI_samz  GI_samphi')
            spec_file.append('#o0 samx samz samphi') #RV do I need this line?
        spec_file.append('')

        # Create the SPEC file scan info (and image and parfile data for SMB)
        par_file = []
        image_sets = []
        num_scan = 0
        count_time = 1
        for schema, nxroot in configs.items():
            detector = nxroot.entry.instrument.detector
            if 'z_translation' in detector:
                z_translations = list(np.asarray(detector.z_translation))
            else:
                z_translations = [0.]
            frame_start_number = int(detector.frame_start_number)
            thetas = np.asarray(detector.thetas)[frame_start_number:]
            num_theta = thetas.size
            if schema == 'TomoDarkField':
                if station in ('id1a3', 'id3a'):
                    macro = f'slew_ome {thetas[0]} {thetas[-1]} ' \
                        f'{num_theta} {count_time} darkfield'
                    scan_type = 'df1'
                else:
                    macro = f'flyscan {num_theta} {count_time}'
            elif schema == 'TomoBrightField':
                if station in ('id1a3', 'id3a'):
                    macro = f'slew_ome {thetas[0]} {thetas[-1]} ' \
                        f'{num_theta} {count_time}'
                    scan_type = 'bf1'
                else:
                    macro = f'flyscan {num_theta} {count_time}'
            elif schema == 'TomoSimField':
                if station in ('id1a3', 'id3a'):
                    macro = f'slew_ome {thetas[0]} {thetas[-1]} ' \
                        f'{num_theta} {count_time}'
                    scan_type = 'ts1'
                else:
                    macro = f'flyscan samphi {thetas[0]} ' \
                        f'{thetas[-1]} {num_theta-1} {count_time}'
            for  n, z_translation in enumerate(z_translations):
                spec_file.append(f'#S {scan_numbers[num_scan]}  {macro}')
                spec_file.append(
                    f'#D {datetime.now().strftime("%a %b %d %I:%M:%S %Y")}')
                if station in ('id1a3', 'id3a'):
                    spec_file.append(f'#P0 0.0 {z_translation}')
                    spec_file.append('#N 1')
                    spec_file.append('#L  ome')
                    if scan_type == 'ts1':
                        image_sets.append(detector.data[n])
                    else:
                        image_sets.append(detector.data)
                    par_file.append(
                        f'{datetime.now().strftime("%Y%m%d")} '
                        f'{datetime.now().strftime("%H%M%S")} '
                        f'{scan_numbers[num_scan]} '
#                        '2.0 '
#                        '1.0 '
                        '0 '
                        f'{frame_start_number} '
                        '0.0 '
                        f'{z_translation} '
                        f'{thetas[0]} '
                        f'{thetas[-1]} '
                        f'{num_theta} '
                        f'{count_time} '
                        f'{scan_type}')
                else:
                    spec_file.append(f'#P0 0.0 {z_translation} 0.0')
                    spec_file.append('#N 1')
                    if schema in ('TomoDarkField', 'TomoBrightField'):
                        spec_file.append('#L Time')
                        spec_file += [str(count_time*time)
                            for time in range(num_theta)]
                    elif schema == 'TomoSimField':
                        spec_file.append('#L GI_samphi')
                        spec_file += [str(theta) for theta in thetas]
                spec_file.append('')
                num_scan += 1
        
        if station in ('id1a3', 'id3a'):

            # Write the SPEC file
            self.write_txt(spec_file, os_path.join(spec_folder, 'spec.log'))

            # Write the JSON file
            parfile_header = {
                '0': 'date',
                '1': 'time',
                '2': 'SCAN_N',
#                '3': 'beam_width',
#                '4': 'beam_height',
                '3': 'junkstart',
                '4': 'goodstart',
                '5': 'ramsx',
                '6': 'ramsz',
                '7': 'ome_start_real',
                '8': 'ome_end_real',
                '9': 'nframes_real',
                '10': 'count_time',
                '11': 'tomotype',
            }
            json_filename = os_path.join(
                spec_folder,
                f'{station}-tomo_sim-{os_path.basename(spec_folder)}.json')
            with open(json_filename, 'w') as f:
                dump(parfile_header, f)
        
            # Write the par file
            par_filename = os_path.join(
                spec_folder,
                f'{station}-tomo_sim-{os_path.basename(spec_folder)}.par')
            self.write_txt(par_file, par_filename)

            # Write image files as individual tiffs
            for scan_number, image_set in zip(scan_numbers, image_sets):
                image_folder = os_path.join(spec_folder, str(scan_number))
                if not os_path.isdir(image_folder):
                    mkdir(image_folder)
                image_folder = os_path.join(image_folder, 'nf')
                if not os_path.isdir(image_folder):
                    mkdir(image_folder)
                self.write_tiffs(image_set, image_folder)

        return spec_file

    def get_config(self, data):
        """Get an instance of the configuration object needed by this
        `Processor` from a returned value of `Reader.read`

        :param data: Result of `Reader.read` where at least one item
            has the value `'TomoSimField'` for the `'schema'` key.
        :type data: list[dict[str,object]]
        :raises Exception: If a valid config object cannot be
            constructed from `data`.
        :return: a valid instance of a configuration object with field
            values taken from `data`.
        :rtype: nexusformat.nexus.NXroot
        """
        configs = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    schema = item.get('schema')
                    if schema == 'TomoDarkField':
                        configs[schema] = item.get('data')
                    elif schema == 'TomoBrightField':
                        configs[schema] = item.get('data')
                    elif schema == 'TomoSimField':
                        configs[schema] = item.get('data')
        if not len(configs):
            raise ValueError(
                'No tomography data set found in input data')

        return configs

    def write_tiffs(self, data, image_folder):
        # Third party modules
        from imageio import imwrite
        for n in range(data.shape[0]):
            imwrite(
                os_path.join(image_folder, f'nf_{n:06d}.tif'), data[n])

    def write_txt(self, data, filepath, force_overwrite=True):
        # Local modules
        from CHAP.common import TXTWriter
        from CHAP.pipeline import PipelineData
        writer = TXTWriter()
        writer.write(
            [PipelineData(data=data)], filepath,
            force_overwrite=force_overwrite)


if __name__ == '__main__':
    main()
