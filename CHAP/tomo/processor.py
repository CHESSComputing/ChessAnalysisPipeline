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
from CHAP.utils.general import (
    is_num,
    is_int_pair,
    input_int,
    input_num,
    input_yesno,
    select_image_indices,
    select_roi_1d,
    select_roi_2d,
    quick_plot,
    quick_imshow,
)
from CHAP.utils.fit import Fit
from CHAP.processor import Processor
from CHAP.reader import main

NUM_CORE_TOMOPY_LIMIT = 24


def get_nxroot(data, schema=None, remove=True):
    """Look through `data` for an item whose value for the `'schema'`
    key matches `schema` (if supplied) and whose value for the `'data'`
    key matches a nexusformat.nexus.NXobject object and return this
    object.

    :param data: Input list of `PipelineData` objects.
    :type data: list[PipelineData]
    :param schema: Name associated with the nexusformat.nexus.NXobject
        object to match in `data`.
    :type schema: str, optional
    :param remove: Removes the matching entry in `data` when found,
        defaults to `True`.
    :type remove: bool, optional
    :raises ValueError: Found an invalid matching object or multiple
        matching objects.
    :return: Object matching with `schema` or None when not found.
    :rtype: None, nexusformat.nexus.NXroot
    """
    # System modules
    from copy import deepcopy

    # Local modules
    from nexusformat.nexus import NXobject
    nxobject = None
    if isinstance(data, list):
        for i, item in enumerate(deepcopy(data)):
            if isinstance(item, dict):
                if schema is None or item.get('schema') == schema:
                    item_data = item.get('data')
                    if isinstance(item_data, NXobject):
                        if nxobject is not None:
                            raise ValueError(
                                'Multiple NXobject objects found in input'
                                f' data matching schema = {schema}')
                        nxobject = item_data
                        if remove:
                            data.pop(i)
                    elif schema is not None:
                        raise ValueError(
                            'Invalid NXobject object found in input data')

    return nxobject


class TomoCHESSMapConverter(Processor):
    """
    A processor to convert a CHESS style tomography map with dark and
    bright field configurations to an nexusformat.nexus.NXtomo style
    input format.
    """

    def process(self, data):
        """
        Process the input map and configuration and return a
        nexusformat.nexus.NXroot object based on the
        nexusformat.nexus.NXtomo style format.

        :param data: Input map and configuration for tomographic image
            reduction/reconstruction.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: nexusformat.nexus.NXtomo style tomography input
            configuration.
        :rtype: nexusformat.nexus.NXroot
        """
        # System modules
        from copy import deepcopy

        # Third party modules
        from json import loads
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXentry,
            NXinstrument,
            NXroot,
            NXsample,
            NXsource,
        )

        # Local modules
        from CHAP.common.models.map import MapConfig

        darkfield = get_nxroot(data, 'darkfield')
        brightfield = get_nxroot(data, 'brightfield')
        tomofields = get_nxroot(data, 'tomofields')
        detector_config = self.get_config(data, 'tomo.models.Detector')

        if darkfield is not None and not isinstance(darkfield, NXentry):
            raise ValueError('Invalid parameter darkfield ({darkfield})')
        if not isinstance(brightfield, NXentry):
            raise ValueError('Invalid parameter brightfield ({brightfield})')
        if not isinstance(tomofields, NXentry):
            raise ValueError('Invalid parameter tomofields {tomofields})')

        # Construct NXroot
        nxroot = NXroot()

        # Validate map
        map_config = MapConfig(**loads(str(tomofields.map_config)))

        # Check available independent dimensions
        independent_dimensions = tomofields.data.attrs['axes']
        if isinstance(independent_dimensions, str):
            independent_dimensions = [independent_dimensions]
        matched_dimensions = deepcopy(independent_dimensions)
        if 'rotation_angles' not in independent_dimensions:
            raise ValueError('Data for rotation angles is unavailable '
                             '(available independent dimensions: '
                             f'{independent_dimensions})')
        rotation_angles_index = \
            tomofields.data.attrs['rotation_angles_indices']
        rotation_angle_data_type = \
            tomofields.data.rotation_angles.attrs['data_type']
        if rotation_angle_data_type != 'scan_column':
            raise ValueError('Invalid data type for rotation angles '
                             f'({rotation_angle_data_type})')
        matched_dimensions.pop(matched_dimensions.index('rotation_angles'))
        if 'x_translation' in independent_dimensions:
            x_translation_index = \
                tomofields.data.attrs['x_translation_indices']
            x_translation_data_type = \
                tomofields.data.x_translation.attrs['data_type']
            x_translation_name = \
                tomofields.data.x_translation.attrs['local_name']
            if x_translation_data_type not in ('spec_motor', 'smb_par'):
                raise ValueError('Invalid data type for x translation '
                                 f'({x_translation_data_type})')
            matched_dimensions.pop(matched_dimensions.index('x_translation'))
        else:
            x_translation_data_type = None
        if 'z_translation' in independent_dimensions:
            z_translation_index = \
                tomofields.data.attrs['z_translation_indices']
            z_translation_data_type = \
                tomofields.data.z_translation.attrs['data_type']
            z_translation_name = \
                tomofields.data.z_translation.attrs['local_name']
            if z_translation_data_type not in ('spec_motor', 'smb_par'):
                raise ValueError('Invalid data type for x translation '
                                 f'({z_translation_data_type})')
            matched_dimensions.pop(matched_dimensions.index('z_translation'))
        else:
            z_translation_data_type = None
        if matched_dimensions:
            raise ValueError('Unknown independent dimension '
                             f'({matched_dimensions}), independent dimensions '
                             'must be in {"z_translation", "x_translation", '
                             '"rotation_angles"}')

        # Construct base NXentry and add to NXroot
        nxentry = NXentry()
        nxroot[map_config.title] = nxentry
        nxroot.attrs['default'] = map_config.title
        nxentry.definition = 'NXtomo'
        nxentry.map_config = tomofields.map_config

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
#        nxsource.attrs['cycle'] = cycle
#        nxsource.attrs['btr'] = btr
        nxsource.attrs['station'] = tomofields.station
        nxsource.attrs['experiment_type'] = map_config.experiment_type

        # Add an NXdetector to the NXinstrument
        # (do not fill in data fields yet)
        detector_prefix = detector_config.prefix
        detectors = list(set(tomofields.data.entries)
                         - set(independent_dimensions))
        if detector_prefix not in detectors:
            raise ValueError(f'Data for detector {detector_prefix} is '
                             f'unavailable (available detectors: {detectors})')
        tomo_stacks = np.asarray(tomofields.data[detector_prefix])
        tomo_stack_shape = tomo_stacks.shape
        assert len(tomo_stack_shape) == 2+len(independent_dimensions)
        assert tomo_stack_shape[-2] == detector_config.rows
        assert tomo_stack_shape[-1] == detector_config.columns
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = detector_prefix
        pixel_size = detector_config.pixel_size
        if len(pixel_size) == 1:
            nxdetector.row_pixel_size = \
                pixel_size[0]/detector_config.lens_magnification
            nxdetector.column_pixel_size = \
                pixel_size[0]/detector_config.lens_magnification
        else:
            nxdetector.row_pixel_size = \
                pixel_size[0]/detector_config.lens_magnification
            nxdetector.column_pixel_size = \
                pixel_size[1]/detector_config.lens_magnification
        nxdetector.row_pixel_size.units = 'mm'
        nxdetector.column_pixel_size.units = 'mm'
        nxdetector.rows = detector_config.rows
        nxdetector.columns = detector_config.columns
        nxdetector.rows.units = 'pixels'
        nxdetector.columns.units = 'pixels'

        # Add an NXsample to NXentry
        # (do not fill in data fields yet)
        nxsample = NXsample()
        nxentry.sample = nxsample
        nxsample.name = map_config.sample.name
        if map_config.sample.description is not None:
            nxsample.description = map_config.sample.description

        # Collect dark field data
        image_keys = []
        sequence_numbers = []
        image_stacks = []
        rotation_angles = []
        x_translations = []
        z_translations = []
        if darkfield is not None:
            nxentry.dark_field_config = darkfield.spec_config
            for scan_name, scan in darkfield.spec_scans.items():
                for scan_number, nxcollection in scan.items():
                    scan_columns = loads(str(nxcollection.scan_columns))
                    data_shape = nxcollection.data[detector_prefix].shape
                    assert len(data_shape) == 3
                    assert data_shape[1] == detector_config.rows
                    assert data_shape[2] == detector_config.columns
                    num_image = data_shape[0]
                    image_keys += num_image*[2]
                    sequence_numbers += list(range(num_image))
                    image_stacks.append(np.asarray(
                        nxcollection.data[detector_prefix]))
                    rotation_angles += num_image*[0.0]
                    if (x_translation_data_type == 'spec_motor' or
                            z_translation_data_type == 'spec_motor'):
                        spec_motors = loads(str(nxcollection.spec_motors))
                    if (x_translation_data_type == 'smb_par' or
                            z_translation_data_type == 'smb_par'):
                        smb_pars = loads(str(nxcollection.smb_pars))
                    if x_translation_data_type is None:
                        x_translations += num_image*[0.0]
                    else:
                        if x_translation_data_type == 'spec_motor':
                            x_translations += \
                                num_image*[spec_motors[x_translation_name]]
                        else:
                            x_translations += \
                                num_image*[smb_pars[x_translation_name]]
                    if z_translation_data_type is None:
                        z_translations += num_image*[0.0]
                    else:
                        if z_translation_data_type == 'spec_motor':
                            z_translations += \
                                num_image*[spec_motors[z_translation_name]]
                        else:
                            z_translations += \
                                num_image*[smb_pars[z_translation_name]]

        # Collect bright field data
        nxentry.bright_field_config = brightfield.spec_config
        for scan_name, scan in brightfield.spec_scans.items():
            for scan_number, nxcollection in scan.items():
                scan_columns = loads(str(nxcollection.scan_columns))
                data_shape = nxcollection.data[detector_prefix].shape
                assert len(data_shape) == 3
                assert data_shape[1] == detector_config.rows
                assert data_shape[2] == detector_config.columns
                num_image = data_shape[0]
                image_keys += num_image*[1]
                sequence_numbers += list(range(num_image))
                image_stacks.append(np.asarray(
                    nxcollection.data[detector_prefix]))
                rotation_angles += num_image*[0.0]
                if (x_translation_data_type == 'spec_motor' or
                        z_translation_data_type == 'spec_motor'):
                    spec_motors = loads(str(nxcollection.spec_motors))
                if (x_translation_data_type == 'smb_par' or
                        z_translation_data_type == 'smb_par'):
                    smb_pars = loads(str(nxcollection.smb_pars))
                if x_translation_data_type is None:
                    x_translations += num_image*[0.0]
                else:
                    if x_translation_data_type == 'spec_motor':
                        x_translations += \
                            num_image*[spec_motors[x_translation_name]]
                    else:
                        x_translations += \
                            num_image*[smb_pars[x_translation_name]]
                if z_translation_data_type is None:
                    z_translations += num_image*[0.0]
                if z_translation_data_type is not None:
                    if z_translation_data_type == 'spec_motor':
                        z_translations += \
                            num_image*[spec_motors[z_translation_name]]
                    else:
                        z_translations += \
                            num_image*[smb_pars[z_translation_name]]

        # Collect tomography fields data
        if x_translation_data_type is None:
            x_trans = [0.0]
            if z_translation_data_type is None:
                z_trans = [0.0]
                tomo_stacks = np.reshape(tomo_stacks, (1,1,*tomo_stacks.shape))
            else:
                if len(list(tomofields.data.z_translation)):
                    z_trans = list(tomofields.data.z_translation)
                else:
                    z_trans = [float(tomofields.data.z_translation)]
                if rotation_angles_index < z_translation_index:
                    tomo_stacks = np.swapaxes(
                        tomo_stacks, rotation_angles_index,
                        z_translation_index)
                tomo_stacks = np.expand_dims(tomo_stacks, z_translation_index)
        elif z_translation_data_type is None:
            z_trans = [0.0]
            if rotation_angles_index < x_translation_index:
                tomo_stacks = np.swapaxes(
                    tomo_stacks, rotation_angles_index, x_translation_index)
            tomo_stacks = np.expand_dims(tomo_stacks, 0)
        else:
            if len(list(tomofields.data.x_translation)):
                x_trans = list(tomofields.data.x_translation)
            else:
                x_trans = [float(tomofields.data.x_translation)]
            if len(list(tomofields.data.z_translation)):
                z_trans = list(tomofields.data.z_translation)
            else:
                z_trans = [float(tomofields.data.z_translation)]
            if (rotation_angles_index
                    < max(x_translation_index, z_translation_index)):
                tomo_stacks = np.swapaxes(
                    tomo_stacks, rotation_angles_index,
                    max(x_translation_index, z_translation_index))
            if x_translation_index < z_translation_index:
                tomo_stacks = np.swapaxes(
                    tomo_stacks, x_translation_index, z_translation_index)
        # Restrict to 180 degrees set of data for now to match old code
        thetas = np.asarray(tomofields.data.rotation_angles)
#RV        num_image = len(tomofields.data.rotation_angles)
        assert len(thetas) > 2
        from CHAP.utils.general import index_nearest
        delta_theta = thetas[1]-thetas[0]
        if thetas[-1]-thetas[0] > 180-delta_theta:
            image_end = index_nearest(thetas, thetas[0]+180)
        else:
            image_end = len(thetas)
        thetas = thetas[:image_end]
        num_image = len(thetas)
        for i, z in enumerate(z_trans):
            for j, x in enumerate(x_trans):
                image_keys += num_image*[0]
                sequence_numbers += list(range(num_image))
                image_stacks.append(np.asarray(
                    tomo_stacks[i,j][:image_end,:,:]))
                rotation_angles += list(thetas)
#RV                rotation_angles += list(tomofields.data.rotation_angles)
                x_translations += num_image*[x]
                z_translations += num_image*[z]

        # Add image data to NXdetector
        nxinstrument.detector.image_key = image_keys
        nxinstrument.detector.sequence_number = sequence_numbers
        nxinstrument.detector.data = np.concatenate(image_stacks)

        # Add image data to NXsample
        nxsample.rotation_angle = rotation_angles
        nxsample.rotation_angle.units = 'degrees'
        nxsample.x_translation = x_translations
        nxsample.x_translation.units = 'mm'
        nxsample.z_translation = z_translations
        nxsample.z_translation.units = 'mm'

        # Add an NXdata to NXentry
        nxdata = NXdata()
        nxentry.data = nxdata
        nxdata.makelink(nxentry.instrument.detector.data, name='data')
        nxdata.makelink(nxentry.instrument.detector.image_key)
        nxdata.makelink(nxentry.sample.rotation_angle)
        nxdata.makelink(nxentry.sample.x_translation)
        nxdata.makelink(nxentry.sample.z_translation)
        nxdata.attrs['signal'] = 'data'
#        nxdata.attrs['axes'] = ['field', 'row', 'column']
#        nxdata.attrs['field_indices'] = 0
#        nxdata.attrs['row_indices'] = 1
#        nxdata.attrs['column_indices'] = 2

        return nxroot


class TomoDataProcessor(Processor):
    """
    A processor to reconstruct a set of tomographic images returning
    either a dictionary or a nexusformat.nexus.NXroot object
    containing the (meta) data after processing each individual step.
    """

    def process(
            self, data, interactive=False, reduce_data=False,
            find_center=False, reconstruct_data=False, combine_data=False,
            output_folder='.', save_figs='no', **kwargs):
        """
        Process the input map or configuration with the step specific
        instructions and return either a dictionary or a
        nexusformat.nexus.NXroot object with the processed result.

        :param data: Input configuration and specific step instructions
            for tomographic image reduction.
        :type data: list[PipelineData]
        :param interactive: Allows for user interactions,
            defaults to False.
        :type interactive: bool, optional
        :param reduce_data: Generate reduced tomography images,
            defaults to False.
        :type reduce_data: bool, optional
        :param find_center: Find the calibrated center axis info,
            defaults to False.
        :type find_center: bool, optional
        :param reconstruct_data: Reconstruct the tomography data,
            defaults to False.
        :type reconstruct_data: bool, optional
        :param combine_data: Combine the reconstructed tomography
            stacks, defaults to False.
        :type combine_data: bool, optional
        :param output_folder: Output folder name, defaults to '.'.
        :type output_folder:: str, optional
        :param save_figs: Safe figures to file ('yes' or 'only') and/or
            display figures ('yes' or 'no'), defaults to 'no'.
        :type save_figs: Literal['yes', 'no', 'only'], optional
        :raises ValueError: Invalid input or configuration parameter.
        :raises RuntimeError: Missing map configuration to generate
            reduced tomography images.
        :return: Processed (meta)data of the last step.
        :rtype: Union[dict, nexusformat.nexus.NXroot]
        """
        # Local modules
        from nexusformat.nexus import (
            nxsetconfig,
            NXroot,
        )
        from CHAP.pipeline import PipelineItem
        from CHAP.tomo.models import (
            TomoReduceConfig,
            TomoFindCenterConfig,
            TomoReconstructConfig,
            TomoCombineConfig,
        )

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

        try:
            reduce_data_config = self.get_config(
                data, 'tomo.models.TomoReduceConfig')
        except:
            reduce_data_config = None
        try:
            find_center_config = self.get_config(
                data, 'tomo.models.TomoFindCenterConfig')
        except:
            find_center_config = None
        try:
            reconstruct_data_config = self.get_config(
                data, 'tomo.models.TomoReconstructConfig')
        except:
            reconstruct_data_config = None
        try:
            combine_data_config = self.get_config(
                data, 'tomo.models.TomoCombineConfig')
        except:
            combine_data_config = None
        nxroot = get_nxroot(data)

        tomo = Tomo(
            interactive=interactive, output_folder=output_folder,
            save_figs=save_figs)

        nxsetconfig(memory=100000)

        # Reduce tomography images
        if reduce_data or reduce_data_config is not None:
            if nxroot is None:
                raise RuntimeError('Map info required to reduce the '
                                   'tomography images')
            nxroot = tomo.reduce_data(nxroot, reduce_data_config)

        # Find rotation axis centers for the tomography stacks
        center_config = None
        if find_center or find_center_config is not None:
            run_find_centers = False
            if find_center_config is None:
                find_center_config = TomoFindCenterConfig()
                run_find_centers = True
            else:
                if (find_center_config.center_rows is None
                        or find_center_config.center_offsets is None):
                    run_find_centers = True
            if run_find_centers:
                center_config = tomo.find_centers(nxroot, find_center_config)
            else:
                # RV make a convert to dict in basemodel?
                center_config = {
                    'center_rows': find_center_config.center_rows,
                    'center_offsets': 
                        find_center_config.center_offsets,
                    'center_stack_index':
                         find_center_config.center_stack_index,
                }

        # Reconstruct tomography stacks
        # RV pass reconstruct_data_config and center_config directly to
        #     tomo.reconstruct_data?
        if reconstruct_data or reconstruct_data_config is not None:
            if reconstruct_data_config is None:
                reconstruct_data_config = TomoReconstructConfig()
            nxroot = tomo.reconstruct_data(
                nxroot, center_config, reconstruct_data_config)
            center_config = None

        # Combine reconstructed tomography stacks
        if combine_data or combine_data_config is not None:
            if combine_data_config is None:
                combine_data_config = TomoCombineConfig()
            nxroot = tomo.combine_data(nxroot, combine_data_config)

        if center_config is not None:
            return center_config
        return nxroot

def nxcopy(nxobject, exclude_nxpaths=None, nxpath_prefix=''):
    """
    Function that returns a copy of a nexus object, optionally exluding
    certain child items.

    :param nxobject: The input nexus object to "copy".
    :type nxobject: nexusformat.nexus.NXobject
    :param exlude_nxpaths: A list of paths to child nexus objects that
        should be excluded from the returned "copy", defaults to `[]`.
    :type exclude_nxpaths: list[str], optional
    :param nxpath_prefix: For use in recursive calls from inside this
        function only.
    :type nxpath_prefix: str
    :return: Copy of the input `nxobject` with some children optionally
        exluded.
    :rtype: nexusformat.nexus.NXobject
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
    """

    def __init__(self, num_core):
        """
        Initialize SetNumexprThreads.

        :param num_core: Number of processors used by the num_expr
            package.
        :type num_core: int
        """
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
            save_figs='no', test_mode=False):
        """
        Initialize Tomo.

        :param interactive: Allows for user interactions,
            defaults to False.
        :type interactive: bool, optional
        :param num_core: Number of processors.
        :type num_core: int
        :param output_folder: Output folder name, defaults to '.'.
        :type output_folder:: str, optional
        :param save_figs: Safe figures to file ('yes' or 'only') and/or
            display figures ('yes' or 'no'), defaults to 'no'.
        :type save_figs: Literal['yes', 'no', 'only'], optional
        :param test_mode: Run in test mode (non-interactively), defaults
            to False.
        :type test_mode: bool, optional
        :raises ValueError: Invalid input parameter.
        """
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
                f'of available processors and reduced to {cpu_count()}')
            self._num_core = cpu_count()

    def reduce_data(self, nxroot, tool_config=None):
        """
        Reduced the tomography images.

        :param nxroot: Data object containing the raw data info and
            metadata required for a tomography data reduction.
        :type nxroot: nexusformat.nexus.NXroot
        :param tool_config: Tool configuration.
        :type tool_config: CHAP.tomo.models.TomoReduceConfig, optional
        :raises ValueError: Invalid input or configuration parameter.
        :return: Reduced tomography data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXprocess,
            NXroot,
        )

        self._logger.info('Generate the reduced tomography images')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.attrs['default']]
        else:
            raise ValueError(
                f'Invalid parameter nxroot {type(nxroot)}:\n{nxroot}')
        if tool_config is None:
            delta_theta = None
            img_row_bounds = None
        else:
            delta_theta = tool_config.delta_theta
            img_row_bounds = tool_config.img_row_bounds

        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is None or 'data' not in nxentry.instrument.detector:
            raise ValueError(f'Unable to find image_key or data in '
                             'instrument.detector '
                             f'({nxentry.instrument.detector.tree})')
        image_key = np.asarray(image_key)

        # Create an NXprocess to store data reduction (meta)data
        reduced_data = NXprocess()

        # Generate dark field
        reduced_data = self._gen_dark(nxentry, reduced_data, image_key)

        # Generate bright field
        reduced_data = self._gen_bright(nxentry, reduced_data, image_key)

        # Get rotation angles for image stacks
        thetas = self._gen_thetas(nxentry, image_key)

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
            reduced_data.image_mask = image_mask
            thetas = thetas[image_mask]
        self._logger.debug(f'thetas = {thetas}')
        reduced_data.rotation_angle = thetas
        reduced_data.rotation_angle.units = 'degrees'

        # Set vertical detector bounds for image stack
        img_row_bounds = self._set_detector_bounds(
            nxentry, reduced_data, image_key, thetas[0],
            img_row_bounds=img_row_bounds)
        self._logger.info(f'img_row_bounds = {img_row_bounds}')
        reduced_data.img_row_bounds = img_row_bounds
        reduced_data.img_row_bounds.units = 'pixels'

        # Generate reduced tomography fields
        reduced_data = self._gen_tomo(nxentry, reduced_data, image_key)

        # Create a copy of the input Nexus object and remove raw and
        #     any existing reduced data
        if isinstance(nxroot, NXroot):
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
            nxroot = nxcopy(nxroot, exclude_nxpaths=exclude_items)
            nxentry = nxroot[nxroot.attrs['default']]

        # Add the reduced data NXprocess
        nxentry.reduced_data = reduced_data

        if 'data' not in nxentry:
            nxentry.data = NXdata()
        nxentry.data.makelink(
            nxentry.reduced_data.data.tomo_fields, name='reduced_data')
        nxentry.data.makelink(
            nxentry.reduced_data.rotation_angle, name='rotation_angle')
        nxentry.data.attrs['signal'] = 'reduced_data'

        return nxroot

    def find_centers(self, nxroot, tool_config):
        """
        Find the calibrated center axis info

        :param nxroot: Data object containing the reduced data and
            metadata required to find the calibrated center axis info.
        :type data: nexusformat.nexus.NXroot
        :param tool_config: Tool configuration.
        :type tool_config: CHAP.tomo.models.TomoFindCenterConfig
        :raises ValueError: Invalid or missing input or configuration
            parameter.
        :return: Calibrated center axis info.
        :rtype: dict
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )
        from yaml import safe_dump

        self._logger.info('Find the calibrated center axis info')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.attrs['default']]
        else:
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        center_rows = tool_config.center_rows
        center_stack_index = tool_config.center_stack_index
        if (center_stack_index is not None
                and (not isinstance(center_stack_index, int)
                     or center_stack_index < 0)):
            raise ValueError(
                'Invalid parameter center_stack_index '
                f'({center_stack_index})')

        # Check if reduced data is available
        if ('reduced_data' not in nxentry
                or 'reduced_data' not in nxentry.data):
            raise ValueError(f'Unable to find valid reduced data in {nxentry}.')

        # Get full bright field
        tbf = np.asarray(nxentry.reduced_data.data.bright_field)
        tbf_shape = tbf.shape

        # Get image bounds
        img_row_bounds = tuple(
            nxentry.reduced_data.get('img_row_bounds', (0, tbf_shape[0])))
        img_column_bounds = tuple(
            nxentry.reduced_data.get('img_column_bounds', (0, tbf_shape[1])))

        # Select the image stack to calibrate the center axis
        #     reduced data axes order: stack,theta,row,column
        # Note: Nexus can't follow a link if the data it points to is
        #     too big get the data from the actual place, not from
        #     nxentry.data
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        img_shape = nxentry.reduced_data.data.bright_field.shape
        num_row = int(img_row_bounds[1] - img_row_bounds[0])
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
                        'center axis', ge=0, lt=num_tomo_stacks,
                        default=int(num_tomo_stacks/2))
            else:
                if center_stack_index is None:
                    center_stack_index = int(num_tomo_stacks/2)
                    self._logger.warning(
                        'center_stack_index unspecified, use stack '
                        f'{center_stack_index} to find centers')
            default = 'y'

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Get effective pixel_size
        if 'zoom_perc' in nxentry.reduced_data:
            eff_pixel_size = float(
                100. * (nxentry.instrument.detector.row_pixel_size
                         / nxentry.reduced_data.attrs['zoom_perc']))
        else:
            eff_pixel_size = float(nxentry.instrument.detector.row_pixel_size)

        # Get cross sectional diameter
        cross_sectional_dim = img_shape[1]*eff_pixel_size
        self._logger.debug(f'cross_sectional_dim = {cross_sectional_dim}')

        # Determine center offset at sample row boundaries
        self._logger.info('Determine center offset at sample row boundaries')

        # Select center rows
        if self._test_mode:
            center_rows = tuple(self._test_config['center_rows'])
        else:
            # Third party modules
            import matplotlib.pyplot as plt

            if center_rows is None:
                if num_tomo_stacks == 1:
                    # Add a small margin to avoid edge effects
                    offset = min(5, int(0.1*num_row))
                    center_rows = (offset, num_row-1-offset)
                else:
                    if not self._interactive:
                        self._logger.warning('center_rows unspecified, find '
                                             'centers at reduced data bounds')
                    center_rows = (0, num_row-1)
            fig, center_rows = select_image_indices(
                nxentry.reduced_data.data.tomo_fields[
                    center_stack_index,0,:,:],
                0,
                b=tbf[img_row_bounds[0]:img_row_bounds[1],
                      img_column_bounds[0]:img_column_bounds[1]], 
                preselected_indices=center_rows,
                title='Select or adjust two detector image row indices to '
                    f'find center axis (in range [0, {num_row-1}])',
                title_a=r'Tomography image at $\theta$ = '
                        f'{round(thetas[0], 2)+0}',
                title_b='Bright field', interactive=self._interactive)
            if center_rows[1] == num_row:
                center_rows = (center_rows[0], center_rows[1]-1)
            # Plot results
            if self._save_figs:
                fig.savefig(
                    os_path.join(
                        self._output_folder, 'center_finding_rows.png'))
            plt.close()

        # Find the center offsets at each of the center rows
        center_offsets = []
        for i, center_row in enumerate(center_rows):
            t0 = time()
            center_offsets.append(
                self._find_center_one_plane(
                    nxentry.reduced_data.data.tomo_fields[
                        center_stack_index,:,center_row,:],
                    center_row, thetas, eff_pixel_size, cross_sectional_dim,
                    path=self._output_folder, num_core=self._num_core,
                    search_range=tool_config.search_range,
                    search_step=tool_config.search_step,
                    gaussian_sigma=tool_config.gaussian_sigma,
                    ring_width=tool_config.ring_width))
            self._logger.info(
                f'Finding center {i} took {time()-t0:.2f} seconds')
            self._logger.debug(f'center_row {i} = {center_rows[i]:.2f}')
            self._logger.debug(f'center_offset {i} = {center_offsets[i]:.2f}')

        center_config = {
            'center_rows': list(center_rows),
            'center_offsets': center_offsets,
        }
        if num_tomo_stacks > 1:
            center_config['center_stack_index'] = center_stack_index

        # Save test data to file
        if self._test_mode:
            with open(f'{self._output_folder}/center_config.yaml', 'w',
                      encoding='utf8') as f:
                safe_dump(center_config, f)

        return center_config

    def reconstruct_data(self, nxroot, center_info, tool_config):
        """
        Reconstruct the tomography data.

        :param nxroot: Data object containing the reduced data and
            metadata required for a tomography data reconstruction.
        :type data: nexusformat.nexus.NXroot
        :param center_info: Calibrated center axis info.
        :type center_info: dict
        :param tool_config: Tool configuration.
        :type tool_config: CHAP.tomo.models.TomoReconstructConfig
        :raises ValueError: Invalid or missing input or configuration
            parameter.
        :return: Reconstructed tomography data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            nxgetconfig,
            NXdata,
            NXentry,
            NXprocess,
            NXroot,
        )

        self._logger.info('Reconstruct the tomography data')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.attrs['default']]
        else:
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        if not isinstance(center_info, dict):
            raise ValueError(f'Invalid parameter center_info ({center_info})')

        # Check if reduced data is available
        if ('reduced_data' not in nxentry
                or 'reduced_data' not in nxentry.data):
            raise ValueError(f'Unable to find valid reduced data in {nxentry}.')

        # Create an NXprocess to store image reconstruction (meta)data
        nxprocess = NXprocess()

        # Get rotation axis rows and centers
        center_rows = center_info.get('center_rows')
        center_offsets = center_info.get('center_offsets')
        if center_rows is None or center_offsets is None:
            raise KeyError(
                'Unable to find valid calibrated center axis info in '
                f'{center_info}.')
        center_slope = (center_offsets[1]-center_offsets[0]) \
            / (center_rows[1]-center_rows[0])

        # Get thetas (in degrees)
        thetas = np.asarray(nxentry.reduced_data.rotation_angle)

        # Reconstruct tomography data
        #     reduced data axes order: stack,theta,row,column
        #     reconstructed data: row/-z,y,x
        # Note: Nexus can't follow a link if the data it points to is
        #     too big get the data from the actual place, not from
        #     nxentry.data
        if 'zoom_perc' in nxentry.reduced_data:
            res_title = f'{nxentry.reduced_data.attrs["zoom_perc"]}p'
        else:
            res_title = 'fullres'
        tomo_stacks = np.asarray(nxentry.reduced_data.data.tomo_fields)
        num_tomo_stacks = tomo_stacks.shape[0]
        tomo_recon_stacks = num_tomo_stacks*[np.array([])]
        for i in range(num_tomo_stacks):
            # Convert reduced data stack from theta,row,column to
            #     row,theta,column
            t0 = time()
            tomo_stack = tomo_stacks[i]
            self._logger.info(
                f'Reading reduced data stack {i} took {time()-t0:.2f} '
                'seconds')
            if (len(tomo_stack.shape) != 3
                    or any(True for dim in tomo_stack.shape if not dim)):
                raise RuntimeError(
                    f'Unable to load tomography stack {i} for '
                    'reconstruction')
            tomo_stack = np.swapaxes(tomo_stack, 0, 1)
            assert len(thetas) == tomo_stack.shape[1]
            assert 0 <= center_rows[0] < center_rows[1] < tomo_stack.shape[0]
            center_offsets = [
                center_offsets[0]-center_rows[0]*center_slope,
                center_offsets[1] + center_slope * (
                    tomo_stack.shape[0]-1-center_rows[1]),
            ]
            t0 = time()
            tomo_recon_stack = self._reconstruct_one_tomo_stack(
                tomo_stack, thetas, center_offsets=center_offsets,
                num_core=self._num_core, algorithm='gridrec',
                secondary_iters=tool_config.secondary_iters,
                remove_stripe_sigma=tool_config.remove_stripe_sigma,
                ring_width=tool_config.ring_width)
            self._logger.info(
                f'Reconstruction of stack {i} took {time()-t0:.2f} seconds')

            # Combine stacks
            tomo_recon_stacks[i] = tomo_recon_stack

        # Resize the reconstructed tomography data
        #     reconstructed data order in each stack: row/-z,y,x
        tomo_recon_shape = tomo_recon_stacks[0].shape
        if self._test_mode:
            x_bounds = tuple(self._test_config.get('x_bounds'))
            y_bounds = tuple(self._test_config.get('y_bounds'))
            z_bounds = (0, tomo_recon_shape[0])
        else:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
                tomo_recon_stacks, x_bounds=tool_config.x_bounds,
                y_bounds=tool_config.y_bounds, z_bounds=tool_config.z_bounds)
        if x_bounds is None:
            x_range = (0, tomo_recon_shape[2])
            x_slice = int(x_range[1]/2)
        else:
            x_range = (min(x_bounds), max(x_bounds))
            x_slice = int((x_bounds[0]+x_bounds[1]) / 2)
        if y_bounds is None:
            y_range = (0, tomo_recon_shape[1])
            y_slice = int(y_range[1] / 2)
        else:
            y_range = (min(y_bounds), max(y_bounds))
            y_slice = int((y_bounds[0]+y_bounds[1]) / 2)
        if z_bounds is None:
            z_range = (0, tomo_recon_shape[0])
            z_slice = int(z_range[1] / 2)
        else:
            z_range = (min(z_bounds), max(z_bounds))
            z_slice = int((z_bounds[0]+z_bounds[1]) / 2)
        z_dim_org = tomo_recon_shape[0]
        for i, stack in enumerate(tomo_recon_stacks):
            tomo_recon_stacks[i] = stack[
                z_range[0]:z_range[1],y_range[0]:y_range[1],
                x_range[0]:x_range[1]]
        tomo_recon_stacks = np.asarray(tomo_recon_stacks)

        row_pixel_size = float(
            nxentry.instrument.detector.row_pixel_size)
        column_pixel_size = float(
            nxentry.instrument.detector.column_pixel_size)
        if num_tomo_stacks == 1:
            # Convert the reconstructed tomography data from internal
            #     coordinate frame row/-z,y,x with the origin on the
            #     near-left-top corner to an z,y,x coordinate frame
            #     with the origin on the par file x,z values, halfway
            #     in the y-dimension.
            #     Here x is to the right, y along the beam direction
            #     and z upwards in the lab frame of reference
            tomo_recon_stack = np.flip(tomo_recon_stacks[0], 0)
            z_range = (z_dim_org-z_range[1], z_dim_org-z_range[0])

            # Get coordinate axes
            x = column_pixel_size * (
                np.linspace(
                    x_range[0], x_range[1], x_range[1]-x_range[0], False)
                - 0.5*nxentry.instrument.detector.columns
                + 0.5)
            x = np.asarray(x + nxentry.reduced_data.x_translation[0])
            y = np.asarray(
                    column_pixel_size * (
                    np.linspace(
                        y_range[0], y_range[1], y_range[1]-y_range[0], False)
                    - 0.5*nxentry.instrument.detector.columns
                    + 0.5))
            z = row_pixel_size*(
                np.linspace(
                    z_range[0], z_range[1], z_range[1]-z_range[0], False)
                + nxentry.instrument.detector.rows
                - int(nxentry.reduced_data.img_row_bounds[1])
                + 0.5)
            z = np.asarray(z + nxentry.reduced_data.z_translation[0])

            # Plot a few reconstructed image slices
            if self._save_figs:
                x_index = x_slice-x_range[0]
                extent = (
                    y[0],
                    y[-1],
                    z[0],
                    z[-1])
                quick_imshow(
                    tomo_recon_stack[:,:,x_index], 
                    title=f'recon {res_title} x={x[x_index]:.4f}',
                    origin='lower', extent=extent, path=self._output_folder,
                    save_fig=True, save_only=True)
                y_index = y_slice-y_range[0]
                extent = (
                    x[0],
                    x[-1],
                    z[0],
                    z[-1])
                quick_imshow(
                    tomo_recon_stack[:,y_index,:],
                    title=f'recon {res_title} y={y[y_index]:.4f}',
                    origin='lower', extent=extent, path=self._output_folder,
                    save_fig=True, save_only=True)
                z_index = z_slice-z_range[0]
                extent = (
                    x[0],
                    x[-1],
                    y[0],
                    y[-1])
                quick_imshow(
                    tomo_recon_stack[z_index,:,:],
                    title=f'recon {res_title} z={z[z_index]:.4f}',
                    origin='lower', extent=extent, path=self._output_folder,
                    save_fig=True, save_only=True)

            # Save test data to file
            #     reconstructed data order in each stack: z,y,x
            if self._test_mode:
                np.savetxt(
                    f'{self._output_folder}/recon_stack.txt',
                    tomo_recon_stacks[z_slice-z_range[0],:,:],
                    fmt='%.6e')
        else:
            # Plot a few reconstructed image slices
            if self._save_figs:
                for i in range(tomo_recon_stacks.shape[0]):
                    basetitle = f'recon stack {i}'
                    title = f'{basetitle} {res_title} xslice{x_slice}'
                    quick_imshow(
                        tomo_recon_stacks[i,:,:,x_slice-x_range[0]],
                        title=title, path=self._output_folder,
                        save_fig=True, save_only=True)
                    title = f'{basetitle} {res_title} yslice{y_slice}'
                    quick_imshow(
                        tomo_recon_stacks[i,:,y_slice-y_range[0],:],
                        title=title, path=self._output_folder,
                        save_fig=True, save_only=True)
                    title = f'{basetitle} {res_title} zslice{z_slice}'
                    quick_imshow(
                        tomo_recon_stacks[i,z_slice-z_range[0],:,:],
                        title=title, path=self._output_folder,
                        save_fig=True, save_only=True)

            # Save test data to file
            #     reconstructed data order in each stack: row/-z,y,x
            if self._test_mode:
                for i in range(tomo_recon_shape[0]):
                    np.savetxt(
                        f'{self._output_folder}/recon_stack_{i}.txt',
                        tomo_recon_stacks[i,z_slice-z_range[0],:,:],
                        fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #     reconstructed data order:
        #     - for one stack: z,y,x
        #     - for multiple stacks: row/-z,y,x
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        for k, v in center_info.items():
            nxprocess[k] = v
            if k == 'center_rows' or k == 'center_offsets':
                nxprocess[k].units = 'pixels'
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
            nxprocess.x_bounds.units = 'pixels'
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
            nxprocess.y_bounds.units = 'pixels'
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
            nxprocess.z_bounds.units = 'pixels'
        nxprocess.data.attrs['signal'] = 'reconstructed_data'
        if num_tomo_stacks == 1:
            nxprocess.data.reconstructed_data = tomo_recon_stack
            nxprocess.data.attrs['axes'] = ['z', 'y', 'x']
            nxprocess.data.attrs['x_indices'] = 2
            nxprocess.data.attrs['y_indices'] = 1
            nxprocess.data.attrs['z_indices'] = 0
            nxprocess.data.x = x
            nxprocess.data.x.units = \
                nxentry.instrument.detector.column_pixel_size.units
            nxprocess.data.y = y
            nxprocess.data.y.units = \
                nxentry.instrument.detector.column_pixel_size.units
            nxprocess.data.z = z
            nxprocess.data.z.units = \
                nxentry.instrument.detector.row_pixel_size.units
        else:
            nxprocess.data.reconstructed_data = tomo_recon_stacks

        # Create a copy of the input Nexus object and remove reduced
        #     data
        exclude_items = [
            f'{nxentry.nxname}/reduced_data/data',
            f'{nxentry.nxname}/data/reduced_data',
            f'{nxentry.nxname}/data/rotation_angle',
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
        if num_tomo_stacks == 1:
            nxentry_copy.data.attrs['axes'] = ['z', 'y', 'x']
            nxentry_copy.data.attrs['x_indices'] = 2
            nxentry_copy.data.attrs['y_indices'] = 1
            nxentry_copy.data.attrs['z_indices'] = 0
            nxentry_copy.data.makelink(nxprocess.data.x, name='x')
            nxentry_copy.data.makelink(nxprocess.data.y, name='y')
            nxentry_copy.data.makelink(nxprocess.data.z, name='z')

        return nxroot_copy

    def combine_data(self, nxroot, tool_config):
        """Combine the reconstructed tomography stacks.

        :param nxroot: Data object containing the reconstructed data
            and metadata required to combine the tomography stacks.
        :type data: nexusformat.nexus.NXroot
        :param tool_config: Tool configuration.
        :type tool_config: CHAP.tomo.models.TomoCombineConfig
        :raises ValueError: Invalid or missing input or configuration
            parameter.
        :return: Combined reconstructed tomography data.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXentry,
            NXprocess,
            NXroot,
        )

        self._logger.info('Combine the reconstructed tomography stacks')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.attrs['default']]
        else:
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')

        # Check if reconstructed image data is available
        if ('reconstructed_data' not in nxentry
                or 'reconstructed_data' not in nxentry.data):
            raise KeyError(
                f'Unable to find valid reconstructed image data in {nxentry}')

        # Create an NXprocess to store combined image reconstruction
        #     (meta)data
        nxprocess = NXprocess()

        num_tomo_stacks = \
            nxentry.reconstructed_data.data.reconstructed_data.shape[0]
        if num_tomo_stacks == 1:
            self._logger.info('Only one stack available: leaving combine_data')
            return nxroot

        # Get and combine the reconstructed stacks
        #     reconstructed data order: stack,row/-z,y,x
        # Note: Nexus can't follow a link if the data it points to is
        #     too big. So get the data from the actual place, not from
        #     nxentry.data
        # (load one stack at a time to reduce risk of hitting Nexus
        #     data access limit)
        t0 = time()
        tomo_recon_combined = \
            nxentry.reconstructed_data.data.reconstructed_data[0,:,:,:]
        tomo_recon_combined = np.concatenate(
            [nxentry.reconstructed_data.data.reconstructed_data[i,:,:,:]
               for i in range(num_tomo_stacks-1,0,-1)]
            + [tomo_recon_combined])
        self._logger.info(
            f'Combining the reconstructed stacks took {time()-t0:.2f} seconds')

        # Resize the combined tomography data stacks
        #     combined data order: row/-z,y,x
        if self._test_mode:
            x_bounds = None
            y_bounds = None
            z_bounds = tuple(self._test_config.get('z_bounds'))
        elif self._interactive:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
                tomo_recon_combined, combine_data=True)
        else:
            x_bounds = tool_config.x_bounds
            if x_bounds is None:
                self._logger.warning(
                    'x_bounds unspecified, reconstruct data for full x-range')
            elif not is_int_pair(
                    x_bounds, ge=0, le=tomo_recon_combined.shape[2]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            y_bounds = tool_config.y_bounds
            if y_bounds is None:
                self._logger.warning(
                    'y_bounds unspecified, reconstruct data for full y-range')
            elif not is_int_pair(
                    y_bounds, ge=0, le=tomo_recon_combined.shape[1]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = tool_config.z_bounds
            if z_bounds is None:
                self._logger.warning(
                    'z_bounds unspecified, reconstruct data for full z-range')
            elif not is_int_pair(
                    z_bounds, ge=0, le=tomo_recon_combined.shape[0]):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
        if x_bounds is None:
            x_range = (0, tomo_recon_combined.shape[2])
            x_slice = int(x_range[1]/2)
        else:
            x_range = (min(x_bounds), max(x_bounds))
            x_slice = int((x_bounds[0]+x_bounds[1]) / 2)
        if y_bounds is None:
            y_range = (0, tomo_recon_combined.shape[1])
            y_slice = int(y_range[1]/2)
        else:
            y_range = (min(y_bounds), max(y_bounds))
            y_slice = int((y_bounds[0]+y_bounds[1]) / 2)
        if z_bounds is None:
            z_range = (0, tomo_recon_combined.shape[0])
            z_slice = int(z_range[1]/2)
        else:
            z_range = (min(z_bounds), max(z_bounds))
            z_slice = int((z_bounds[0]+z_bounds[1]) / 2)
        z_dim_org = tomo_recon_combined.shape[0]
        tomo_recon_combined = tomo_recon_combined[
            z_range[0]:z_range[1],y_range[0]:y_range[1],x_range[0]:x_range[1]]

        # Convert the reconstructed tomography data from internal
        #     coordinate frame row/-z,y,x with the origin on the
        #     near-left-top corner to an z,y,x coordinate frame.
        #     Here x is to the right, y along the beam direction
        #     and z upwards in the lab frame of reference
        tomo_recon_combined = np.flip(tomo_recon_combined, 0)
        z_range = (z_dim_org-z_range[1], z_dim_org-z_range[0])

        # Get coordinate axes
        row_pixel_size = float(
            nxentry.instrument.detector.row_pixel_size)
        column_pixel_size = float(
            nxentry.instrument.detector.column_pixel_size)
        x = column_pixel_size * (
            np.linspace(x_range[0], x_range[1], x_range[1]-x_range[0], False)
            - 0.5*nxentry.instrument.detector.columns
            + 0.5)
        if nxentry.reconstructed_data.get('x_bounds', None) is not None:
            x += column_pixel_size*nxentry.reconstructed_data.x_bounds[0]
        x = np.asarray(x + nxentry.reduced_data.x_translation[0])
        y = column_pixel_size * (
            np.linspace(y_range[0], y_range[1], y_range[1]-y_range[0], False)
            - 0.5*nxentry.instrument.detector.columns
            + 0.5)
        if nxentry.reconstructed_data.get('y_bounds', None) is not None:
            y += column_pixel_size*nxentry.reconstructed_data.y_bounds[0]
        y = np.asarray(y)
        z = row_pixel_size*(
            np.linspace(z_range[0], z_range[1], z_range[1]-z_range[0], False)
            - int(nxentry.reduced_data.img_row_bounds[0])
            + 0.5*(nxentry.instrument.detector.rows)
            -0.5)
        z = np.asarray(z + nxentry.reduced_data.z_translation[0])

        # Plot a few combined image slices
        if self._save_figs:
            extent = (
                y[0],
                y[-1],
                z[0],
                z[-1])
            x_slice = int(tomo_recon_combined.shape[2]/2)
            quick_imshow(
                tomo_recon_combined[:,:,x_slice],
                title=f'recon combined x={x[x_slice]:.4f}', origin='lower',
                extent=extent, path=self._output_folder, save_fig=True,
                save_only=True)
            extent = (
                x[0],
                x[-1],
                z[0],
                z[-1])
            y_slice = int(tomo_recon_combined.shape[1]/2)
            quick_imshow(
                tomo_recon_combined[:,y_slice,:],
                title=f'recon combined y={y[y_slice]:.4f}', origin='lower',
                extent=extent, path=self._output_folder, save_fig=True,
                save_only=True)
            extent = (
                x[0],
                x[-1],
                y[0],
                y[-1])
            z_slice = int(tomo_recon_combined.shape[0]/2)
            quick_imshow(
                tomo_recon_combined[z_slice,:,:],
                title=f'recon combined z={z[z_slice]:.4f}', origin='lower',
                extent=extent, path=self._output_folder, save_fig=True,
                save_only=True)

        # Save test data to file
        #     combined data order: z,y,x
        if self._test_mode:
            z_slice = int(tomo_recon_combined.shape[0]/2)
            np.savetxt(
                f'{self._output_folder}/recon_combined.txt',
                tomo_recon_combined[z_slice,:,:], fmt='%.6e')

        # Add image reconstruction to reconstructed data NXprocess
        #     combined data order: z,y,x
        nxprocess.data = NXdata()
        nxprocess.attrs['default'] = 'data'
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
            nxprocess.x_bounds.units = 'pixels'
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
            nxprocess.y_bounds.units = 'pixels'
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
            nxprocess.z_bounds.units = 'pixels'
        nxprocess.data.combined_data = tomo_recon_combined
        nxprocess.data.attrs['signal'] = 'combined_data'
        nxprocess.data.attrs['axes'] = ['z', 'y', 'x']
        nxprocess.data.attrs['x_indices'] = 2
        nxprocess.data.attrs['y_indices'] = 1
        nxprocess.data.attrs['z_indices'] = 0
        nxprocess.data.x = x
        nxprocess.data.x.units = \
            nxentry.instrument.detector.column_pixel_size.units
        nxprocess.data.y = y
        nxprocess.data.y.units = \
            nxentry.instrument.detector.column_pixel_size.units
        nxprocess.data.z = z
        nxprocess.data.z.units = \
            nxentry.instrument.detector.row_pixel_size.units

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
        nxentry_copy.data.attrs['axes'] = ['z', 'y', 'x']
        nxentry_copy.data.attrs['x_indices'] = 2
        nxentry_copy.data.attrs['y_indices'] = 1
        nxentry_copy.data.attrs['z_indices'] = 0
        nxentry_copy.data.makelink(nxprocess.data.x, name='x')
        nxentry_copy.data.makelink(nxprocess.data.y, name='y')
        nxentry_copy.data.makelink(nxprocess.data.z, name='z')

        return nxroot_copy

    def _gen_dark(self, nxentry, reduced_data, image_key):
        """Generate dark field."""
        # Third party modules
        from nexusformat.nexus import NXdata

        # Get the dark field images
        field_indices = [
            index for index, key in enumerate(image_key) if key == 2]
        if field_indices:
            tdf_stack = np.asarray(
                nxentry.instrument.detector.data[field_indices,:,:])
        else:
            self._logger.warning('Dark field unavailable')
            return reduced_data

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
                tdf, title='Dark field', name='dark_field',
                path=self._output_folder, save_fig=True, save_only=True)

        # Add dark field to reduced data NXprocess
        reduced_data.data = NXdata()
        reduced_data.data.dark_field = tdf

        return reduced_data

    def _gen_bright(self, nxentry, reduced_data, image_key):
        """Generate bright field."""
        # Third party modules
        from nexusformat.nexus import NXdata

        # Get the bright field images
        field_indices = [
            index for index, key in enumerate(image_key) if key == 1]
        if field_indices:
            tbf_stack = np.asarray(
                nxentry.instrument.detector.data[field_indices,:,:])
        else:
            raise ValueError('Bright field unavailable')

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
            tbf -= np.asarray(reduced_data.data.dark_field)
        else:
            self._logger.warning('Dark field unavailable')

        # Set any non-positive values to one
        # (avoid negative bright field values for spikes in dark field)
        tbf[tbf < 1] = 1

        # Plot bright field
        if self._save_figs:
            quick_imshow(
                tbf, title='Bright field', name='bright_field',
                path=self._output_folder, save_fig=True, save_only=True)

        # Add bright field to reduced data NXprocess
        if 'data' not in reduced_data:
            reduced_data.data = NXdata()
        reduced_data.data.bright_field = tbf

        return reduced_data

    def _set_detector_bounds(self, nxentry, reduced_data, image_key, theta,
            img_row_bounds=None):
        """
        Set vertical detector bounds for each image stack.Right now the
        range is the same for each set in the image stack.
        """
        # Third party modules
        import matplotlib.pyplot as plt

        # Local modules
        from CHAP.utils.general import is_index_range

        if self._test_mode:
            return tuple(self._test_config['img_row_bounds'])

        # Get the first tomography image and the reference heights
        image_mask = reduced_data.get('image_mask')
        if image_mask is None:
            first_image_index = 0
        else:
            raise RuntimeError('image_mask not tested yet')
            image_mask = np.asarray(image_mask)
            first_image_index = int(np.argmax(image_mask))
        field_indices_all = [
            index for index, key in enumerate(image_key) if key == 0]
        if not field_indices_all:
            raise ValueError('Tomography field(s) unavailable')
        z_translation_all = np.asarray(
            nxentry.sample.z_translation)[field_indices_all]
        z_translation_levels = sorted(list(set(z_translation_all)))
        num_tomo_stacks = len(z_translation_levels)
        center_stack_index = int(num_tomo_stacks/2)
        z_translation = z_translation_levels[center_stack_index]
        try:
            field_indices = [
                field_indices_all[index]
                for index, z in enumerate(z_translation_all)
                if z == z_translation]
            first_image = np.asarray(nxentry.instrument.detector.data[
                field_indices[first_image_index]])
        except:
            raise RuntimeError('Unable to load the tomography images '
                               f'for stack {i}')

        # Select image bounds
        tbf = np.asarray(reduced_data.data.bright_field)
        if nxentry.instrument.source.attrs['station'] in ('id1a3', 'id3a'):
            pixel_size = float(nxentry.instrument.detector.row_pixel_size)
            # Try to get a fit from the bright field
            row_sum = np.sum(tbf, 1)
            fit = Fit.fit_data(
                row_sum, 'rectangle', x=np.array(range(len(row_sum))),
                form='atan', guess=True)
            parameters = fit.best_values
            row_low_fit = parameters.get('center1', None)
            row_upp_fit = parameters.get('center2', None)
            sig_low = parameters.get('sigma1', None)
            sig_upp = parameters.get('sigma2', None)
            have_fit = (fit.success and row_low_fit is not None
                and row_upp_fit is not None and sig_low is not None
                and sig_upp is not None
                and 0 <= row_low_fit < row_upp_fit <= row_sum.size
                and (sig_low+sig_upp) / (row_upp_fit-row_low_fit) < 0.1)
            if num_tomo_stacks == 1:
                if have_fit:
                    delta_z = (row_upp_fit-row_low_fit) * pixel_size
                else:
                    # Set a default range of 1 mm
                    # RV can we get this from the slits?
                    delta_z = 1.0
            else:
                # Get the default range from the reference heights
                delta_z = z_translation_levels[1]-z_translation_levels[0]
                for i in range(2, num_tomo_stacks):
                    delta_z = min(
                        delta_z,
                        z_translation_levels[i]-z_translation_levels[i-1])
            self._logger.debug(f'delta_z = {delta_z}')
            num_row_min = int((delta_z + 0.5*pixel_size) / pixel_size)
            if num_row_min > tbf.shape[0]:
                self._logger.warning(
                    'Image bounds and pixel size prevent seamless '
                    'stacking')
                row_low = 0
                row_upp = tbf.shape[0]
            else:
                self._logger.debug(f'num_row_min = {num_row_min}')
                if have_fit:
                    # Center the default range relative to the fitted
                    #     window
                    row_low = int((row_low_fit+row_upp_fit-num_row_min) / 2)
                    row_upp = row_low+num_row_min
                else:
                    # Center the default range
                    row_low = int((tbf.shape[0]-num_row_min) / 2)
                    row_upp = row_low+num_row_min
            img_row_bounds = (row_low, row_upp)
        else:
            if num_tomo_stacks > 1:
                raise NotImplementedError(
                    'Selecting image bounds for multiple stacks on FMB')
            # For FMB: use the first tomography image to select range
            # RV revisit if they do tomography with multiple stacks
            if img_row_bounds is None:
                if not self._interactive:
                    self._logger.warning(
                        'img_row_bounds unspecified, reduce data for entire '
                        'detector range')
                    img_row_bounds = (0, first_image.shape[0])
        fig, img_row_bounds = select_image_indices(
            first_image, 0, b=tbf, preselected_indices=img_row_bounds,
            title='Select or adjust detector image row bounds for data '
                  f'reduction (in range {[0, first_image.shape[0]]})',
            title_a=r'Tomography image at $\theta$ = 'f'{round(theta, 2)+0}',
            title_b='Bright field',
            interactive=self._interactive)
        if (num_tomo_stacks > 1
                and (img_row_bounds[1]-img_row_bounds[0]+1)
                     < int((delta_z - 0.5*pixel_size) / pixel_size)):
            self._logger.warning(
                'Image bounds and pixel size prevent seamless stacking')

        # Plot results
        if self._save_figs:
            fig.savefig(
                os_path.join(self._output_folder, 'detector_image_bounds.png'))
        plt.close()

        return img_row_bounds

    def _gen_thetas(self, nxentry, image_key):
        """Get the rotation angles for the image stacks."""
        # Get the rotation angles
        field_indices_all = [
            index for index, key in enumerate(image_key) if key == 0]
        z_translation_all = np.asarray(
            nxentry.sample.z_translation)[field_indices_all]
        z_translation_levels = sorted(list(set(z_translation_all)))
        thetas = None
        for i, z_translation in enumerate(z_translation_levels):
            field_indices = [
                field_indices_all[index]
                for index, z in enumerate(z_translation_all)
                if z == z_translation]
            sequence_numbers = np.asarray(
                nxentry.instrument.detector.sequence_number)[field_indices]
            assert (list(sequence_numbers)
                    == list(range((len(sequence_numbers)))))
            if thetas is None:
                thetas = np.asarray(
                    nxentry.sample.rotation_angle)[
                        field_indices][sequence_numbers]
            else:
                assert all(
                    thetas[i] == np.asarray(
                        nxentry.sample.rotation_angle)[
                            field_indices[index]]
                    for i, index in enumerate(sequence_numbers))

        return thetas

    def _set_zoom_or_delta_theta(self, thetas, delta_theta=None):
        """
        Set zoom and/or delta theta to reduce memory the requirement
        for the analysis.
        """
        # Local modules
        from CHAP.utils.general import index_nearest

        if self._test_mode:
            return tuple(self._test_config['delta_theta'])

#        if input_yesno(
#                '\nDo you want to zoom in to reduce memory '
#                'requirement (y/n)?', 'n'):
#            zoom_perc = input_int(
#                '    Enter zoom percentage', ge=1, le=100)
#        else:
#            zoom_perc = None
        zoom_perc = None

        if delta_theta is not None and not is_num(delta_theta, gt=0):
            self._logger.warning(
                f'Invalid parameter delta_theta ({delta_theta}), '
                'ignoring delta_theta')
            delta_theta = None
        if self._interactive:
            if delta_theta is None:
                delta_theta = thetas[1]-thetas[0]
            print(f'Available \u03b8 range: [{thetas[0]}, {thetas[-1]}]')
            print(f'Current \u03b8 interval: {delta_theta}')
            if input_yesno(
                    'Do you want to change the \u03b8 interval to reduce the '
                    'memory requirement (y/n)?', 'n'):
                delta_theta = input_num(
                    '    Enter the desired \u03b8 interval',
                    ge=thetas[1]-thetas[0], lt=(thetas[-1]-thetas[0])/2)
        if delta_theta is not None:
            delta_theta = index_nearest(thetas, thetas[0]+delta_theta)
            if delta_theta <= 1:
                delta_theta = None

        return zoom_perc, delta_theta

    def _gen_tomo(self, nxentry, reduced_data, image_key):
        """Generate tomography fields."""
        # Third party modules
        from numexpr import evaluate
        from scipy.ndimage import zoom

        # Get full bright field
        tbf = np.asarray(reduced_data.data.bright_field)
        tbf_shape = tbf.shape

        # Get image bounds
        img_row_bounds = tuple(
            reduced_data.get('img_row_bounds', (0, tbf_shape[0])))
        img_column_bounds = tuple(
            reduced_data.get('img_column_bounds', (0, tbf_shape[1])))

        # Get resized dark field
        if 'dark_field' in reduced_data.data:
            tdf = np.asarray(
                reduced_data.data.dark_field)[
                    img_row_bounds[0]:img_row_bounds[1],
                    img_column_bounds[0]:img_column_bounds[1]]
        else:
            self._logger.warning('Dark field unavailable')
            tdf = None

        # Resize bright field
        if (img_row_bounds != (0, tbf.shape[0])
                or img_column_bounds != (0, tbf.shape[1])):
            tbf = tbf[
                img_row_bounds[0]:img_row_bounds[1],
                img_column_bounds[0]:img_column_bounds[1]]

        # Get thetas (in degrees)
        thetas = np.asarray(reduced_data.rotation_angle)

        # Get or create image mask
        image_mask = reduced_data.get('image_mask')
        if image_mask is None:
            image_mask = np.ones(len(thetas), dtype=bool)
        else:
            image_mask = np.asarray(image_mask)

        # Get the tomography images
        field_indices_all = [
            index for index, key in enumerate(image_key) if key == 0]
        if not field_indices_all:
            raise ValueError('Tomography field(s) unavailable')
        z_translation_all = np.asarray(
            nxentry.sample.z_translation)[field_indices_all]
        z_translation_levels = sorted(list(set(z_translation_all)))
        num_tomo_stacks = len(z_translation_levels)
        tomo_stacks = num_tomo_stacks*[np.array([])]
        horizontal_shifts = []
        vertical_shifts = []
        tomo_stacks = []
        for i, z_translation in enumerate(z_translation_levels):
            try:
                field_indices = [
                    field_indices_all[index]
                    for index, z in enumerate(z_translation_all)
                    if z == z_translation]
                field_indices_masked = np.asarray(field_indices)[image_mask]
                horizontal_shift = list(set(np.asarray(
                    nxentry.sample.x_translation)[field_indices_masked]))
                assert len(horizontal_shift) == 1
                horizontal_shifts += horizontal_shift
                vertical_shift = list(set(np.asarray(
                    nxentry.sample.z_translation)[field_indices_masked]))
                assert len(vertical_shift) == 1
                vertical_shifts += vertical_shift
                sequence_numbers = np.asarray(
                    nxentry.instrument.detector.sequence_number)[
                        field_indices]
                assert (list(sequence_numbers)
                        == list(range((len(sequence_numbers)))))
                tomo_stack = np.asarray(
                    nxentry.instrument.detector.data)[field_indices_masked]
            except:
                raise RuntimeError('Unable to load the tomography images '
                                   f'for stack {i}')
            tomo_stacks.append(tomo_stack)
            if not i:
                tomo_stack_shape = tomo_stack.shape
            else:
                assert tomo_stack_shape == tomo_stack.shape

        row_pixel_size = float(nxentry.instrument.detector.row_pixel_size)
        column_pixel_size = float(
            nxentry.instrument.detector.column_pixel_size)
        reduced_tomo_stacks = []
        for i, tomo_stack in enumerate(tomo_stacks):
            # Resize the tomography images
            # Right now the range is the same for each set in the stack
            assert len(thetas) == tomo_stack.shape[0]
            if (img_row_bounds != (0, tomo_stack.shape[1])
                    or img_column_bounds != (0, tomo_stack.shape[2])):
                tomo_stack = tomo_stack[
                    :,img_row_bounds[0]:img_row_bounds[1],
                    img_column_bounds[0]:img_column_bounds[1]].astype(
                        'float64', copy=False)
            else:
                tomo_stack = tomo_stack.astype('float64', copy=False)

            # Subtract dark field
            if tdf is not None:
                try:
                    with SetNumexprThreads(self._num_core):
                        evaluate('tomo_stack-tdf', out=tomo_stack)
                except TypeError as e:
                    sys_exit(
                        f'\nA {type(e).__name__} occured while subtracting '
                        'the dark field with num_expr.evaluate()'
                        '\nTry reducing the detector range'
                        f'\n(currently img_row_bounds = {img_row_bounds}, and '
                        f'img_column_bounds = {img_column_bounds})\n')

            # Normalize
            try:
                with SetNumexprThreads(self._num_core):
                    evaluate('tomo_stack/tbf', out=tomo_stack, truediv=True)
            except TypeError as e:
                sys_exit(
                    f'\nA {type(e).__name__} occured while normalizing the '
                    'tomography data with num_expr.evaluate()'
                    '\nTry reducing the detector range'
                    f'\n(currently img_row_bounds = {img_row_bounds}, and '
                    f'img_column_bounds = {img_column_bounds})\n')

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
            if not self._test_mode and (self._save_figs or self._save_only):
                theta = round(thetas[0], 2)
                if len(tomo_stacks) == 1:
                    title = r'Reduced data, $\theta$ = 'f'{theta}'
                    name = f'reduced_data_theta_{theta}'
                else:
                    title = f'Reduced data stack {i}, 'r'$\theta$ = 'f'{theta}'
                    name = f'reduced_data_stack_{i}_theta_{theta}'
                quick_imshow(
                    tomo_stack[0,:,:], title=title, name=name,
                    path=self._output_folder, save_fig=self._save_figs,
                    save_only=self._save_only, block=self._block)
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
                        f'{round(thetas[0], 2)+0}'
                    quick_imshow(
                        tomo_stack[0,:,:], title=title, 
                        path=self._output_folder, save_fig=self._save_figs,
                        save_only=self._save_only, block=self._block)

            # Save test data to file
            if self._test_mode:
                row_index = int(tomo_stack.shape[1]/2)
                np.savetxt(
                    f'{self._output_folder}/red_stack_{i}.txt',
                    tomo_stack[:,row_index,:], fmt='%.6e')

            # Combine resized stacks
            reduced_tomo_stacks.append(tomo_stack)

        # Add tomo field info to reduced data NXprocess
        reduced_data.x_translation = np.asarray(horizontal_shifts)
        reduced_data.x_translation.units = 'mm'
        reduced_data.z_translation = np.asarray(vertical_shifts)
        reduced_data.z_translation.units = 'mm'
        reduced_data.data.tomo_fields = np.asarray(reduced_tomo_stacks)
        reduced_data.data.attrs['signal'] = 'tomo_fields'

        if tdf is not None:
            del tdf
        del tbf

        return reduced_data

    def _find_center_one_plane(
            self, sinogram, row, thetas, eff_pixel_size, cross_sectional_dim,
            path=None, num_core=1, search_range=None, search_step=None,
            gaussian_sigma=None, ring_width=None):
        """Find center for a single tomography plane."""
        # Third party modules
        import matplotlib.pyplot as plt
        from tomopy import find_center_vo

        if not gaussian_sigma:
            gaussian_sigma = None
        if not ring_width:
            ring_width = None
        # Try automatic center finding routines for initial value
        # sinogram index order: theta,column
        # need column,theta for iradon, so take transpose
        sinogram = np.asarray(sinogram)
        sinogram_t = sinogram.T
        center = sinogram.shape[1]/2

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
            'seconds')
        center_offset_vo = float(tomo_center-center)
        self._logger.info(
            f'Center at row {row} using Nghia Vo’s method = '
            f'{center_offset_vo:.2f}')

        if self._interactive or self._save_figs:

            # Reconstruct the plane for Nghia Vo’s center offset
            t0 = time()
            recon_plane = self._reconstruct_one_plane(
                sinogram_t, center_offset_vo, thetas, eff_pixel_size,
                cross_sectional_dim, False, num_core, gaussian_sigma,
                ring_width)
            self._logger.info(
                f'Reconstructing row {row} took {time()-t0:.2f} seconds')
            recon_edges = [self._get_edges_one_plane(recon_plane)]
            fig, accept, selected_center_offset = self._select_center_offset(
                recon_edges, row, center_offset_vo)
            # Plot results
            if self._save_figs:
                fig.savefig(
                    os_path.join(
                        self._output_folder,
                        f'edges_default_center_row_{row}.png'))
            plt.close()


            # Perform center finding search
            center_offsets = [center_offset_vo]
            step_size = 4
            indices = 3*[-1]
            prev_index = None
            up = True
            while not accept and step_size:
                selected_center_offset = round(selected_center_offset)
                preselected_offsets = (
                    selected_center_offset-step_size,
                    selected_center_offset,
                    selected_center_offset+step_size)
                for i, preselected_offset in enumerate(preselected_offsets):
                    if preselected_offset in center_offsets:
                        indices[i] = center_offsets.index(preselected_offset)
                    else:
                        recon_plane = self._reconstruct_one_plane(
                            sinogram_t, preselected_offset, thetas,
                            eff_pixel_size, cross_sectional_dim, False,
                            num_core, gaussian_sigma, ring_width)
                        indices[i] = len(center_offsets)
                        center_offsets.append(preselected_offset)
                        recon_edges.append(
                            self._get_edges_one_plane(recon_plane))
                fig, accept, selected_center_offset = \
                    self._select_center_offset(
                        [recon_edges[i] for i in indices],
                        row, preselected_offsets, center_offset_vo)
                index = preselected_offsets.index(selected_center_offset)
                if index != 1 and prev_index in (None, index) and up:
                    step_size *=2
                else:
                    step_size = int(step_size/2)
                    up = False
                prev_index = index
                # Plot results
                if self._save_figs:
                    fig.savefig(
                        os_path.join(
                            self._output_folder,
                            f'edges_center_{row}_{min(preselected_offsets)}_'\
                                f'{max(preselected_offsets)}.png'))
                plt.close()

        # Select center location
        if self._interactive:
            center_offset = selected_center_offset
        else:
            center_offset = center_offset_vo

        del sinogram_t
        if recon_plane is not None:
            del recon_plane

        return float(center_offset)

    def _reconstruct_one_plane(
            self, tomo_plane_t, center_offset, thetas, eff_pixel_size,
            cross_sectional_dim, plot_sinogram=False, num_core=1,
            gaussian_sigma=None, ring_width=None):
        """Invert the sinogram for a single tomography plane."""
        # Third party modules
        from scipy.ndimage import gaussian_filter
        from skimage.transform import iradon
        from tomopy import misc

        # tomo_plane_t index order: column,theta
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
                f'{-dist_from_edge}]')
            sinogram = tomo_plane_t[
                two_offset+dist_from_edge:-dist_from_edge,:]
        else:
            self._logger.debug(
                f'sinogram range = [{dist_from_edge}, '
                f'{two_offset-dist_from_edge}]')
            sinogram = tomo_plane_t[dist_from_edge:two_offset-dist_from_edge,:]
        if plot_sinogram:
            quick_imshow(
                sinogram.T,
                title=f'Sinogram for a center offset of {center_offset:.2f}',
                name=f'sinogram_center_offset{center_offset:.2f}',
                path=self._output_folder, save_fig=self._save_figs,
                save_only=self._save_only, block=self._block, aspect='auto')

        # Inverting sinogram
        t0 = time()
        recon_sinogram = iradon(sinogram, theta=thetas, circle=True)
        self._logger.info(f'Inverting sinogram took {time()-t0:.2f} seconds')
        del sinogram

        # Performing Gaussian filtering and removing ring artifacts
        if gaussian_sigma is not None and gaussian_sigma:
            recon_sinogram = gaussian_filter(
                recon_sinogram, gaussian_sigma, mode='nearest')
        recon_clean = np.expand_dims(recon_sinogram, axis=0)
        del recon_sinogram
        if ring_width is not None and ring_width:
            recon_clean = misc.corr.remove_ring(
                recon_clean, rwidth=ring_width, ncore=num_core)

        return recon_clean

    def _get_edges_one_plane(self, recon_plane):
        """
        Create an "edges plot" image for a single reconstructed
        tomography data plane.
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
                    f'Invalid weight ({weight}) in _get_edges_one_plane, '
                    'set to a default of 0.1')
                weight = 0.1
        return denoise_tv_chambolle(recon_plane, weight=weight)[0]

    def _select_center_offset(
            self, recon_edges, row, preselected_offsets,
            center_offset_vo=None):
        """Select a center offset value from an "edges plot" image
        for a single reconstructed tomography data plane."""
        # Third party modules
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Button
        from matplotlib.widgets import Button

        def select_offset(offset):
            """Callback function for the "Select offset" input."""
            selected_offset.append(
                ((False,
                  preselected_offsets[preselected_offsets.index(
                      float(radio_btn.value_selected))])))
            plt.close()

        def reject(event):
            """Callback function for the "Reject" button."""
            selected_offset.append((False, preselected_offsets[0]))
            plt.close()

        def accept(event):
            """Callback function for the "Accept" button."""
            if num_plots == 1:
                selected_offset.append((True, preselected_offsets[0]))
            else:
                selected_offset.append(
                    ((False,
                      preselected_offsets[preselected_offsets.index(
                          float(radio_btn.value_selected))])))
            plt.close()

        if not isinstance(recon_edges, (tuple, list)):
            recon_edges = [recon_edges]
        if not isinstance(preselected_offsets, (tuple, list)):
            preselected_offsets = [preselected_offsets]
        assert len(recon_edges) == len(preselected_offsets)

        selected_offset = []

        title_pos = (0.5, 0.95)
        title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        subtitle_pos = (0.5, 0.90)
        subtitle_props = {'fontsize': 'xx-large',
                          'horizontalalignment': 'center',
                          'verticalalignment': 'bottom'}

        num_plots = len(recon_edges)
        if num_plots == 1:
            fig, axs = plt.subplots(figsize=(11, 8.5))
            axs = [axs]
            vmax = np.max(recon_edges[0][:,:])
        else:
            fig, axs = plt.subplots(ncols=num_plots, figsize=(17, 8.5))
            axs = list(axs)
            vmax = np.max(recon_edges[1][:,:])
        for i, (ax, recon_edge, preselected_offset) in enumerate(zip(
                axs, recon_edges, preselected_offsets)):
            ax.imshow(recon_edge, vmin=-vmax, vmax=vmax, cmap='gray')
            if num_plots == 1:
                ax.set_title(
                    f'Reconstruction for row {row}, center offset: ' \
                    f'{preselected_offset:.2f}', fontsize='x-large')
            else:
                ax.set_title(
                    f'Center offset: {preselected_offset}',
                    fontsize='x-large')
            ax.set_xlabel('x', fontsize='x-large')
            if not i:
                ax.set_ylabel('y', fontsize='x-large')

        if len(recon_edges) > 1:
            fig_title = plt.figtext(
                *title_pos,
                f'Reconstructions for row {row} (default center offset: '
                    f'{center_offset_vo})',
                **title_props)
            fig_subtitle = plt.figtext(
                *subtitle_pos,
                'Select the best offset or press "Accept" to accept the '
                f'default value of {preselected_offsets[1]}',
                **subtitle_props)
        else:
            fig_title = plt.figtext(
                *title_pos,
                'Press "Accept" to accept this value or "Reject" to start a '
                    'center calibration search',
                **title_props)

        if not self._interactive:

            selected_offset.append((True, preselected_offsets[0]))

        else:

            fig.subplots_adjust(bottom=0.2)

            if num_plots == 1:

                # Setup "Reject" button
                reject_btn = Button(
                    plt.axes([0.15, 0.05, 0.15, 0.075]), 'Reject')
                reject_cid = reject_btn.on_clicked(reject)


            else:

                # Setup RadioButtons
                select_text = plt.figtext(
                    0.225, 0.175, 'Select offset', fontsize='x-large',
                    horizontalalignment='center', verticalalignment='center')
                radio_btn = RadioButtons(
                    plt.axes([0.175, 0.05, 0.1, 0.1]),
                    labels = preselected_offsets, active=1)
                radio_cid = radio_btn.on_clicked(select_offset)

            # Setup "Accept" button
            accept_btn = Button(
                plt.axes([0.7, 0.05, 0.15, 0.075]), 'Accept')
            accept_cid = accept_btn.on_clicked(accept)

            plt.show()

            # Disconnect all widget callbacks when figure is closed
            # and remove the buttons before returning the figure
            if num_plots == 1:
                reject_btn.disconnect(reject_cid)
                reject_btn.ax.remove()
            else:
                radio_btn.disconnect(radio_cid)
                radio_btn.ax.remove()
            accept_btn.disconnect(accept_cid)
            accept_btn.ax.remove()

        if num_plots == 1:
            fig_title.remove()
        else:
            fig_title.set_in_layout(True)
            fig_subtitle.remove()
            select_text.remove()
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return fig, *selected_offset[0]

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
                    'reconstruct_one_tomo_stack')
            centers = center_offsets
        centers += tomo_stack.shape[2]/2

#        tomo_recon_stack = []
#        eff_pixel_size = 0.05
#        cross_sectional_dim = 20.0 
#        gaussian_sigma = 0.05
#        ring_width = 1
#        for i in range(tomo_stack.shape[0]):
#            sinogram_t = tomo_stack[i,:,:].T
#            recon_plane = self._reconstruct_one_plane(
#                sinogram_t, centers[i], thetas, eff_pixel_size,
#                cross_sectional_dim, False, num_core, gaussian_sigma,
#                ring_width)
#            tomo_recon_stack.append(recon_plane[0,:,:])
#        tomo_recon_stack = np.asarray(tomo_recon_stack)
#        return tomo_recon_stack

        # Remove horizontal stripe
        # RV prep.stripe.remove_stripe_fw seems flawed for hollow brick
        #     accross multiple stacks
        if remove_stripe_sigma is not None and remove_stripe_sigma:
            self._logger.warning('Ignoring remove_stripe_sigma')
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
            'seconds')

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
                'seconds')

        # Remove ring artifacts
        if ring_width is not None and ring_width:
            misc.corr.remove_ring(
                tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_core)

        return tomo_recon_stack

    def _resize_reconstructed_data(
            self, data, x_bounds=None, y_bounds=None, z_bounds=None,
            combine_data=False):
        """Resize the reconstructed tomography data."""
        # Third party modules
        import matplotlib.pyplot as plt

        # Data order: row/-z,y,x or stack,row/-z,y,x
        if isinstance(data, list):
            for i, stack in enumerate(data):
                assert stack.ndim == 3
                if i:
                    assert stack.shape[1:] == data[0].shape[1:]
            num_tomo_stacks = len(data)
            tomo_recon_stacks = data
        else:
            assert data.ndim == 3
            num_tomo_stacks = 1
            tomo_recon_stacks = [data]

        # Selecting x an y bounds (in z-plane)
        if x_bounds is None:
            if not self._interactive:
                self._logger.warning('x_bounds unspecified, reconstruct '
                                     'data for full x-range')
                x_bounds = (0, tomo_recon_stacks[0].shape[2])
        elif not is_int_pair(
                x_bounds, ge=0, le=tomo_recon_stacks[0].shape[2]):
            raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
        if y_bounds is None:
            if not self._interactive:
                self._logger.warning('y_bounds unspecified, reconstruct '
                                     'data for full y-range')
                y_bounds = (0, tomo_recon_stacks[0].shape[1])
        elif not is_int_pair(
                y_bounds, ge=0, le=tomo_recon_stacks[0].shape[1]):
            raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
        if x_bounds is None and y_bounds is None:
            preselected_roi = None
        elif x_bounds is None:
            preselected_roi = (
                0, tomo_recon_stacks[0].shape[2],
                y_bounds[0], y_bounds[1])
        elif y_bounds is None:
            preselected_roi = (
                x_bounds[0], x_bounds[1],
                0, tomo_recon_stacks[0].shape[1])
        else:
            preselected_roi = (
                x_bounds[0], x_bounds[1],
                y_bounds[0], y_bounds[1])
        tomosum = 0
        for i in range(num_tomo_stacks):
            tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=0)
        fig, roi = select_roi_2d(
            tomosum, preselected_roi=preselected_roi,
            title_a='Reconstructed data summed over z',
            row_label='y', column_label='x',
            interactive=self._interactive)
        if roi is None:
            x_bounds = (0, tomo_recon_stacks[0].shape[2])
            y_bounds = (0, tomo_recon_stacks[0].shape[1])
        else:
            x_bounds = (int(roi[0]), int(roi[1]))
            y_bounds = (int(roi[2]), int(roi[3]))
        self._logger.debug(f'x_bounds = {x_bounds}')
        self._logger.debug(f'y_bounds = {y_bounds}')
        # Plot results
        if self._save_figs:
            fig.savefig(
                os_path.join(
                    self._output_folder, 'reconstructed_data_xy_roi.png'))
        plt.close()

        # Selecting z bounds (in xy-plane)
        # (only valid for a single image stack or when combining a stack)
        if num_tomo_stacks == 1 or combine_data:
            if z_bounds is None:
                if not self._interactive:
                    if combine_data:
                        self._logger.warning(
                            'z_bounds unspecified, combine reconstructed data '
                            'for full z-range')
                    else:
                        self._logger.warning(
                            'z_bounds unspecified, reconstruct data for '
                            'full z-range')
                z_bounds = (0, tomo_recon_stacks[0].shape[0])
            elif not is_int_pair(
                    z_bounds, ge=0, le=tomo_recon_stacks[0].shape[0]):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            tomosum = 0
            for i in range(num_tomo_stacks):
                tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=(1,2))
            fig, z_bounds = select_roi_1d(
                tomosum, preselected_roi=z_bounds,
                xlabel='z', ylabel='Reconstructed data summed over x and y',
                interactive=self._interactive)
            self._logger.debug(f'z_bounds = {z_bounds}')
            # Plot results
            if self._save_figs:
                fig.savefig(
                    os_path.join(
                        self._output_folder, 'reconstructed_data_z_roi.png'))
            plt.close()

        return x_bounds, y_bounds, z_bounds


class TomoSimFieldProcessor(Processor):
    """
    A processor to create a simulated tomography data set returning a
    `nexusformat.nexus.NXroot` object containing the simulated
    tomography detector images.
    """

    def process(self, data, **kwargs):
        """
        Process the input configuration and return a
        `nexusformat.nexus.NXroot` object with the simulated
        tomography detector images.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: Simulated tomographic images.
        :rtype: nexusformat.nexus.NXroot
        """
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
        config = self.get_config(data, 'tomo.models.TomoSimConfig')

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
#RV        theta_end = theta_start + 360.
        theta_end = theta_start + 180.
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
        img_row_offset = -0.5 * (detector_size[0]*pixel_size[0]
                               + slit_size * (num_tomo_stack-1))
        img_row_coords = np.flip(img_row_offset
            + pixel_size[0] * (0.5 + np.asarray(range(int(detector_size[0])))))

        # Get the transmitted intensities
        num_theta = len(thetas)
        vertical_shifts = []
        tomo_fields_stack = []
        img_dim = (len(img_row_coords), path_lengths_solid.shape[1])
        intensities_solid = None
        intensities_hollow = None
        for n in range(num_tomo_stack):
            vertical_shifts.append(img_row_offset + n*slit_size
                                   + 0.5*detector_size[0]*pixel_size[0])
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
                    np.where(abs(img_row_coords) <= sample_size[0]/2)[0]
                inner_indices = np.where(
                    abs(img_row_coords) < sample_size[0]/2 - wall_thickness)[0]
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
                img_row_coords += slit_size

        # Add dummy snapshots at each end to mimic FMB/SMB
        if station in ('id1a3', 'id3a'):
            num_dummy_start = 5
            num_dummy_end = 0
            starting_image_index = 345000
        else:
            num_dummy_start = 1
            num_dummy_end = 0
            starting_image_index = 0
        starting_image_offset = num_dummy_start
#        thetas = [theta_start-n*theta_step
#            for n in range(num_dummy_start, 0, -1)] + thetas
#        thetas += [theta_end+n*theta_step
#            for n in range(1, num_dummy_end+1)]
        if num_dummy_start:
            dummy_fields = background_intensity * np.ones(
                (num_dummy_start, *img_dim), dtype=np.int64)
            for n, tomo_field in enumerate(tomo_fields_stack):
                tomo_fields_stack[n] = np.concatenate(
                    (dummy_fields, tomo_field))
        if num_dummy_end:
            dummy_fields = background_intensity * np.ones(
                (num_dummy_end, *img_dim), dtype=np.int64)
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
        nxdetector.row_pixel_size = pixel_size[0]
        nxdetector.column_pixel_size = pixel_size[1]
        nxdetector.row_pixel_size.units = 'mm'
        nxdetector.column_pixel_size.units = 'mm'
        nxdetector.data = tomo_fields_stack
        nxdetector.thetas = thetas
        nxdetector.z_translation = vertical_shifts
        nxdetector.starting_image_index = starting_image_index
        nxdetector.starting_image_offset = starting_image_offset
#        nxdetector.path_lengths_solid = path_lengths_solid
#        nxdetector.path_lengths_hollow = path_lengths_hollow
#        nxdetector.intensities_solid = intensities_solid
#        nxdetector.intensities_hollow = intensities_hollow

        return nxroot

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
    A processor to create the dark field associated with a simulated
    tomography data set created by TomoSimProcessor.
    """

    def process(self, data, num_image=5, **kwargs):
        """
        Process the input configuration and return a
        `nexusformat.nexus.NXroot` object with the simulated
        dark field detector images.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :param num_image: Number of dark field images, defaults to 5.
        :type num_image: int, optional.
        :raises ValueError: Missing or invalid input or configuration
            parameter.
        :return: Simulated dark field images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXroot,
            NXentry,
            NXinstrument,
            NXdetector,
        )

        # Get and validate the TomoSimField configuration object in data
        nxroot = get_nxroot(data, 'tomo.models.TomoSimField', remove=False)
        if nxroot is None:
            raise ValueError('No valid TomoSimField configuration found in '
                             'input data')
        source = nxroot.entry.instrument.source
        detector = nxroot.entry.instrument.detector
        background_intensity = source.background_intensity
        detector_size = detector.data.shape[-2:]

        # Add dummy snapshots at start to mimic SMB
        if source.station in ('id1a3', 'id3a'):
            num_dummy_start = 5
            starting_image_index = 123000
        else:
            num_dummy_start = 1
            starting_image_index = 0
        starting_image_offset = num_dummy_start
        num_image += num_dummy_start

        # Create the dark field
        dark_field = int(background_intensity) * np.ones(
            (num_image, detector_size[0], detector_size[1]), dtype=np.int64)

        # Create Nexus object and write to file
        nxdark = NXroot()
        nxdark.entry = NXentry()
        nxdark.entry.sample = nxroot.entry.sample
        nxinstrument = NXinstrument()
        nxdark.entry.instrument = nxinstrument
        nxinstrument.source = source
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = detector.local_name
        nxdetector.row_pixel_size = detector.row_pixel_size
        nxdetector.column_pixel_size = detector.column_pixel_size
        nxdetector.data = dark_field
        nxdetector.thetas = np.asarray((num_image-num_dummy_start)*[0])
        nxdetector.starting_image_index = starting_image_index
        nxdetector.starting_image_offset = starting_image_offset

        return nxdark


class TomoBrightFieldProcessor(Processor):
    """
    A processor to create the bright field associated with a simulated
    tomography data set created by TomoSimProcessor.
    """

    def process(self, data, num_image=5, **kwargs):
        """
        Process the input configuration and return a
        `nexusformat.nexus.NXroot` object with the simulated
        bright field detector images.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :param num_image: Number of bright field images, defaults to 5.
        :type num_image: int, optional.
        :raises ValueError: Missing or invalid input or configuration
            parameter.
        :return: Simulated bright field images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NeXusError,
            NXroot,
            NXentry,
            NXinstrument,
            NXdetector,
        )

        # Get and validate the TomoSimField configuration object in data
        nxroot = get_nxroot(data, 'tomo.models.TomoSimField', remove=False)
        if nxroot is None:
            raise ValueError('No valid TomoSimField configuration found in '
                             'input data')
        source = nxroot.entry.instrument.source
        detector = nxroot.entry.instrument.detector
        beam_intensity = source.beam_intensity
        background_intensity = source.background_intensity
        detector_size = detector.data.shape[-2:]

        # Add dummy snapshots at start to mimic SMB
        if source.station in ('id1a3', 'id3a'):
            num_dummy_start = 5
            starting_image_index = 234000
        else:
            num_dummy_start = 1
            starting_image_index = 0
        starting_image_offset = num_dummy_start

        # Create the bright field
        bright_field = int(background_intensity+beam_intensity) * np.ones(
            (num_image, detector_size[0], detector_size[1]), dtype=np.int64)
        if num_dummy_start:
            dummy_fields = int(background_intensity) * np.ones(
                (num_dummy_start, detector_size[0], detector_size[1]),
                dtype=np.int64)
            bright_field = np.concatenate((dummy_fields, bright_field))
            num_image += num_dummy_start
        # Add 10% to slit size to make the bright beam slightly taller
        #     than the vertical displacements between stacks
        slit_size = 1.10*source.slit_size
        if slit_size < float(detector.row_pixel_size*detector_size[0]):
            img_row_coords = float(detector.row_pixel_size) \
                * (0.5 + np.asarray(range(int(detector_size[0])))
                   - 0.5*detector_size[0])
            outer_indices = np.where(abs(img_row_coords) > slit_size/2)[0]
            bright_field[:,outer_indices,:] = 0

        # Create Nexus object and write to file
        nxbright = NXroot()
        nxbright.entry = NXentry()
        nxbright.entry.sample = nxroot.entry.sample
        nxinstrument = NXinstrument()
        nxbright.entry.instrument = nxinstrument
        nxinstrument.source = source
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = detector.local_name
        nxdetector.row_pixel_size = detector.row_pixel_size
        nxdetector.column_pixel_size = detector.column_pixel_size
        nxdetector.data = bright_field
        nxdetector.thetas = np.asarray((num_image-num_dummy_start)*[0])
        nxdetector.starting_image_index = starting_image_index
        nxdetector.starting_image_offset = starting_image_offset

        return nxbright


class TomoSpecProcessor(Processor):
    """
    A processor to create a tomography SPEC file associated with a
    simulated tomography data set created by TomoSimProcessor.
    """

    def process(self, data, scan_numbers=[1], **kwargs):
        """
        Process the input configuration and return a list of strings
        representing a plain text SPEC file.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :param scan_numbers: List of SPEC scan numbers,
            defaults to [1].
        :type scan_numbers: list[int]
        :raises ValueError: Invalid input or configuration parameter.
        :return: Simulated SPEC file.
        :rtype: list[str]
        """
        # System modules
        from json import dumps
        from datetime import datetime

        # Third party modules
        from nexusformat.nexus import (
            NeXusError,
            NXcollection,
            NXentry,
            NXroot,
            NXsubentry,
        )

        # Get and validate the TomoSimField, TomoDarkField, or
        #     TomoBrightField configuration object in data
        configs = {}
        nxroot = get_nxroot(data, 'tomo.models.TomoDarkField')
        if nxroot is not None:
            configs['tomo.models.TomoDarkField'] = nxroot
        nxroot = get_nxroot(data, 'tomo.models.TomoBrightField')
        if nxroot is not None:
            configs['tomo.models.TomoBrightField'] = nxroot
        nxroot = get_nxroot(data, 'tomo.models.TomoSimField')
        if nxroot is not None:
            configs['tomo.models.TomoSimField'] = nxroot
        scan_numbers = list(set(scan_numbers))
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

        # Create the output data structure in Nexus format
        nxentry = NXentry()

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
        starting_image_indices = []
        num_scan = 0
        count_time = 1
        for schema, nxroot in configs.items():
            detector = nxroot.entry.instrument.detector
            if 'z_translation' in detector:
                z_translations = list(np.asarray(detector.z_translation))
            else:
                z_translations = [0.]
            thetas = np.asarray(detector.thetas)
            num_theta = thetas.size
            if schema == 'tomo.models.TomoDarkField':
                if station in ('id1a3', 'id3a'):
                    macro = f'slew_ome {thetas[0]} {thetas[-1]} ' \
                        f'{num_theta} {count_time} darkfield'
                    scan_type = 'df1'
                else:
                    macro = f'flyscan {num_theta-1} {count_time}'
                    field_type = 'dark_field'
            elif schema == 'tomo.models.TomoBrightField':
                if station in ('id1a3', 'id3a'):
                    macro = f'slew_ome {thetas[0]} {thetas[-1]} ' \
                        f'{num_theta} {count_time}'
                    scan_type = 'bf1'
                else:
                    macro = f'flyscan {num_theta-1} {count_time}'
                    field_type = 'bright_field'
            elif schema == 'tomo.models.TomoSimField':
                if station in ('id1a3', 'id3a'):
                    macro = f'slew_ome {thetas[0]} {thetas[-1]} ' \
                        f'{num_theta} {count_time}'
                    scan_type = 'ts1'
                else:
                    macro = f'flyscan samphi {thetas[0]} ' \
                        f'{thetas[-1]} {num_theta-1} {count_time}'
                    field_type = 'tomo_field'
            starting_image_index = int(detector.starting_image_index)
            starting_image_offset = int(detector.starting_image_offset)
            for n, z_translation in enumerate(z_translations):
                scan_number = scan_numbers[num_scan]
                spec_file.append(f'#S {scan_number}  {macro}')
                spec_file.append(
                    f'#D {datetime.now().strftime("%a %b %d %I:%M:%S %Y")}')
                if station in ('id1a3', 'id3a'):
                    spec_file.append(f'#P0 0.0 {z_translation}')
                    spec_file.append('#N 1')
                    spec_file.append('#L  ome')
                    if scan_type == 'ts1':
                        image_sets.append(np.asarray(detector.data)[n])
                    else:
                        image_sets.append(np.asarray(detector.data))
                    par_file.append(
                        f'{datetime.now().strftime("%Y%m%d")} '
                        f'{datetime.now().strftime("%H%M%S")} '
                        f'{scan_number} '
#                        '2.0 '
#                        '1.0 '
                        f'{starting_image_index} '
                        f'{starting_image_index+starting_image_offset} '
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
                    spec_file.append('#L theta')
                    spec_file += [str(theta) for theta in thetas]
                    # Add the h5 file to output
                    prefix = str(detector.local_name).upper()
                    field_name = f'{field_type}_{scan_number:03d}'
                    nxentry[field_name] = nxroot.entry
                    nxentry[field_name].attrs['schema'] = 'h5'
                    nxentry[field_name].attrs['filename'] = \
                        f'{sample_type}_{prefix}_{scan_number:03d}.h5'
                starting_image_indices.append(starting_image_index)
                spec_file.append('')
                num_scan += 1

        if station in ('id1a3', 'id3a'):

            spec_filename = 'spec.log'

            # Add the JSON file to output
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
            nxentry.json = NXsubentry()
            nxentry.json.data = dumps(parfile_header)
            nxentry.json.attrs['schema'] = 'json'
            nxentry.json.attrs['filename'] = \
                f'{station}-tomo_sim-{sample_type}.json'
        
            # Add the par file to output
            nxentry.par = NXsubentry()
            nxentry.par.data = par_file
            nxentry.par.attrs['schema'] = 'txt'
            nxentry.par.attrs['filename'] = \
                f'{station}-tomo_sim-{sample_type}.par'

            # Add image files as individual tiffs to output
            for scan_number, image_set, starting_image_index in zip(
                    scan_numbers, image_sets, starting_image_indices):
                nxentry[f'{scan_number}'] = NXsubentry()
                nxsubentry = NXsubentry()
                nxentry[f'{scan_number}']['nf'] = nxsubentry
                for n in range(image_set.shape[0]):
                    nxsubentry[f'tiff_{n}'] = NXsubentry()
                    nxsubentry[f'tiff_{n}'].data = image_set[n]
                    nxsubentry[f'tiff_{n}'].attrs['schema'] = 'tif'
                    nxsubentry[f'tiff_{n}'].attrs['filename'] = \
                        f'nf_{(n+starting_image_index):06d}.tif'
        else:

            spec_filename = sample_type

        # Add spec file to output
        nxentry.spec = NXsubentry()
        nxentry.spec.data = spec_file
        nxentry.spec.attrs['schema'] = 'txt'
        nxentry.spec.attrs['filename'] = spec_filename

        nxroot = NXroot()
        nxroot[sample_type] = nxentry
        nxroot.attrs['default'] = sample_type

        return nxroot


if __name__ == '__main__':
    main()
