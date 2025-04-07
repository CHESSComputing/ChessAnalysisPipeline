#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Rolf Verberg <rolfverberg AT gmail dot com>
Description: Module for Processors used only by tomography experiments
"""

# System modules
from os import path as os_path
from sys import exit as sys_exit
from time import time

# Third party modules
import numpy as np

# Local modules
from CHAP.utils.general import (
    is_num,
    is_num_series,
    is_int_pair,
    input_int,
    input_num,
    input_num_list,
    input_yesno,
    select_image_indices,
    select_roi_1d,
    select_roi_2d,
    quick_imshow,
    nxcopy,
)
from CHAP.processor import Processor

NUM_CORE_TOMOPY_LIMIT = 24


def get_nxroot(data, schema=None, remove=True):
    """Look through `data` for an item whose value for the `'schema'`
    key matches `schema` (if supplied) and whose value for the `'data'`
    key matches a `nexusformat.nexus.NXobject` object and return this
    object.

    :param data: Input list of `PipelineData` objects.
    :type data: list[PipelineData]
    :param schema: Name associated with the
        `nexusformat.nexus.NXobject` object to match in `data`.
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


class TomoMetadataProcessor(Processor):
    """A processor that takes data from the FOXDEN Data Discovery or
    Metadata service and extracts what's available to create
    a `CHAP.common.models.map.MapConfig` object for a tomography
    experiment.
    """
    def process(self, data, config):
        """Process the meta data and return a dictionary with
        extracted data to create a `MapConfig` for the tomography
        experiment.

        :param data: Input data.
        :type data: list[PipelineData]
        :param config: Any additional input data required to create a
            `MapConfig` that is unavailable from the Metadata service.
        :type config: dict
        :return: Metadata from the tomography experiment.
        :rtype: CHAP.common.models.map.MapConfig
        """
        # Local modules
        from CHAP.common.models.map import MapConfig

        try:
            data = self.unwrap_pipelinedata(data)[0]
            if isinstance(data, list) and len(data) != 1:
                raise ValueError(f'Invalid PipelineData input data ({data})')
            data = data[0]
            if not isinstance(data, dict):
                raise ValueError(f'Invalid PipelineData input data ({data})')
        except Exception as exc:
            raise

        # Extracted any available MapConfig info
        map_config = {}
        map_config['title'] = data.get('sample_name')
        station = data.get('beamline')[0]
        if station == '3A':
            station = 'id3a'
        else:
            raise ValueError(f'Invalid beamline parameter ({beamline})')
        map_config['station'] = station
        experiment_type = data.get('technique')
        assert 'tomography' in experiment_type
        map_config['experiment_type'] = 'TOMO'
        map_config['sample'] = {'name': map_config['title'],
                                'description': data.get('description')}
        if station == 'id3a':
            scan_numbers = config['scan_numbers']
            if isinstance(scan_numbers, list):
                if isinstance(scan_numbers[0], list):
                    scan_numbers = scan_numbers[0]
            map_config['spec_scans'] = [{
                'spec_file': os_path.join(
                    data.get('data_location_raw'), 'spec.log'),
                'scan_numbers': scan_numbers}]
        map_config['independent_dimensions'] = config['independent_dimensions']

        # Validate the MapConfig info
        MapConfig(**map_config)

        return map_config


class TomoCHESSMapConverter(Processor):
    """A processor to convert a CHESS style tomography map with dark
    and bright field configurations to an NeXus style input format.
    """
    def process(self, data):
        """Process the input map and configuration and return a
        `nexusformat.nexus.NXroot` object based on the
        `nexusformat.nexus.NXtomo` style format.

        :param data: Input map and configuration for tomographic image
            reduction/reconstruction.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: NeXus style tomography input configuration.
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
            NXlink,
            NXroot,
            NXsample,
            NXsource,
        )

        # Local modules
        from CHAP.common.models.map import MapConfig
        from CHAP.utils.general import index_nearest

        # Load and validate the tomography fields
        tomofields = get_nxroot(data, 'tomofields')
        if isinstance(tomofields, NXroot):
            tomofields = tomofields[tomofields.default]
        if not isinstance(tomofields, NXentry):
            raise ValueError(f'Invalid parameter tomofields {tomofields})')
        detector_prefix = str(tomofields.detector_ids)
        tomo_stacks = tomofields.data[detector_prefix].nxdata
        tomo_stack_shape = tomo_stacks.shape
        assert len(tomo_stack_shape) == 3

        # Validate map
        map_config = MapConfig(**loads(str(tomofields.map_config)))
        assert len(map_config.spec_scans) == 1
        spec_scan = map_config.spec_scans[0]
        scan_numbers = spec_scan.scan_numbers

        # Load and validate dark field
        darkfield = get_nxroot(data, 'darkfield')
        if darkfield is None:
            for scan_number in range(min(scan_numbers), 0, -1):
                scanparser = spec_scan.get_scanparser(scan_number)
                scan_type = scanparser.get_scan_type()
                if scan_type == 'df1':
                    darkfield = scanparser
                    break
            else:
                self.logger.warning(f'Unable to load dark field')
        else:
            if isinstance(darkfield, NXroot):
                darkfield = darkfield[darkfield.default]
            if not isinstance(darkfield, NXentry):
                raise ValueError(f'Invalid parameter darkfield ({darkfield})')

        # Load and validate bright field
        brightfield = get_nxroot(data, 'brightfield')
        if brightfield is None:
            for scan_number in range(min(scan_numbers), 0, -1):
                scanparser = spec_scan.get_scanparser(scan_number)
                scan_type = scanparser.get_scan_type()
                if scan_type == 'bf1':
                    brightfield = scanparser
                    break
            else:
                raise ValueError(f'Unable to load bright field')
        else:
            if isinstance(brightfield, NXroot):
                brightfield = brightfield[brightfield.default]
            if not isinstance(brightfield, NXentry):
                raise ValueError(
                    f'Invalid parameter brightfield ({brightfield})')

        # Load and validate detector config if supplied
        try:
            detector_config = self.get_config(
                data=data, schema='tomo.models.Detector')
        except:
            detector_config = None

        # Construct NXroot
        nxroot = NXroot()

        # Check available independent dimensions
        if 'axes' in tomofields.data.attrs:
            independent_dimensions = tomofields.data.attrs['axes']
        else:
            independent_dimensions = tomofields.data.attrs['unstructured_axes']
        if isinstance(independent_dimensions, str):
            independent_dimensions = [independent_dimensions]
        matched_dimensions = deepcopy(independent_dimensions)
        if 'rotation_angles' not in independent_dimensions:
            raise ValueError('Data for rotation angles is unavailable '
                             '(available independent dimensions: '
                             f'{independent_dimensions})')
        rotation_angle_data_type = \
            tomofields.data.rotation_angles.attrs['data_type']
        if rotation_angle_data_type != 'scan_column':
            raise ValueError('Invalid data type for rotation angles '
                             f'({rotation_angle_data_type})')
        matched_dimensions.pop(matched_dimensions.index('rotation_angles'))
        if 'x_translation' in independent_dimensions:
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
        nxentry = NXentry(name=map_config.title)
        nxroot[nxentry.nxname] = nxentry
        nxentry.set_default()

        # Add configuration fields
        nxentry.definition = 'NXtomo'
        nxentry.map_config = map_config.model_dump_json()

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
        nxdetector = NXdetector()
        nxinstrument.detector = nxdetector
        nxdetector.local_name = detector_prefix
        if detector_config is None:
            detector_attrs = tomofields.data[detector_prefix].attrs
        else:
            detector_attrs = {
                'pixel_size': detector_config.pixel_size,
                'lens_magnification': detector_config.lens_magnification}
        pixel_size = detector_attrs['pixel_size']
        if isinstance(pixel_size, (int, float)):
            pixel_size = [pixel_size]
        if len(pixel_size) == 1:
            nxdetector.row_pixel_size = \
                pixel_size[0]/detector_attrs['lens_magnification']
            nxdetector.column_pixel_size = \
                pixel_size[0]/detector_attrs['lens_magnification']
        else:
            nxdetector.row_pixel_size = \
                pixel_size[0]/detector_attrs['lens_magnification']
            nxdetector.column_pixel_size = \
                pixel_size[1]/detector_attrs['lens_magnification']
        nxdetector.row_pixel_size.units = 'mm'
        nxdetector.column_pixel_size.units = 'mm'
        nxdetector.rows = tomo_stack_shape[1]
        nxdetector.columns = tomo_stack_shape[2]
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
        if isinstance(darkfield, NXentry):
            nxentry.dark_field_config = darkfield.config
            for scan in darkfield.spec_scans.values():
                for nxcollection in scan.values():
                    data_shape = nxcollection.data[detector_prefix].shape
                    assert len(data_shape) == 3
                    assert data_shape[1] == nxdetector.rows
                    assert data_shape[2] == nxdetector.columns
                    num_image = data_shape[0]
                    image_keys += num_image*[2]
                    sequence_numbers += list(range(num_image))
                    image_stacks.append(
                        nxcollection.data[detector_prefix].nxdata)
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
        elif darkfield is not None:
            data = darkfield.get_detector_data(detector_prefix)
            data_shape = data.shape
            assert len(data_shape) == 3
            assert data_shape[1] == nxdetector.rows
            assert data_shape[2] == nxdetector.columns
            num_image = data_shape[0]
            image_keys += num_image*[2]
            sequence_numbers += list(range(num_image))
            image_stacks.append(data)
            rotation_angles += num_image*[0.0]
            if (x_translation_data_type == 'spec_motor' or
                    z_translation_data_type == 'spec_motor'):
                spec_motors = darkfield.spec_positioner_values
#                    {k:float(v)
#                    for k, v in darkfield.spec_positioner_values.items()}
            if (x_translation_data_type == 'smb_par' or
                    z_translation_data_type == 'smb_par'):
                smb_pars = scanparser.pars
#                    {k:v for k,v in scanparser.pars.items()}
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
        if isinstance(brightfield, NXentry):
            nxentry.bright_field_config = brightfield.config
            for scan in brightfield.spec_scans.values():
                for nxcollection in scan.values():
                    data_shape = nxcollection.data[detector_prefix].shape
                    assert len(data_shape) == 3
                    assert data_shape[1] == nxdetector.rows
                    assert data_shape[2] == nxdetector.columns
                    num_image = data_shape[0]
                    image_keys += num_image*[1]
                    sequence_numbers += list(range(num_image))
                    image_stacks.append(
                        nxcollection.data[detector_prefix].nxdata)
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
        else:
            data = brightfield.get_detector_data(detector_prefix)
            data_shape = data.shape
            assert len(data_shape) == 3
            assert data_shape[1] == nxdetector.rows
            assert data_shape[2] == nxdetector.columns
            num_image = data_shape[0]
            image_keys += num_image*[1]
            sequence_numbers += list(range(num_image))
            image_stacks.append(data)
            rotation_angles += num_image*[0.0]
            if (x_translation_data_type == 'spec_motor' or
                    z_translation_data_type == 'spec_motor'):
                spec_motors = brightfield.spec_positioner_values
#                    {k:float(v)
#                    for k, v in brightfield.spec_positioner_values.items()}
            if (x_translation_data_type == 'smb_par' or
                    z_translation_data_type == 'smb_par'):
                smb_pars = scanparser.pars
#                    {k:v for k,v in scanparser.pars.items()}
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

        # Collect tomography fields data
        num_tomo_stack = len(scan_numbers)
        assert not tomo_stack_shape[0] % num_tomo_stack
        # Restrict to 180 degrees set of data for now to match old code
        thetas_stacks = tomofields.data.rotation_angles.nxdata
        num_theta = tomo_stack_shape[0] // num_tomo_stack
        assert num_theta > 2
        thetas = thetas_stacks[0:num_theta]
        delta_theta = thetas[1] - thetas[0]
        if thetas[num_theta-1] - thetas[0] > 180 - delta_theta:
            image_end = index_nearest(thetas, thetas[0] + 180)
        else:
            image_end = thetas.size
        thetas = thetas[:image_end]
        num_image = thetas.size
        n_start = 0
        image_keys += num_tomo_stack * num_image * [0]
        sequence_numbers += num_tomo_stack * list(range(num_image))
        if x_translation_data_type is None:
            x_translations += num_tomo_stack * num_image * [0.0]
        if z_translation_data_type is None:
            z_translations += num_tomo_stack * num_image * [0.0]
        for _ in range(num_tomo_stack):
            image_stacks.append(tomo_stacks[n_start:n_start+num_image])
            if not np.array_equal(
                    thetas, thetas_stacks[n_start:n_start+num_image]):
                raise RuntimeError(
                    'Inconsistent thetas among tomography image stacks')
            rotation_angles += list(thetas)
            if x_translation_data_type is not None:
                x_translations += list(
                    tomofields.data.x_translation[n_start:n_start+num_image])
            if z_translation_data_type is not None:
                z_translations += list(
                    tomofields.data.z_translation[n_start:n_start+num_image])
            n_start += num_theta

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
        nxentry.data = NXdata(NXlink(nxentry.instrument.detector.data))
        nxentry.data.makelink(nxentry.instrument.detector.image_key)
        nxentry.data.makelink(nxentry.sample.rotation_angle)
        nxentry.data.makelink(nxentry.sample.x_translation)
        nxentry.data.makelink(nxentry.sample.z_translation)
        nxentry.data.set_default()

        return nxroot


class TomoDataProcessor(Processor):
    """A processor to reconstruct a set of tomographic images returning
    either a dictionary or a `nexusformat.nexus.NXroot` object
    containing the (meta) data after processing each individual step.
    """
    def process(
            self, data, outputdir='.', interactive=False, reduce_data=False,
            find_center=False, calibrate_center=False, reconstruct_data=False,
            combine_data=False, save_figs='no'):
        """Process the input map or configuration with the step
        specific instructions and return either a dictionary or a
        `nexusformat.nexus.NXroot` object with the processed result.

        :param data: Input configuration and specific step instructions
            for tomographic image reduction.
        :type data: list[PipelineData]
        :param outputdir: Output folder name, defaults to `'.'`.
        :type outputdir: str, optional
        :param interactive: Allows for user interactions,
            defaults to `False`.
        :type interactive: bool, optional
        :param reduce_data: Generate reduced tomography images,
            defaults to `False`.
        :type reduce_data: bool, optional
        :param find_center: Generate calibrated center axis info,
            defaults to `False`.
        :type find_center: bool, optional
        :param calibrate_center: Calibrate the rotation axis,
            defaults to `False`.
        :type calibrate_center: bool, optional
        :param reconstruct_data: Reconstruct the tomography data,
            defaults to `False`.
        :type reconstruct_data: bool, optional
        :param combine_data: Combine the reconstructed tomography
            stacks, defaults to `False`.
        :type combine_data: bool, optional
        :param save_figs: Safe figures to file ('yes' or 'only') and/or
            display figures ('yes' or 'no'), defaults to `'no'`.
        :type save_figs: Literal['yes', 'no', 'only'], optional
        :raises ValueError: Invalid input or configuration parameter.
        :raises RuntimeError: Missing map configuration to generate
            reduced tomography images.
        :return: Processed (meta)data of the last step.
        :rtype: Union[dict, nexusformat.nexus.NXroot]
        """
        # Third party modules
        from nexusformat.nexus import nxsetconfig

        # Local modules
        from CHAP.tomo.models import (
            TomoFindCenterConfig,
            TomoReconstructConfig,
            TomoCombineConfig,
        )

        if not isinstance(reduce_data, bool):
            raise ValueError(f'Invalid parameter reduce_data ({reduce_data})')
        if not isinstance(find_center, bool):
            raise ValueError(f'Invalid parameter find_center ({find_center})')
        if not isinstance(calibrate_center, bool):
            raise ValueError(
                f'Invalid parameter calibrate_center ({calibrate_center})')
        if not isinstance(reconstruct_data, bool):
            raise ValueError(
                f'Invalid parameter reconstruct_data ({reconstruct_data})')
        if not isinstance(combine_data, bool):
            raise ValueError(
                f'Invalid parameter combine_data ({combine_data})')

        try:
            reduce_data_config = self.get_config(
                data=data, schema='tomo.models.TomoReduceConfig')
        except ValueError:
            reduce_data_config = None
        try:
            find_center_config = self.get_config(
                data=data, schema='tomo.models.TomoFindCenterConfig')
        except ValueError:
            find_center_config = None
        try:
            reconstruct_data_config = self.get_config(
                data=data, schema='tomo.models.TomoReconstructConfig')
        except ValueError:
            reconstruct_data_config = None
        try:
            combine_data_config = self.get_config(
                data=data, schema='tomo.models.TomoCombineConfig')
        except ValueError:
            combine_data_config = None
        nxroot = get_nxroot(data)

        tomo = Tomo(
            logger=self.logger, interactive=interactive,
            outputdir=outputdir, save_figs=save_figs)

        nxsetconfig(memory=100000)

        # Calibrate the rotation axis
        if calibrate_center:
            if (reduce_data or find_center
                    or reconstruct_data or reconstruct_data_config is not None
                    or combine_data or combine_data_config is not None):
                self.logger.warning('Ignoring any step specific instructions '
                                    'during center calibration')
            if nxroot is None:
                raise RuntimeError('Map info required to calibrate the '
                                   'rotation axis')
            if find_center_config is None:
                find_center_config = TomoFindCenterConfig()
                calibrate_center_rows = True
            else:
                calibrate_center_rows = find_center_config.center_rows
                if calibrate_center_rows is None:
                    calibrate_center_rows = True
            nxroot, calibrate_center_rows = tomo.reduce_data(
                nxroot, reduce_data_config, calibrate_center_rows)
            return tomo.find_centers(
                nxroot, find_center_config, calibrate_center_rows)

        # Reduce tomography images
        if reduce_data or reduce_data_config is not None:
            if nxroot is None:
                raise RuntimeError('Map info required to reduce the '
                                   'tomography images')
            nxroot, _ = tomo.reduce_data(nxroot, reduce_data_config)

        # Find calibrated center axis info for the tomography stacks
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
        # tomo.reconstruct_data?
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


class SetNumexprThreads:
    """Class that sets and keeps track of the number of processors used
    by the code in general and by the num_expr package specifically.
    """
    def __init__(self, num_core):
        """Initialize SetNumexprThreads.

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
            self, logger=None, outputdir='.', interactive=False, num_core=-1,
            save_figs='no'):
        """Initialize Tomo.

        :param interactive: Allows for user interactions,
            defaults to `False`.
        :type interactive: bool, optional
        :param num_core: Number of processors.
        :type num_core: int
        :param outputdir: Output folder name, defaults to `'.'`.
        :type outputdir: str, optional
        :param save_figs: Safe figures to file ('yes' or 'only') and/or
            display figures ('yes' or 'no'), defaults to `'no'`.
        :type save_figs: Literal['yes', 'no', 'only'], optional
        :raises ValueError: Invalid input parameter.
        """
        # System modules
        from multiprocessing import cpu_count

        self.__name__ = self.__class__.__name__
        if logger is None:
            # System modules
            from logging import getLogger

            self._logger = getLogger(self.__name__)
            self._logger.propagate = False
        else:
            self._logger = logger

        if not isinstance(interactive, bool):
            raise ValueError(f'Invalid parameter interactive ({interactive})')
        self._outputdir = outputdir
        self._interactive = interactive
        self._num_core = num_core
        self._test_config = {}
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
        # Tompy py uses numexpr with NUMEXPR_MAX_THREADS = 64
        if self._num_core > 64:
            self._logger.warning(
                f'num_core = {self._num_core} is larger than the number '
                f'of processors suitable to Tomopy and reduced to 64')
            self._num_core = 64

    def reduce_data(
            self, nxroot, tool_config=None, calibrate_center_rows=False):
        """Reduced the tomography images.

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

        # Validate input parameter
        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.default]
        else:
            raise ValueError(
                f'Invalid parameter nxroot {type(nxroot)}:\n{nxroot}')
        if tool_config is None:
            delta_theta = None
            img_row_bounds = None
        else:
            delta_theta = tool_config.delta_theta
            img_row_bounds = tuple(tool_config.img_row_bounds)
            if img_row_bounds is not None:
                if (nxentry.instrument.source.attrs['station']
                        in ('id1a3', 'id3a')):
                    self._logger.warning('Ignoring parameter img_row_bounds '
                                        'for id1a3 and id3a')
                    img_row_bounds = None
                elif calibrate_center_rows:
                    self._logger.warning('Ignoring parameter img_row_bounds '
                                        'during rotation axis calibration')
                    img_row_bounds = None
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is None or 'data' not in nxentry.instrument.detector:
            raise ValueError(f'Unable to find image_key or data in '
                             'instrument.detector '
                             f'({nxentry.instrument.detector.tree})')

        # Create an NXprocess to store data reduction (meta)data
        reduced_data = NXprocess()

        # Generate dark field
        reduced_data = self._gen_dark(nxentry, reduced_data, image_key)

        # Generate bright field
        reduced_data = self._gen_bright(nxentry, reduced_data, image_key)

        # Get rotation angles for image stacks (in degrees)
        thetas = self._gen_thetas(nxentry, image_key)

        # Get the image stack mask to remove bad images from stack
        image_mask = None
        drop_fraction = 0 # fraction of images dropped as a percentage
        if drop_fraction:
            if delta_theta is not None:
                delta_theta = None
                self._logger.warning(
                    'Ignoring delta_theta when an image mask is used')
            np.random.seed(0)
            image_mask = np.where(np.random.rand(
                len(thetas)) < drop_fraction/100, 0, 1).astype(bool)

        # Set zoom and/or rotation angle interval to reduce memory
        # requirement
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

        # Set vertical detector bounds for image stack or rotation
        # axis calibration rows
        img_row_bounds = self._set_detector_bounds(
            nxentry, reduced_data, image_key, thetas[0],
            img_row_bounds, calibrate_center_rows)
        self._logger.debug(f'img_row_bounds = {img_row_bounds}')
        if calibrate_center_rows:
            calibrate_center_rows = tuple(sorted(img_row_bounds))
            img_row_bounds = None
        if img_row_bounds is None:
            tbf_shape = reduced_data.data.bright_field.shape
            img_row_bounds = (0, tbf_shape[0])
        reduced_data.img_row_bounds = img_row_bounds
        reduced_data.img_row_bounds.units = 'pixels'
        reduced_data.img_row_bounds.attrs['long_name'] = \
            'image row boundaries in detector frame of reference'

        # Store rotation angles for image stacks
        self._logger.debug(f'thetas = {thetas}')
        reduced_data.rotation_angle = thetas
        reduced_data.rotation_angle.units = 'degrees'

        # Generate reduced tomography fields
        reduced_data = self._gen_tomo(
            nxentry, reduced_data, image_key, calibrate_center_rows)

        # Create a copy of the input NeXus object and remove raw and
        # any existing reduced data
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

        # Add the reduced data NXprocess
        nxentry = nxroot[nxroot.default]
        nxentry.reduced_data = reduced_data

        if 'data' not in nxentry:
            nxentry.data = NXdata()
            nxentry.data.set_default()
        nxentry.data.makelink(
            nxentry.reduced_data.data.tomo_fields, name='reduced_data')
        nxentry.data.makelink(nxentry.reduced_data.rotation_angle)
        nxentry.data.attrs['signal'] = 'reduced_data'

        return nxroot, calibrate_center_rows

    def find_centers(self, nxroot, tool_config, calibrate_center_rows=False):
        """Find the calibrated center axis info

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
        from nexusformat.nexus import NXroot

        self._logger.info('Find the calibrated center axis info')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.default]
        else:
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')

        # Check if reduced data is available
        if 'reduced_data' not in nxentry:
            raise ValueError(f'Unable to find valid reduced data in {nxentry}.')

        # Select the image stack to find the calibrated center axis
        # reduced data axes order: stack,theta,row,column
        # Note: NeXus can't follow a link if the data it points to is
        # too big get the data from the actual place, not from
        # nxentry.data
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        self._logger.debug(f'num_tomo_stacks = {num_tomo_stacks}')
        if num_tomo_stacks == 1:
            center_stack_index = 0
        else:
            center_stack_index = tool_config.center_stack_index
            if calibrate_center_rows:
                center_stack_index = num_tomo_stacks//2
            elif self._interactive:
                if center_stack_index is None:
                    center_stack_index = input_int(
                        '\nEnter tomography stack index to calibrate the '
                        'center axis', ge=0, lt=num_tomo_stacks,
                        default=num_tomo_stacks//2)
            else:
                if center_stack_index is None:
                    center_stack_index = num_tomo_stacks//2
                    self._logger.warning(
                        'center_stack_index unspecified, use stack '
                        f'{center_stack_index} to find center axis info')

        # Get thetas (in degrees)
        thetas = nxentry.reduced_data.rotation_angle.nxdata

        # Select center rows
        if calibrate_center_rows:
            center_rows = calibrate_center_rows
            offset_center_rows = (0, 1)
        else:
            # Third party modules
            import matplotlib.pyplot as plt

            # Get full bright field
            tbf = nxentry.reduced_data.data.bright_field.nxdata
            tbf_shape = tbf.shape

            # Get image bounds
            img_row_bounds = nxentry.reduced_data.get(
                'img_row_bounds', (0, tbf_shape[0]))
            img_row_bounds = (int(img_row_bounds[0]), int(img_row_bounds[1]))
            img_column_bounds = nxentry.reduced_data.get(
                'img_column_bounds', (0, tbf_shape[1]))
            img_column_bounds = (
                int(img_column_bounds[0]), int(img_column_bounds[1]))

            center_rows = tool_config.center_rows
            if center_rows is None:
                if num_tomo_stacks == 1:
                    # Add a small margin to avoid edge effects
                    offset = min(
                        5, int(0.1*(img_row_bounds[1] - img_row_bounds[0])))
                    center_rows = (
                        img_row_bounds[0]+offset, img_row_bounds[1]-1-offset)
                else:
                    if not self._interactive:
                        self._logger.warning('center_rows unspecified, find '
                                             'centers at reduced data bounds')
                    center_rows = (img_row_bounds[0], img_row_bounds[1]-1)
            fig, center_rows = select_image_indices(
                nxentry.reduced_data.data.tomo_fields[
                    center_stack_index,0,:,:],
                0,
                b=tbf[img_row_bounds[0]:img_row_bounds[1],
                      img_column_bounds[0]:img_column_bounds[1]],
                preselected_indices=center_rows,
                axis_index_offset=img_row_bounds[0],
                title='Select two detector image row indices to find center '
                    f'axis (in range [{img_row_bounds[0]}, '
                    f'{img_row_bounds[1]-1}])',
                title_a=r'Tomography image at $\theta$ = '
                        f'{round(thetas[0], 2)+0}',
                title_b='Bright field', interactive=self._interactive)
            if center_rows[1] == img_row_bounds[1]:
                center_rows = (center_rows[0], center_rows[1]-1)
            offset_center_rows = (
                center_rows[0] - img_row_bounds[0],
                center_rows[1] - img_row_bounds[0])
            # Plot results
            if self._save_figs:
                fig.savefig(
                    os_path.join(self._outputdir, 'center_finding_rows.png'))
            plt.close()

        # Find the center offsets at each of the center rows
        prev_center_offset = None
        center_offsets = []
        for row, offset_row in zip(center_rows, offset_center_rows):
            t0 = time()
            center_offsets.append(
                self._find_center_one_plane(
                    nxentry.reduced_data.data.tomo_fields, center_stack_index,
                    row, offset_row, np.radians(thetas),
                    num_core=self._num_core,
                    center_offset_min=tool_config.center_offset_min,
                    center_offset_max=tool_config.center_offset_max,
                    center_search_range=tool_config.center_search_range,
                    gaussian_sigma=tool_config.gaussian_sigma,
                    ring_width=tool_config.ring_width,
                    prev_center_offset=prev_center_offset))
            self._logger.info(
                f'Finding center row {row} took {time()-t0:.2f} seconds')
            self._logger.debug(f'center_row = {row:.2f}')
            self._logger.debug(f'center_offset = {center_offsets[-1]:.2f}')
            prev_center_offset = center_offsets[-1]

        center_config = {
            'center_rows': list(center_rows),
            'center_offsets': center_offsets,
        }
        if num_tomo_stacks > 1:
            center_config['center_stack_index'] = center_stack_index
        if tool_config.center_offset_min is not None:
            center_config['center_offset_min'] = tool_config.center_offset_min
        if tool_config.center_offset_max is not None:
            center_config['center_offset_max'] = tool_config.center_offset_max
        if tool_config.gaussian_sigma is not None:
            center_config['gaussian_sigma'] = tool_config.gaussian_sigma
        if tool_config.ring_width is not None:
            center_config['ring_width'] = tool_config.ring_width

        return center_config

    def reconstruct_data(self, nxroot, center_info, tool_config):
        """Reconstruct the tomography data.

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
            NXdata,
            NXfield,
            NXprocess,
            NXroot,
        )

        self._logger.info('Reconstruct the tomography data')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.default]
        else:
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')
        if not isinstance(center_info, dict):
            raise ValueError(f'Invalid parameter center_info ({center_info})')

        # Check if reduced data is available
        if 'reduced_data' not in nxentry:
            raise ValueError(f'Unable to find valid reduced data in {nxentry}.')

        # Create an NXprocess to store image reconstruction (meta)data
        nxprocess = NXprocess()

        # Get calibrated center axis rows and centers
        center_rows = center_info.get('center_rows')
        center_offsets = center_info.get('center_offsets')
        if center_rows is None or center_offsets is None:
            raise KeyError(
                'Unable to find valid calibrated center axis info in '
                f'{center_info}.')
        center_slope = (center_offsets[1]-center_offsets[0]) \
            / (center_rows[1]-center_rows[0])

        # Get thetas (in degrees)
        thetas = nxentry.reduced_data.rotation_angle.nxdata

        # Reconstruct tomography data
        # - reduced data axes order: stack,theta,row,column
        # - reconstructed data axes order: row/-z,y,x
        # Note: NeXus can't follow a link if the data it points to is
        # too big get the data from the actual place, not from
        # nxentry.data
        if 'zoom_perc' in nxentry.reduced_data:
            res_title = f'{nxentry.reduced_data.attrs["zoom_perc"]}p'
        else:
            res_title = 'fullres'
        tomo_stacks = nxentry.reduced_data.data.tomo_fields
        num_tomo_stacks = tomo_stacks.shape[0]
        tomo_recon_stacks = []
        img_row_bounds = tuple(nxentry.reduced_data.get(
            'img_row_bounds', (0, tomo_stacks.shape[2])))
        center_rows -= img_row_bounds[0]
        for i in range(num_tomo_stacks):
            # Convert reduced data stack from theta,row,column to
            # row,theta,column
            tomo_stack = np.swapaxes(tomo_stacks[i,:,:,:], 0, 1)
            assert len(thetas) == tomo_stack.shape[1]
            assert 0 <= center_rows[0] < center_rows[1] < tomo_stack.shape[0]
            center_offsets = [
                center_offsets[0]-center_rows[0]*center_slope,
                center_offsets[1] + center_slope * (
                    tomo_stack.shape[0]-1-center_rows[1]),
            ]
            t0 = time()
            tomo_recon_stack = self._reconstruct_one_tomo_stack(
                tomo_stack, np.radians(thetas), center_offsets=center_offsets,
                num_core=self._num_core, algorithm='gridrec',
                secondary_iters=tool_config.secondary_iters,
                gaussian_sigma=tool_config.gaussian_sigma,
                remove_stripe_sigma=tool_config.remove_stripe_sigma,
                ring_width=tool_config.ring_width)
            self._logger.info(
                f'Reconstruction of stack {i} took {time()-t0:.2f} seconds')

            # Combine stacks
            tomo_recon_stacks.append(tomo_recon_stack)

        # Resize the reconstructed tomography data
        # - reconstructed axis data order in each stack: row/-z,y,x
        tomo_recon_shape = tomo_recon_stacks[0].shape
        x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
            tomo_recon_stacks, x_bounds=tool_config.x_bounds,
            y_bounds=tool_config.y_bounds, z_bounds=tool_config.z_bounds)
        if x_bounds is None:
            x_range = (0, tomo_recon_shape[2])
            x_slice = x_range[1]//2
        else:
            x_range = (min(x_bounds), max(x_bounds))
            x_slice = (x_bounds[0]+x_bounds[1])//2
        if y_bounds is None:
            y_range = (0, tomo_recon_shape[1])
            y_slice = y_range[1]//2
        else:
            y_range = (min(y_bounds), max(y_bounds))
            y_slice = (y_bounds[0]+y_bounds[1])//2
        if z_bounds is None:
            z_range = (0, tomo_recon_shape[0])
            z_slice = z_range[1]//2
        else:
            z_range = (min(z_bounds), max(z_bounds))
            z_slice = (z_bounds[0]+z_bounds[1])//2
        z_dim_org = tomo_recon_shape[0]
        for i, stack in enumerate(tomo_recon_stacks):
            tomo_recon_stacks[i] = stack[
                z_range[0]:z_range[1],y_range[0]:y_range[1],
                x_range[0]:x_range[1]]
        tomo_recon_stacks = np.asarray(tomo_recon_stacks)

        detector = nxentry.instrument.detector
        row_pixel_size = float(detector.row_pixel_size)
        column_pixel_size = float(detector.column_pixel_size)
        if num_tomo_stacks == 1:
            # Convert the reconstructed tomography data from internal
            # coordinate frame row/-z,y,x with the origin on the
            # near-left-top corner to an z,y,x coordinate frame with
            # the origin on the par file x,z values, halfway in the
            # y-dimension.
            # Here x is to the right, y along the beam direction and
            # z upwards in the lab frame of reference
            tomo_recon_stack = np.flip(tomo_recon_stacks[0], 0)
            z_range = (z_dim_org-z_range[1], z_dim_org-z_range[0])

            # Get coordinate axes
            x = column_pixel_size * (
                np.linspace(
                    x_range[0], x_range[1], x_range[1]-x_range[0], False)
                - 0.5*detector.columns + 0.5)
            x = np.asarray(x + nxentry.reduced_data.x_translation[0])
            y = np.asarray(
                    column_pixel_size * (
                    np.linspace(
                        y_range[0], y_range[1], y_range[1]-y_range[0], False)
                    - 0.5*detector.columns + 0.5))
            z = row_pixel_size*(
                np.linspace(
                    z_range[0], z_range[1], z_range[1]-z_range[0], False)
                + detector.rows
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
                    origin='lower', extent=extent, path=self._outputdir,
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
                    origin='lower', extent=extent, path=self._outputdir,
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
                    origin='lower', extent=extent, path=self._outputdir,
                    save_fig=True, save_only=True)
        else:
            # Plot a few reconstructed image slices
            if self._save_figs:
                for i in range(tomo_recon_stacks.shape[0]):
                    basetitle = f'recon stack {i}'
                    title = f'{basetitle} {res_title} xslice{x_slice}'
                    quick_imshow(
                        tomo_recon_stacks[i,:,:,x_slice-x_range[0]],
                        title=title, path=self._outputdir, save_fig=True,
                        save_only=True)
                    title = f'{basetitle} {res_title} yslice{y_slice}'
                    quick_imshow(
                        tomo_recon_stacks[i,:,y_slice-y_range[0],:],
                        title=title, path=self._outputdir, save_fig=True,
                        save_only=True)
                    title = f'{basetitle} {res_title} zslice{z_slice}'
                    quick_imshow(
                        tomo_recon_stacks[i,z_slice-z_range[0],:,:],
                        title=title, path=self._outputdir, save_fig=True,
                        save_only=True)

        # Add image reconstruction to reconstructed data NXprocess
        # reconstructed axis data order:
        # - for one stack: z,y,x
        # - for multiple stacks: row/-z,y,x
        for k, v in center_info.items():
            nxprocess[k] = v
            if k in ('center_rows', 'center_offsets'):
                nxprocess[k].units = 'pixels'
            if k == 'center_rows':
                nxprocess[k].attrs['long_name'] = \
                    'center row indices in detector frame of reference'
        if x_bounds is not None:
            nxprocess.x_bounds = x_bounds
            nxprocess.x_bounds.units = 'pixels'
            nxprocess.x_bounds.attrs['long_name'] = \
                'x range indices in reduced data frame of reference'
        if y_bounds is not None:
            nxprocess.y_bounds = y_bounds
            nxprocess.y_bounds.units = 'pixels'
            nxprocess.y_bounds.attrs['long_name'] = \
                'y range indices in reduced data frame of reference'
        if z_bounds is not None:
            nxprocess.z_bounds = z_bounds
            nxprocess.z_bounds.units = 'pixels'
            nxprocess.z_bounds.attrs['long_name'] = \
                'z range indices in reduced data frame of reference'
        if num_tomo_stacks == 1:
            nxprocess.data = NXdata(
                NXfield(tomo_recon_stack, 'reconstructed_data'),
                (NXfield(
                     z, 'z', attrs={'units': detector.row_pixel_size.units}),
                 NXfield(
                     y, 'y',
                     attrs={'units': detector.column_pixel_size.units}),
                 NXfield(
                     x, 'x',
                     attrs={'units': detector.column_pixel_size.units}),))
        else:
            nxprocess.data = NXdata(
                NXfield(tomo_recon_stacks, 'reconstructed_data'))

        # Create a copy of the input NeXus object and remove reduced
        # data
        exclude_items = [
            f'{nxentry.nxname}/reduced_data/data',
            f'{nxentry.nxname}/data/reduced_data',
            f'{nxentry.nxname}/data/rotation_angle',
        ]
        nxroot = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the reconstructed data NXprocess to the new NeXus object
        nxentry = nxroot[nxroot.default]
        nxentry.reconstructed_data = nxprocess
        if 'data' not in nxentry:
            nxentry.data = NXdata()
            nxentry.data.set_default()
        nxentry.data.makelink(nxprocess.data.reconstructed_data)
        if num_tomo_stacks == 1:
            nxentry.data.attrs['axes'] = ['z', 'y', 'x']
            nxentry.data.makelink(nxprocess.data.x)
            nxentry.data.makelink(nxprocess.data.y)
            nxentry.data.makelink(nxprocess.data.z)
        nxentry.data.attrs['signal'] = 'reconstructed_data'

        return nxroot

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
            NXfield,
            NXprocess,
            NXroot,
        )

        self._logger.info('Combine the reconstructed tomography stacks')

        if isinstance(nxroot, NXroot):
            nxentry = nxroot[nxroot.default]
        else:
            raise ValueError(f'Invalid parameter nxroot ({nxroot})')

        # Check if reconstructed image data is available
        if 'reconstructed_data' not in nxentry:
            raise KeyError(
                f'Unable to find valid reconstructed image data in {nxentry}')

        # Create an NXprocess to store combined image reconstruction
        # (meta)data
        nxprocess = NXprocess()

        if nxentry.reconstructed_data.data.reconstructed_data.ndim == 3:
            num_tomo_stacks = 1
        else:
            num_tomo_stacks = \
                nxentry.reconstructed_data.data.reconstructed_data.shape[0]
        if num_tomo_stacks == 1:
            self._logger.info('Only one stack available: leaving combine_data')
            return nxroot

        # Get and combine the reconstructed stacks
        # - reconstructed axis data order: stack,row/-z,y,x
        # Note: NeXus can't follow a link if the data it points to is
        # too big. So get the data from the actual place, not from
        # nxentry.data
        # Also load one stack at a time to reduce risk of hitting NeXus
        # data access limit
        t0 = time()
        tomo_recon_combined = \
            nxentry.reconstructed_data.data.reconstructed_data[0,:,:,:]
# RV check this out more
#        tomo_recon_combined = np.concatenate(
#            [tomo_recon_combined]
#            + [nxentry.reconstructed_data.data.reconstructed_data[i,:,:,:]
#               for i in range(1, num_tomo_stacks)])
        tomo_recon_combined = np.concatenate(
            [nxentry.reconstructed_data.data.reconstructed_data[i,:,:,:]
               for i in range(num_tomo_stacks-1, 0, -1)]
            + [tomo_recon_combined])
        self._logger.info(
            f'Combining the reconstructed stacks took {time()-t0:.2f} seconds')
        tomo_shape = tomo_recon_combined.shape

        # Resize the combined tomography data stacks
        # - combined axis data order: row/-z,y,x
        if self._interactive or self._save_figs:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
                tomo_recon_combined, combine_data=True)
        else:
            x_bounds = tool_config.x_bounds
            if x_bounds is None:
                self._logger.warning(
                    'x_bounds unspecified, combine data for full x-range')
            elif not is_int_pair(
                    x_bounds, ge=0, le=tomo_shape[2]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            y_bounds = tool_config.y_bounds
            if y_bounds is None:
                self._logger.warning(
                    'y_bounds unspecified, combine data for full y-range')
            elif not is_int_pair(
                    y_bounds, ge=0, le=tomo_shape[1]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = tool_config.z_bounds
            if z_bounds is None:
                self._logger.warning(
                    'z_bounds unspecified, combine data for full z-range')
            elif not is_int_pair(
                    z_bounds, ge=0, le=tomo_shape[0]):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
        if x_bounds is None:
            x_range = (0, tomo_shape[2])
            x_slice = x_range[1]//2
        else:
            x_range = (min(x_bounds), max(x_bounds))
            x_slice = (x_bounds[0]+x_bounds[1])//2
        if y_bounds is None:
            y_range = (0, tomo_shape[1])
            y_slice = y_range[1]//2
        else:
            y_range = (min(y_bounds), max(y_bounds))
            y_slice = (y_bounds[0]+y_bounds[1])//2
        if z_bounds is None:
            z_range = (0, tomo_shape[0])
            z_slice = z_range[1]//2
        else:
            z_range = (min(z_bounds), max(z_bounds))
            z_slice = (z_bounds[0]+z_bounds[1])//2
        z_dim_org = tomo_shape[0]
        tomo_recon_combined = tomo_recon_combined[
            z_range[0]:z_range[1],y_range[0]:y_range[1],x_range[0]:x_range[1]]

        # Convert the reconstructed tomography data from internal
        # coordinate frame row/-z,y,x with the origin on the
        # near-left-top corner to an z,y,x coordinate frame.
        # Here x is to the right, y along the beam direction and
        # z upwards in the lab frame of reference
        tomo_recon_combined = np.flip(tomo_recon_combined, 0)
        tomo_shape = tomo_recon_combined.shape
        z_range = (z_dim_org-z_range[1], z_dim_org-z_range[0])

        # Get coordinate axes
        detector = nxentry.instrument.detector
        row_pixel_size = float(detector.row_pixel_size)
        column_pixel_size = float(detector.column_pixel_size)
        x = column_pixel_size * (
            np.linspace(x_range[0], x_range[1], x_range[1]-x_range[0], False)
            - 0.5*detector.columns + 0.5)
        if nxentry.reconstructed_data.get('x_bounds', None) is not None:
            x += column_pixel_size*nxentry.reconstructed_data.x_bounds[0]
        x = np.asarray(x + nxentry.reduced_data.x_translation[0])
        y = column_pixel_size * (
            np.linspace(y_range[0], y_range[1], y_range[1]-y_range[0], False)
            - 0.5*detector.columns + 0.5)
        if nxentry.reconstructed_data.get('y_bounds', None) is not None:
            y += column_pixel_size*nxentry.reconstructed_data.y_bounds[0]
        y = np.asarray(y)
        z = row_pixel_size*(
            np.linspace(z_range[0], z_range[1], z_range[1]-z_range[0], False)
            - int(nxentry.reduced_data.img_row_bounds[0])
            + 0.5*detector.rows - 0.5)
        z = np.asarray(z + nxentry.reduced_data.z_translation[0])

        # Plot a few combined image slices
        if self._save_figs:
            extent = (
                y[0],
                y[-1],
                z[0],
                z[-1])
            x_slice = tomo_shape[2]//2
            quick_imshow(
                tomo_recon_combined[:,:,x_slice],
                title=f'recon combined x={x[x_slice]:.4f}', origin='lower',
                extent=extent, path=self._outputdir, save_fig=True,
                save_only=True)
            extent = (
                x[0],
                x[-1],
                z[0],
                z[-1])
            y_slice = tomo_shape[1]//2
            quick_imshow(
                tomo_recon_combined[:,y_slice,:],
                title=f'recon combined y={y[y_slice]:.4f}', origin='lower',
                extent=extent, path=self._outputdir, save_fig=True,
                save_only=True)
            extent = (
                x[0],
                x[-1],
                y[0],
                y[-1])
            z_slice = tomo_shape[0]//2
            quick_imshow(
                tomo_recon_combined[z_slice,:,:],
                title=f'recon combined z={z[z_slice]:.4f}', origin='lower',
                extent=extent, path=self._outputdir, save_fig=True,
                save_only=True)

        # Add image reconstruction to reconstructed data NXprocess
        # - combined axis data order: z,y,x
        if x_bounds is not None and x_bounds != (0, tomo_shape[2]):
            nxprocess.x_bounds = x_bounds
            nxprocess.x_bounds.units = 'pixels'
            nxprocess.x_bounds.attrs['long_name'] = \
                'x range indices in reconstructed data frame of reference'
        if y_bounds is not None and y_bounds != (0, tomo_shape[1]):
            nxprocess.y_bounds = y_bounds
            nxprocess.y_bounds.units = 'pixels'
            nxprocess.y_bounds.attrs['long_name'] = \
                'y range indices in reconstructed data frame of reference'
        if z_bounds is not None and z_bounds != (0, tomo_shape[0]):
            nxprocess.z_bounds = z_bounds
            nxprocess.z_bounds.units = 'pixels'
            nxprocess.z_bounds.attrs['long_name'] = \
                'z range indices in reconstructed data frame of reference'
        nxprocess.data = NXdata(
            NXfield(tomo_recon_combined, 'combined_data'),
            (NXfield(z, 'z', attrs={'units': detector.row_pixel_size.units}),
             NXfield(
                 y, 'y', attrs={'units': detector.column_pixel_size.units}),
             NXfield(
                 x, 'x', attrs={'units': detector.column_pixel_size.units}),))

        # Create a copy of the input NeXus object and remove
        # reconstructed data
        exclude_items = [
            f'{nxentry.nxname}/reconstructed_data/data',
            f'{nxentry.nxname}/data/reconstructed_data',
        ]
        nxroot = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the combined data NXprocess to the new NeXus object
        nxentry = nxroot[nxroot.default]
        nxentry.combined_data = nxprocess
        if 'data' not in nxentry:
            nxentry.data = NXdata()
            nxentry.data.set_default()
        nxentry.data.makelink(nxprocess.data.combined_data)
        nxentry.data.attrs['axes'] = ['z', 'y', 'x']
        nxentry.data.makelink(nxprocess.data.x)
        nxentry.data.makelink(nxprocess.data.y)
        nxentry.data.makelink(nxprocess.data.z)
        nxentry.data.attrs['signal'] = 'combined_data'

        return nxroot

    def _gen_dark(self, nxentry, reduced_data, image_key):
        """Generate dark field."""
        # Third party modules
        from nexusformat.nexus import NXdata

        # Get the dark field images
        field_indices = [
            index for index, key in enumerate(image_key) if key == 2]
        if field_indices:
            tdf_stack = nxentry.instrument.detector.data[field_indices,:,:]
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
                path=self._outputdir, save_fig=True, save_only=True)

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
            tbf_stack = nxentry.instrument.detector.data[field_indices,:,:]
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

        # Set any non-positive values to one
        # (avoid negative bright field values for spikes in dark field)
        tbf[tbf < 1] = 1

        # Plot bright field
        if self._save_figs:
            quick_imshow(
                tbf, title='Bright field', name='bright_field',
                path=self._outputdir, save_fig=True, save_only=True)

        # Add bright field to reduced data NXprocess
        if 'data' not in reduced_data:
            reduced_data.data = NXdata()
        reduced_data.data.bright_field = tbf

        return reduced_data

    def _set_detector_bounds(
            self, nxentry, reduced_data, image_key, theta, img_row_bounds,
            calibrate_center_rows):
        """Set vertical detector bounds for each image stack. Right
        now the range is the same for each set in the image stack.
        """
        # Third party modules
        import matplotlib.pyplot as plt

        # Get the first tomography image and the reference heights
        image_mask = reduced_data.get('image_mask')
        if image_mask is None:
            first_image_index = 0
        else:
            first_image_index = int(np.argmax(image_mask))
        field_indices_all = [
            index for index, key in enumerate(image_key) if key == 0]
        if not field_indices_all:
            raise ValueError('Tomography field(s) unavailable')
        z_translation_all = nxentry.sample.z_translation[field_indices_all]
        z_translation_levels = sorted(list(set(z_translation_all)))
        num_tomo_stacks = len(z_translation_levels)
        center_stack_index = num_tomo_stacks//2
        z_translation = z_translation_levels[center_stack_index]
        try:
            field_indices = [
                field_indices_all[index]
                for index, z in enumerate(z_translation_all)
                if z == z_translation]
            first_image = nxentry.instrument.detector.data[
                field_indices[first_image_index]]
        except Exception as exc:
            raise RuntimeError('Unable to load the tomography images') from exc

        # Set initial image bounds or rotation calibration rows
        tbf = reduced_data.data.bright_field.nxdata
        if (not isinstance(calibrate_center_rows, bool)
                and is_int_pair(calibrate_center_rows)):
            img_row_bounds = calibrate_center_rows
        else:
            if nxentry.instrument.source.attrs['station'] in ('id1a3', 'id3a'):
                # System modules
                from sys import float_info

                # Third party modules
                from nexusformat.nexus import (
                    NXdata,
                    NXfield,
                )

                # Local modules
                from CHAP.utils.fit import FitProcessor

                pixel_size = float(nxentry.instrument.detector.row_pixel_size)
                # Try to get a fit from the bright field
                row_sum = np.sum(tbf, 1)
                num = len(row_sum)
                fit = FitProcessor()
                model = {'model': 'rectangle',
                         'parameters': [
                             {'name': 'amplitude',
                              'value': row_sum.max()-row_sum.min(),
                              'min': 0.0},
                             {'name': 'center1', 'value': 0.25*num,
                                 'min': 0.0, 'max': num},
                             {'name': 'sigma1', 'value': num/7.0,
                              'min': float_info.min},
                             {'name': 'center2', 'value': 0.75*num,
                              'min': 0.0, 'max': num},
                             {'name': 'sigma2', 'value': num/7.0,
                              'min': float_info.min}]}
                bounds_fit = fit.process(
                    data=NXdata(
                        NXfield(row_sum, 'y'),
                        NXfield(np.array(range(num)), 'x')),
                    config={'models': [model], 'method': 'trf'})
                parameters = bounds_fit.best_values
                row_low_fit = parameters.get('center1', None)
                row_upp_fit = parameters.get('center2', None)
                sig_low = parameters.get('sigma1', None)
                sig_upp = parameters.get('sigma2', None)
                have_fit = (bounds_fit.success and row_low_fit is not None
                    and row_upp_fit is not None and sig_low is not None
                    and sig_upp is not None
                    and 0 <= row_low_fit < row_upp_fit <= row_sum.size
                    and (sig_low+sig_upp) / (row_upp_fit-row_low_fit) < 0.1)
                if num_tomo_stacks == 1:
                    if have_fit:
                        # Add a pixel margin for roundoff effects in fit
                        row_low_fit += 1
                        row_upp_fit -= 1
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
                        # window
                        row_low = int((row_low_fit+row_upp_fit-num_row_min)/2)
                        row_upp = row_low+num_row_min
                    else:
                        # Center the default range
                        row_low = int((tbf.shape[0]-num_row_min)/2)
                        row_upp = row_low+num_row_min
                img_row_bounds = (row_low, row_upp)
                if calibrate_center_rows:
                    # Add a small margin to avoid edge effects
                    offset = int(min(5, 0.1*(row_upp-row_low)))
                    img_row_bounds = (row_low+offset, row_upp-1-offset)
            else:
                if num_tomo_stacks > 1:
                    raise NotImplementedError(
                        'Selecting image bounds or calibrating rotation axis '
                        'for multiple stacks on FMB')
                # For FMB: use the first tomography image to select range
                # RV revisit if they do tomography with multiple stacks
                if img_row_bounds is None and not self._interactive:
                    if calibrate_center_rows:
                        self._logger.warning(
                            'calibrate_center_rows unspecified, find rotation '
                            'axis at detector bounds (with a small margin)')
                        # Add a small margin to avoid edge effects
                        offset = min(5, 0.1*first_image.shape[0])
                        img_row_bounds = (
                            offset, first_image.shape[0]-1-offset)
                    else:
                        self._logger.warning(
                            'img_row_bounds unspecified, reduce data for '
                            'entire detector range')
                        img_row_bounds = (0, first_image.shape[0])
        if calibrate_center_rows:
            title='Select two detector image row indices to '\
                  'calibrate rotation axis (in range '\
                  f'[0, {first_image.shape[0]-1}])'
        else:
            title='Select detector image row bounds for data '\
                  f'reduction (in range [0, {first_image.shape[0]}])'
        fig, img_row_bounds = select_image_indices(
            first_image, 0, b=tbf, preselected_indices=img_row_bounds,
            title=title,
            title_a=r'Tomography image at $\theta$ = 'f'{round(theta, 2)+0}',
            title_b='Bright field',
            interactive=self._interactive)
        if not calibrate_center_rows and (num_tomo_stacks > 1
                and (img_row_bounds[1]-img_row_bounds[0]+1)
                     < int((delta_z - 0.5*pixel_size) / pixel_size)):
            self._logger.warning(
                'Image bounds and pixel size prevent seamless stacking')

        # Plot results
        if self._save_figs:
            if calibrate_center_rows:
                fig.savefig(os_path.join(
                    self._outputdir, 'rotation_calibration_rows.png'))
            else:
                fig.savefig(os_path.join(
                    self._outputdir, 'detector_image_bounds.png'))
        plt.close()

        return img_row_bounds

    def _gen_thetas(self, nxentry, image_key):
        """Get the rotation angles for the image stacks."""
        # Get the rotation angles (in degrees)
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
                thetas = nxentry.sample.rotation_angle[
                    field_indices][sequence_numbers]
            else:
                assert all(
                    thetas[i] == nxentry.sample.rotation_angle[
                        field_indices[index]]
                    for i, index in enumerate(sequence_numbers))

        return np.asarray(thetas)

    def _set_zoom_or_delta_theta(self, thetas, delta_theta=None):
        """Set zoom and/or delta theta to reduce memory the requirement
        for the analysis.
        """
        # Local modules
        from CHAP.utils.general import index_nearest

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
            print(f'\nAvailable \u03b8 range: [{thetas[0]}, {thetas[-1]}]')
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

    def _gen_tomo(
            self, nxentry, reduced_data, image_key, calibrate_center_rows):
        """Generate tomography fields."""
        # Third party modules
        from numexpr import evaluate
        from scipy.ndimage import zoom

        # Get dark field
        if 'dark_field' in reduced_data.data:
            tdf = reduced_data.data.dark_field.nxdata
        else:
            self._logger.warning('Dark field unavailable')
            tdf = None

        # Get bright field
        tbf = reduced_data.data.bright_field.nxdata
        tbf_shape = tbf.shape

        # Subtract dark field
        if tdf is not None:
            try:
                with SetNumexprThreads(self._num_core):
                    evaluate('tbf-tdf', out=tbf)
            except TypeError as e:
                sys_exit(
                    f'\nA {type(e).__name__} occured while subtracting '
                    'the dark field with num_expr.evaluate()'
                    '\nTry reducing the detector range')

        # Get image bounds
        img_row_bounds = tuple(reduced_data.get('img_row_bounds'))
        img_column_bounds = tuple(
            reduced_data.get('img_column_bounds', (0, tbf_shape[1])))

        # Check if this run is a rotation axis calibration
        # and resize dark and bright fields accordingly
        if calibrate_center_rows:
            if tdf is not None:
                tdf = tdf[calibrate_center_rows,:]
            tbf = tbf[calibrate_center_rows,:]
        else:
            if (img_row_bounds != (0, tbf.shape[0])
                    or img_column_bounds != (0, tbf.shape[1])):
                if tdf is not None:
                    tdf = tdf[
                        img_row_bounds[0]:img_row_bounds[1],
                        img_column_bounds[0]:img_column_bounds[1]]
                tbf = tbf[
                    img_row_bounds[0]:img_row_bounds[1],
                    img_column_bounds[0]:img_column_bounds[1]]

        # Get thetas (in degrees)
        thetas = reduced_data.rotation_angle.nxdata

        # Get or create image mask
        image_mask = reduced_data.get('image_mask')
        if image_mask is None:
            image_mask = [True]*len(thetas)
        else:
            image_mask = list(image_mask)

        # Get the tomography images
        field_indices_all = [
            index for index, key in enumerate(image_key) if key == 0]
        if not field_indices_all:
            raise ValueError('Tomography field(s) unavailable')
        z_translation_all = nxentry.sample.z_translation[
            field_indices_all]
        z_translation_levels = sorted(list(set(z_translation_all)))
        num_tomo_stacks = len(z_translation_levels)
        if calibrate_center_rows:
            center_stack_index = num_tomo_stacks//2
        tomo_stacks = num_tomo_stacks*[np.array([])]
        horizontal_shifts = []
        vertical_shifts = []
        for i, z_translation in enumerate(z_translation_levels):
            if calibrate_center_rows and i != center_stack_index:
                continue
            try:
                field_indices = [
                    field_indices_all[i]
                    for i, z in enumerate(z_translation_all)
                    if z == z_translation]
                field_indices_masked = [
                     v for i, v in enumerate(field_indices) if image_mask[i]]
                horizontal_shift = list(
                    set(nxentry.sample.x_translation[field_indices_masked]))
                assert len(horizontal_shift) == 1
                horizontal_shifts += horizontal_shift
                vertical_shift = list(
                    set(nxentry.sample.z_translation[field_indices_masked]))
                assert len(vertical_shift) == 1
                vertical_shifts += vertical_shift
                sequence_numbers = \
                    nxentry.instrument.detector.sequence_number[field_indices]
                assert (list(sequence_numbers)
                        == list(range((len(sequence_numbers)))))
                tomo_stack = nxentry.instrument.detector.data[
                    field_indices_masked]
            except Exception as exc:
                raise RuntimeError('Unable to load the tomography images '
                                   f'for stack {i}') from exc
            tomo_stacks[i] = tomo_stack
            if not calibrate_center_rows:
                if not i:
                    tomo_stack_shape = tomo_stack.shape
                else:
                    assert tomo_stack_shape == tomo_stack.shape

        reduced_tomo_stacks = num_tomo_stacks*[np.array([])]
        tomo_stack_shape = None
        for i, tomo_stack in enumerate(tomo_stacks):
            if not tomo_stack.size:
                continue
            # Resize the tomography images
            # Right now the range is the same for each set in the stack
            if calibrate_center_rows:
                tomo_stack = tomo_stack[:,calibrate_center_rows,:].astype(
                        'float64', copy=False)
            else:
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
            tomo_stack = np.where(np.isfinite(tomo_stack), tomo_stack, 0.)

            # Downsize tomography stack to smaller size
            tomo_stack = tomo_stack.astype('float32', copy=False)
            if self._save_figs or self._save_only:
                theta = round(thetas[0], 2)
                if len(tomo_stacks) == 1:
                    title = r'Reduced data, $\theta$ = 'f'{theta}'
                    name = f'reduced_data_theta_{theta}'
                else:
                    title = f'Reduced data stack {i}, 'r'$\theta$ = 'f'{theta}'
                    name = f'reduced_data_stack_{i}_theta_{theta}'
                quick_imshow(
                    tomo_stack[0,:,:], title=title, name=name,
                    path=self._outputdir, save_fig=self._save_figs,
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
                title = f'red stack {zoom_perc}p theta ' \
                    f'{round(thetas[0], 2)+0}'
                quick_imshow(
                    tomo_stack[0,:,:], title=title,
                    path=self._outputdir, save_fig=self._save_figs,
                    save_only=self._save_only, block=self._block)
                del tomo_zoom_list

            # Combine resized stacks
            reduced_tomo_stacks[i] = tomo_stack
            if tomo_stack_shape is None:
                tomo_stack_shape = tomo_stack.shape
            else:
                assert tomo_stack_shape == tomo_stack.shape

        for i, stack in enumerate(reduced_tomo_stacks):
            if not stack.size:
                reduced_tomo_stacks[i] = np.zeros(tomo_stack_shape)

        # Add tomo field info to reduced data NXprocess
        reduced_data.x_translation = horizontal_shifts
        reduced_data.x_translation.units = 'mm'
        reduced_data.z_translation = vertical_shifts
        reduced_data.z_translation.units = 'mm'
        reduced_data.data.tomo_fields = reduced_tomo_stacks
        reduced_data.data.attrs['signal'] = 'tomo_fields'

        if tdf is not None:
            del tdf
        del tbf

        return reduced_data

    def _find_center_one_plane(
            self, tomo_stacks, stack_index, row, offset_row, thetas,
            num_core=1, center_offset_min=-50, center_offset_max=50,
            center_search_range=None, gaussian_sigma=None, ring_width=None,
            prev_center_offset=None):
        """Find center for a single tomography plane.

        tomo_stacks data axes order: stack,theta,row,column
        thetas in radians
        """
        # Third party modules
        import matplotlib.pyplot as plt
        from tomopy import (
#            find_center,
            find_center_vo,
            find_center_pc,
        )

        if not gaussian_sigma:
            gaussian_sigma = None
        if not ring_width:
            ring_width = None

        # Get the sinogram for the selected plane
        sinogram = tomo_stacks[stack_index,:,offset_row,:]
        center_offset_range = sinogram.shape[1]/2

        # Try Nghia Vo's method to find the center
        t0 = time()
        if center_offset_min is None:
            center_offset_min = -50
        if center_offset_max is None:
            center_offset_max = 50
        if num_core > NUM_CORE_TOMOPY_LIMIT:
            self._logger.debug(
                f'Running find_center_vo on {NUM_CORE_TOMOPY_LIMIT} '
                'cores ...')
            tomo_center = find_center_vo(
                sinogram, ncore=NUM_CORE_TOMOPY_LIMIT, smin=center_offset_min,
                smax=center_offset_max)
        else:
            tomo_center = find_center_vo(
                sinogram, ncore=num_core, smin=center_offset_min,
                smax=center_offset_max)
        self._logger.info(
            f'Finding center using Nghia Vo\'s method took {time()-t0:.2f} '
            'seconds')
        center_offset_vo = float(tomo_center-center_offset_range)
        self._logger.info(
            f'Center at row {row} using Nghia Vo\'s method = '
            f'{center_offset_vo:.2f}')

        selected_center_offset = center_offset_vo
        if self._interactive or self._save_figs:

            # Try Guizar-Sicairos's phase correlation method to find
            # the center
            t0 = time()
            tomo_center = find_center_pc(
                tomo_stacks[stack_index,0,:,:],
                tomo_stacks[stack_index,-1,:,:])
            self._logger.info(
                'Finding center using Guizar-Sicairos\'s phase correlation '
                f'method took {time()-t0:.2f} seconds')
            center_offset_pc = float(tomo_center-center_offset_range)
            self._logger.info(
                f'Center at row {row} using Guizar-Sicairos\'s image entropy '
                f'method = {center_offset_pc:.2f}')

            # Try Donath's image entropy method to find the center
# Skip this method, it seems flawed somehow or I'm doing something wrong
#            t0 = time()
#            tomo_center = find_center(
#                tomo_stacks[stack_index,:,:,:], thetas,
#                ind=offset_row)
#            self._logger.info(
#                'Finding center using Donath\'s image entropy method took '
#                f'{time()-t0:.2f} seconds')
#            center_offset_ie = float(tomo_center-center_offset_range)
#            self._logger.info(
#                f'Center at row {row} using Donath\'s image entropy method = '
#                f'{center_offset_ie:.2f}')

            # Reconstruct the plane for the Nghia Vo's center
            t0 = time()
            center_offsets = [center_offset_vo]
            fig_titles = [f'Vo\'s method: center offset = '
                         f'{center_offset_vo:.2f}']
            recon_planes = [self._reconstruct_planes(
                    sinogram, center_offset_vo, thetas, num_core=num_core,
                    gaussian_sigma=gaussian_sigma, ring_width=ring_width)]
            self._logger.info(
                f'Reconstructing row {row} with center at '
                f'{center_offset_vo} took {time()-t0:.2f} seconds')

            # Reconstruct the plane for the Guizar-Sicairos's center
            t0 = time()
            center_offsets.append(center_offset_pc)
            fig_titles.append(f'Guizar-Sicairos\'s method: center offset = '
                          f'{center_offset_pc:.2f}')
            recon_planes.append(self._reconstruct_planes(
                    sinogram, center_offset_pc, thetas, num_core=num_core,
                    gaussian_sigma=gaussian_sigma, ring_width=ring_width))
            self._logger.info(
                f'Reconstructing row {row} with center at '
                f'{center_offset_pc} took {time()-t0:.2f} seconds')

            # Reconstruct the plane for the Donath's center
#            t0 = time()
#            center_offsets.append(center_offset_ie)
#            fig_titles.append(f'Donath\'s method: center offset = '
#                              f'{center_offset_ie:.2f}')
#            recon_planes.append(self._reconstruct_planes(
#                sinogram, center_offset_ie, thetas, num_core=num_core,
#                gaussian_sigma=gaussian_sigma, ring_width=ring_width))
#            self._logger.info(
#                f'Reconstructing row {row} with center at '
#                f'{center_offset_ie} took {time()-t0:.2f} seconds')

            # Reconstruct the plane at the previous row's center
            if (prev_center_offset is not None
                    and prev_center_offset not in center_offsets):
                t0 = time()
                center_offsets.append(prev_center_offset)
                fig_titles.append(f'Previous row\'s: center offset = '
                                  f'{prev_center_offset:.2f}')
                recon_planes.append(self._reconstruct_planes(
                    sinogram, prev_center_offset, thetas, num_core=num_core,
                    gaussian_sigma=gaussian_sigma, ring_width=ring_width))
                self._logger.info(
                    f'Reconstructing row {row} with center at '
                    f'{prev_center_offset} took {time()-t0:.2f} seconds')

#            t0 = time()
#            recon_edges = []
#            for recon_plane in recon_planes:
#                recon_edges.append(self._get_edges_one_plane(recon_plane))
#            print(f'\nGetting edges for row {row} with centers at '
#                  f'{center_offsets} took {time()-t0:.2f} seconds\n')

            # Select the best center
            fig, accept, selected_center_offset = \
                self._select_center_offset(
                    recon_planes, row, center_offsets, default_offset_index=0,
                    fig_titles=fig_titles, search_button=False,
                    include_all_bad=True)

            # Plot results
            if self._save_figs:
                fig.savefig(
                    os_path.join(
                        self._outputdir,
                        f'recon_row_{row}_default_centers.png'))
            plt.close()

        # Create reconstructions for a specified search range
        if self._interactive:
            if (center_search_range is None
                    and input_yesno('\nDo you want to reconstruct images '
                                    'for a range of rotation centers', 'n')):
                center_search_range = input_num_list(
                    'Enter up to 3 numbers (start, end, step), '
                    '(range, step), or range', remove_duplicates=False,
                    sort=False)
        if center_search_range is not None:
            if len(center_search_range) != 3:
                search_range = center_search_range[0]
                if len(center_search_range) == 1:
                    step = search_range
                else:
                    step = center_search_range[1]
                if selected_center_offset == 'all bad':
                    center_search_range = [
                        - search_range/2, search_range/2, step]
                else:
                    center_search_range = [
                        selected_center_offset - search_range/2,
                        selected_center_offset + search_range/2,
                        step]
            center_search_range[1] += 1 # Make upper bound inclusive
            search_center_offsets = list(np.arange(*center_search_range))
            search_recon_planes = self._reconstruct_planes(
                sinogram, search_center_offsets, thetas, num_core=num_core,
                gaussian_sigma=gaussian_sigma, ring_width=ring_width)
            for i, center in enumerate(search_center_offsets):
                title = f'Reconstruction for row {row}, center offset: ' \
                        f'{center:.2f}'
                name = f'recon_row_{row}_center_{center:.2f}.png'
                if self._interactive:
                    save_only = False
                    block = True
                else:
                    save_only = True
                    block = False
                quick_imshow(
                    search_recon_planes[i], title=title, row_label='y',
                    column_label='x', path=self._outputdir, name=name,
                    save_only=save_only, save_fig=True, block=block)
                center_offsets.append(center)
                recon_planes.append(search_recon_planes[i])

        # Perform an interactive center finding search
        calibrate_interactively = False
        if self._interactive:
            if selected_center_offset == 'all bad':
                calibrate_interactively = input_yesno(
                    '\nDo you want to perform an interactive search to '
                    'calibrate the rotation center (y/n)?', 'n')
            else:
                calibrate_interactively = input_yesno(
                    '\nDo you want to perform an interactive search to '
                    'calibrate the rotation center around the selected value '
                    f'of {selected_center_offset} (y/n)?', 'n')
        if calibrate_interactively:
            include_all_bad = True
            low = None
            upp = None
            if selected_center_offset == 'all bad':
                selected_center_offset = None
            selected_center_offset = input_num(
                '\nEnter the initial center offset in the center calibration '
                'search', ge=-center_offset_range, le=center_offset_range,
                default=selected_center_offset)
            max_step_size = min(
                center_offset_range+selected_center_offset,
                center_offset_range-selected_center_offset-1)
            max_step_size = 1 << int(np.log2(max_step_size))-1
            step_size = input_int(
                '\nEnter the intial step size in the center calibration '
                'search (will be truncated to the nearest lower power of 2)',
                ge=2, le=max_step_size, default=4)
            step_size = 1 << int(np.log2(step_size))
            selected_center_offset_prev = round(selected_center_offset)
            while step_size:
                preselected_offsets = (
                    selected_center_offset_prev-step_size,
                    selected_center_offset_prev,
                    selected_center_offset_prev+step_size)
                indices = []
                for i, preselected_offset in enumerate(preselected_offsets):
                    if preselected_offset in center_offsets:
                        indices.append(
                            center_offsets.index(preselected_offset))
                    else:
                        indices.append(len(center_offsets))
                        center_offsets.append(preselected_offset)
                        recon_planes.append(self._reconstruct_planes(
                            sinogram, preselected_offset, thetas,
                            num_core=num_core, gaussian_sigma=gaussian_sigma,
                            ring_width=ring_width))
                fig, accept, selected_center_offset = \
                    self._select_center_offset(
                        [recon_planes[i] for i in indices],
                        row, preselected_offsets, default_offset_index=1,
                        include_all_bad=include_all_bad)
                # Plot results
                if self._save_figs:
                    fig.savefig(
                        os_path.join(
                            self._outputdir,
                            f'recon_row_{row}_center_range_'
                                f'{min(preselected_offsets)}_'\
                                f'{max(preselected_offsets)}.png'))
                plt.close()
                if accept and input_yesno(
                        f'Accept center offset {selected_center_offset} '
                        f'for row {row}? (y/n)', 'y'):
                    break
                if selected_center_offset   == 'all bad':
                    step_size *=2
                else:
                    if selected_center_offset == preselected_offsets[0]:
                        upp = preselected_offsets[1]
                    elif selected_center_offset == preselected_offsets[1]:
                        low = preselected_offsets[0]
                        upp = preselected_offsets[2]
                    else:
                        low = preselected_offsets[1]
                    if None in (low, upp):
                        step_size *= 2
                    else:
                        step_size = step_size//2
                        include_all_bad = False
                    selected_center_offset_prev = round(selected_center_offset)
                if step_size > max_step_size:
                    self._logger.warning(
                        'Exceeding maximum step size of {max_step_size}')
                    step_size = max_step_size

            # Collect info for the currently selected center
            recon_planes = [recon_planes[
                center_offsets.index(selected_center_offset)]]
            center_offsets = [selected_center_offset]
            fig_titles = [f'Reconstruction for center offset = '
                         f'{selected_center_offset:.2f}']

            # Try Nghia Vo's method with the selected center
            step_size = min(step_size, 10)
            center_offset_min = selected_center_offset-step_size
            center_offset_max = selected_center_offset+step_size
            if num_core > NUM_CORE_TOMOPY_LIMIT:
                self._logger.debug(
                    f'Running find_center_vo on {NUM_CORE_TOMOPY_LIMIT} '
                    'cores ...')
                tomo_center = find_center_vo(
                    sinogram, ncore=NUM_CORE_TOMOPY_LIMIT,
                    smin=center_offset_min, smax=center_offset_max)
            else:
                tomo_center = find_center_vo(
                    sinogram, ncore=num_core, smin=center_offset_min,
                    smax=center_offset_max)
            center_offset_vo = float(tomo_center-center_offset_range)
            self._logger.info(
                f'Center at row {row} using Nghia Vo\'s method = '
                f'{center_offset_vo:.2f}')

            # Reconstruct the plane for the Nghia Vo's center
            center_offsets.append(center_offset_vo)
            fig_titles.append(f'Vo\'s method: center offset = '
                         f'{center_offset_vo:.2f}')
            recon_planes.append(self._reconstruct_planes(
                    sinogram, center_offset_vo, thetas, num_core=num_core,
                    gaussian_sigma=gaussian_sigma, ring_width=ring_width))

            # Select the best center
            fig, accept, selected_center_offset = \
                self._select_center_offset(
                    recon_planes, row, center_offsets, default_offset_index=0,
                    fig_titles=fig_titles, search_button=False)

            # Plot results
            if self._save_figs:
                fig.savefig(
                    os_path.join(
                        self._outputdir,
                        f'recon_row_{row}_center_'
                            f'{selected_center_offset:.2f}.png'))
            plt.close()

            del recon_planes

        del sinogram

        # Return the center location
        if self._interactive:
            if selected_center_offset == 'all bad':
                print('\nUnable to successfully calibrate center axis')
                selected_center_offset = input_num(
                    'Enter the center offset for row {row}',
                    ge=-center_offset_range, le=center_offset_range)
            return float(selected_center_offset)
        return float(center_offset_vo)

    def _reconstruct_planes(
            self, tomo_planes, center_offset, thetas, num_core=1,
            gaussian_sigma=None, ring_width=None):
        """Invert the sinogram for a single or multiple tomography
        planes using tomopy's recon routine."""
        # Third party modules
        from scipy.ndimage import gaussian_filter
        from tomopy import (
            misc,
            recon,
        )

        # Reconstruct the planes
        # tomo_planes axis data order: (row,)theta,column
        # thetas in radians
        if isinstance(center_offset, (int, float)):
            tomo_planes = np.expand_dims(tomo_planes, 0)
            center_offset = center_offset + tomo_planes.shape[2]/2
        elif is_num_series(center_offset):
            tomo_planes = np.array([tomo_planes]*len(center_offset))
            center_offset = np.asarray(center_offset) + tomo_planes.shape[2]/2
        else:
            raise ValueError(
                f'Invalid parameter center_offset ({center_offset})')
        recon_planes = recon(
            tomo_planes, thetas, center=center_offset, sinogram_order=True,
            algorithm='gridrec', ncore=num_core)

        # Performing Gaussian filtering and removing ring artifacts
        if gaussian_sigma is not None and gaussian_sigma:
            recon_planes = gaussian_filter(
                recon_planes, gaussian_sigma, mode='nearest')
        if ring_width is not None and ring_width:
            recon_planes = misc.corr.remove_ring(
                recon_planes, rwidth=ring_width, ncore=num_core)

        # Apply a circular mask
        recon_planes = misc.corr.circ_mask(recon_planes, axis=0)

        return np.squeeze(recon_planes)

#    def _get_edges_one_plane(self, recon_plane):
#        """Create an "edges plot" image for a single reconstructed
#        tomography data plane.
#        """
#        # Third party modules
#        from skimage.restoration import denoise_tv_chambolle
#
#        vis_parameters = None  # RV self._config.get('vis_parameters')
#        if vis_parameters is None:
#            weight = 0.1
#        else:
#            weight = vis_parameters.get('denoise_weight', 0.1)
#            if not is_num(weight, ge=0.):
#                self._logger.warning(
#                    f'Invalid weight ({weight}) in _get_edges_one_plane, '
#                    'set to a default of 0.1')
#                weight = 0.1
#        return denoise_tv_chambolle(recon_plane, weight=weight)

    def _select_center_offset(
            self, recon_planes, row, preselected_offsets,
            default_offset_index=0, fig_titles=None, search_button=True,
            include_all_bad=False):
        """Select a center offset value from reconstructed images
        for a single reconstructed tomography data plane."""
        # Third party modules
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Button

        def select_offset(offset):
            """Callback function for the "Select offset" input."""
            pass

        def search(event):
            """Callback function for the "Search" button."""
            if num_plots == 1:
                selected_offset.append(
                    (False, preselected_offsets[default_offset_index]))
            else:
                offset = radio_btn.value_selected
                if offset in ('both bad', 'all bad'):
                    selected_offset.append((False, 'all bad'))
                else:
                    selected_offset.append((False, float(offset)))
            plt.close()

        def accept(event):
            """Callback function for the "Accept" button."""
            if num_plots == 1:
                selected_offset.append(
                    (True, preselected_offsets[default_offset_index]))
            else:
                offset = radio_btn.value_selected
                if offset in ('both bad', 'all bad'):
                    selected_offset.append((False, 'all bad'))
                else:
                    selected_offset.append((True, float(offset)))
            plt.close()

        if not isinstance(recon_planes, (tuple, list)):
            recon_planes = [recon_planes]
        if not isinstance(preselected_offsets, (tuple, list)):
            preselected_offsets = [preselected_offsets]
        assert len(recon_planes) == len(preselected_offsets)
        if fig_titles is not None:
            assert len(fig_titles) == len(preselected_offsets)

        selected_offset = []

        title_pos = (0.5, 0.95)
        title_props = {'fontsize': 'xx-large', 'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        subtitle_pos = (0.5, 0.90)
        subtitle_props = {'fontsize': 'xx-large',
                          'horizontalalignment': 'center',
                          'verticalalignment': 'bottom'}

        num_plots = len(recon_planes)
        if num_plots == 1:
            fig, axs = plt.subplots(figsize=(11, 8.5))
            axs = [axs]
            vmax = np.max(recon_planes[0][:,:])
        else:
            fig, axs = plt.subplots(ncols=num_plots, figsize=(17, 8.5))
            axs = list(axs)
            vmax = np.max(recon_planes[1][:,:])
        for i, (ax, recon_plane, preselected_offset) in enumerate(zip(
                axs, recon_planes, preselected_offsets)):
            ax.imshow(recon_plane, vmin=-vmax, vmax=vmax, cmap='gray')
            if fig_titles is None:
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
        if fig_titles is not None:
            for (ax, fig_title) in zip(axs, fig_titles):
                ax.set_title(fig_title, fontsize='x-large')

        fig_title = plt.figtext(
            *title_pos, f'Reconstruction for row {row}', **title_props)
        if num_plots == 1:
            fig_subtitle = plt.figtext(
                *subtitle_pos,
                'Press "Accept" to accept this value or "Reject" if not',
                **subtitle_props)
        else:
            if search_button:
                fig_subtitle = plt.figtext(
                    *subtitle_pos,
                    'Select the best offset and press "Accept" to accept or '
                    '"Search" to continue the search',
                    **subtitle_props)
            else:
                fig_subtitle = plt.figtext(
                    *subtitle_pos,
                    'Select the best offset and press "Accept" to accept',
                    **subtitle_props)

        if not self._interactive:

            selected_offset.append(
                (True, preselected_offsets[default_offset_index]))

        else:

            fig.subplots_adjust(bottom=0.25, top=0.85)

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
                if include_all_bad:
                    if num_plots == 2:
                        labels = (*preselected_offsets, 'both bad')
                    else:
                        labels = (*preselected_offsets, 'all bad')
                else:
                    labels = preselected_offsets
                radio_btn = RadioButtons(
                    plt.axes([0.175, 0.05, 0.1, 0.1]),
                    labels = labels, active=default_offset_index)
                radio_cid = radio_btn.on_clicked(select_offset)

                # Setup "Search" button
                if search_button:
                    search_btn = Button(
                        plt.axes([0.4125, 0.05, 0.15, 0.075]), 'Search')
                    search_cid = search_btn.on_clicked(search)

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
                if search_button:
                    search_btn.disconnect(search_cid)
                    search_btn.ax.remove()
            accept_btn.disconnect(accept_cid)
            accept_btn.ax.remove()

        if num_plots == 1:
            fig_title.remove()
        else:
            fig_title.set_in_layout(True)
            if self._interactive:
                select_text.remove()
        fig_subtitle.remove()
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if not selected_offset:# and num_plots == 1:
            selected_offset.append(
                (True, preselected_offsets[default_offset_index]))

        return fig, *selected_offset[0]

    def _reconstruct_one_tomo_stack(
            self, tomo_stack, thetas, center_offsets=None, num_core=1,
            algorithm='gridrec', secondary_iters=0, gaussian_sigma=None,
            remove_stripe_sigma=None, ring_width=None):
        """Reconstruct a single tomography stack."""
        # Third party modules
        from tomopy import (
            astra,
            misc,
            prep,
            recon,
        )

        # tomo_stack axis data order: row,theta,column
        # thetas in radians
        # centers_offset: tomography axis shift in pixels relative
        # to column center
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

        # Remove horizontal stripe
        # RV prep.stripe.remove_stripe_fw seems flawed for hollow brick
        # accross multiple stacks
        if remove_stripe_sigma is not None and remove_stripe_sigma:
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
            tomo_stack, thetas, centers, sinogram_order=True,
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
                tomo_stack, thetas, centers, init_recon=tomo_recon_stack,
                options=options, sinogram_order=True, algorithm=astra,
                ncore=num_core)
            self._logger.info(
                f'Performing secondary iterations took {time()-t0:.2f} '
                'seconds')

        # Remove ring artifacts
        if ring_width is not None and ring_width:
            misc.corr.remove_ring(
                tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_core)

        # Performing Gaussian filtering
        if gaussian_sigma is not None and gaussian_sigma:
            tomo_recon_stack = misc.corr.gaussian_filter(
                tomo_recon_stack, sigma=gaussian_sigma, ncore=num_core)

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
                self._logger.warning('x_bounds unspecified, use data for '
                                     'full x-range')
                x_bounds = (0, tomo_recon_stacks[0].shape[2])
        elif not is_int_pair(
                x_bounds, ge=0, le=tomo_recon_stacks[0].shape[2]):
            raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
        if y_bounds is None:
            if not self._interactive:
                self._logger.warning('y_bounds unspecified, use data for '
                                     'full y-range')
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
        if self._save_figs:
            if combine_data:
                filename = os_path.join(
                    self._outputdir, 'combined_data_xy_roi.png')
            else:
                filename = os_path.join(
                    self._outputdir, 'reconstructed_data_xy_roi.png')
        else:
            filename = None
        roi = select_roi_2d(
            tomosum, preselected_roi=preselected_roi,
            title_a='Reconstructed data summed over z',
            row_label='y', column_label='x',
            interactive=self._interactive, filename=filename)
        if roi is None:
            x_bounds = (0, tomo_recon_stacks[0].shape[2])
            y_bounds = (0, tomo_recon_stacks[0].shape[1])
        else:
            x_bounds = (int(roi[0]), int(roi[1]))
            y_bounds = (int(roi[2]), int(roi[3]))
        self._logger.debug(f'x_bounds = {x_bounds}')
        self._logger.debug(f'y_bounds = {y_bounds}')

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
            if self._save_figs:
                if combine_data:
                    filename = os_path.join(
                        self._outputdir, 'combined_data_z_roi.png')
                else:
                    filename = os_path.join(
                        self._outputdir, 'reconstructed_data_z_roi.png')
            else:
                filename = None
            z_bounds = select_roi_1d(
                tomosum, preselected_roi=z_bounds,
                xlabel='z', ylabel='Reconstructed data summed over x and y',
                interactive=self._interactive, filename=filename)
            self._logger.debug(f'z_bounds = {z_bounds}')

        return x_bounds, y_bounds, z_bounds


class TomoSimFieldProcessor(Processor):
    """A processor to create a simulated tomography data set returning
    a `nexusformat.nexus.NXroot` object containing the simulated
    tomography detector images.
    """
    def process(self, data):
        """Process the input configuration and return a
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
        config = self.get_config(data=data, schema='tomo.models.TomoSimConfig')

        station = config.station
        sample_type = config.sample_type
        sample_size = config.sample_size
        if len(sample_size) == 1:
            sample_size = (sample_size[0], sample_size[0])
        if sample_type == 'hollow_pyramid' and len(sample_size) != 3:
            raise ValueError('Invalid combindation of sample_type '
                             f'({sample_type}) and sample_size ({sample_size}')
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
        # and add thetas for a full 360 degrees rotation series)
        if station in ('id1a3', 'id3a'):
            theta_start = 0.
        else:
            theta_start = -17
# RV        theta_end = theta_start + 360.
        theta_end = theta_start + 180.
        thetas = list(
            np.arange(theta_start, theta_end+0.5*theta_step, theta_step))

        # Get the number of horizontal stacks bases on the diagonal
        # of the square and for now don't allow more than one
        if (sample_size) == 3:
            num_tomo_stack = 1 + int(
                (max(sample_size[1:2])*np.sqrt(2)-pixel_size[1])
                / (detector_size[1]*pixel_size[1]))
        else:
            num_tomo_stack = 1 + int((sample_size[1]*np.sqrt(2)-pixel_size[1])
                                     / (detector_size[1]*pixel_size[1]))
        if num_tomo_stack > 1:
            raise ValueError('Sample is too wide for the detector')

        # Create the x-ray path length through a solid square
        # crosssection for a set of rotation angles.
        path_lengths_solid = None
        if sample_type != 'hollow_pyramid':
            path_lengths_solid = self._create_pathlength_solid_square(
                    sample_size[1], thetas, pixel_size[1], detector_size[1])

        # Create the x-ray path length through a hollow square
        # crosssection for a set of rotation angles.
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
        len_img_y = (detector_size[1]+1)//2
        if len_img_y%2:
            len_img_y = 2*len_img_y - 1
        else:
            len_img_y = 2*len_img_y
        img_dim = (len(img_row_coords), len_img_y)
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
            elif sample_type == 'hollow_pyramid':
                outer_indices = \
                    np.where(abs(img_row_coords) <= sample_size[0]/2)[0]
                inner_indices = np.where(
                    abs(img_row_coords) < sample_size[0]/2 - wall_thickness)[0]
                wall_indices = list(set(outer_indices)-set(inner_indices))
                ratio = abs(sample_size[1]-sample_size[2])/sample_size[0]
                baselength = max(sample_size[1:2])
                for i in wall_indices:
                    path_lengths_solid = self._create_pathlength_solid_square(
                        baselength - ratio*(
                            img_row_coords[i] + 0.5*sample_size[0]),
                        thetas, pixel_size[1], detector_size[1])
                    intensities_solid = \
                        beam_intensity * np.exp(-mu*path_lengths_solid)
                    for n in range(num_theta):
                        tomo_field[n,i] = intensities_solid[n]
                for i in inner_indices:
                    path_lengths_hollow = (
                        self._create_pathlength_solid_square(
                            baselength - ratio*(
                                img_row_coords[i] + 0.5*sample_size[0]),
                            thetas, pixel_size[1], detector_size[1])
                        - self._create_pathlength_solid_square(
                            baselength - 2*wall_thickness - ratio*(
                                img_row_coords[i] + 0.5*sample_size[0]),
                            thetas, pixel_size[1], detector_size[1]))
                    intensities_hollow = \
                        beam_intensity * np.exp(-mu*path_lengths_hollow)
                    for n in range(num_theta):
                        tomo_field[n,i] = intensities_hollow[n]
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

        # Create a NeXus object and write to file
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

        return nxroot

    def _create_pathlength_solid_square(self, dim, thetas, pixel_size,
            detector_size):
        """Create the x-ray path length through a solid square
        crosssection for a set of rotation angles.
        """
        # Get the column coordinates
        img_y_coords = pixel_size * (0.5 * (1 - detector_size%2)
            + np.asarray(range((detector_size+1)//2)))

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
    """A processor to create the dark field associated with a simulated
    tomography data set created by TomoSimProcessor.
    """
    def process(self, data, num_image=5):
        """Process the input configuration and return a
        `nexusformat.nexus.NXroot` object with the simulated
        dark field detector images.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :param num_image: Number of dark field images, defaults to `5`.
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

        # Create a NeXus object and write to file
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
    """A processor to create the bright field associated with a
    simulated tomography data set created by TomoSimProcessor.
    """
    def process(self, data, num_image=5):
        """Process the input configuration and return a
        `nexusformat.nexus.NXroot` object with the simulated
        bright field detector images.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :param num_image: Number of bright field images,
            defaults to `5`.
        :type num_image: int, optional.
        :raises ValueError: Missing or invalid input or configuration
            parameter.
        :return: Simulated bright field images.
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
        # Add 20% to slit size to make the bright beam slightly taller
        # than the vertical displacements between stacks
        slit_size = 1.2*source.slit_size
        if slit_size < float(detector.row_pixel_size*detector_size[0]):
            img_row_coords = float(detector.row_pixel_size) \
                * (0.5 + np.asarray(range(int(detector_size[0])))
                   - 0.5*detector_size[0])
            outer_indices = np.where(abs(img_row_coords) > slit_size/2)[0]
            bright_field[:,outer_indices,:] = 0

        # Create a NeXus object and write to file
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
    """A processor to create a tomography SPEC file associated with a
    simulated tomography data set created by TomoSimProcessor.
    """
    def process(self, data, scan_numbers=None):
        """Process the input configuration and return a list of strings
        representing a plain text SPEC file.

        :param data: Input configuration for the simulation.
        :type data: list[PipelineData]
        :param scan_numbers: List of SPEC scan numbers.
        :type scan_numbers: list[int], optional
        :raises ValueError: Invalid input or configuration parameter.
        :return: Simulated SPEC file.
        :rtype: list[str]
        """
        # System modules
        from json import dumps
        from datetime import datetime

        from nexusformat.nexus import (
            NXentry,
            NXroot,
            NXsubentry,
        )

        # Get and validate the TomoSimField, TomoDarkField, or
        # TomoBrightField configuration object in data
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
        if scan_numbers is None:
            scan_numbers = [1]
        else:
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
                num_stack = detector.z_translation.size
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

        # Create the output data structure in NeXus format
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
            # RV Fix main code to use independent dim info
            spec_file.append('#O0 GI_samx  GI_samz  GI_samphi')
            spec_file.append('#o0 samx samz samphi') # RV do I need this line?
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
                z_translations = list(detector.z_translation.nxdata)
            else:
                z_translations = [0.]
            thetas = detector.thetas
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
            else:
                raise ValueError(f'Invalid schema {schema}')
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
                        #image_sets.append(detector.data.nxdata[n])
                        image_sets.append(detector.data[n])
                    else:
                        #image_sets.append(detector.data.nxdata)
                        image_sets.append(detector.data)
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
        nxroot[sample_type].set_default()

        return nxroot


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
