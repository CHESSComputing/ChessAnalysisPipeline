#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module for Processors unique to the tomography workflow.

Tomographic reconstruction refers to the process of recovering 3D
spatial information on an object from a set of projected images
acquired under different angles after transmission of an x-ray beam
through the sample. This module contains the CHAP processors that
perform the steps in a typical tomographic reconstruction workflow.
It also contains CHAP processors to create simulated 3D image data to
test the workflow.
"""

# System modules
import os
import re
import sys
from time import time
from typing import (
    Annotated,
    Optional,
)
import tkinter as tk

# Third party modules
from json import loads
import numpy as np
from pydantic import (
#    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    conint,
    confloat,
    conlist,
    constr,
    field_validator,
    model_validator,
)
import tkinter as tk

# Local modules
from CHAP.common.models.map import (
    DetectorConfig,
    MapConfig,
)
from CHAP.pipeline import PipelineData
from CHAP.processor import Processor
from CHAP.tomo.models import (
    TomoReduceConfig,
    TomoFindCenterConfig,
    TomoReconstructConfig,
    TomoCombineConfig,
    TomoSimConfig,
)
from CHAP.utils.general import (
    #input_num,
    #input_yesno,
    is_int_pair,
    is_num,
    is_num_series,
    nxcopy,
    select_image_indices,
    select_roi_1d,
    select_roi_2d,
    quick_imshow,
)


NUM_CORE_TOMOPY_LIMIT = 24
"""int: Maximum number of cores in Tomopy routines."""


def read_metadata_provenance(data, logger=None, remove=True):
    """Retrieve metadata and provenance records from the data pipeline.

    :param data: Input data.
    :type data: list[PipelineData]
    :param logger: A python Logger object.
    :type logger: logging.Logger, optional
    :param remove: If there is a matching entry in `data`, remove it
        from the list, defaults to `True`.
    :type remove: bool, optional
    :return: The metadata and provenance records.
    :rtype: dict, dict
    """
    # Local modules
    from CHAP.pipeline import PipelineItem

    try:
        metadata = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenMetadataReader', remove=remove)
    except ValueError:
        try:
            metadata = PipelineItem.get_data(
                data, schema='foxden.reader.FoxdenDataDiscoveryReader',
                remove=remove)
            if len(metadata) > 1:
                logger.warning('Unable to get unique metadata from pipeline')
            metadata = metadata[0]
        except ValueError:
            if logger is None:
                print('WARNING: Unable to get metadata from pipeline')
            else:
                logger.warning('Unable to get metadata from pipeline')
            metadata = {}
    # FIX right now the provenance service returns input and output
    # info, not the actual record, so always remove it from the
    # pipeline and create a new record using the metadata
    # This means that you also need to read a metadata record to get
    # the did
    try:
        provenance = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenProvenanceReader')
            #data, schema='foxden.reader.FoxdenProvenanceReader', remove=remove)
    except ValueError:
        if logger is None:
            print('WARNING: Unable to get provedance from pipeline')
        else:
            logger.warning('Unable to get provedance from pipeline')
        provenance = {}
    return metadata, provenance


def create_metadata_provenance(
        did_suffix, data=None, *, metadata=None, provenance=None,
        user_metadata=None, logger=None, update=False, read=True):
    """Create metadata and provenance for CHAP processors with the
    correct schema to submit to the
    `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    Data Discovery and Metadata services.

    :param did_suffix: The FOXDEN DID suffix to add to the parent
        record's DID.
    :type did_suffix: str
    :param data: Input data pipeline, only required when `read` is
        `True` (its default value).
    :type data: list[PipelineData], optional
    :param metadata: Parent metadata to create the output metadata
        record, only used when `read` is `False`.
    :type metadata: dict, optional
    :param provenance: Parent provenance to create the output
        provenance record, only used when `read` is `False`.
    :type provenance: dict, optional
    :param user_metadata: Any workflow specific  metadata to add to the
        `'metadata'` field of the output metadata record.
    :type user_metadata: dict, optional
    :param logger: A python Logger object.
    :type logger: logging.Logger, optional
    :param update: Update the parent metadata record if `True`,
        otherwise return a clean child metadata record with only the
        updated fields required by the FOXDEN schema in addition to the
        `'metadata'` field, defaults to `False`.
    :type update: bool, optional
    :param read: Read the parent metadata and provenance from the data
        pipeline when `True`, defaults to `True`.
    :type read: bool, optional
    :raises AssertionError: When the DID doesn't match between the metadata
        and the provenance records.
    :raises RuntimeError: When `read` is True and the `data` is omitted.
    :return: The metadata and provenance information.
    :rtype: dict, dict
    """
    def validate_read(data, metadata, provenance):
        if data is None:
            raise ValueError('Missing required input parameter "data"')
        if metadata or provenance:
            if logger is None:
                print('WARNING: Ignoring inputs for metadata and provenance '
                      'when reading them from pipeline data in '
                      'create_metadata_provenance')
            else:
                logger.warning('Ignoring inputs for metadata and provenance '
                      'when reading them from pipeline data')

    if metadata is None:
        metadata = {}
    if provenance is None:
        provenance = {}
    if read:
        validate_read(data, metadata, provenance)
        metadata, provenance = read_metadata_provenance(data, logger)

    did = metadata.get('did')
    if provenance:
        # FIX: right now no multiple parent_did's inplemented
        parent_did = provenance.get('parent_did')
        if did is None:
            did = provenance.get('did')
        elif 'did' in provenance:
            assert did == provenance.get('did')
    else:
        parent_did = did
    did = f'/workflow={did_suffix}' \
        if parent_did is None \
        else f'{parent_did}/workflow={did_suffix}'
    btr = metadata.pop('btr', None)
    if btr is None:
        try:
            btr = did.split('btr=')[1].split('/')[0]
            assert isinstance(btr, str)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            logger.warning(
                f'Unable to get a valid btr from did ({did}): {exc}')
            btr = 'unknown'
    user_metadata = {} \
        if user_metadata is None \
        else metadata.pop('user_metadata', {}) | user_metadata
    if not update:
        metadata = {}
    metadata.update({
        'btr': btr,
        'did': did,
        'parent_did': parent_did,
        'schema': 'user',
        'user_metadata': user_metadata})
    provenance.update({
        'did': did,
        'parent_did': parent_did,
        'input_files': [{'name': 'todo.fix: pipeline.yaml'}]})
    return metadata, provenance


def read_metadata_provenance(data, logger=None, remove=True):
    # Local modules
    from CHAP.pipeline import PipelineItem

    try:
        metadata = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenMetadataReader', remove=remove)
    except Exception:
        try:
            metadata = PipelineItem.get_data(
                data, schema='foxden.reader.FoxdenDataDiscoveryReader',
                remove=remove)
            if len(metadata) > 1:
                logger.warning(f'Unable to get unique metadata from pipeline')
            metadata = metadata[0]
        except Exception:
            if logger is None:
                print(f'WARNING: Unable to get metadata from pipeline')
            else:
                logger.warning(f'Unable to get metadata from pipeline')
            metadata = {}
    # FIX right now the provenance service returns input and output
    # info, not the actual record, so always remove it from the
    # pipeline and create a new record using the metadata
    # This means that you also need to read a metadata record to get
    # the did
    try:
        provenance = PipelineItem.get_data(
            data, schema='foxden.reader.FoxdenProvenanceReader')
            #data, schema='foxden.reader.FoxdenProvenanceReader', remove=remove)
    except Exception:
        if logger is None:
            print(f'WARNING: Unable to get provedance from pipeline')
        else:
            logger.warning(f'Unable to get provedance from pipeline')
        provenance = {}
    return metadata, provenance


def create_metadata_provenance(
        did_suffix, data=None, metadata=None, provenance=None,
        user_metadata=None, logger=None, update=False, read=True):
    if read:
        if None in (metadata, provenance):
            if logger is None:
                print('WARNING: Ignoring inputs for metadata and provenance '
                      'when reading them from pipeline data in '
                      'create_metadata_provenance')
            else:
                logger.warning('Ignoring inputs for metadata and provenance '
                      'when reading them from pipeline data')
        metadata, provenance = read_metadata_provenance(data, logger)
    else:
        if metadata is None:
            metadata = {}
        if provenance is None:
            provenance = {}

    did = metadata.get('did')
    if provenance:
        # FIX: right now no multiple parent_did's inplemented
        parent_did = provenance.get('parent_did')
        if did is None:
            did = provenance.get('did')
        elif 'did' in provenance:
            assert did == provenance.get('did')
    else:
        parent_did = did
    if parent_did is None:
        did = f'/workflow={did_suffix}'
    else:
        did = f'{parent_did}/workflow={did_suffix}'
    btr = metadata.pop('btr', None)
    if btr is None:
        try:
            btr = did.split('btr=')[1].split('/')[0]
            assert isinstance(btr, str)
        except Exception:
            logger.warning(f'Unable to get a valid btr from did ({did})')
            btr = 'unknown'
    if user_metadata is None:
        user_metadata = {}
    else:
        user_metadata = metadata.pop('user_metadata', {}) | user_metadata
    if not update:
        metadata = {}
    metadata.update({
        'btr': btr,
        'did': did,
        'parent_did': parent_did,
        'schema': 'user',
        'user_metadata': user_metadata})
    provenance.update({
        'did': did,
        'parent_did': parent_did,
        'input_files': [{'name': 'todo.fix: pipeline.yaml'}]})
    return metadata, provenance


class TomoMetadataProcessor(Processor):
    """A processor that takes data from the
    `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    Data Discovery or Metadata service and extracts what's available
    to create a :class:`~CHAP.common.models.map.MapConfig` instance
    for a tomography experiment.

    :ivar config: Configuration dictionary containing all fields
        required to create a
        :class:`~CHAP.common.mocelc.map.MapConfig`
        instance that are not available from the metadata record.
    :vartype config: dict
    """

    config: dict

    def process(self, data):
        """Process the metadata and return a dictionary with extracted
        data to create a :class:`~CHAP.common.mocelc.map.MapConfig`
        instance for the tomography experiment.

        :param data: Input data.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: Metadata from the tomography experiment.
        :rtype: MapConfig
        """
        try:
            record = self.get_data(
                data, schema='foxden.reader.FoxdenMetadataReader',
                remove=False)
        except ValueError as exc:
            raise ValueError(
                f'Unable to get the metadata from the pipeline ({data}), '
                f'check FOXDEN read token') from exc

        # Extract any available MapConfig info
        map_config = {}
        map_config['did'] = record.get('did')
        map_config['title'] = record.get('sample_name')
        station = record.get('beamline')[0]
        if station == '3A':
            station = 'id3a'
        elif station == '1A3':
            station = 'id1a3'
        else:
            raise ValueError(f'Invalid beamline parameter ({station})')
        map_config['station'] = station
        experiment_type = record.get('technique')
        assert 'Tomography' in experiment_type
        map_config['experiment_type'] = 'TOMO'
        map_config['sample'] = {'name': map_config['title'],
                                'description': record.get('description')}
        if station in ('id1a3', 'id3a'):
            map_config['spec_scans'] = [{
                'spec_file': os.path.join(
                    record.get('data_location_raw'), 'spec.log'),
                'scan_numbers': self.config['scan_numbers']}]
        map_config['independent_dimensions'] = \
            self.config['independent_dimensions']

        # Validate the MapConfig info
        map_config = MapConfig(**map_config)

        return map_config.model_dump()


class TomoCHESSMapConverter(Processor):
    """A processor to convert a CHESS style tomography experiment map
    with dark and bright field configurations to a NeXus style
    `NXtomo <https://manual.nexusformat.org/classes/applications/NXtomo.html#nxtomo>`__
    input format.

    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    """

    nxmemory: Optional[conint(gt=0)] = 100000

    def process(self, data):
        """Process the input map and configuration and return a NeXus
        style 
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object based on the
        `NXtomo <https://manual.nexusformat.org/classes/applications/NXtomo.html#nxtomo>`__
        format.

        :param data: Input map and configuration for tomographic image
            reduction/reconstruction.
        :type data: list[PipelineData]
        :raises RuntimeError: Inconsistent thetas among tomography
            image stacks.
        :raises ValueError: Invalid input or configuration parameter.
        :return: NeXus style tomography input configuration.
        :rtype: nexusformat.nexus.NXroot
        """
        # System modules
        from copy import deepcopy

        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXdata,
            NXdetector,
            NXentry,
            NXinstrument,
            NXlink,
            NXroot,
            NXsample,
            NXsource,
            nxsetconfig,
        )
        # pylint: enable=no-name-in-module

        # Local modules
        from CHAP.utils.general import index_nearest

        # FIX make a config input
        nxsetconfig(memory=self.nxmemory)

        # Load and validate the tomography fields
        tomofields = self.get_default_nxentry(
            self.get_data(data, schema='tomofields'))
        detector_prefix = str(tomofields.detector_ids)
        tomo_stacks = tomofields.data[detector_prefix].nxdata
        tomo_stack_shape = tomo_stacks.shape
        assert len(tomo_stack_shape) == 3

        # Validate map
        map_config = MapConfig(**loads(str(tomofields.map_config)))
        assert len(map_config.spec_scans) == 1
        spec_scan = map_config.spec_scans[0]
        scan_numbers = spec_scan.scan_numbers

        # Load and validate dark field (look upstream and downstream
        # in the SPEC log file)
        try:
            darkfield = self.get_data(data, schema='darkfield')
        except ValueError:
            self.logger.warning('Unable to load dark field from pipeline')
            darkfield = None
        data_darkfield = None
        if darkfield is None:
            try:
                for scan_number in range(min(scan_numbers), 0, -1):
                    scanparser = spec_scan.get_scanparser(scan_number)
                    scan_type = scanparser.get_scan_type()
                    if scan_type == 'df1':
                        darkfield = scanparser
                        data_darkfield = darkfield.get_detector_data(
                            detector_prefix)
                        data_shape = data_darkfield.shape
                        break
            except (IOError, OSError, RuntimeError, ValueError) as exc:
                self.logger.warning(f'Unable to load valid dark field: {exc}')
            if data_darkfield is None:
                try:
                    for scan_number in range(
                            1 + max(scan_numbers), 3 + max(scan_numbers)):
                        scanparser = spec_scan.get_scanparser(scan_number)
                        scan_type = scanparser.get_scan_type()
                        if scan_type == 'df2':
                            darkfield = scanparser
                            data_darkfield = darkfield.get_detector_data(
                                detector_prefix)
                            data_shape = data_darkfield.shape
                            break
                except (IOError, OSError, RuntimeError, ValueError) as exc:
                    self.logger.warning(
                        f'Unable to load valid dark field: {exc}')
            if data_darkfield is None:
                self.logger.warning('Unable to load dark field')
        else:
            darkfield = self.get_default_nxentry(darkfield)

        # Load and validate bright field (FIX look upstream and
        # downstream # in the SPEC log file)
        try:
            brightfield = self.get_data(data, schema='brightfield')
        except ValueError:
            self.logger.warning('Unable to load bright field from pipeline')
            brightfield = None
        if brightfield is None:
            for scan_number in range(min(scan_numbers), 0, -1):
                scanparser = spec_scan.get_scanparser(scan_number)
                scan_type = scanparser.get_scan_type()
                if scan_type == 'bf1':
                    brightfield = scanparser
                    break
            else:
                raise ValueError('Unable to load bright field')
        else:
            brightfield = self.get_default_nxentry(brightfield)

        # Load and validate detector config if supplied
        try:
            detector_config = self.get_config(
                data=data, schema='tomo.models.Detector')
        except ValueError:
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
            x_translation_name = None
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
            z_translation_name = None
        if matched_dimensions:
            raise ValueError('Unknown independent dimension '
                             f'({matched_dimensions}), independent dimensions '
                             'must be in {"z_translation", "x_translation", '
                             '"rotation_angles"}')

        # Check for presample intensity
        if 'presample_intensity' in tomofields.scalar_data:
            presample_intensity = \
                tomofields.scalar_data.presample_intensity
        else:
            presample_intensity = None

        # Construct base NXentry and add to NXroot
        nxentry = NXentry(name=map_config.title)
        nxroot[nxentry.nxname] = nxentry

        # Add configuration fields
        nxentry.definition = 'NXtomo'
        nxentry.map_config = map_config.model_dump_json()
        nxentry.detector_config = DetectorConfig(
            **loads(str(tomofields.detector_config))).model_dump_json()

        # Add a NXinstrument to the NXentry
        nxinstrument = NXinstrument()
        nxentry.instrument = nxinstrument

        # Add a NXsource to the NXinstrument
        nxsource = NXsource()
        nxinstrument.source = nxsource
        nxsource.type = 'Synchrotron X-ray Source'
        nxsource.name = 'CHESS'
        nxsource.probe = 'x-ray'

        # Tag the NXsource with the runinfo (as an attribute)
        nxsource.attrs['station'] = tomofields.station
        nxsource.attrs['experiment_type'] = map_config.experiment_type

        # Add a NXdetector to the NXinstrument
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
        if is_num(pixel_size, log=False):
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

        # Add a NXsample to NXentry
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
        presample_intensities = []
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
                    if presample_intensity is not None:
                        scan_columns = loads(str(nxcollection.scan_columns))
                        presample_intensities += scan_columns[
                            presample_intensity.attrs['local_name']]
        elif data_darkfield is not None:
            data_shape = data_darkfield.shape
            assert len(data_shape) == 3
            assert data_shape[1] == nxdetector.rows
            assert data_shape[2] == nxdetector.columns
            num_image = data_shape[0]
            image_keys += num_image*[2]
            sequence_numbers += list(range(num_image))
            image_stacks.append(data_darkfield)
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
            if presample_intensity is not None:
                scan_columns = scanparser.spec_scan_data
                presample_intensities += scan_columns[
                    presample_intensity.attrs['local_name']]

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
                    if presample_intensity is not None:
                        scan_columns = loads(str(nxcollection.scan_columns))
                        presample_intensities += scan_columns[
                            presample_intensity.attrs['local_name']]
        else:
            data_brightfield = brightfield.get_detector_data(detector_prefix)
            data_shape = data_brightfield.shape
            assert len(data_shape) == 3
            assert data_shape[1] == nxdetector.rows
            assert data_shape[2] == nxdetector.columns
            num_image = data_shape[0]
            image_keys += num_image*[1]
            sequence_numbers += list(range(num_image))
            image_stacks.append(data_brightfield)
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
            if presample_intensity is not None:
                scan_columns = scanparser.spec_scan_data
                presample_intensities += scan_columns[
                    presample_intensity.attrs['local_name']]

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
            # FIX convert to using CHAPslice
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
            if presample_intensity is not None:
                presample_intensities += list(
                    presample_intensity[n_start:n_start+num_image])
            n_start += num_theta

        # Add image data to NXdetector
        nxinstrument.detector.image_key = image_keys
        nxinstrument.detector.sequence_number = sequence_numbers
        nxinstrument.detector.data = np.concatenate(image_stacks)
        del image_stacks

        # Add image data to NXsample
        nxsample.rotation_angle = rotation_angles
        nxsample.rotation_angle.units = 'degrees'
        nxsample.x_translation = x_translations
        nxsample.x_translation.units = 'mm'
        nxsample.z_translation = z_translations
        nxsample.z_translation.units = 'mm'
        nxsample.presample_intensity = presample_intensities
        if presample_intensities:
            nxsample.presample_intensity.units = \
                presample_intensity.attrs['units']

        # Add a NXdata to NXentry
        nxentry.data = NXdata(NXlink(nxentry.instrument.detector.data))
        nxentry.data.makelink(nxentry.instrument.detector.image_key)
        nxentry.data.makelink(nxentry.sample.rotation_angle)
        nxentry.data.makelink(nxentry.sample.x_translation)
        nxentry.data.makelink(nxentry.sample.z_translation)
        nxentry.data.set_default()

        # Update metadata and provenance
        metadata, provenance = create_metadata_provenance(
            'tomo_convert', data, logger=self.logger)

        return (
            PipelineData(
                name=self.name, data=metadata,
                schema='foxden.reader.FoxdenMetadataReader'),
            PipelineData(
                name=self.name, data=provenance,
                schema='foxden.reader.FoxdenProvenanceReader'),
            PipelineData(
                    name=self.name, data=nxroot, schema=self.get_schema()))


class SetNumexprThreads:
    """Class that sets and keeps track of the number of processors used
    by the code in general and by the
    `numexpr <https://pypi.org/project/numexpr/>`__
    package specifically.
    """

    def __init__(self, num_proc):
        """Initialize SetNumexprThreads.

        :param num_proc: Number of processors to be used by the
            num_expr package.
        :type num_proc: int
        """
        # System modules
        from multiprocessing import cpu_count

        if num_proc is None or num_proc < 1 or num_proc > cpu_count():
            self._num_proc = cpu_count()
        else:
            self._num_proc = num_proc
        self._num_proc_org = self._num_proc

    def __enter__(self):
        # Third party modules
        from numexpr import (
            MAX_THREADS,
            set_num_threads,
        )

        self._num_proc_org = set_num_threads(
            min(self._num_proc, MAX_THREADS))

    def __exit__(self, exc_type, exc_value, traceback):
        # Third party modules
        from numexpr import set_num_threads

        set_num_threads(self._num_proc_org)


class TomoReduceProcessor(Processor):
    """A processor to reduce a set of raw tomographic images returning
    a NeXus style
    `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
    object containing the data after
    correcting the images for the presample intensity (optionally) and
    normalization with dark and bright field, an optional list of byte
    stream representions of Matplotlib figures, and the metadata
    associated with the data reduction step.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.tomo.models.TomoReduceConfig`.
    :vartype config: dict, optional
    :ivar num_proc: Number of processors, defaults to `64`.
    :vartype num_proc: int, optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    :ivar save_figures: Create Matplotlib figures that can be saved to
        file downstream in the workflow, defaults to `True`.
    :vartype save_figures: bool, optional
    """

    pipeline_fields: dict = Field(
        default = {'config': 'tomo.models.TomoReduceConfig'},
        init_var=True)
    config: Optional[TomoReduceConfig] = TomoReduceConfig()
    num_proc: Optional[conint(gt=0)] = 64
    nxmemory: Optional[conint(gt=0)] = 100000
    save_figures: Optional[bool] = True

    _figures: list = PrivateAttr(default=[])

    def process(self, data):
        """Reduced the tomography images.

        :param data: Input data containing the raw data as a NeXus
            style
            `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
            object.
        :type data: list[PipelineData]
        :raises RuntimeError: Invalid dark or bring field shape or
            unable to load valid (the) tomography image stack(s).
        :raises TypeError: Error progagated from numexpr.evaluate().
        :raises ValueError: Invalid input or configuration parameter.
        :return: Metadata associated with the workflow, a list of byte
            stream representions of Matplotlib figures, and the result
            of the data reduction.
        :rtype: PipelineData, PipelineData, PipelineData
        """
        # System modules
        from multiprocessing import cpu_count

        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXprocess,
            nxsetconfig,
        )

        self.logger.info('Generate the reduced tomography images')

        # FIX make a config input
        nxsetconfig(memory=self.nxmemory)

        # Load the tomography data
        nxroot = self.get_data(data)
        nxentry = self.get_default_nxentry(nxroot)

        # Check the number of processors
        if self.num_proc > cpu_count():
            self.logger.warning(
                f'num_proc = {self.num_proc} is larger than the number '
                f'of available processors and reduced to {cpu_count()}')
            self.num_proc = cpu_count()
        # Tompy py uses numexpr with NUMEXPR_MAX_THREADS = 64
        if self.num_proc > 64:
            self.logger.warning(
                f'num_proc = {self.num_proc} is larger than the number '
                f'of processors suitable to Tomopy and reduced to 64')
            self.num_proc = 64

        # Validate input parameters
        detector_config = DetectorConfig(**loads(str(nxentry.detector_config)))
        img_row_bounds = self.config.img_row_bounds
        if img_row_bounds is not None and detector_config.roi is not None:
            self.logger.warning('Ignoring parameter img_row_bounds '
                                'when detector ROI is specified')
            img_row_bounds = None
        image_key = nxentry.instrument.detector.get('image_key', None)
        if image_key is None or 'data' not in nxentry.instrument.detector:
            raise ValueError(f'Unable to find image_key or data in '
                             'instrument.detector '
                             f'({nxentry.instrument.detector.tree})')

        # Create a NXprocess to store data reduction (meta)data
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
        delta_theta = self.config.delta_theta
        if drop_fraction:
            if delta_theta is not None:
                delta_theta = None
                self.logger.warning(
                    'Ignoring delta_theta when an image mask is used')
            np.random.seed(0)
            image_mask = np.where(np.random.rand(
                len(thetas)) < drop_fraction/100, 0, 1).astype(bool)

        # Set rotation angle interval to reduce memory requirement
        if image_mask is None:
            delta_theta = self._set_delta_theta(thetas, delta_theta)
            if delta_theta is not None:
                image_mask = np.asarray(
                    [0 if i%delta_theta else 1
                        for i in range(len(thetas))], dtype=bool)
            self.logger.debug(f'delta_theta: {delta_theta}')
        self.config.delta_theta = delta_theta
        if image_mask is not None:
            self.logger.debug(f'image_mask = {image_mask}')
            reduced_data.image_mask = image_mask
            thetas = thetas[image_mask]

        # Set vertical detector bounds for image stack or rotation
        # axis calibration rows
        if detector_config.roi is None:
            img_row_bounds = self._set_detector_bounds(
                nxentry, reduced_data, image_key, thetas[0],
                img_row_bounds)
        self.logger.debug(f'img_row_bounds = {img_row_bounds}')
        if img_row_bounds is None:
            tbf_shape = reduced_data.data.bright_field.shape
            img_row_bounds = (0, tbf_shape[0])
        self.config.img_row_bounds = list(img_row_bounds)
        reduced_data.img_row_bounds = self.config.img_row_bounds
        reduced_data.img_row_bounds.units = 'pixels'
        reduced_data.img_row_bounds.attrs['long_name'] = \
            'image row boundaries in detector frame of reference'

        # Store rotation angles for image stacks
        self.logger.debug(f'thetas = {thetas}')
        reduced_data.rotation_angle = thetas
        reduced_data.rotation_angle.units = 'degrees'

        # Generate reduced tomography fields
        reduced_data = self._gen_tomo(nxentry, reduced_data, image_key)

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
            f'{nxentry.nxname}/sample/presample_intensity',
            f'{nxentry.nxname}/data/data',
            f'{nxentry.nxname}/data/image_key',
            f'{nxentry.nxname}/data/rotation_angle',
            f'{nxentry.nxname}/data/x_translation',
            f'{nxentry.nxname}/data/z_translation',
        ]
        nxroot = nxcopy(nxroot, exclude_nxpaths=exclude_items)

        # Add the reduced data NXprocess
        nxentry = self.get_default_nxentry(nxroot)
        nxentry.reduced_data = reduced_data

        if 'data' not in nxentry:
            nxentry.data = NXdata()
            nxentry.data.set_default()
        nxentry.data.makelink(
            nxentry.reduced_data.data.tomo_fields, name='reduced_data')
        nxentry.data.makelink(nxentry.reduced_data.rotation_angle)
        nxentry.data.attrs['signal'] = 'reduced_data'

        # Update metadata and provenance
        metadata, provenance = create_metadata_provenance(
            'tomo_reduce',
            data,
            user_metadata={'reduced_data': self.config.model_dump()},
            logger=self.logger)
        nxentry.reduced_data.attrs['did'] = metadata.get('did')
        nxentry.reduced_data.attrs['parent_did'] = metadata.get('parent_did')

        return (
            PipelineData(
                name=self.name, data=metadata,
                schema='foxden.reader.FoxdenMetadataReader'),
            PipelineData(
                name=self.name, data=provenance,
                schema='foxden.reader.FoxdenProvenanceReader'),
            PipelineData(
                name=self.name, data=self._figures,
                schema='common.write.ImageWriter'),
            PipelineData(name=self.name, data=nxroot, schema='tomodata'))

    def _gen_bright(self, nxentry, reduced_data, image_key):
        """Generate bright field."""
        # Third party modules
        from nexusformat.nexus import NXdata
        from numexpr import evaluate

        def subtrack_dark_and_normalize(nxentry, tdf, tbf_stack):
            """Subtract dark field and normalize with presample intensity."""
            if nxentry.sample.presample_intensity.size:
                presam_int = \
                    nxentry.sample.presample_intensity.nxdata[
                        field_indices].reshape(-1, 1, 1)
                if tdf is None:
                    try:
                        with SetNumexprThreads(self.num_proc):
                            evaluate('tbf_stack/presam_int', out=tbf_stack)
                    except TypeError as exc:
                        raise TypeError(
                            f'\nA {type(exc).__name__} occured while '
                            'normalizing with num_expr.evaluate(). '
                            'Try reducing the detector\'s roi') from exc
                else:
                    try:
                        with SetNumexprThreads(self.num_proc):
                            evaluate(
                                '(tbf_stack-tdf)/presam_int', out=tbf_stack)
                    except TypeError as exc:
                        raise TypeError(
                            f'\nA {type(exc).__name__} occured while '
                            'subtracting the dark field and normalizing with '
                            'num_expr.evaluate(). Try reducing the detector\'s'
                            'roi') from exc
            elif tdf is not None:
                try:
                    with SetNumexprThreads(self.num_proc):
                        evaluate('tbf_stack-tdf', out=tbf_stack)
                except TypeError as exc:
                    raise TypeError(
                        f'\nA {type(exc).__name__} occured while subtracting '
                        'the dark field with num_expr.evaluate()'
                        '\nTry reducing the detector range') from exc

        # Get dark field
        if 'dark_field' in reduced_data.data:
            tdf = reduced_data.data.dark_field.nxdata
        else:
            self.logger.warning('Dark field unavailable')
            tdf = None

        # Get the bright field images
        field_indices = [
            index for index, key in enumerate(image_key) if key == 1]
        try:
            assert field_indices
            tbf_stack = nxentry.instrument.detector.data.nxdata[
                field_indices,:,:]
        except AssertionError as exc:
            raise ValueError('Bright field unavailable') from exc

        # Subtract dark field and normalize with presample intensity
        subtrack_dark_and_normalize(nxentry, tdf, tbf_stack)

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
            tbf = np.asarray(tbf_stack)
        elif tbf_stack.ndim == 3:
            tbf = np.median(tbf_stack, axis=0)
            del tbf_stack
        else:
            raise RuntimeError(f'Invalid tbf_stack shape ({tbf_stack.shape})')

        # Remove non-positive values
        # (avoid negative bright field values for spikes in dark field)
        tbf[tbf < 0] = 0

        # Save bright field
        if self.save_figures:
            self._figures.append(
                (quick_imshow(
                    tbf, title='Bright field', cmap='gray', show_fig=False,
                    return_fig=True),
                'bright_field'))

        # Add bright field to reduced data NXprocess
        if 'data' not in reduced_data:
            reduced_data.data = NXdata()
        reduced_data.data.bright_field = tbf

        return reduced_data

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
            self.logger.warning('Dark field unavailable')
            return reduced_data

        # Take median
        if tdf_stack.ndim == 2:
            tdf = np.asarray(tdf_stack)
        elif tdf_stack.ndim == 3:
            tdf = np.median(tdf_stack, axis=0)
            del tdf_stack
        else:
            raise RuntimeError(f'Invalid tdf_stack shape ({tdf_stack.shape})')

        # Remove dark field intensities above the cutoff
#        tdf_cutoff = tdf.min() + 2 * (np.median(tdf)-tdf.min())
#        if tdf_cutoff is not None:
#            if not is_num(tdf_cutoff) or tdf_cutoff < 0:
#                self.logger.warning(
#                    f'Ignoring illegal value of tdf_cutoff {tdf_cutoff}')
#            else:
#                tdf[tdf > tdf_cutoff] = np.nan
#                self.logger.debug(f'tdf_cutoff = {tdf_cutoff}')

        # Remove nans
        tdf_mean = np.nanmean(tdf)
        self.logger.debug(f'tdf_mean = {tdf_mean}')
        np.nan_to_num(
            tdf, copy=False, nan=tdf_mean, posinf=tdf_mean, neginf=0.)

        # Save dark field
        if self.save_figures:
            self._figures.append(
                (quick_imshow(
                    tdf, title='Dark field', cmap='gray', show_fig=False,
                    return_fig=True),
                'dark_field'))

        # Add dark field to reduced data NXprocess
        reduced_data.data = NXdata()
        reduced_data.data.dark_field = tdf

        return reduced_data

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

    def _gen_tomo(self, nxentry, reduced_data, image_key):
        """Generate tomography fields."""
        # Third party modules
        from numexpr import evaluate

        # Get dark field
        if 'dark_field' in reduced_data.data:
            tdf = reduced_data.data.dark_field.nxdata
        else:
            self.logger.warning('Dark field unavailable')
            tdf = None

        # Get bright field
        tbf = reduced_data.data.bright_field.nxdata
        tbf_shape = tbf.shape

        # Get image bounds
        img_row_bounds = tuple(reduced_data.get('img_row_bounds'))
        img_column_bounds = tuple(
            reduced_data.get('img_column_bounds', (0, tbf_shape[1])))

        # Check if this run is a rotation axis calibration
        # and resize dark and bright fields accordingly
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
        tomo_stacks = num_tomo_stacks*[None]
        horizontal_shifts = []
        vertical_shifts = []
        presam_ints = []
        for i, z_translation in enumerate(z_translation_levels):
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
                tomo_stacks[i] = nxentry.instrument.detector.data.nxdata[
                    field_indices_masked]
                if nxentry.sample.presample_intensity.size:
                    presam_ints.append(
                        nxentry.sample.presample_intensity.nxdata[
                            field_indices_masked].reshape(-1, 1, 1))
            except Exception as exc:
                raise RuntimeError('Unable to load the tomography images '
                                   f'for stack {i}') from exc
            if not i:
                tomo_stack_shape = tomo_stacks[0].shape
            else:
                assert tomo_stacks[i].shape == tomo_stack_shape

        tomo_stack_shape = None
        for i in range(num_tomo_stacks):
            tomo_stack = tomo_stacks[i]
            if tomo_stack is None:
                continue
            # Resize the tomography images
            # Right now the range is the same for each set in the stack
            if (img_row_bounds != (0, tomo_stack.shape[1])
                    or img_column_bounds != (0, tomo_stack.shape[2])):
                tomo_stack = tomo_stack[:,
                    img_row_bounds[0]:img_row_bounds[1],
                    img_column_bounds[0]:img_column_bounds[1]]

            # Subtract dark field and normalize with bright field and
            # presample intensity
            if nxentry.sample.presample_intensity.size:
                presam_int = presam_ints[i]
                if tdf is not None:
                    try:
                        with SetNumexprThreads(self.num_proc):
                            evaluate(
                                '((tomo_stack-tdf)/presam_int)/tbf',
                                out=tomo_stack)
                    except TypeError as exc:
                        raise TypeError(
                            f'\nA {type(exc).__name__} occured while '
                            'subtracting the dark field and normalizing with '
                            'num_expr.evaluate(). Try reducing the detector '
                            'range (currently img_row_bounds = '
                            f'{img_row_bounds}, and img_column_bounds = '
                            f'{img_column_bounds})') from exc
                else:
                    try:
                        with SetNumexprThreads(self.num_proc):
                            evaluate(
                                '(tomo_stack//presam_int)/tbf', out=tomo_stack)
                    except TypeError as exc:
                        raise TypeError(
                            f'\nA {type(exc).__name__} occured while '
                            'normalizing with num_expr.evaluate(). Try '
                            'reducing the detector range (currently '
                            f'img_row_bounds = {img_row_bounds}, and '
                            f'img_column_bounds = {img_column_bounds})'
                        ) from exc
            elif tdf is not None:
                try:
                    with SetNumexprThreads(self.num_proc):
                        evaluate('(tomo_stack-tdf)/tbf', out=tomo_stack)
                except TypeError as exc:
                    raise TypeError(
                        f'\nA {type(exc).__name__} occured while subtracting '
                        'the dark field with num_expr.evaluate(). Try '
                        'reducing the detector range (currently '
                        f'img_row_bounds = {img_row_bounds}, and '
                        f'img_column_bounds = {img_column_bounds})') from exc

            # Save the slice if only one slice is reduced
            if tomo_stack.shape[1] == 1 and self.save_figures:
                self._figures.append(
                    (quick_imshow(
                        tomo_stack[:,0,:], title='Reduced intensity',
                        cmap='gray', colorbar=True, show_fig=False,
                        return_fig=True),
                    'reduced_intensity'))

            # Remove non-positive values and linearize data
            # RV make input argument? cutoff = 1.e-6
            with SetNumexprThreads(self.num_proc):
                cutoff = np.float32(1.e-6)
                evaluate(
                    'where(tomo_stack < cutoff, cutoff, tomo_stack)',
                    out=tomo_stack)
            with SetNumexprThreads(self.num_proc):
                evaluate('-log(tomo_stack)', out=tomo_stack)

            # Get rid of nans/infs that may be introduced by normalization
            tomo_stack[~np.isfinite(tomo_stack)] = 0

            # Remove stripes
            if (self.config.remove_stripe is not None
                    and self.config.remove_stripe):
                # Third party modules
                from tomopy.prep import stripe
                for method_name, kwargs in self.config.remove_stripe.items():
                    method = getattr(stripe, method_name)
                    self.logger.info(f'Running {method_name}')
                    tomo_stack = method(
                        tomo_stack,
                        ncore=kwargs.get('ncore', NUM_CORE_TOMOPY_LIMIT),
                        **kwargs)

            # Save the slice if only one slice is reduced
            if tomo_stack.shape[1] == 1 and self.save_figures:
                self._figures.append(
                    (quick_imshow(
                        tomo_stack[:,0,:], title='Reduced intensity',
                        cmap='gray', colorbar=True, show_fig=False,
                        return_fig=True),
                    'sinogram_after_remove_stripe'))

            # Combine resized stacks
            tomo_stacks[i] = tomo_stack
            if tomo_stack_shape is None:
                tomo_stack_shape = tomo_stack.shape
            else:
                assert tomo_stack_shape == tomo_stack.shape

        for i in range(num_tomo_stacks):
            if tomo_stacks[i] is None:
                tomo_stacks[i] = np.zeros(tomo_stack_shape)

        # Add tomo field info to reduced data NXprocess
        reduced_data.x_translation = horizontal_shifts
        reduced_data.x_translation.units = 'mm'
        reduced_data.z_translation = vertical_shifts
        reduced_data.z_translation.units = 'mm'
        reduced_data.data.tomo_fields = tomo_stacks
        reduced_data.data.attrs['signal'] = 'tomo_fields'

        if tdf is not None:
            del tdf
        del tbf
        del tomo_stacks

        return reduced_data

    def _set_delta_theta(self, thetas, delta_theta=None):
        """Set delta theta to reduce memory the requirement for the
        analysis.
        """
        # Local modules
        from CHAP.utils.general import index_nearest

        # For now eliminate from interactive use
        #if self.interactive:
        #    if delta_theta is None:
        #        delta_theta = thetas[1]-thetas[0]
        #    print(f'\nAvailable \u03b8 range: [{thetas[0]}, {thetas[-1]}]')
        #    print(f'Current \u03b8 interval: {delta_theta}')
        #    if input_yesno(
        #            'Do you want to change the \u03b8 interval to reduce the '
        #            'memory requirement (y/n)?', 'n'):
        #        delta_theta = input_num(
        #            '    Enter the desired \u03b8 interval',
        #            ge=thetas[1]-thetas[0], lt=(thetas[-1]-thetas[0])/2)
        if delta_theta is not None:
            delta_theta = index_nearest(thetas, thetas[0]+delta_theta)
            if delta_theta <= 1:
                delta_theta = None
        return delta_theta

    def _set_detector_bounds(
            self, nxentry, reduced_data, image_key, theta, img_row_bounds):
        """Set vertical detector bounds for each image stack. Right
        now the range is the same for each set in the image stack.
        """
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
        if num_tomo_stacks > 1 and (nxentry.instrument.source.attrs['station']
                in ('id1a3', 'id3a')):
            self.logger.warning('Ignoring parameter img_row_bounds '
                                 'for id1a3 and id3a for an image stack')
            img_row_bounds = None
        tbf = reduced_data.data.bright_field.nxdata
        if img_row_bounds is None:
            if nxentry.instrument.source.attrs['station'] in ('id1a3', 'id3a'):
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
                fit = FitProcessor(**self.run_config)
                model = {'model': 'rectangle',
                         'parameters': [
                             {'name': 'amplitude',
                              'value': row_sum.max()-row_sum.min(),
                              'min': 0.0},
                             {'name': 'center1', 'value': 0.25*num,
                                 'min': 0.0, 'max': num},
                             {'name': 'sigma1', 'value': num/7.0,
                              'min': sys.float_info.min},
                             {'name': 'center2', 'value': 0.75*num,
                              'min': 0.0, 'max': num},
                             {'name': 'sigma2', 'value': num/7.0,
                              'min': sys.float_info.min}]}
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
                self.logger.debug(f'delta_z = {delta_z}')
                num_row_min = int((delta_z + 0.5*pixel_size) / pixel_size)
                if num_row_min > tbf.shape[0]:
                    self.logger.warning(
                        'Image bounds and pixel size prevent seamless '
                        'stacking')
                    row_low = 0
                    row_upp = tbf.shape[0]
                else:
                    self.logger.debug(f'num_row_min = {num_row_min}')
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
            else:
                if num_tomo_stacks > 1:
                    raise NotImplementedError(
                        'Selecting image bounds or calibrating rotation axis '
                        'for multiple stacks on FMB')
                # For FMB: use the first tomography image to select range
                # RV revisit if they do tomography with multiple stacks
                if img_row_bounds is None and not self.interactive:
                    self.logger.warning(
                        'img_row_bounds unspecified, reduce data for '
                        'entire detector range')
                    img_row_bounds = (0, first_image.shape[0])
        buf, img_row_bounds = select_image_indices(
            first_image, 0, b=tbf, preselected_indices=img_row_bounds,
            title='Select detector image row bounds for data '\
                  f'reduction (in range [0, {first_image.shape[0]}])',
            title_a=r'Tomography image at $\theta$ = 'f'{round(theta, 2)+0}',
            title_b='Bright field',
            interactive=self.interactive, return_buf=self.save_figures)
        if (num_tomo_stacks > 1
                and (img_row_bounds[1]-img_row_bounds[0]+1)
                     < int((delta_z - 0.5*pixel_size) / pixel_size)):
            self.logger.warning(
                'Image bounds and pixel size prevent seamless stacking')

        # Save figure
        if self.save_figures:
            self._figures.append((buf, 'rotation_calibration_rows'))

        return img_row_bounds


class TomoFindCenterGui(Processor):
    """A processor that creates and opens a GUI to interactively find
    and return the calibrated center axis information from a set of
    reduced tomographic images.

    :ivar tk_root: tkinter root window.
    :vartype tk_root: tkinter.Tk
    :ivar tomo_stacks: Reduced image data stack(s).
    :vartype tomo_stacks: numpy.ndarray
    :ivar tbf: Bright field image data.
    :vartype tbf: numpy.ndarray
    :ivar thetas: Rotation angle of the images in each stack in
        degrees.
    :vartype thetas: numpy.ndarray
    :ivar img_row_bounds: Detector image bounds in the row-direction.
    :vartype img_row_bounds: [int, int]
    :ivar img_column_bounds: Detector image bounds in the
        column-direction
    :vartype img_column_bounds: [int, int]
    :ivar center_stack_index: Stack index of the tomography set to find
        the center axis.
    :vartype center_stack_index: int
    :ivar center_rows: Detector image row indices for the center
        finding processor.
    :vartype center_rows: list[int], optional
    :ivar num_center_rows: Number of rows to find the center at,
        defaults to the 2 or 1 if the reduced data stack only contains
        one row.
    :vartype num_center_rows: int, optional
    :ivar num_proc: Number of processors, defaults to `64`.
    :vartype num_proc: int, optional
    :ivar gaussian_sigma: Standard deviation for the
        `Gaussian filter <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.gaussian_filter>`__
        applied to image reconstruction visualizations, defaults to no
        filtering performed.
    :vartype gaussian_sigma: float, optional
    :ivar ring_width: Maximum ring width in pixels used to
        `filter ring artifacts <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.remove_ring>`__
        during image reconstruction, defaults to no ring removal performed.
    :vartype ring_width: float, optional
    """

    tk_root: Annotated[tk.Tk, SkipValidation]
    tomo_stacks: np.ndarray
    tbf: np.ndarray
    thetas: np.ndarray
    img_row_bounds: conlist(min_length=2, max_length=2, item_type=conint(ge=0))
    img_column_bounds: conlist(
        min_length=2, max_length=2, item_type=conint(ge=0))
    center_stack_index: conint(ge=0)
    center_rows: Optional[conlist(item_type=conint(ge=0))] = []
    num_center_rows: Optional[conint(gt=0)] = 2
    num_proc: Optional[conint(gt=0)] = 64
    gaussian_sigma: Optional[confloat(ge=0, allow_inf_nan=False)] = None
    ring_width: Optional[confloat(ge=0, allow_inf_nan=False)] = None

    _content: tk.Frame = PrivateAttr(default=None)
    _center_offsets: list = PrivateAttr(default=[])
    _range_x: tuple = PrivateAttr(default=None)
    _range_y: tuple = PrivateAttr(default=None)
    _recon_planes: list = PrivateAttr(default=[])
    _rects: list = PrivateAttr(default=[])
    _selected_rows: list = PrivateAttr(default=[])
    _selected_offset: tk.StringVar = PrivateAttr(default=None)
    _zoom_window: tuple = PrivateAttr(default=None)

    _fig_title: list = PrivateAttr(default=[])
    _error_text: list = PrivateAttr(default=[])

    _exclude = {'tk_root'}

    #FIX model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def center_offsets(self):
        """Return the selected centers at the specified or selected
        center finding rows.

        :type: list[float]
        """
        return self._center_offsets

    @property
    def recon_planes(self):
        """Return the reconstructed images at the specified or selected
        center finding row indices.

        :type: list[numpy.ndarray]
        """
        return self._recon_planes

    def __init__(self, tk_root=None, config=None):
        """Initialize TomoFindCenterGui.

        :param tk_root: tkinter root window.
        :type tk_root: tkinter.Tk
        :param config: Any keyword arguments to pass along to the
            base processor (:class:`~CHAP.processor.Processor`).
        :type config: dict
        """
        super().__init__(tk_root=tk_root, **config)

        # Initialize the main application window
        self.tk_root.title('Center axes calibration')
        self.tk_root.columnconfigure(0, weight=1)
        self.tk_root.rowconfigure(0, weight=6)

        # Build initial content frame
        if 1 <= self.img_row_bounds[1] - self.img_row_bounds[0] <= 2:
            self.num_center_rows = \
                self.img_row_bounds[1] - self.img_row_bounds[0]
            self.center_rows = list(range(self.num_center_rows))
            self._build_gui(
                self._find_center_offset_one_plane,
                self._on_confirm_find_center_offset_one_plane,
                num_row=21, num_column=7)
        else:
            self._build_gui(
                self._find_center_rows, self._on_confirm_find_center_rows,
                num_row=8, num_column=5)

        # Start the GUI event loop
        self.tk_root.mainloop()

    def _change_fig_title(self, title, plt, title_pos=None):
        if title_pos is None:
            title_pos = (0.5, 0.9)
        title_props = {'fontsize': 'xx-large',
                       'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        if self._fig_title:
            self._fig_title[0].remove()
            self._fig_title.pop()
        self._fig_title.append(plt.figtext(*title_pos, title, **title_props))

    def _change_error_text(self, error, plt, error_pos=None):
        if error_pos is None:
            error_pos = (0.5, 0.82)
        error_props = {'fontsize': 'x-large',
                       'horizontalalignment': 'center',
                       'verticalalignment': 'bottom'}
        if self._error_text:
            self._error_text[0].remove()
            self._error_text.pop()
        self._error_text.append(plt.figtext(*error_pos, error, **error_props))

    def _build_gui(self, task, confirm_callback, num_row=1, num_column=1):
        """Build the GUI."""
        # Third party modules
        import matplotlib.pyplot as plt

        # Clear out the old figure title and error text
        if self._fig_title:
            self._fig_title[0].remove()
            self._fig_title.pop()
        if self._error_text:
            self._error_text[0].remove()
            self._error_text.pop()

        # Clear out the old content frame
        if self._content:
            self._content.destroy()

        # Create the main content frame
        self._content = tk.Frame(self.tk_root)
        self._content.grid(row=0, column=0, sticky='nsew')
        for n_row in range(num_row):
            # pass weights, for now all equal
            self._content.rowconfigure(n_row, weight=1)
        for n_column in range(num_column):
            # pass weights, for now all equal
            self._content.columnconfigure(n_column, weight=1)

        # Setup the "Confirm" button
        confirm_text = tk.StringVar(value='Confirm')
        confirm_button = tk.Button(
            self._content, textvariable=confirm_text, height=3,
            command=lambda: confirm_callback(plt))
        confirm_button.grid(row=num_row-3, column=num_column-1,
                rowspan=3, padx=5, pady=5)

        # Run the task
        task(confirm_text, plt)

    def _on_confirm_find_center_rows(self, plt):
        """Callback function for the "Confirm" button during
        `_find_center_rows`.
        """
        if len(self._selected_rows) >= self.num_center_rows:
            plt.close()
            self.center_rows = tuple(sorted(self._selected_rows))
            self._build_gui(
                self._find_center_offset_one_plane,
                self._on_confirm_find_center_offset_one_plane,
                num_row=21, num_column=7)

    def _on_confirm_find_center_offset_one_plane(self, plt):
        """Callback function for the "Confirm" button during
        `_find_center_offset_one_plane`.
        """
        plt.close()
        self._center_offsets.append(float(self._selected_offset.get()))
        if len(self._center_offsets) < self.num_center_rows:
            self._build_gui(
                self._find_center_offset_one_plane,
                self._on_confirm_find_center_offset_one_plane,
                num_row=21, num_column=7)
        else:
            self.tk_root.destroy()  # Close the tkinter root window

    def _find_center_rows(self, confirm_text, plt):
        """Method used to create and interact with the select center
        rows GUI frame.
        """
        # Third party modules
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        def add_center_row(center_row):
            """Add a new row index to the list of center rows."""
            if center_row in self._selected_rows:
                raise ValueError('Ignoring duplicate of selected rows')
            self._selected_rows.append(int(center_row))
            for ax in axs:
                lines.append(ax.axhline(self._selected_rows[-1], c='r', lw=2))

        def get_selected_rows(plt, change_fnc):
            """Update the figure or error text in the GUI giving the
            currently selected center rows.
            """
            selected_rows = tuple(sorted(self._selected_rows))
            if len(selected_rows) == self.num_center_rows:
                self._change_fig_title(
                    'Click "Reset"/"Confirm" to change/confirm the selected '
                    'rows', plt)
                text = f'Selected rows: {selected_rows}'
            elif selected_rows:
                text = f'Selected row(s): {selected_rows}, select ' + \
                       f'{self.num_center_rows-len(selected_rows)} more'
            else:
                text = 'Enter the first row index in "Select row"'
            change_fnc(text, plt)
            return selected_rows

        def on_select_center_row(*args):
            """Callback function for the "Select center row" TextBox.
            """
            if self._error_text:
                self._error_text[0].remove()
                self._error_text.pop()
            input_str = entry.get()
            err = f'Invalid center_row ({input_str}), enter an integer ' + \
                f'between {self.img_row_bounds[0]} and ' + \
                f'{self.img_row_bounds[1]-1}'
            try:
                center_row = int(input_str)
                if (center_row < self.img_row_bounds[0]
                        or center_row >= self.img_row_bounds[1]):
                    raise ValueError
                if len(self._selected_rows) >= self.num_center_rows:
                    err = 'Exceeding the number of required rows ' + \
                        f'({self.num_center_rows}), click "Reset"/' + \
                        '"Confirm" to change/confirm the selected rows'
                    raise ValueError
            except ValueError:
                self._change_error_text(err, plt)
            else:
                try:
                    add_center_row(center_row)
                    get_selected_rows(plt, self._change_error_text)
                except ValueError as exc:
                    self._change_error_text(exc, plt)
            entry.delete(0, 'end')
            canvas.draw()

        def on_reset():
            """Callback function for the "Reset" button."""
            if self._error_text:
                self._error_text[0].remove()
                self._error_text.pop()
            for line in reversed(lines):
                line.remove()
            self._selected_rows.clear()
            lines.clear()
            self._change_fig_title(
                f'Select {self.num_center_rows} detector image rows to find '
                f'center axis (in range ([{self.img_row_bounds[0]}, '
                f'{self.img_row_bounds[1]-1}])', plt)
            get_selected_rows(plt, self._change_error_text)
            canvas.draw()

        lines = []

        data = self.tomo_stacks[self.center_stack_index,0,:,:]
        data_shape = data.shape
        assert self.tbf.shape == data_shape

        # Create the figure
        ratio = data_shape[0] / data_shape[1]
        if ratio > 1:
            fig, axs = plt.subplots(
                1, 2, figsize=(ratio*8.5+2, 8.5))
        else:
            fig, axs = plt.subplots(
                2, 1, figsize=(11, ratio*11+4))
        extent = (
            0, data_shape[1],
            self.img_row_bounds[0] + data_shape[0], self.img_row_bounds[0])
        axs[0].imshow(data, extent=extent, cmap='gray')
        axs[0].set_title(
            r'Tomography image at $\theta$ = 'f'{round(self.thetas[0], 2)+0}',
            fontsize='xx-large')
        axs[1].imshow(self.tbf, extent=extent, cmap='gray')
        axs[1].set_title('Bright field', fontsize='xx-large')
        if ratio > 1:
            axs[0].set_xlabel('column_label', fontsize='x-large')
            axs[0].set_ylabel('row_label', fontsize='x-large')
            axs[1].set_xlabel('column_label', fontsize='x-large')
        else:
            axs[0].set_ylabel('row_label', fontsize='x-large')
            axs[1].set_xlabel('column_label', fontsize='x-large')
            axs[1].set_ylabel('row_label', fontsize='x-large')
        for ax in axs:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        fig.subplots_adjust(bottom=0.0, top=0.8)

        # Setup the preselected rows
        for center_row in sorted(list(self.center_rows)):
            add_center_row(center_row)
        self._change_fig_title(
            f'Select {self.num_center_rows} detector image rows to find '
            f'center axis (in range ([{self.img_row_bounds[0]}, '
            f'{self.img_row_bounds[1]-1}])', plt)
        get_selected_rows(plt, self._change_error_text)

        # Setup the figure canvas
        canvas = FigureCanvasTkAgg(fig, master=self._content)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(
            row=0, column=0, rowspan=5, columnspan=4, sticky='nsew')

        # Setup selector
        label_text = tk.StringVar(value='Select row')
        label = tk.Label(self._content, textvariable=label_text)
        label.grid(row=6, column=0, sticky='e', padx=5, pady=5)
        entry = tk.Entry(self._content)
        entry.grid(row=6, column=1, sticky='w', padx=5, pady=5)
        entry.bind('<Return>', on_select_center_row)

        # Setup the "Reset" button
        reset_button = tk.Button(
            self._content, text='Reset', command=on_reset)
        reset_button.grid(row=0, column=4, padx=5)

    def _find_center_offset_one_plane(self, confirm_text, plt):
        """Method used to create and interact with the find center
        offset GUI frame for a given row index.
        """
        # System modules
        from copy import deepcopy

        # Third party modules
        from tomopy import find_center_vo
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.widgets import RectangleSelector

        def draw_images(current_images):
            """(Re)draw the plots in the current graph window."""
            imslices = np.asarray(current_images)[
                    :,slice(*self._range_y),slice(*self._range_x)]
            vmin = imslices.min()
            vmax = imslices.max()
            for i in range(num_plot+num_plot_extra):
                axs[i].set_xlim(*self._zoom_window[0])
                axs[i].set_ylim(*self._zoom_window[1])
                ims[i].set_clim(vmin, vmax)
            cbar[0].remove()
            cbar.pop(0)
            fig.subplots_adjust(bottom=0.0, top=0.88)
            cbar.append(fig.colorbar(
                ims[0], ax=axs, pad=0.03, shrink=0.5, location='bottom'))
            canvas.draw()

        def on_rect_select(eclick, erelease):
            """Callback function for the RectangleSelector widget."""
            self._range_x = (
                max(0, int(eclick.xdata+1)),
                min(image_shape[1], int(erelease.xdata)+1))
            self._range_y = (
                max(0, int(eclick.ydata+1)),
                min(image_shape[0], int(erelease.ydata)+1))
            self._zoom_window = ((self._range_x[0]-0.5, self._range_x[1]-0.5),
                                 (self._range_y[1]-0.5, self._range_y[0]-0.5))
            draw_images(images)

        def on_reset():
            """Callback function for the "Reset" button."""
            for i in range(num_plot+num_plot_extra):
                offset_choices[i] = deepcopy(offset_choices_original[i])
                images[i] = deepcopy(images_original[i])
                ims[i] = axs[i].imshow(images[i], cmap='gray')
                if num_plot_extra and i == num_plot:
                    axs[i].set_title(
                        f'previous row: {previous_center_row}, '
                        f'offset: {offset_choices[i]}')
                else:
                    axs[i].set_title(f'offset: {offset_choices[i]}')
                    rb[i].configure(
                        text=f'{float(offset_choices[i]):.1f}',
                        value=offset_choices[i])
            on_zoom_out()

        def on_select(event):
            """Callback function for the "Select offset" TextBoxes."""
            try:
                index = entries.index(event.widget)
                value = float(f'{float(event.widget.get()):.1f}')
                if not -center_offset_range < value < center_offset_range:
                    raise ValueError
            except ValueError:
                value = None
            if value is not None and value not in offset_choices[
                    :len(offset_choices)-num_plot_extra]:
                offset_choice_save = offset_choices[index]
                image_save = images[index]
                try:
                    offset_choices[index] = value
                    images[index] = self.reconstruct_planes(
                        sinogram, value, np.radians(self.thetas),
                        num_proc=self.num_proc,
                        gaussian_sigma=self.gaussian_sigma,
                        ring_width=self.ring_width)
                except ValueError as exc:
                    self._change_error_text(exc, plt)
                    offset_choices[index] = offset_choice_save
                    images[index] = image_save
                ims[index] = axs[index].imshow(images[index], cmap='gray')
                axs[index].set_title(f'offset: {offset_choices[index]}')
                axs[index].set_xlim(*self._zoom_window[0])
                axs[index].set_ylim(*self._zoom_window[1])
                rb[index].configure(text=f'{float(value):.1f}', value=value)
            draw_images(images)
            event.widget.delete(0, 'end')

        def on_select_offset(*args):
            """Callback function for the selected offset radio
            buttons.
            """
            offset = self._selected_offset.get()
            self._recon_planes[-1] = images[
               offset_choices.index(float(offset))]
            confirm_text.set(f'Confirm  {offset}')

        def on_zoom_out():
            """Callback function for the "Zoom Out" button."""
            self._range_x = (0, image_shape[1])
            self._range_y = (0, image_shape[0])
            self._zoom_window = ((self._range_x[0]-0.5, self._range_x[1]-0.5),
                                 (self._range_y[1]-0.5, self._range_y[0]-0.5))
            draw_images(images)

        num_plot = 5

        # Get the sinogram for the selected plane
        current_center_row = self.center_rows[len(self._center_offsets)]
        sinogram = self.tomo_stacks[
            self.center_stack_index,:,
            current_center_row-self.img_row_bounds[0],:]
        sinogram_shape = sinogram.shape
        center_offset_range = sinogram_shape[1]/2

        # Get the sinogram for the previous plane:
        if 0 < len(self._center_offsets) < self.num_center_rows:
            num_plot_extra = 1
            previous_center_row = self.center_rows[len(self._center_offsets)-1]
            previous_sinogram = self.tomo_stacks[
                self.center_stack_index,:,
                previous_center_row-self.img_row_bounds[0],:]
            previous_center_offset = self._center_offsets[-1]
        else:
            num_plot_extra = 0
            previous_center_row = None
            previous_sinogram = None
            previous_center_offset = None

        # Create the figure
        fig = plt.figure(figsize=(14, 10))
        self._change_fig_title(
            'Select the calibrated center axis offset for row '
            f'{current_center_row}', plt, title_pos = (0.5, 0.95))

        # Setup the figure canvas
        canvas = FigureCanvasTkAgg(fig, master=self._content)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(
            row=0, column=0, rowspan=18, columnspan=6, sticky='nsew')

        # Setup selectors
        tk.Label(self._content, text='Change any of the offsets:').grid(
            row=18, column=0, sticky='w', padx=5, pady=5)
        labels = []
        entries = []
        places = ['Upper left:', 'Upper middle:', 'Upper right:',
                  'Lower left:', 'Lower middle:']
        for i in range(num_plot):
            labels.append(tk.Label(self._content, text=places[i]))
            labels[i].grid(
                row=19+i//3, column=2*(i%3), sticky='e', padx=5, pady=5)
            entries.append(tk.Entry(self._content))
            entries[i].grid(
                row=19+i//3, column=2*(i%3)+1, sticky='w', padx=5, pady=5)
            entries[i].bind('<Return>', on_select)
        entries[0].focus_set()

        # Setup the "Reset" button
        reset_button = tk.Button(
            self._content, text='Reset', command=on_reset)
        reset_button.grid(row=0, column=6, padx=5)

        # Try Nghia Vo's method to find the center
#        if center_offset_min is None:
#            center_offset_min = -50
#        if center_offset_max is None:
#            center_offset_max = 50
        # RV FIX num_proc
        tomo_center = find_center_vo(
            sinogram, ncore=min(1, NUM_CORE_TOMOPY_LIMIT),
            smin=-50, smax=50)
            #smin=center_offset_min, smax=center_offset_max)
        center_offset_vo = round(float(tomo_center-center_offset_range), 1)
        center_offset_vo_text = f'{center_offset_vo:.1f}'

        # Reconstruct the plane for Nghia Vo's center offset
        thetas = np.radians(self.thetas)
        recon_plane_vo = self.reconstruct_planes(
            sinogram, center_offset_vo, thetas, num_proc=self.num_proc,
            gaussian_sigma=self.gaussian_sigma, ring_width=self.ring_width)
        self._recon_planes.append(recon_plane_vo)

        # Reconstruct plane for center offset on both sides of Vo's
        delta = 1.0 # Could become an input parameter
        offset_choices = [
            center_offset_vo-2*delta, center_offset_vo-delta, center_offset_vo,
            center_offset_vo+delta, center_offset_vo+2*delta]
        images = []
        for offset in offset_choices:
            if offset == center_offset_vo:
                images.append(recon_plane_vo)
            else:
                images.append(self.reconstruct_planes(
                    sinogram, offset, thetas, num_proc=self.num_proc,
                    gaussian_sigma=self.gaussian_sigma,
                    ring_width=self.ring_width))
        if num_plot_extra:
            offset_choices.append(previous_center_offset)
            images.append(self.reconstruct_planes(
                previous_sinogram, previous_center_offset, thetas,
                num_proc=self.num_proc, gaussian_sigma=self.gaussian_sigma,
                ring_width=self.ring_width))
        image_shape = images[0].shape
        self._range_x = (0, image_shape[1])
        self._range_y = (0, image_shape[0])
        self._zoom_window = ((self._range_x[0]-0.5, self._range_x[1]-0.5),
                             (self._range_y[1]-0.5, self._range_y[0]-0.5))
        offset_choices_original = deepcopy(offset_choices)
        images_original = deepcopy(images)

        # Add the reconstructions to the figure
        axs = []
        ims = []
        for i in range(num_plot+num_plot_extra):
            axs.append(fig.add_subplot(2, 3, i+1))
            ims.append(axs[i].imshow(images[i], cmap='gray'))
            if num_plot_extra and i == num_plot:
                axs[i].set_title(
                    f'previous row: {previous_center_row}, '
                    f'offset: {offset_choices[i]}')
            else:
                axs[i].set_title(f'offset: {offset_choices[i]}')
            axs[i].set_axis_off()
        self._change_error_text(
            f'Center axis offset obtained with Nghia Vo\'s method: '
            f'{center_offset_vo_text}', plt, error_pos=(0.5, 0.91))
        fig.subplots_adjust(bottom=0.0, top=0.88)
        cbar = [fig.colorbar(
            ims[0], ax=axs, pad=0.03, shrink=0.5, location='bottom')]

        # Setup the figure "Zoom" function
        rect_props = {
            'alpha': 0.5, 'facecolor': 'tab:blue', 'edgecolor': 'blue'}
        self._rects = []
        for i in range(num_plot+num_plot_extra):
            self._rects.append(
                RectangleSelector(
                    axs[i], on_rect_select, props=rect_props, useblit=True,
                    minspanx=2, minspany=2))

        # Setup the "Zoom Out" button
        zoom_button = tk.Button(
            self._content, text='Zoom Out', command=on_zoom_out)
        zoom_button.grid(row=1, column=6, padx=5)

        # Setup the selected offset radio buttons
        self._selected_offset = tk.StringVar(value=f'{center_offset_vo_text}')
        self._selected_offset.trace_add('write', on_select_offset)
        choices_label = tk.Label(self._content, text='Choose offset:')
        choices_label.grid(row=3, column=6, padx=5, sticky='w')
        rb = []
        for i in range(num_plot):
            rb.append(tk.Radiobutton(
                self._content, text=f'{offset_choices[i]:.1f}',
                variable=self._selected_offset, value=offset_choices[i],
                command=None))
            rb[i].grid(row=4+i, column=6, padx=5, sticky='w')
        confirm_text.set(f'Confirm {center_offset_vo_text}')

    @staticmethod
    def reconstruct_planes(
            tomo_planes, center_offset, thetas, *, num_proc=1,
            gaussian_sigma=None, ring_width=None):
        """Invert the sinogram for a single or multiple tomography
        planes using tomopy's recon routine.

        :param tomo_planes: The (set of) sinogram(s).
        :type tomo_planes: numpy.ndarray
        :param center_offset: Rotation center axis for the current
            sinograms in `tomo_planes`.
        :type center_offset: int, list[int]
        :param thetas: Rotation angles in degrees for each sinogram in
            `tomo_planes`.
        :type thetas: numpy.ndarray
        :param num_proc: Number of processors, defaults to `1`.
        :type num_proc: int, optional
        :param gaussian_sigma: Standard deviation for the
            `Gaussian filter <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.gaussian_filter>`__
            applied to image reconstruction visualizations, defaults to
            no filtering performed.
        :type gaussian_sigma: float, optional
        :param ring_width: Maximum ring width in pixels used to
            `filter ring artifacts <https://tomopy.readthedocs.io/en/stable/api/tomopy.misc.corr.html#tomopy.misc.corr.remove_ring>`__
            during image reconstruction, defaults to no ring removal
            performed.
        :type ring_width: float, optional
        :raises ValueError: Invalid `center_offset` input parameter.
        :return: Reconstructed plane(s)
        :rtype: numpy.ndarray
        """
        # Third party modules
        from scipy.ndimage import gaussian_filter
        from tomopy import (
            misc,
            recon,
        )

        # Reconstruct the planes
        # tomo_planes axis data order: (row,)theta,column
        # thetas in radians
        if is_num(center_offset):
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
            algorithm='gridrec', ncore=num_proc)

        # Performing Gaussian filtering and removing ring artifacts
        if gaussian_sigma is not None and gaussian_sigma:
            recon_planes = gaussian_filter(
                recon_planes, gaussian_sigma, mode='nearest')
        if ring_width is not None and ring_width:
            recon_planes = misc.corr.remove_ring(
                recon_planes, rwidth=ring_width, ncore=num_proc)

        # Apply a circular mask
        recon_planes = misc.corr.circ_mask(recon_planes, axis=0) #RV

        return np.squeeze(recon_planes)


class TomoFindCenterProcessor(Processor):
    """A processor to find and return the calibrated center axis
    information from a set of reduced tomographic images. In addition,
    it returns an optional list of byte stream representions of
    Matplotlib figures, and the metadata associated with the center
    calibration step.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.tomo.models.TomoFindCenterConfig`.
    :vartype config: dict, optional
    :ivar num_proc: Number of processors, defaults to `64`.
    :vartype num_proc: int, optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    :ivar save_figures: Create Matplotlib figures that can be saved to
        file downstream in the workflow, defaults to `True`.
    :vartype save_figures: bool, optional
    """

    pipeline_fields: dict = Field(
        default = {'config': 'tomo.models.TomoFindCenterConfig'},
        init_var=True)
    config: Optional[TomoFindCenterConfig] = TomoFindCenterConfig()
    num_proc: Optional[conint(gt=0)] = 64
    nxmemory: Optional[conint(gt=0)] = 100000
    save_figures: Optional[bool] = True

    _figures: list = PrivateAttr(default=[])

    def process(self, data):
        """Find the calibrated center axis information

        :param data: Input data containing the reduced data as a
            NeXus style
            `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
            object and optionally a center axis calibration processor
            configuration.
        :type data: list[PipelineData]
        :raises ValueError: Unable to find valid reduced data in input
            data.
        :return: Metadata associated with the workflow, a list of byte
            stream representions of Matplotlib figures, and the
            calibrated center axis information.
        :rtype: PipelineData, PipelineData, PipelineData
        """
        # Third party modules
        from nexusformat.nexus import nxsetconfig

        self.logger.info('Find the calibrated center axis information')

        # FIX make a config input
        nxsetconfig(memory=self.nxmemory)

        # Load the reduced tomography data
        nxroot = self.get_data(data, remove=False)
        nxentry = self.get_default_nxentry(nxroot)

        # Get thetas (in degrees)
        thetas = nxentry.reduced_data.rotation_angle.nxdata

        # Get full bright field
        tbf = nxentry.reduced_data.data.bright_field.nxdata

        # Get image bounds and default image stack index and center
        # rows plus offsets
        img_row_bounds = nxentry.reduced_data.get(
            'img_row_bounds', (0, tbf.shape[0]))
        img_row_bounds = (int(img_row_bounds[0]), int(img_row_bounds[1]))
        img_column_bounds = nxentry.reduced_data.get(
            'img_column_bounds', (0, tbf.shape[1]))
        img_column_bounds = (
            int(img_column_bounds[0]), int(img_column_bounds[1]))
        num_tomo_stacks = nxentry.reduced_data.data.tomo_fields.shape[0]
        if num_tomo_stacks == 1:
            center_stack_index = 0
        elif self.config.center_stack_index is not None:
            center_stack_index = self.config.center_stack_index
        else:
            center_stack_index = num_tomo_stacks//2
        self.config.center_stack_index = center_stack_index
        if 1 <= img_row_bounds[1] - img_row_bounds[0] <= 2:
            center_rows = list(range(img_row_bounds[1] - img_row_bounds[0]))
        else:
            center_rows = self.config.center_rows
            if center_rows is None:
                if num_tomo_stacks == 1:
                    # Add a small margin to avoid edge effects
                    offset = min(
                        5,
                        int(0.1*(img_row_bounds[1] - img_row_bounds[0])))
                    center_rows = (
                        img_row_bounds[0] + offset,
                        img_row_bounds[1] - 1 - offset)
                else:
                    self.logger.info(
                        'center_rows unspecified, find center_rows at reduced '
                        'data bounds')
                    center_rows = (img_row_bounds[0], img_row_bounds[1]-1)
            elif center_rows[1] == img_row_bounds[1]:
                center_rows = (center_rows[0], center_rows[1]-1)
        self.config.center_rows = list(center_rows)

        # Calibrate the center axis
        if self.interactive:
            # Create the center finding GUI to allow the user to
            # interactively choose the center rows and find the
            # optimal center axis
            gui_config = {
                'tomo_stacks': nxentry.reduced_data.data.tomo_fields.nxdata,
                'tbf': tbf[img_row_bounds[0]:img_row_bounds[1],
                           img_column_bounds[0]:img_column_bounds[1]],
                'thetas': thetas,
                'img_row_bounds': img_row_bounds,
                'img_column_bounds': img_column_bounds,
                'center_stack_index': self.config.center_stack_index,
                'center_rows':self.config.center_rows,
                'gaussian_sigma':self.config.gaussian_sigma,
                'ring_width':self.config.ring_width,
            }
            recon_planes = self._find_center_gui(config=gui_config)
        else:
            recon_planes = self._find_center(
                nxentry.reduced_data.data.tomo_fields, img_row_bounds[0],
                np.radians(thetas))

        if self.save_figures:
            # Create and save center rows figure
            buf, _ = select_image_indices(
                nxentry.reduced_data.data.tomo_fields[
                    self.config.center_stack_index,0,:,:],
                0,
                b=tbf[img_row_bounds[0]:img_row_bounds[1],
                    img_column_bounds[0]:img_column_bounds[1]],
                preselected_indices=self.config.center_rows,
                axis_index_offset=img_row_bounds[0],
                title_a=r'Tomography image at $\theta$ = '
                        f'{round(thetas[0], 2)+0}',
                title_b='Bright field',
                interactive=False, return_buf=True)
            self._figures.append((buf, 'center_finding_rows'))

            # Create and save reconstructed plane figures
            for row, offset, recon_plane in zip(
                    self.config.center_rows, self.config.center_offsets,
                    recon_planes):
                self._figures.append(
                    (quick_imshow(
                        recon_plane,
                        title=f'Reconstruction for row {row} and center '
                              f'offset: {offset}',
                        cmap='gray', colorbar=True, show_fig=False,
                        return_fig=True),
                    f'reconstruction_row_{row}_offset_{offset}'))

        # Update metadata and provenance
        metadata, provenance = create_metadata_provenance(
            'tomo_center',
            data,
            user_metadata={'findcenter': self.config.model_dump()},
            logger=self.logger)

        return (
            PipelineData(
                name=self.name, data=metadata,
                schema='foxden.reader.FoxdenMetadataReader'),
            PipelineData(
                name=self.name, data=provenance,
                schema='foxden.reader.FoxdenProvenanceReader'),
            PipelineData(
                name=self.name, data=self._figures,
                schema='common.write.ImageWriter'),
            PipelineData(
                name=self.name, data=self.config.model_dump(),
                schema='tomo.models.TomoFindCenterConfig'))

    def _find_center(self, tomo_stack, row_offset, thetas):
        """Find calibrated center axis using Nghia Vo's method

        tomo_stacks data axes order: stack,theta,row,column
        thetas in radians
        """
        # Third party modules
        from tomopy import find_center_vo

        # Use Nghia Vo's method to find the optimal center axis
        recon_planes = []
        self.config.center_offsets = []
        for row in self.config.center_rows:
            sinogram = tomo_stack[
                self.config.center_stack_index,:,row-row_offset,:]
            #RV FIX num_core
            tomo_center = find_center_vo(
                sinogram, ncore=min(1, NUM_CORE_TOMOPY_LIMIT),
                smin=-50, smax=50)
                #smin=center_offset_min, smax=center_offset_max)
            offset_vo = round(float(tomo_center-sinogram.shape[1]/2), 1)
            self.config.center_offsets.append(offset_vo)
            if self.save_figures:
                recon_planes.append(TomoFindCenterGui.reconstruct_planes(
                    sinogram, offset_vo, thetas, num_proc=self.num_proc,
                    gaussian_sigma=self.config.gaussian_sigma,
                    ring_width=self.config.ring_width))
        return recon_planes

    def _find_center_gui(self, config):
        """Find calibrated center axis interactively

        tomo_stacks data axes order: stack,theta,row,column
        thetas in radians
        """
        # System modules
        from copy import deepcopy

        # Third party modules
        from tkinter import messagebox

        def on_closing():
            if messagebox.askyesno(
                    'Exit Confirmation',
                    'Are you sure you want to quit? This will kill CHAP! '
                    'Use the `Confirm` button to accept your choice and close '
                    'the GUI.', default='no'):
                tk_root.destroy()
                raise SystemExit

        config_save = deepcopy(config)
        try:
            # Initialize the main application window
            tk_root = tk.Tk()

            # Create the center calibration GUI within the main window
            tk_root.protocol("WM_DELETE_WINDOW", on_closing)
            app = TomoFindCenterGui(tk_root=tk_root, config=config_save)

            tk_root.mainloop()

            assert len(app.center_offsets) == app.num_center_rows
        except (AssertionError, KeyboardInterrupt, SystemExit) as exc:
            raise exc
        self.config.center_stack_index = app.center_stack_index
        self.config.center_rows = list(app.center_rows)
        self.config.center_offsets = list(app.center_offsets)
        return app.recon_planes


class TomoReconstructProcessor(Processor):
    """A processor to reconstruct a set of reduced images returning
    a NeXus style
    `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
    object containing the reconstructed data, an optional list of byte
    stream representions of Matplotlib figures, and the metadata
    associated with the data reduction step.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.tomo.models.TomoReconstructConfig`.
    :vartype config: dict, optional
    :ivar center_config: Center axis calibration configuration.
    :vartype center_config: dict, optional
    :ivar num_proc: Number of processors, defaults to `64`.
    :vartype num_proc: int, optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    :ivar save_figures: Create Matplotlib figures that can be saved to
        file downstream in the workflow, defaults to `True`.
    :vartype save_figures: bool, optional
    """

    pipeline_fields: dict = Field(
        default = {'config': 'tomo.models.TomoReconstructConfig',
                   'center_config': 'tomo.models.TomoFindCenterConfig'},
        init_var=True)
    config: Optional[TomoReconstructConfig] = TomoReconstructConfig()
    center_config: Optional[TomoFindCenterConfig] = TomoFindCenterConfig()
    num_proc: Optional[conint(gt=0)] = 64
    nxmemory: Optional[conint(gt=0)] = 100000
    save_figures: Optional[bool] = True

    _figures: list = PrivateAttr(default=[])

    def process(self, data):
        """Reconstruct the tomography data.

        :param data: Input data containing the reduced data as a
            NeXus style
            `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
            object, the center axis information and optionally a
            reconstruct data processor configuration.
        :type data: list[PipelineData]
        :raises RuntimeError: Dimension mismatch in `center_offsets`.
        :raises ValueError: Invalid input or configuration parameter.
        :return: Metadata associated with the workflow, a list of byte
            stream representions of Matplotlib figures, and the result
            of the data reconstruction.
        :rtype: PipelineData, PipelineData, PipelineData
        """
        # System modules
        from multiprocessing import cpu_count

        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXprocess,
            nxsetconfig,
        )

        self.logger.info('Reconstruct the tomography data')

        # FIX make a config input
        nxsetconfig(memory=self.nxmemory)

        # Load the reduced tomography data
        nxroot = self.get_data(data)
        nxentry = self.get_default_nxentry(nxroot)

        # Check if reduced data is available
        if 'reduced_data' not in nxentry:
            raise ValueError(f'Unable to find valid reduced data in {nxentry}.')

        # Check the number of processors
        if self.num_proc > cpu_count():
            self.logger.warning(
                f'num_proc = {self.num_proc} is larger than the number '
                f'of available processors and reduced to {cpu_count()}')
            self.num_proc = cpu_count()
        # Tompy py uses numexpr with NUMEXPR_MAX_THREADS = 64
        if self.num_proc > 64:
            self.logger.warning(
                f'num_proc = {self.num_proc} is larger than the number '
                f'of processors suitable to Tomopy and reduced to 64')
            self.num_proc = 64

        # Create a NXprocess to store image reconstruction (meta)data
        nxprocess = NXprocess()

        # Get calibrated center axis rows and centers
        center_rows = self.center_config.center_rows
        center_offsets = self.center_config.center_offsets
        if center_rows is None or center_offsets is None:
            self.logger.warning(
                'Unable to find valid calibrated center axis info in '
                f'{self.center_config}, try getting it from metadata')
            try:
                metadata, _ = read_metadata_provenance(
                    data, self.logger, remove=False)
                self.center_config = TomoFindCenterConfig(
                    **metadata['user_metadata']['findcenter'])
                center_rows = self.center_config.center_rows
                center_offsets = self.center_config.center_offsets
            except Exception:
                metadata = {}
            if center_rows is None or center_offsets is None:
                raise ValueError(
                    'Unable to find valid calibrated center axis info from '
                    'metadata {metadata}')
        if len(center_rows) == 1:
            center_slope = 0.0
        else:
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
            if len(center_rows) == 1:
                assert 0 <= center_rows[0] < tomo_stack.shape[0]
            else:
                assert (0 <= center_rows[0] < center_rows[1]
                        < tomo_stack.shape[0])
                center_offsets = [
                    center_offsets[0]-center_rows[0]*center_slope,
                    center_offsets[1] + center_slope * (
                        tomo_stack.shape[0]-1-center_rows[1]),
                ]
            t0 = time()
            tomo_recon_stack = self._reconstruct_one_tomo_stack(
                tomo_stack, np.radians(thetas), center_offsets=center_offsets,
                num_proc=self.num_proc, algorithm='gridrec',
                secondary_iters=self.config.secondary_iters,
                gaussian_sigma=self.config.gaussian_sigma,
                #remove_stripe_sigma=self.config.remove_stripe_sigma,
                ring_width=self.config.ring_width)
            self.logger.info(
                f'Reconstruction of stack {i} took {time()-t0:.2f} seconds')

            # Combine stacks
            tomo_recon_stacks.append(tomo_recon_stack)

        # Resize the reconstructed tomography data
        # - reconstructed axis data order in each stack: row/-z,y,x
        tomo_recon_shape = tomo_recon_stacks[0].shape
        x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
            tomo_recon_stacks, x_bounds=self.config.x_bounds,
            y_bounds=self.config.y_bounds, z_bounds=self.config.z_bounds)
        self.config.x_bounds = None if x_bounds is None else list(x_bounds)
        self.config.y_bounds = None if y_bounds is None else list(y_bounds)
        self.config.z_bounds = None if z_bounds is None else list(z_bounds)
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
        tomo_recon_stacks = np.asarray(tomo_recon_stacks)[:,
            z_range[0]:z_range[1],y_range[0]:y_range[1],x_range[0]:x_range[1]]

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

            # Save a few reconstructed image slices
            if self.save_figures:
                x_index = x_slice-x_range[0]
                title = f'recon x={x[x_index]:.4f}'
                self._figures.append(
                    (quick_imshow(
                        tomo_recon_stack[:,:,x_index], title=title,
                        origin='lower', extent=(y[0], y[-1], z[0], z[-1]),
                        cmap='gray', show_fig=False, return_fig=True),
                    re.sub(r'\s+', '_', title)))
                y_index = y_slice-y_range[0]
                title = f'recon y={y[y_index]:.4f}'
                self._figures.append(
                    (quick_imshow(
                        tomo_recon_stack[:,y_index,:], title=title,
                        origin='lower', extent=(x[0], x[-1], z[0], z[-1]),
                        cmap='gray', show_fig=False, return_fig=True),
                    re.sub(r'\s+', '_', title)))
                z_index = z_slice-z_range[0]
                title = f'recon z={z[z_index]:.4f}'
                self._figures.append(
                    (quick_imshow(
                        tomo_recon_stack[z_index,:,:], title=title,
                        origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
                        cmap='gray', show_fig=False, return_fig=True),
                    re.sub(r'\s+', '_', title)))
        else:
            # Save a few reconstructed image slices
            if self.save_figures:
                for i in range(tomo_recon_stacks.shape[0]):
                    basetitle = f'recon stack {i}'
                    title = f'{basetitle} xslice{x_slice}'
                    self._figures.append(
                        (quick_imshow(
                            tomo_recon_stacks[i,:,:,x_slice-x_range[0]],
                            title=title, show_fig=False, return_fig=True,
                            cmap='gray'),
                        re.sub(r'\s+', '_', title)))
                    title = f'{basetitle} yslice{y_slice}'
                    self._figures.append(
                        (quick_imshow(
                            tomo_recon_stacks[i,:,y_slice-y_range[0],:],
                            title=title, show_fig=False, return_fig=True,
                            cmap='gray'),
                        re.sub(r'\s+', '_', title)))
                    title = f'{basetitle} zslice{z_slice}'
                    self._figures.append(
                        (quick_imshow(
                            tomo_recon_stacks[i,z_slice-z_range[0],:,:],
                            title=title, show_fig=False, return_fig=True,
                            cmap='gray'),
                        re.sub(r'\s+', '_', title)))

        # Add image reconstruction to reconstructed data NXprocess
        # reconstructed axis data order:
        # - for one stack: z,y,x
        # - for multiple stacks: row/-z,y,x
        for k, v in self.center_config.model_dump().items():
            if k == 'center_stack_index':
                nxprocess[k] = v
            if k in ('center_rows', 'center_offsets'):
                nxprocess[k] = v
                nxprocess[k].units = 'pixels'
            if k == 'center_rows':
                nxprocess[k] = v
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
        nxentry = self.get_default_nxentry(nxroot)
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

        # Add the center info to the new NeXus object

        # Update metadata and provenance
        metadata, provenance = create_metadata_provenance(
            'tomo_reconstruct',
            data,
            user_metadata={'reconstructed_data': self.config.model_dump()},
            logger=self.logger)
        nxentry.reconstructed_data.attrs['did'] = metadata.get('did')
        nxentry.reconstructed_data.attrs['parent_did'] = \
            metadata.get('parent_did')

        return (
            PipelineData(name=self.name, data=metadata,
                schema='foxden.reader.FoxdenMetadataReader'),
            PipelineData(
                name=self.name, data=provenance,
                schema='foxden.reader.FoxdenProvenanceReader'),
            PipelineData(
                name=self.name, data=self._figures,
                schema='common.write.ImageWriter'),
            PipelineData(name=self.name, data=nxroot, schema='tomodata'))

    def _reconstruct_one_tomo_stack(
            self, tomo_stack, thetas, *, center_offsets=None, num_proc=1,
            algorithm='gridrec', secondary_iters=0, gaussian_sigma=None,
            ring_width=None):
            #remove_stripe_sigma=None, ring_width=None):
        """Reconstruct a single tomography stack."""
        # Third party modules
        from tomopy import (
            astra,
            misc,
            #prep,
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
            if len(center_offsets) != tomo_stack.shape[0]:
                raise RuntimeError(
                    'center_offsets dimension mismatch in '
                    'reconstruct_one_tomo_stack')
            centers = center_offsets
        centers = np.asarray(centers) + tomo_stack.shape[2]/2

        # Remove horizontal stripe
        # RV prep.stripe.remove_stripe_fw seems flawed for hollow brick
        # accross multiple stacks
        #if remove_stripe_sigma is not None and remove_stripe_sigma:
        #    if num_proc > NUM_CORE_TOMOPY_LIMIT:
        #        tomo_stack = prep.stripe.remove_stripe_fw(
        #            tomo_stack, sigma=remove_stripe_sigma,
        #            ncore=NUM_CORE_TOMOPY_LIMIT)
        #    else:
        #        tomo_stack = prep.stripe.remove_stripe_fw(
        #            tomo_stack, sigma=remove_stripe_sigma, ncore=num_proc)

        # Perform initial image reconstruction
        self.logger.debug('Performing initial image reconstruction')
        t0 = time()
        tomo_recon_stack = recon(
            tomo_stack, thetas, centers, sinogram_order=True,
            algorithm=algorithm, ncore=num_proc)
        self.logger.info(
            f'Performing initial image reconstruction took {time()-t0:.2f} '
            'seconds')

        # Run optional secondary iterations
        if secondary_iters > 0:
            self.logger.debug(
                f'Running {secondary_iters} secondary iterations')
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
                ncore=num_proc)
            self.logger.info(
                f'Performing secondary iterations took {time()-t0:.2f} '
                'seconds')

        # Remove ring artifacts
        if ring_width is not None and ring_width:
            misc.corr.remove_ring(
                tomo_recon_stack, rwidth=ring_width, out=tomo_recon_stack,
                ncore=num_proc)

        # Performing Gaussian filtering
        if gaussian_sigma is not None and gaussian_sigma:
            tomo_recon_stack = misc.corr.gaussian_filter(
                tomo_recon_stack, sigma=gaussian_sigma, ncore=num_proc)

        return tomo_recon_stack

    def _resize_reconstructed_data(
            self, data, *, x_bounds=None, y_bounds=None, z_bounds=None,
            combine_data=False):
        """Resize the reconstructed tomography data."""
        # Data order: row/-z,y,x or stack,row/-z,y,x
        if isinstance(data, list):
            num_tomo_stacks = len(data)
            for i in range(num_tomo_stacks):
                assert data[i].ndim == 3
                if i:
                    assert data[i].shape[1:] == data[0].shape[1:]
            tomo_recon_stacks = data
        else:
            assert data.ndim == 3
            num_tomo_stacks = 1
            tomo_recon_stacks = [data]

        # Selecting x and y bounds (in z-plane)
        if x_bounds is None:
            if not self.interactive:
                self.logger.warning('x_bounds unspecified, use data for '
                                     'full x-range')
                x_bounds = (0, tomo_recon_stacks[0].shape[2])
        elif not is_int_pair(
                x_bounds, ge=0, le=tomo_recon_stacks[0].shape[2]):
            raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
        if y_bounds is None:
            if not self.interactive:
                self.logger.warning('y_bounds unspecified, use data for '
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
        buf, roi = select_roi_2d(
            tomosum, preselected_roi=preselected_roi,
            title_a='Reconstructed data summed over z',
            row_label='y', column_label='x',
            interactive=self.interactive, return_buf=self.save_figures)
        if self.save_figures:
            if combine_data:
                filename = 'combined_data_xy_roi'
            else:
                filename = 'reconstructed_data_xy_roi'
            self._figures.append((buf, filename))
        if roi is None:
            x_bounds = (0, tomo_recon_stacks[0].shape[2])
            y_bounds = (0, tomo_recon_stacks[0].shape[1])
        else:
            x_bounds = (int(roi[0]), int(roi[1]))
            y_bounds = (int(roi[2]), int(roi[3]))
        self.logger.debug(f'x_bounds = {x_bounds}')
        self.logger.debug(f'y_bounds = {y_bounds}')

        # Selecting z bounds (in xy-plane)
        # (only valid for a single image stack or when combining a stack)
        if ((num_tomo_stacks == 1 or combine_data)
                and tomo_recon_stacks[0].shape[0] > 1):
            if z_bounds is None:
                if not self.interactive:
                    if combine_data:
                        self.logger.warning(
                            'z_bounds unspecified, combine reconstructed data '
                            'for full z-range')
                    else:
                        self.logger.warning(
                            'z_bounds unspecified, reconstruct data for '
                            'full z-range')
                z_bounds = (0, tomo_recon_stacks[0].shape[0])
            elif not is_int_pair(
                    z_bounds, ge=0, le=tomo_recon_stacks[0].shape[0]):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            tomosum = 0
            for i in range(num_tomo_stacks):
                tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=(1,2))
            buf, z_bounds = select_roi_1d(
                tomosum, preselected_roi=z_bounds,
                xlabel='z', ylabel='Reconstructed data summed over x and y',
                interactive=self.interactive, return_buf=self.save_figures)
            self.logger.debug(f'z_bounds = {z_bounds}')
            if self.save_figures:
                if combine_data:
                    filename = 'combined_data_z_roi'
                else:
                    filename = 'reconstructed_data_z_roi'
                self._figures.append((buf, filename))

        return x_bounds, y_bounds, z_bounds


class TomoCombineProcessor(Processor):
    """A processor to combine a stack of reconstructed images returning
    a NeXus style
    `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
    object containing the combined data, an optional list of byte
    stream representions of Matplotlib figures, and the metadata
    associated with the data reduction step.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.tomo.models.TomoCombineConfig`.
    :vartype config: dict, optional
    :ivar num_proc: Number of processors, defaults to `64`.
    :vartype num_proc: int, optional
    :ivar nxmemory: Maximum memory usage when reading NeXus files.
    :vartype nxmemory: int, optional
    :ivar save_figures: Create Matplotlib figures that can be saved to
        file downstream in the workflow, defaults to `True`.
    :vartype save_figures: bool, optional
    """

    pipeline_fields: dict = Field(
        default = {'config': 'tomo.models.TomoCombineConfig'},
        init_var=True)
    config: Optional[TomoCombineConfig] = TomoCombineConfig()
    num_proc: Optional[conint(gt=0)] = 64
    nxmemory: Optional[conint(gt=0)] = 100000
    save_figures: Optional[bool] = True

    _figures: list = PrivateAttr(default=[])

    def process(self, data):
        """Combine the reconstructed tomography stacks.

        :param data: Input data containing the reconstructed data as a
            NeXus style
            `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
            object.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: Metadata associated with the workflow, a list of byte
            stream representions of Matplotlib figures, and the result
            of the data combination.
        :rtype: PipelineData, PipelineData, PipelineData
        """
        # Third party modules
        from nexusformat.nexus import (
            NXdata,
            NXfield,
            NXprocess,
            nxsetconfig,
        )

        self.logger.info('Combine the reconstructed tomography stacks')

        # FIX make a config input
        nxsetconfig(memory=self.nxmemory)

        # Load the reduced tomography data
        nxroot = self.get_data(data)
        nxentry = self.get_default_nxentry(nxroot)

        # Check the number of stacks
        if nxentry.reconstructed_data.data.reconstructed_data.ndim == 3:
            num_tomo_stacks = 1
        else:
            num_tomo_stacks = \
                nxentry.reconstructed_data.data.reconstructed_data.shape[0]
        if num_tomo_stacks == 1:
            self.logger.info('Only one stack available: leaving combine_data')
            return nxroot

        # Create a NXprocess to store combined image reconstruction
        # (meta)data
        nxprocess = NXprocess()

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
        self.logger.info(
            f'Combining the reconstructed stacks took {time()-t0:.2f} seconds')
        tomo_shape = tomo_recon_combined.shape

        # Resize the combined tomography data stacks
        # - combined axis data order: row/-z,y,x
        if self.interactive or self.save_figures:
            x_bounds, y_bounds, z_bounds = self._resize_reconstructed_data(
                tomo_recon_combined, combine_data=True)
            self.config.x_bounds = None if x_bounds is None else list(x_bounds)
            self.config.y_bounds = None if y_bounds is None else list(y_bounds)
            self.config.z_bounds = None if z_bounds is None else list(z_bounds)
        else:
            x_bounds = self.config.x_bounds
            if x_bounds is None:
                self.logger.warning(
                    'x_bounds unspecified, combine data for full x-range')
            elif not is_int_pair(
                    x_bounds, ge=0, le=tomo_shape[2]):
                raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
            y_bounds = self.config.y_bounds
            if y_bounds is None:
                self.logger.warning(
                    'y_bounds unspecified, combine data for full y-range')
            elif not is_int_pair(
                    y_bounds, ge=0, le=tomo_shape[1]):
                raise ValueError(f'Invalid parameter y_bounds ({y_bounds})')
            z_bounds = self.config.z_bounds
            if z_bounds is None:
                self.logger.warning(
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

        # Save a few combined image slices
        if self.save_figures:
            x_slice = tomo_shape[2]//2
            title = f'recon combined x={x[x_slice]:.4f}'
            self._figures.append(
                (quick_imshow(
                    tomo_recon_combined[:,:,x_slice], title=title,
                    origin='lower', extent=(y[0], y[-1], z[0], z[-1]),
                    cmap='gray', show_fig=False, return_fig=True),
                re.sub(r'\s+', '_', title)))
            y_slice = tomo_shape[1]//2
            title = f'recon combined y={y[y_slice]:.4f}'
            self._figures.append(
                (quick_imshow(
                    tomo_recon_combined[:,y_slice,:], title=title,
                    origin='lower', extent=(x[0], x[-1], z[0], z[-1]),
                    cmap='gray', show_fig=False, return_fig=True),
                re.sub(r'\s+', '_', title)))
            z_slice = tomo_shape[0]//2
            title = f'recon combined z={z[z_slice]:.4f}'
            self._figures.append(
                (quick_imshow(
                    tomo_recon_combined[z_slice,:,:], title=title,
                    origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
                    cmap='gray', show_fig=False, return_fig=True),
                re.sub(r'\s+', '_', title)))

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
        nxentry = self.get_default_nxentry(nxroot)
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

        # Update metadata and provenance
        metadata, provenance = create_metadata_provenance(
            'tomo_combine',
            data,
            user_metadata={'combined_data': self.config.model_dump()},
            logger=self.logger)
        nxentry.combined_data.attrs['did'] = metadata.get('did')
        nxentry.combined_data.attrs['parent_did'] = metadata.get('parent_did')

        return (
            PipelineData(
                name=self.name, data=metadata,
                schema='foxden.reader.FoxdenMetadataReader'),
            PipelineData(
                name=self.name, data=provenance,
                schema='foxden.reader.FoxdenProvenanceReader'),
            PipelineData(
                name=self.name, data=self._figures,
                schema='common.write.ImageWriter'),
            PipelineData(name=self.name, data=nxroot, schema='tomodata'))

    def _resize_reconstructed_data(
            self, data, *, x_bounds=None, y_bounds=None, z_bounds=None,
            combine_data=False):
        """Resize the reconstructed tomography data."""
        # Data order: row/-z,y,x or stack,row/-z,y,x
        if isinstance(data, list):
            num_tomo_stacks = len(data)
            for i in range(num_tomo_stacks):
                assert data[i].ndim == 3
                if i:
                    assert data[i].shape[1:] == data[0].shape[1:]
            tomo_recon_stacks = data
        else:
            assert data.ndim == 3
            num_tomo_stacks = 1
            tomo_recon_stacks = [data]

        # Selecting x an y bounds (in z-plane)
        if x_bounds is None:
            if not self.interactive:
                self.logger.warning('x_bounds unspecified, use data for '
                                     'full x-range')
                x_bounds = (0, tomo_recon_stacks[0].shape[2])
        elif not is_int_pair(
                x_bounds, ge=0, le=tomo_recon_stacks[0].shape[2]):
            raise ValueError(f'Invalid parameter x_bounds ({x_bounds})')
        if y_bounds is None:
            if not self.interactive:
                self.logger.warning('y_bounds unspecified, use data for '
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
        buf, roi = select_roi_2d(
            tomosum, preselected_roi=preselected_roi,
            title_a='Reconstructed data summed over z',
            row_label='y', column_label='x',
            interactive=self.interactive, return_buf=self.save_figures)
        if self.save_figures:
            if combine_data:
                filename = 'combined_data_xy_roi'
            else:
                filename = 'reconstructed_data_xy_roi'
            self._figures.append((buf, filename))
        if roi is None:
            x_bounds = (0, tomo_recon_stacks[0].shape[2])
            y_bounds = (0, tomo_recon_stacks[0].shape[1])
        else:
            x_bounds = (int(roi[0]), int(roi[1]))
            y_bounds = (int(roi[2]), int(roi[3]))
        self.logger.debug(f'x_bounds = {x_bounds}')
        self.logger.debug(f'y_bounds = {y_bounds}')

        # Selecting z bounds (in xy-plane)
        # (only valid for a single image stack or when combining a stack)
        if num_tomo_stacks == 1 or combine_data:
            if z_bounds is None:
                if not self.interactive:
                    if combine_data:
                        self.logger.warning(
                            'z_bounds unspecified, combine reconstructed data '
                            'for full z-range')
                    else:
                        self.logger.warning(
                            'z_bounds unspecified, reconstruct data for '
                            'full z-range')
                z_bounds = (0, tomo_recon_stacks[0].shape[0])
            elif not is_int_pair(
                    z_bounds, ge=0, le=tomo_recon_stacks[0].shape[0]):
                raise ValueError(f'Invalid parameter z_bounds ({z_bounds})')
            tomosum = 0
            for i in range(num_tomo_stacks):
                tomosum = tomosum + np.sum(tomo_recon_stacks[i], axis=(1,2))
            buf, z_bounds = select_roi_1d(
                tomosum, preselected_roi=z_bounds,
                xlabel='z', ylabel='Reconstructed data summed over x and y',
                interactive=self.interactive, return_buf=self.save_figures)
            self.logger.debug(f'z_bounds = {z_bounds}')
            if self.save_figures:
                if combine_data:
                    filename = 'combined_data_z_roi'
                else:
                    filename = 'reconstructed_data_z_roi'
                self._figures.append((buf, filename))

        return x_bounds, y_bounds, z_bounds


class TomoSimFieldProcessor(Processor):
    """A processor to create a simulated tomography data set returning
    a NeXus style
    `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
    object containing the simulated tomography detector images.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.tomo.models.TomoSimConfig`.
    :vartype config: dict, optional
    """

    pipeline_fields: dict = Field(
        default = {'config': 'tomo.models.TomoSimConfig'}, init_var=True)
    config: TomoSimConfig

    def process(self, data):
        """Process the input configuration and return a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object with the simulated tomography detector images.

        :param data: Input data.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: Simulated tomographic images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXdetector,
            NXentry,
            NXinstrument,
            NXroot,
            NXsample,
            NXsource,
        )
        # pylint: enable=no-name-in-module

        station = self.config.station
        sample_type = self.config.sample_type
        sample_size = self.config.sample_size
        if len(sample_size) == 1:
            sample_size = (sample_size[0], sample_size[0])
        if sample_type == 'hollow_pyramid' and len(sample_size) != 3:
            raise ValueError('Invalid combindation of sample_type '
                             f'({sample_type}) and sample_size ({sample_size}')
        wall_thickness = self.config.wall_thickness
        mu = self.config.mu
        theta_step = self.config.theta_step
        beam_intensity = self.config.beam_intensity
        background_intensity = self.config.background_intensity
        slit_size = self.config.slit_size
        detector = self.config.detector
        pixel_size = detector.pixel_size
        if len(pixel_size) == 1:
            pixel_size = (
                pixel_size[0]/detector.lens_magnification,
                pixel_size[0]/detector.lens_magnification,
            )
        else:
            pixel_size = (
                pixel_size[0]/detector.lens_magnification,
                pixel_size[1]/detector.lens_magnification,
            )
        detector_size = (detector.rows, detector.columns)
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
            num_dummy_start = 0
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
        nxdetector.local_name = detector.prefix
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

    :ivar num_image: Number of dark field images, defaults to `5`.
    :vartype num_image: int, optional.
    """

    num_image: Optional[conint(gt=0)] = 5

    def process(self, data):
        """Process the input configuration and return a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object with the simulated dark field detector images.

        :param data: Input data.
        :type data: list[PipelineData]
        :raises ValueError: Missing or invalid input or configuration
            parameter.
        :return: Simulated dark field images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXroot,
            NXentry,
            NXinstrument,
            NXdetector,
        )
        # pylint: enable=no-name-in-module

        # Get and validate the TomoSimField configuration object in data
        nxroot = self.get_data(
            data, schema='tomo.models.TomoSimField', remove=False)
        source = nxroot.entry.instrument.source
        detector = nxroot.entry.instrument.detector
        background_intensity = source.background_intensity
        detector_size = detector.data.shape[-2:]

        # Add dummy snapshots at start to mimic SMB
        if source.station in ('id1a3', 'id3a'):
            num_dummy_start = 5
            starting_image_index = 123000
        else:
            num_dummy_start = 0
            starting_image_index = 0
        starting_image_offset = num_dummy_start
        self.num_image += num_dummy_start

        # Create the dark field
        dark_field = int(background_intensity) * np.ones(
            (self.num_image, detector_size[0], detector_size[1]),
            dtype=np.int64)

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
        nxdetector.thetas = np.asarray((self.num_image-num_dummy_start)*[0])
        nxdetector.starting_image_index = starting_image_index
        nxdetector.starting_image_offset = starting_image_offset

        return nxdark


class TomoBrightFieldProcessor(Processor):
    """A processor to create the bright field associated with a
    simulated tomography data set created by TomoSimProcessor.

    :ivar num_image: Number of bright field images, defaults to `5`.
    :vartype num_image: int, optional.
    """

    num_image: Optional[conint(gt=0)] = 5

    def process(self, data):
        """Process the input configuration and return a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#nxroot>`__
        object with the simulated bright field detector images.

        :param data: Input data.
        :type data: list[PipelineData]
        :raises ValueError: Missing or invalid input or configuration
            parameter.
        :return: Simulated bright field images.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        # pylint: disable=no-name-in-module
        from nexusformat.nexus import (
            NXroot,
            NXentry,
            NXinstrument,
            NXdetector,
        )
        # pylint: enable=no-name-in-module

        # Get and validate the TomoSimField configuration object in data
        nxroot = self.get_data(
            data, schema='tomo.models.TomoSimField', remove=False)
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
            num_dummy_start = 0
            starting_image_index = 0
        starting_image_offset = num_dummy_start

        # Create the bright field
        bright_field = int(background_intensity+beam_intensity) * np.ones(
            (self.num_image, detector_size[0], detector_size[1]),
            dtype=np.int64)
        if num_dummy_start:
            dummy_fields = int(background_intensity) * np.ones(
                (num_dummy_start, detector_size[0], detector_size[1]),
                dtype=np.int64)
            bright_field = np.concatenate((dummy_fields, bright_field))
            self.num_image += num_dummy_start
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
        nxdetector.thetas = np.asarray((self.num_image-num_dummy_start)*[0])
        nxdetector.starting_image_index = starting_image_index
        nxdetector.starting_image_offset = starting_image_offset

        return nxbright


class TomoSpecProcessor(Processor):
    """A processor to create a tomography SPEC file associated with a
    simulated tomography data set created by TomoSimProcessor.

    :var filename: Metadata input filename, when running with FOXDEN.
    :vartype filename: str, optional
    :ivar scan_numbers: List of SPEC scan numbers.
    :vartype scan_numbers: list[int], optional
    """

    filename: Optional[constr(strip_whitespace=True, min_length=1)] = None
    scan_numbers: Optional[
        conlist(min_length=1, item_type=conint(gt=0))] = None

    @model_validator(mode='after')
    def validate_tomospecprocessor_after(self):
        """Validate the `TomoSpecProcessor` configuration.

        :return: Validated model configuration
        :rtype: TomoSpecProcessor
        """
        if self.filename is None:
            return self

        # Local modules
        from CHAP.reader import validate_reader_model
        return  validate_reader_model(self)

    @field_validator('scan_numbers', mode='before')
    @classmethod
    def validate_scan_numbers(cls, scan_numbers):
        """Validate the specified list of scan numbers.

        :param scan_numbers: List of scan numbers.
        :type scan_numbers: int or list[int] or str
        :return: Validated scan numbers.
        :rtype: list[int]
        """
        if isinstance(scan_numbers, int):
            scan_numbers = [scan_numbers]
        elif isinstance(scan_numbers, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_numbers = string_to_list(scan_numbers)
        return scan_numbers

    def process(self, data):
        """Process the input configuration and return a list of strings
        representing a plain text SPEC file.

        :param data: Input data.
        :type data: list[PipelineData]
        :raises ValueError: Invalid input or configuration parameter.
        :return: Simulated SPEC file.
        :rtype: nexusformat.nexus.NXroot or
            (PipelineData, PipelineData)
        """
        # System modules
        from json import dumps, load
        from datetime import datetime

        from nexusformat.nexus import (
            NXentry,
            NXroot,
            NXsubentry,
        )

        # Get and validate the TomoSimField, TomoDarkField, or
        # TomoBrightField configuration object in data
        configs = {}
        try:
            nxroot = self.get_data(data, schema='tomo.models.TomoDarkField')
            configs['tomo.models.TomoDarkField'] = nxroot
        except ValueError:
            pass
        try:
            nxroot = self.get_data(data, schema='tomo.models.TomoBrightField')
            configs['tomo.models.TomoBrightField'] = nxroot
        except ValueError:
            pass
        try:
            nxroot = self.get_data(data, schema='tomo.models.TomoSimField')
            configs['tomo.models.TomoSimField'] = nxroot
        except ValueError:
            pass
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
                sample_type = str(nxroot.entry.sample.sample_type)
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
        if self.scan_numbers is None:
            self.scan_numbers = list(range(1, num_scan+1))
        elif len(self.scan_numbers) != num_scan:
            raise ValueError(
                f'Inconsistent number of scans ({num_scan}), '
                f'len(self.scan_numbers) = {len(self.scan_numbers)})')

        # Create the output data structure in NeXus format
        nxentry = NXentry()
        output_filenames = []

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
            field_type = None
            scan_type = None
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
                scan_number = self.scan_numbers[num_scan]
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
                    filename = f'{sample_type}_{prefix}_{scan_number:03d}.h5'
                    nxentry[field_name].attrs['filename'] = filename
                    output_filenames.append(f'{field_name}/{filename}')
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
            filename = f'{station}-tomo_sim-{sample_type}.json'
            nxentry.json.attrs['filename'] = filename
            output_filenames.append(filename)

            # Add the par file to output
            nxentry.par = NXsubentry()
            nxentry.par.data = par_file
            nxentry.par.attrs['schema'] = 'txt'
            filename = f'{station}-tomo_sim-{sample_type}.par'
            nxentry.par.attrs['filename'] = filename
            output_filenames.append(filename)

            # Add image files as individual tiffs to output
            for scan_number, image_set, starting_image_index in zip(
                    self.scan_numbers, image_sets, starting_image_indices):
                nxentry[f'{scan_number}'] = NXsubentry()
                nxsubentry = NXsubentry()
                nxentry[f'{scan_number}']['nf'] = nxsubentry
                for n in range(image_set.shape[0]):
                    nxsubentry[f'tiff_{n}'] = NXsubentry()
                    nxsubentry[f'tiff_{n}'].data = image_set[n]
                    nxsubentry[f'tiff_{n}'].attrs['schema'] = 'tif'
                    filename = f'nf_{(n+starting_image_index):06d}.tif'
                    nxsubentry[f'tiff_{n}'].attrs['filename'] = filename
                    output_filenames.append(f'{scan_number}/nf/{filename}')
        else:

            spec_filename = sample_type

        # Add spec file to output
        nxentry.spec = NXsubentry()
        nxentry.spec.data = spec_file
        nxentry.spec.attrs['schema'] = 'txt'
        nxentry.spec.attrs['filename'] = spec_filename
        output_filenames.append(spec_filename)

        nxroot = NXroot()
        nxroot[sample_type] = nxentry

        if station  != 'id1a3':
            return nxroot

        # Create a metadata record
        if self.filename is None:
            metadata = {'beamline': [station.upper()[2:]]}
        else:
            with open(self.filename, 'r', encoding='utf-8') as file:
                metadata = load(file)
        now = datetime.now()
        beamline = metadata['beamline'][0]
        btr = f'tomo-sim-{now.strftime("%m%d%H%M%S")}'
        if now.month < 4:
            cycle = f'{now.year}-1'
        elif now.month < 8:
            cycle = f'{now.year}-2'
        else:
            cycle = f'{now.year}-3'
        did = f'/beamline={beamline.lower()}/btr={btr}/cycle={cycle}/' + \
              f'sample_name={sample_type}'
        metadata['btr'] = btr
        metadata['cycle'] = cycle
        metadata['data_location_meta'] = str(self.inputdir)
        metadata['data_location_raw'] = str(self.outputdir)
        metadata['did'] = did
        metadata['sample_common_name'] = sample_type
        metadata['sample_name'] = sample_type
        metadata['schema'] = station.upper()

        # Add metadata record to output
        nxentry.metadata = NXsubentry()
        #nxentry.metadata.data = dumps(metadata)
        nxentry.metadata.data = dumps({'did': did})
        nxentry.metadata.attrs['schema'] = 'yaml'
        nxentry.metadata.attrs['filename'] = \
            f'../../config/did_{station}_{sample_type}.yaml'

        # Create the provenance info
        provenance = {
            'did': did,
            'input_files': [{'name': 'todo.fix: pipeline.yaml'}],
            'output_files': [{'name': f} for f in output_filenames],
        }

        return (
            PipelineData(
                name=self.name, data=metadata,
                schema='foxden.reader.FoxdenMetadataReader'),
            PipelineData(
                name=self.name, data=provenance,
                schema='foxden.reader.FoxdenProvenanceReader'),
            PipelineData(name=self.name, data=nxroot, schema='simdata'))


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
