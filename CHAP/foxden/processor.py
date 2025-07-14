#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : processor.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Processor module for FOXDEN services
"""

# System modules
import os
from time import time

# Local modules
from CHAP.common.utils import osinfo, environments
from CHAP.processor import Processor


class FoxdenMetadataProcessor(Processor):
    """Processor to collect CHAP workflow metadata from a workflow
    NeXus output file.
    """
    def process(self, data):
        """Extract metadata from a workflow NeXus output file for
        submission to the FOXDEN Metadata service.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: CHAP workflow metadata record.
        :rtype: dict
        """
        # Local modules
        from CHAP.common.models.map import MapConfig

        # Third party modules
        from json import loads
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        # Load and validate the tomography fields
        nxentry = self.get_data(data)
        if isinstance(nxentry, NXroot):
            nxentry = nxentry[nxentry.default]
        if not isinstance(nxentry, NXentry):
            raise ValueError(f'Invalid input data type {type(nxentry)}')

        # Get experiment type
        #map_config = MapConfig(**loads(str(nxentry.map_config)))
        #experiment_type = map_config.experiment_type
        map_config = loads(str(nxentry.map_config))
        did = map_config['did']
        experiment_type = map_config['experiment_type']

        # Extract metadata
        method = getattr(self, f'_get_metadata_{experiment_type.lower()}')
        metadata = method(nxentry)

        if 'reconstructed_data' in metadata:
            did = f'{did}/{experiment_type.lower()}_reconstructed'
        else:
            did = f'{did}/{experiment_type.lower()}_reduced'
        return {'did': did, 'application': 'CHAP', 'metadata': metadata}

    def _get_metadata_tomo(self, nxentry):
        metadata = {}
        if True:#'reduced_data' in nxentry:
            data = nxentry.reduced_data
            metadata.update({
                'reduced_data': {
                    'date': str(data.date),
                    'img_row_bounds': data.img_row_bounds.tolist(),
                }
            })
        if 'reconstructed_data' in nxentry:
            data = nxentry.reconstructed_data
            metadata.update({
                'reconstructed_data': {
                    'date': str(data.date),
                    'center_offsets': data.center_offsets.tolist(),
                    'center_rows': data.center_offsets.tolist(),
                    'center_stack_index': int(data.center_stack_index),
                    'x_bounds': data.x_bounds.tolist(),
                    'y_bounds': data.y_bounds.tolist(),
                }
            })
        if 'combined_data' in nxentry:
            data = nxentry.combined_data
            metadata.update({
                'combined_data': {
                    'date': str(data.date),
                }
            })
        return metadata


class FoxdenProvenanceProcessor(Processor):
    """Processor to communicate with FOXDEN provenance server."""
    def process(self, data, filename):
        """FOXDEN Provenance server communication processor.

        :param data: Input data.
        :type data: list[PipelineData]
        :param filename: CHAP workflow NeXus output file name.
        :type filename: str
        :return: CHAP workflow provenance record.
        :rtype: dict
        """
        did = self.unwrap_pipelinedata(data)[-1]
        if not isinstance(did, str):
            raise ValueError(
                f'Invalid did type in input data ({type(did)}, {did})')

        provenance = {
            'did': did,
            'parent_did': did.rsplit('/', 1)[0],
            'scripts': [
                {'name': 'CHAP', 'parent_script': None, 'order_idx': 1}],
            'site': 'Cornell',
            'osinfo': osinfo(),
            'environments': environments(),
            'input_files': [{'name': 'todo.fix'}],
            'output_files': [{'name': os.path.realpath(filename)}],
            'processing': 'CHAP pipeline',
        }
        return [provenance]


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
