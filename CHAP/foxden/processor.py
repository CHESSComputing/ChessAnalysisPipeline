#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Module for Processors unique to the
`FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
integration with CHAP.

Add discription of FOXDEN
"""

# System modules
import os
from typing import (
    Literal,
    Optional,
)
# Third party modules
from pydantic import conint

# Local modules
from CHAP.processor import Processor


#class FoxdenMetadataProcessor(Processor):
#    """Processor to collect CHAP workflow metadata from a workflow
#    NeXus output object.
#    """
#    def process(self, data):
#        """Extract metadata from a workflow NeXus output object for
#        submission to the FOXDEN Metadata service.
#
#        :param data: Input data.
#        :type data: list[PipelineData]
#        :return: CHAP workflow metadata record.
#        :rtype: dict
#        """
#        # Third party modules
#        from json import loads
#        from nexusformat.nexus import (
#            NXentry,
#            NXroot,
#        )
#
#        # Load and validate the workflow NeXus output object
#        nxentry = self.get_data(data, remove=False)
#        if isinstance(nxentry, NXroot):
#            nxentry = nxentry[nxentry.default]
#        if not isinstance(nxentry, NXentry):
#            raise ValueError(f'Invalid input data type {type(nxentry)}')
#
#        # Get did and experiment type
#        map_config = loads(str(nxentry.map_config))
#        did = map_config['did']
#        experiment_type = map_config['experiment_type']
#
#        # Extract metadata
#        method = getattr(self, f'_get_metadata_{experiment_type.lower()}')
#        metadata = method(nxentry)
#
#        if 'reconstructed_data' in metadata:
#            did = f'{did}/{experiment_type.lower()}_reconstructed'
#        else:
#            did = f'{did}/{experiment_type.lower()}_reduced'
#        return {'did': did, 'application': 'CHAP', 'metadata': metadata}
#
#    def _get_metadata_tomo(self, nxentry):
#        metadata = {}
#        if 'reduced_data' in nxentry:
#            data = nxentry.reduced_data
#            metadata.update({
#                'reduced_data': {
#                    'date': str(data.date),
#                    'img_row_bounds': data.img_row_bounds.tolist(),
#                }
#            })
#        if 'reconstructed_data' in nxentry:
#            data = nxentry.reconstructed_data
#            metadata.update({
#                'reconstructed_data': {
#                    'date': str(data.date),
#                    'center_offsets': data.center_offsets.tolist(),
#                    'center_rows': data.center_offsets.tolist(),
#                    'center_stack_index': int(data.center_stack_index),
#                    'x_bounds': data.x_bounds.tolist(),
#                    'y_bounds': data.y_bounds.tolist(),
#                }
#            })
#        if 'combined_data' in nxentry:
#            data = nxentry.combined_data
#            metadata.update({
#                'combined_data': {
#                    'date': str(data.date),
#                }
#            })
#        return metadata


#class FoxdenProvenanceProcessor(Processor):
#    """Processor to collect CHAP workflow provenance data."""
#    def process(self, data):
#        """Extract provenance data from the pipeline data for
#        submission to the FOXDEN Provenance service.
#
#        :param data: Input data.
#        :type data: list[PipelineData]
#        :return: CHAP workflow provenance record.
#        :rtype: dict
#        """
#        # Local modules
#        from CHAP.common.utils import (
#            osinfo,
#            environments,
#        )
#        # Load the provenance info
#        provenance = self.get_data(data, schema='provenance')
#
#        # Add system info to provenance data
#        provenance.update({
#            'environments': environments(),
#            'osinfo': osinfo(),
#            'processing': 'CHAP pipeline',
#            'scripts': [
#                {'name': 'CHAP', 'parent_script': None, 'order_idx': 1}],
#            'site': 'Cornell',
#        })
#
#        return provenance


class ProvenanceFileProcessor(Processor):
    """A Processor that retrieves a
    `FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__ provenance
    record from the pipeline and returns the content of the in or
    output file listed in the record.

    :ivar file_type: The `'file_type'` in the provenance record,
        defaults to `'output'`.
    :vartype file_type: Literal['input', 'output'], optional.
    :ivar nxmemory: Maximum memory usage when reading NeXus files,
        ignore for any other file type.
    :vartype nxmemory: int, optional
    """

    file_type: Optional[Literal['input', 'output']] = 'output'
    nxmemory: Optional[conint(gt=0)] = None

    def process(self, data):
        """Return the content of in or output files listed in the
        provenance record.

        :return: The file content.
        :rtype: Any
        """
        # Local modules
        from CHAP.tomo.processor import read_metadata_provenance

        try:
            _, provenance = read_metadata_provenance(
                data, logger=self.logger, remove=False)
            filenames = [v['name']
                        for v in provenance if v['file_type'] == 'output']
            if not filenames:
                raise ValueError('Unable to get an output file name from '
                                 f'provenance ({provenance})')
            if len(filenames) > 1:
                raise ValueError('Unable to get a unique output file name '
                                 f'from provenance ({provenance})')
            filename = filenames[0]
        except ValueError:
            raise

        # FIX modify CHAP.reader to be a generic reader, based on ext
        # Can use __import__ as well
        ext = os.path.splitext(filename)[1][1:]
        if ext == 'nxs':
            # Local modules
            from CHAP.common.reader import NexusReader

            reader = NexusReader(filename=filename, **self.model_dump())
        elif ext in ('yml', 'yaml'):
            # Local modules
            from CHAP.common.reader import YAMLReader

            reader = YAMLReader(filename=filename, **self.model_dump())
        else:
            raise ValueError('ProvenanceOutputReader not yet implemented for '
                             f'files with extension {ext}')
        return reader.read()


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
