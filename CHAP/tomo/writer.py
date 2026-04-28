#!/usr/bin/env python
"""Module for Writers unique to the tomography workflow."""

# System modules
import os

# Third party modules
from pydantic import model_validator

# System modules
import os

# Third party modules
from pydantic import model_validator

# Local modules
from CHAP.writer import Writer


class TomoWriter(Writer):
    """Writer for saving tomo data."""

    @model_validator(mode='after')
    def validate_tomowriter_after(self):
        """Validate the filename extension.

        :return: Validated writer configuration
        :rtype: TomoWriter
        """
        ext = os.path.splitext(self.filename)[1][1:]
        if ext not in ('nxs', 'yml', 'yaml'):
            raise ValueError(f'Invalid filename extension {self.filename}')
        return self

    def write(self, data):
        """Write the results of the (partial) tomographic
        reconstruction and add provenance data to the data pipeline.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: Output data.
        :rtype: list[PipelineData]
        """
        # Local modules
        from CHAP.pipeline import PipelineData

        # Load the (partial) tomographic reconstruction result
        ext = os.path.splitext(self.filename)[1][1:]
        if ext == 'nxs':
            tomodata = self.get_data(
                data, schema='tomodata', remove=self.remove)
        elif ext in ('yml', 'yaml'):
            tomodata = self.get_data(
                data, schema='tomo.models.TomoFindCenterConfig',
                remove=self.remove)
        else:
            raise ValueError(f'Invalid filename extension {ext}')

        # Local modules
        if isinstance(tomodata, dict):
            from CHAP.common.writer import YAMLWriter as writer
        else:
            from CHAP.common.writer import NexusWriter as writer

        # Write the (partial) tomographic reconstruction result
        #RV FIX make class methods from the Writer.write functions?
        # or create write function that also accept some default type
        # other than list[PipelineData]?
        writer(**self.model_dump()).write(
            [PipelineData(name=self.__name__, data=tomodata)])

        # Return provenance with the output file name added
        return self._update_provenance(data)


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
