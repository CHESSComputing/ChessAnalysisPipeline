#!/usr/bin/env python
"""Tomography command line writer."""

# Local modules
from CHAP import Writer


class TomoWriter(Writer):
    """Writer for saving ."""
    def write(
            self, data, filename, force_overwrite=False, remove=True):
        """Write the results of the (partial) tomographic
        reconstruction and add provenance data to the data pipeline.

        :param data: Input data.
        :type data: list[PipelineData]
        :param filename: Name of the file to write to.
        :type filename: str
        :param force_overwrite: Flag to allow files to be
            overwritten if they already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :param remove: Remove the result of the (partial)
            reconstruction from the pipeline data list,
            defaults to `True`.
        :type remove: bool, optional
        :return: Output data.
        :rtype: list[PipelineData]
        """
        # System modules
        from os import path as os_path

        # Local modules
        from CHAP.pipeline import PipelineData

        # Load the (partial) tomographic reconstruction result
        tomodata = self.get_data(data, schema='tomodata', remove=remove)

        # Local modules
        if isinstance(tomodata, dict):
            from CHAP.common.writer import YAMLWriter as writer
        else:
            from CHAP.common.writer import NexusWriter as writer

        # Write the (partial) tomographic reconstruction result
        #RV FIX make class methods from the Writer.write functions?
        # or create write function that also accept some default type
        # other than list[PipelineData]?
        writer().write(
            [PipelineData(name=self.__name__, data=tomodata)],
            filename, force_overwrite=force_overwrite)

        # Add provenance info to the data pipeline
        metadata = self.get_data(data, schema='metadata', remove=False)
        did = metadata['did']
        provenance = {
            'did': did,
            'input_files': [{'name': 'todo.fix: pipeline.yaml'}],
            'output_files': [{'name': os_path.realpath(filename)}],
        }
        data.append(PipelineData(
            name=self.__name__, data=provenance, schema='provenance'))

        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
