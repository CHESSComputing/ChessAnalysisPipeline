#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific workflows.
"""

# system modules
import os

# local modules
from CHAP import Writer

class ExtractArchiveWriter(Writer):
    """Writer for tar files from binary data"""
    def write(self, data, filename):
        """Take a .tar archive represented as bytes in `data` and
        write the extracted archive to files.

        :param data: the archive data
        :type data: bytes
        :param filename: the name of a directory to which the archive
            files will be written
        :type filename: str
        :return: the original `data`
        :rtype: bytes
        """

        from io import BytesIO
        import tarfile

        data = self.unwrap_pipelinedata(data)

        with tarfile.open(fileobj=BytesIO(data)) as tar:
            tar.extractall(path=filename)

        return data


class NexusWriter(Writer):
    """Writer for NeXus files from `NXobject`-s"""
    def write(self, data, filename, force_overwrite=False):
        """Write `data` to a NeXus file

        :param data: the data to write to `filename`.
        :type data: nexusformat.nexus.NXobject
        :param filename: name of the file to write to.
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten, if it already exists.
        :return: the original input data
        """

        from nexusformat.nexus import NXobject

        data = self.unwrap_pipelinedata(data)

        if not isinstance(data, NXobject):
            raise TypeError('Cannot write object of type '
                            f'{type(data).__name__} to a NeXus file.')

        mode = 'w' if force_overwrite else 'w-'
        data.save(filename, mode=mode)

        return data


class YAMLWriter(Writer):
    """Writer for YAML files from `dict`-s"""
    def write(self, data, filename, force_overwrite=False):
        """If `data` is a `dict`, write it to `filename`.

        :param data: the dictionary to write to `filename`.
        :type data: dict
        :param filename: name of the file to write to.
        :type filename: str
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten if it already exists.
        :type force_overwrite: bool
        :raises TypeError: if `data` is not a `dict`
        :raises RuntimeError: if `filename` already exists and
            `force_overwrite` is `False`.
        :return: the original input data
        :rtype: dict
        """

        import yaml

        data = self.unwrap_pipelinedata(data)

        if not isinstance(data, (dict, list)):
            raise TypeError(
                f'{self.__name__}.write: input data must be a dict or list.')

        if not force_overwrite:
            if os.path.isfile(filename):
                raise RuntimeError(
                    f'{self.__name__}: {filename} already exists.')

        with open(filename, 'w') as outf:
            yaml.dump(data, outf, sort_keys=False)

        return data


class TXTWriter(Writer):
    """Writer for plain text files from string or list of strings."""
    def write(self, data, filename, force_overwrite=False, append=False):
        """If `data` is a `str`, `tuple[str]` or `list[str]`, write it
        to `filename`.

        :param data: the string or tuple or list of strings to write to
            `filename`.
        :type data: str, tuple, list
        :param filename: name of the file to write to.
        :type filename: str
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten if it already exists.
        :type force_overwrite: bool
        :raises TypeError: if `data` is not a `str`, `tuple[str]` or
            `list[str]`
        :raises RuntimeError: if `filename` already exists and
            `force_overwrite` is `False`.
        :return: the original input data
        :rtype: str, tuple, list
        """
        # Local modules
        from .utils.general import is_str_series

        data = self.unwrap_pipelinedata(data)

        if not isinstance(data, str) and not is_str_series(data, log=False):
            raise TypeError(
                f'{self.__name__}.write: input data must be a str or a tuple or '
                'list of str.')

        if not force_overwrite and not append:
            if os.path.isfile(filename):
                raise RuntimeError(
                    f'{self.__name__}: {filename} already exists.')

        if append:
            with open(filename, 'a') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write('\n'.join(data))
        else:
            with open(filename, 'w') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write('\n'.join(data))

        return data


if __name__ == '__main__':
    from CHAP.writer import main
    main()
