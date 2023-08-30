#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific workflows.
"""

# system modules
from os import mkdir
from os import path as os_path

# local modules
from CHAP import Writer

def write_matplotlibfigure(data, filename, force_overwrite=False):
    # Third party modules
    from matplotlib.figure import Figure

    if not isinstance(data, Figure):
        raise TypeError('Cannot write object of type'
                        f'{type(data)} as a matplotlib Figure.')

    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    data.savefig(filename, **savefig_kw)

def write_nexus(data, filename, force_overwrite=False):
    # Third party modules
    from nexusformat.nexus import NXobject

    if not isinstance(data, NXobject):
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a NeXus file.')

    mode = 'w' if force_overwrite else 'w-'
    data.save(filename, mode=mode)

def write_tif(data, filename, force_overwrite=False):
    # Third party modules
    from imageio import imwrite
    import numpy as np

    data = np.asarray(data)
    if data.ndim != 2:
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a tif file.')

    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    imwrite(filename, data)

def write_txt(data, filename, force_overwrite=False, append=False):
    # Local modules
    from CHAP.utils.general import is_str_series

    if not isinstance(data, str) and not is_str_series(data, log=False):
        raise TypeError('input data must be a str or a tuple or list of str '
                        f'instead of {type(data)} ({data})')

    if not force_overwrite and not append and os_path.isfile(filename):
        raise FileExistsError(f'{filename} already exists')

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

def write_yaml(data, filename, force_overwrite=False):
    # Third party modules
    import yaml

    if not isinstance(data, (dict, list)):
        raise TypeError(f'input data must be a dict or list.')

    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    with open(filename, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def write_filetree(data, outputdir, force_overwrite=False):
    # Third party modules
    from nexusformat.nexus import (
        NXentry,
        NXgroup,
        NXobject,
        NXsubentry,
    )

    if not isinstance(data, NXobject):
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a file tree to disk.')

    if not os_path.isdir(outputdir):
        mkdir(outputdir)

    for k, v in data.items():
        if isinstance(v, NXsubentry) and 'schema' in v.attrs:
            schema = v.attrs['schema']
            filename = os_path.join(outputdir, v.attrs['filename'])
            if schema == 'txt':
                write_txt(list(v.data), filename, force_overwrite)
            elif schema == 'json':
                write_txt(str(v.data), filename, force_overwrite)
            elif schema == 'tif' or schema == 'tiff':
                write_tif(v.data, filename, force_overwrite)
            elif schema == 'h5':
                nxentry = NXentry()
                for kk, vv in v.attrs.items():
                    nxentry.attrs[kk] = vv
                for kk, vv in v.items():
                    nxentry[kk] = vv
                write_nexus(nxentry, filename, force_overwrite)
            else:
                raise TypeError(f'Files of type {schema} not yet implemented')
        elif isinstance(v, NXgroup):
            write_filetree(v, os_path.join(outputdir, k), force_overwrite)


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
        # System modules
        from io import BytesIO
        import tarfile

        data = self.unwrap_pipelinedata(data)[-1]

        with tarfile.open(fileobj=BytesIO(data)) as tar:
            tar.extractall(path=filename)

        return data


class MatplotlibFigureWriter(Writer):
    """Writer for saving matplotlib figures to image files."""
    def write(self, data, filename, savefig_kw={}, force_overwrite=False):
        """Write the matplotlib.fgure.Figure contained in `data` to
        the filename provided.

        :param data: input containing a matplotlib figure
        :type data: CHAP.pipeline.PipelineData
        :param filename: name of the file to write to.
        :param savefig_kw: keyword args to pass to
            matplotlib.figure.Figure.savefig, defaults to {}
        :type savefig_kw: dict, optional
        :param force_overwrite: flag to allow data in `filename` to be
            overwritten, if it already exists.
        :return: the original input data
        """
        data = self.unwrap_pipelinedata(data)[-1]
        write_matplotlibfigure(data, filename, force_overwrite)
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
        data = self.unwrap_pipelinedata(data)[-1]
        write_nexus(data, filename, force_overwrite)
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
        data = self.unwrap_pipelinedata(data)[-1]
        write_txt(data, filename, force_overwrite, append)
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
        data = self.unwrap_pipelinedata(data)[-1]
        write_yaml(data, filename, force_overwrite)
        return data


class FileTreeWriter(Writer):
    """Writer for a file tree in NeXus format"""
    def write(self, data, outputdir, force_overwrite=False):
        """Write `data` to a NeXus file

        :param data: the data to write to disk.
        :type data: Union[nexusformat.nexus.NXroot,
            nexusformat.nexus.NXentry]
        :param force_overwrite: flag to allow data to be overwritten,
            if it already exists.
        :return: the original input data
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        data = self.unwrap_pipelinedata(data)[-1]
        if isinstance(data, NXroot):
            if 'default' in data.attrs:
                nxentry = data[data.attrs['default']]
            else:
                nxentry = [v for v in data.values()
                           if isinstance(data, NXentry)]
                if len(nxentry) == 1:
                    nxentry = nxentry[0]
                else:
                    raise TypeError('Cannot write object of type '
                                    f'{type(data).__name__} as a file tree '
                                    'to disk.')
        elif isinstance(data, NXentry):
            nxentry = data
        else:
            raise TypeError('Cannot write object of type '
                            f'{type(data).__name__} as a file tree to disk.')

        write_filetree(nxentry, outputdir, force_overwrite)
        return data


if __name__ == '__main__':
    from CHAP.writer import main
    main()
