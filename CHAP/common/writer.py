#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific workflows.
"""

# System modules
from os import path as os_path

# Local modules
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
    # System modules
    from os import makedirs

    # Third party modules
    from nexusformat.nexus import (
        NXentry,
        NXsubentry,
        NXgroup,
        NXobject,
        NXroot,
        NXsubentry,
    )

    if not isinstance(data, NXobject):
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a file tree to disk.')

    if not os_path.isdir(outputdir):
        makedirs(outputdir)

    for k, v in data.items():
        if isinstance(v, NXsubentry) and 'schema' in v.attrs:
            schema = v.attrs['schema']
            filename = os_path.join(outputdir, v.attrs['filename'])
            if schema == 'txt':
                write_txt(list(v.data), filename, force_overwrite)
            elif schema == 'json':
                write_txt(str(v.data), filename, force_overwrite)
            elif schema == 'yml' or schema == 'yaml':
                from json import loads
                write_yaml(loads(v.data.nxdata), filename, force_overwrite)
            elif schema == 'tif' or schema == 'tiff':
                write_tif(v.data, filename, force_overwrite)
            elif schema == 'h5':
                if any(isinstance(vv, NXsubentry) for vv in v.values()):
                    nxbase = NXroot()
                else:
                    nxbase = NXentry()
                for kk, vv in v.attrs.items():
                    if kk not in ('schema', 'filename'):
                        nxbase.attrs[kk] = vv
                for kk, vv in v.items():
                    if isinstance(vv, NXsubentry):
                        nxentry = NXentry()
                        nxbase[vv.nxname] = nxentry
                        for kkk, vvv in vv.items():
                            nxentry[kkk] = vvv
                    else:
                        nxbase[kk] = vv
                write_nexus(nxbase, filename, force_overwrite)
            else:
                raise TypeError(f'Files of type {schema} not yet implemented')
        elif isinstance(v, NXgroup):
            write_filetree(v, os_path.join(outputdir, k), force_overwrite)


class ExtractArchiveWriter(Writer):
    """Writer for tar files from binary data"""
    def write(self, data, filename):
        """Take a .tar archive represented as bytes contained in `data`
        and write the extracted archive to files.

        :param data: The data to write to archive.
        :type data: list[PipelineData]
        :param filename: The name of the directory to write the archive
            files to.
        :type filename: str
        :return: The achived data
        :rtype: bytes
        """
        # System modules
        from io import BytesIO
        import tarfile

        data = self.unwrap_pipelinedata(data)[-1]

        with tarfile.open(fileobj=BytesIO(data)) as tar:
            tar.extractall(path=filename)

        return data


class FileTreeWriter(Writer):
    """Writer for a file tree in NeXus format"""
    def write(self, data, outputdir, force_overwrite=False):
        """Write a NeXus format object contained in `data` to a 
        directory tree stuctured like the NeXus tree.

        :param data: The data to write to disk.
        :type data: list[PipelineData]
        :param outputdir: The name of the directory to write to.
        :type outputdir: str
        :param force_overwrite: Flag to allow data to be overwritten
            if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to disk.
        :rtype: Union[nexusformat.nexus.NXroot,
            nexusformat.nexus.NXentry]
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


class H5Writer(Writer):
    """Writer for H5 files from an `nexusformat.nexus.NXdata` object"""
    def write(self, data, filename, force_overwrite=False):
        """Write the NeXus object contained in `data` to hdf5 file.

        :param data: The data to write to file.
        :type data: CHAP.pipeline.PipelineData
        :param filename: The name of the file to write to.
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to file.
        :rtype: nexusformat.nexus.NXobject
        """
        # Third party modules
        from h5py import File
        from nexusformat.nexus import NXdata

        data = self.unwrap_pipelinedata(data)[-1]
        if not isinstance(data, NXdata):
            raise ValueError('Invalid data parameter {(data)}')

        mode = 'w' if force_overwrite else 'w-'
        with File(filename, mode) as f:
            f[data.signal] = data.nxsignal
            for i, axes in enumerate(data.attrs['axes']):
                f[axes] = data[axes]
                f[data.signal].dims[i].label = \
                    f'{axes} ({data[axes].units})' \
                    if 'units' in data[axes].attrs else axes
                f[axes].make_scale(axes)
                f[data.signal].dims[i].attach_scale(f[axes])

        return data


class MatplotlibAnimationWriter(Writer):
    """Writer for saving matplotlib animations."""
    def write(self, data, filename, fps=1):
        """Write the matplotlib.animation.ArtistAnimation object
        contained in `data` to file.

        :param data: The matplotlib animation.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
        :param fps: Movie frame rate (frames per second),
            defaults to `1`
        :type fps: int, optional
        :return: The original animation.
        :rtype: matplotlib.animation.ArtistAnimation
        """
        data = self.unwrap_pipelinedata(data)[-1]
        extension = os_path.splitext(filename)[1]
        if not extension:
            data.save(f'{filename}.gif', fps=fps)
        elif extension == '.gif':
            data.save(filename, fps=fps)
        elif extension == '.mp4':
            data.save(filename, writer='ffmpeg', fps=fps)

        return data


class MatplotlibFigureWriter(Writer):
    """Writer for saving matplotlib figures to image files."""
    def write(self, data, filename, savefig_kw={}, force_overwrite=False):
        """Write the matplotlib.figure.Figure contained in `data` to
        file.

        :param data: The matplotlib figure
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
        :param savefig_kw: Keyword args to pass to
            matplotlib.figure.Figure.savefig, defaults to {}.
        :type savefig_kw: dict, optional
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The original figure object
        :rtype: matplotlib.figure.Figure
        """
        data = self.unwrap_pipelinedata(data)[-1]
        write_matplotlibfigure(data, filename, force_overwrite)

        return data


class NexusWriter(Writer):
    """Writer for NeXus files from `NXobject`-s"""
    def write(self, data, filename, nxpath=None, force_overwrite=False):
        """Write the NeXus object contained in `data` to file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to file.
        :rtype: nexusformat.nexus.NXobject
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXFile,
            NXroot,
        )
        import os
        data = self.unwrap_pipelinedata(data)[-1]
        nxname = data.nxname
        if not os.path.isfile(filename) and nxpath is not None:
            self.logger.warning(
                f'{filename} does not yet exist. Argument for nxpath ({nxpath}) '
                + 'will be ignored.')
            nxpath = None
        if nxpath is None:
            nxclass = data.nxclass
            if nxclass == 'NXentry':
                data = NXroot(data)
                data[nxname].set_default()
            elif nxclass != 'NXroot':
                data = NXroot(NXentry(data))
                if nxclass == 'NXdata':
                    data.entry[nxname].set_default()
                data.entry.set_default()
            write_nexus(data, filename, force_overwrite)
        else:
            nxfile = NXFile(filename, 'rw')
            root = nxfile.readfile()
            if nxfile.get(nxpath) is None:
                nxpath = root.NXentry[0].nxpath
                self.logger.warning(
                    f'Path "{nxpath}" not present in {filename}. '
                    + f'Using {nxpath} instead.')
            full_nxpath = os.path.join(nxpath, nxname)
            self.logger.debug(f'Full path for object to write: {full_nxpath}')
            if nxfile.get(full_nxpath) is not None:
                self.logger.debug(
                    f'{os.path.join(nxpath, nxname)} already exists in {filename}')
                if force_overwrite:
                    self.logger.warning(
                        'Deleting existing NXobject at '
                        + f'{os.path.join(nxpath, nxname)} in {filename}')
                    del root[full_nxpath]
            try:
                root[full_nxpath] = data
            except Exception as e:
                nxfile.close()
                raise e
            nxfile.close()
        return data


class PyfaiResultsWriter(Writer):
    """Writer for results of one or more pyFAI integrations. Able to
    handle multiple output formats. Currently supported formats are:
    .npz, .nxs.
    """
    def write(self, data, filename, force_overwrite=False):
        """Save pyFAI integration results to a file. Format is
        determined automatically form the extension of `filename`.

        :param data: Integration results to save.
        :type data: Union[PipelineData,
            list[pyFAI.containers.IntegrateResult]]
        :param filename: Name of the file to which results will be
            saved. Format of output is determined ffrom the
            extension. Currently supported formats are: `.npz`,
            `.nxs`
        :type filename: str
        """
        import os

        from pyFAI.containers import Integrate1dResult, Integrate2dResult

        try:
            results = self.unwrap_pipelinedata(data)[0]
        except:
            results = data
        if not isinstance(results, list):
            results = [results]
        if not all([isinstance(r, Integrate1dResult) for r in results]) \
           and not all([isinstance(r, Integrate2dResult) for r in results]):
            raise Exception(
                'Bad input data: all items must have the same type -- either '
                + 'all pyFAI.containers.Integrate1dResult, or all '
                + 'pyFAI.containers.Integrate2dResult.')

        if os.path.isfile(filename):
            if force_overwrite:
                self.logger.warning(f'Removing existing file {filename}')
                os.remove(filename)
            else:
                raise Exception(f'{filename} already exists.')
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.npz':
            self.write_npz(results, filename)
        elif ext.lower() == '.nxs':
            self.write_nxs(results, filename)
        else:
            raise Exception(f'Unsupported file format: {ext}')
        self.logger.info(f'Wrote to {filename}')
        return results

    def write_npz(self, results, filename):
        """Save `results` to the .npz file, `filename`"""
        import numpy as np

        data = {'radial': results[0].radial,
                'intensity': [r.intensity for r in results]}
        if hasattr(results[0], 'azimuthal'):
            # 2d results
            data['azimuthal'] = results[0].azimuthal
        if all([r.sigma for r in results]):
            # errors were included
            data['sigma'] = [r.sigma for r in results]

        np.savez(filename, **data)

    def write_nxs(results, filename):
        """Save `results` to the .nxs file, `filename`"""
        raise NotImplementedError



class TXTWriter(Writer):
    """Writer for plain text files from string or tuples or lists of
    strings."""
    def write(self, data, filename, append=False, force_overwrite=False):
        """Write a string or tuple or list of strings contained in 
        `data` to file.

        :param data: The data to write to disk.
        :type data: str, tuple[str], list[str]
        :param filename: The name of the file to write to.
        :type filename: str
        :param append: Flag to allow data in `filename` to be
            be appended, defaults to `False`.
        :type append: bool, optional
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises TypeError: If the object contained in `data` is not a
            `str`, `tuple[str]` or `list[str]`.
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to file.
        :rtype: str, tuple[str], list[str]
        """
        data = self.unwrap_pipelinedata(data)[-1]
        write_txt(data, filename, force_overwrite, append)

        return data


class YAMLWriter(Writer):
    """Writer for YAML files from `dict`-s"""
    def write(self, data, filename, force_overwrite=False):
        """Write the dictionary contained in `data` to file.

        :param data: The data to write to file.
        :type data: dict
        :param filename: The name of the file to write to.
        :type filename: str
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises TypeError: If the object contained in `data` is not a
            `dict`.
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to file.
        :rtype: dict
        """
        data = self.unwrap_pipelinedata(data)[-1]
        try:
            from pydantic import BaseModel
            if isinstance(data, BaseModel):
                data = data.dict()
        except:
            pass
        write_yaml(data, filename, force_overwrite)
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
