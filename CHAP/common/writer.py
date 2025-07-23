#!/usr/bin/env python
"""
File       : writer.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description: Module for Writers used in multiple experiment-specific
             workflows.
"""

# System modules
from os import path as os_path

# Third party modules
import numpy as np

# Local modules
from CHAP import Writer


def write_matplotlibfigure(data, filename, savefig_kw, force_overwrite=False):
    """Write a Matplotlib figure to file.

    :param data: The figure to write to file
    :type data: matplotlib.figure.Figure
    :param filename: File name.
    :type filename: str
    :param savefig_kw: Keyword args to pass to
        matplotlib.figure.Figure.savefig.
    :type savefig_kw: dict, optional
    :param force_overwrite: Flag to allow data to be overwritten if it
        already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    """
    # Third party modules
    from matplotlib.figure import Figure

    if not isinstance(data, Figure):
        raise TypeError('Cannot write object of type'
                        f'{type(data)} as a matplotlib Figure.')

    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    if savefig_kw is None:
        data.savefig(filename)
    else:
        data.savefig(filename, **savefig_kw)

def write_nexus(data, filename, force_overwrite=False):
    """Write a Nexus object to file.

    :param data: The data to write to file
    :type data: nexusformat.nexus.NXobject
    :param filename: File name.
    :type filename: str
    :param force_overwrite: Flag to allow data to be overwritten if it
        already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    """
    # Third party modules
    from nexusformat.nexus import NXobject

    if not isinstance(data, NXobject):
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a NeXus file.')

    mode = 'w' if force_overwrite else 'w-'
    data.save(filename, mode=mode)

def write_tif(data, filename, force_overwrite=False):
    """Write a tif image to file.

    :param data: The data to write to file
    :type data: numpy.ndarray
    :param filename: File name.
    :type filename: str
    :param force_overwrite: Flag to allow data to be overwritten if it
        already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    """
    # Third party modules
    from imageio import imwrite

    data = np.asarray(data)
    if data.ndim != 2:
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a tif file.')

    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    imwrite(filename, data)

def write_txt(data, filename, force_overwrite=False, append=False):
    """Write plain text to file.

    :param data: The data to write to file
    :type data: Union[str, list[str]]
    :param filename: File name.
    :type filename: str
    :param force_overwrite: Flag to allow data to be overwritten if it
        already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    :param append: Flag to allow data to be appended to the file if it
        already exists, defaults to `False`.
    :type append: bool, optional
    """
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
    """Write data to a YAML file.

    :param data: The data to write to file
    :type data: Union[dict, list]
    :param filename: File name.
    :type filename: str
    :param force_overwrite: Flag to allow data to be overwritten if it
        already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    """
    # Third party modules
    import yaml

    if not isinstance(data, (dict, list)):
        raise TypeError('input data must be a dict or list.')

    if os_path.isfile(filename) and not force_overwrite:
        raise FileExistsError(f'{filename} already exists')

    with open(filename, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def write_filetree(data, outputdir, force_overwrite=False):
    """Write data to a file tree.

    :param data: The data to write to files
    :type data: nexusformat.nexus.NXobject
    :param outputdir: Output directory.
    :type filename: str
    :param force_overwrite: Flag to allow data to be overwritten if it
        already exists, defaults to `False`.
    :type force_overwrite: bool, optional
    """
    # System modules
    from os import makedirs

    # Third party modules
    from nexusformat.nexus import (
        NXentry,
        NXgroup,
        NXobject,
        NXroot,
        NXsubentry,
    )

    if not isinstance(data, NXobject):
        raise TypeError('Cannot write object of type'
                        f'{type(data).__name__} as a file tree to disk.')

    # FIX: Right now this can bomb if MultiplePipelineItem
    # is called simultaneously from multiple nodes in MPI
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
            elif schema in ('yml', 'yaml'):
                from json import loads
                write_yaml(loads(v.data.nxdata), filename, force_overwrite)
            elif schema in ('tif',  'tiff'):
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
    """Writer for tar files from binary data."""
    def write(self, data, filename):
        """Take a .tar archive represented as bytes contained in `data`
        and write the extracted archive to files.

        :param data: The data to write to archive.
        :type data: list[PipelineData]
        :param filename: The name of the directory to write the archive
            files to.
        :type filename: str
        :return: The achived data.
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
    """Writer for a file tree in NeXus format."""
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
    """Writer for H5 files from an nexusformat.nexus.NXdata object."""
    def write(self, data, filename, force_overwrite=False):
        """Write the NeXus object contained in `data` to hdf5 file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
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


class ImageWriter(Writer):
    """Writer for saving image files."""
    def write(
            self, data, outputdir, filename=None, force_overwrite=False,
            remove=True):
        """Write the image(s) contained in `data` to file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param outputdir: The name of the directory to write to.
        :type outputdir: str
        :param filename: The name of the file to write to (for a
            single image or a tiff image stack, with a valid extension).
        :type filename: str, optional
        :param force_overwrite: Flag to allow files to be
            overwritten if they already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :param remove: If there is a matching entry in `data`, remove
            it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises RuntimeError: If a file already exists and
            `force_overwrite` is `False`.
        :return: The data written to disk.
        :rtype: list, dict, matplotlib.animation.FuncAnimation,
            numpy.ndarray
        """
        # System modules
        from io import BytesIO

        # Third party modules
        from matplotlib.animation import (
            ArtistAnimation,
            FuncAnimation,
        )

        # Local modules
        from CHAP.utils.general import save_iobuf_fig

        try:
            ddata = self.get_data(
                data, schema='common.write.ImageWriter', remove=remove)
        except ValueError:
            self.logger.warning(
                'Unable to find match with schema `common.write.ImageWriter`: '
                'return without writing')
            return None
        if isinstance(ddata, list):
            for (buf, fileformat), basename in ddata:
                filename = f'{basename}.{fileformat}'
                if not os_path.isabs(filename):
                    filename = os_path.join(outputdir, filename)
                if isinstance(buf, (ArtistAnimation, FuncAnimation)):
                    buf.save(filename)
                else:
                    save_iobuf_fig(
                        buf, filename, force_overwrite=force_overwrite)
            return ddata

        if isinstance(ddata, dict):
            fileformat = ddata['fileformat']
            image_data = ddata['image_data']
        else:
            image_data = ddata
        basename, ext = os_path.splitext(filename)
        if ext[1:] != fileformat:
            filename = f'{filename}.{fileformat}'
        if not os_path.isabs(filename):
            filename = os_path.join(outputdir, filename)
        if os_path.isfile(filename) and not force_overwrite:
            raise FileExistsError(f'{filename} already exists')
        if isinstance(image_data, BytesIO):
            save_iobuf_fig(
                image_data, filename, force_overwrite=force_overwrite)
        elif isinstance(image_data, np.ndarray):
            if image_data.ndim == 2:
                # Third party modules
                from imageio import imwrite

                imwrite(filename, image_data)
            elif image_data.ndim == 3:
                # Third party modules
                from tifffile import imwrite

                kwargs = {'bigtiff': True}
                imwrite(filename, image_data, **kwargs)
        elif isinstance(image_data, (ArtistAnimation, FuncAnimation)):
            image_data.save(filename)
        else:
            raise ValueError(f'Invalid image input type {type(image_data)}')
        return ddata


class MatplotlibAnimationWriter(Writer):
    """Writer for saving matplotlib animations."""
    def write(self, data, filename, fps=1):
        """Write the matplotlib.animation.ArtistAnimation object
        contained in `data` to file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
        :param fps: Movie frame rate (frames per second),
            defaults to `1`.
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
    def write(self, data, filename, savefig_kw=None, force_overwrite=False):
        """Write the matplotlib.figure.Figure contained in `data` to
        file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
        :param savefig_kw: Keyword args to pass to
            matplotlib.figure.Figure.savefig.
        :type savefig_kw: dict, optional
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The original figure object.
        :rtype: matplotlib.figure.Figure
        """
        data = self.unwrap_pipelinedata(data)[-1]
        write_matplotlibfigure(data, filename, savefig_kw, force_overwrite)

        return data


class NexusWriter(Writer):
    """Writer for NeXus files from `NXobject`-s."""
    def write(
            self, data, filename, nxpath=None, force_overwrite=False,
            remove=False):
        """Write the NeXus object contained in `data` to file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :param remove: Flag to remove the NeXus object from `data`.
        :type remove: bool, optional.
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to file.
        :rtype: nexusformat.nexus.NXobject
        """
        # Third party modules
        from nexusformat.nexus import (
            NXFile,
            NXentry,
            NXobject,
            NXroot,
        )

        if remove:
            print(f'remove parameter not implemented yet')
        nxobject = self.get_data(data, remove=remove)

        nxname = nxobject.nxname
        if not os_path.isfile(filename) and nxpath is not None:
            self.logger.warning(
                f'{filename} does not yet exist. Argument for nxpath '
                '({nxpath}) will be ignored.')
            nxpath = None
        if nxpath is None:
            nxclass = nxobject.nxclass
            if nxclass == 'NXroot':
                nxroot = nxobject
            elif nxclass == 'NXentry':
                nxroot = NXroot(nxobject)
                nxroot[nxname].set_default()
            else:
                nxroot = NXroot(NXentry(nxobject))
                if nxclass == 'NXdata':
                    nxroot.entry[nxname].set_default()
                nxroot.entry.set_default()
            write_nexus(nxroot, filename, force_overwrite)
        else:
            nxfile = NXFile(filename, 'rw')
            root = nxfile.readfile()
            if nxfile.get(nxpath) is None:
                if nxfile.get(os_path.dirname(nxpath)) is not None:
                    nxpath, nxname = os_path.split(nxpath)
                else:
                    nxpath = root.NXentry[0].nxpath
                    self.logger.warning(
                        f'Path "{nxpath}" not present in {filename}. '
                        f'Using {nxpath} instead.')
            full_nxpath = os_path.join(nxpath, nxname)
            self.logger.debug(f'Full path for object to write: {full_nxpath}')
            if nxfile.get(full_nxpath) is not None:
                self.logger.debug(
                    f'{os_path.join(nxpath, nxname)} already exists in '
                    f'{filename}')
                if force_overwrite:
                    self.logger.warning(
                        'Deleting existing NXobject at '
                        f'{os_path.join(nxpath, nxname)} in {filename}')
                    del root[full_nxpath]
            try:
                root[full_nxpath] = nxobject
            except Exception as exc:
                nxfile.close()
                raise exc
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

        :param data: The data to write to file.
        :type data: Union[list[PipelineData],
            list[pyFAI.containers.IntegrateResult]]
        :param filename: Name of the file to which results will be
            saved. Format of output is determined ffrom the
            extension. Currently supported formats are: `.npz`,
            `.nxs`.
        :type filename: str
        """
        from os import remove

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
                'all pyFAI.containers.Integrate1dResult, or all '
                'pyFAI.containers.Integrate2dResult.')

        if os_path.isfile(filename):
            if force_overwrite:
                self.logger.warning(f'Removing existing file {filename}')
                remove(filename)
            else:
                raise Exception(f'{filename} already exists.')
        _, ext = os_path.splitext(filename)
        if ext.lower() == '.npz':
            self.write_npz(results, filename)
        elif ext.lower() == '.nxs':
            self.write_nxs(results, filename)
        else:
            raise Exception(f'Unsupported file format: {ext}')
        self.logger.info(f'Wrote to {filename}')
        return results

    def write_npz(self, results, filename):
        """Save `results` to the .npz file, `filename`."""

        data = {'radial': results[0].radial,
                'intensity': [r.intensity for r in results]}
        if hasattr(results[0], 'azimuthal'):
            # 2d results
            data['azimuthal'] = results[0].azimuthal
        if all([r.sigma for r in results]):
            # errors were included
            data['sigma'] = [r.sigma for r in results]

        np.savez(filename, **data)

    def write_nxs(self, results, filename):
        """Save `results` to the .nxs file, `filename`."""
        raise NotImplementedError


class TXTWriter(Writer):
    """Writer for plain text files from string or tuples or lists of
    strings."""
    def write(self, data, filename, append=False, force_overwrite=False):
        """Write a string or tuple or list of strings contained in 
        `data` to file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
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
    """Writer for YAML files from `dict`-s."""
    def write(self, data, filename, force_overwrite=False, remove=False):
        """Write the dictionary contained in `data` to file.

        :param data: The data to write to file.
        :type data: list[PipelineData]
        :param filename: The name of the file to write to.
        :type filename: str
        :param force_overwrite: Flag to allow data in `filename` to be
            overwritten if it already exists, defaults to `False`.
        :type force_overwrite: bool, optional
        :param remove: Flag to remove the NeXus object from `data`.
        :type remove: bool, optional.
        :raises TypeError: If the object contained in `data` is not a
            `dict`.
        :raises RuntimeError: If `filename` already exists and
            `force_overwrite` is `False`.
        :return: The data written to file.
        :rtype: dict
        """
        # Third party modules
        from pydantic import BaseModel

        # Local modules
        from CHAP.models import CHAPBaseModel

        if remove:
            print(f'remove parameter not implemented yet')
        yaml_dict = None
        for i, d in reversed(list(enumerate(data))):
            ddata = d['data']
            if isinstance(ddata, dict):
                yaml_dict = ddata
#                if remove:
#                    data.pop(i)
                break
            if isinstance(ddata, (BaseModel, CHAPBaseModel)):
                try:
                    yaml_dict = ddata.model_dump()
#                    if remove:
#                        data.pop(i)
                    break
                except:
                    pass
        write_yaml(yaml_dict, filename, force_overwrite)
        return data


if __name__ == '__main__':
    # Local modules
    from CHAP.writer import main

    main()
