#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : pipeline.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description:
"""

# System modules
import inspect
import logging
import os
from time import time


class Pipeline():
    """Pipeline represent generic Pipeline class."""
    def __init__(self, items=None, kwds=None):
        """Pipeline class constructor.

        :param items: List of objects.
        :param kwds: List of method args for individual objects.
        """
        self.__name__ = self.__class__.__name__

        self.items = items
        self.kwds = kwds

        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def execute(self):
        """execute API."""
        t0 = time()
        self.logger.info('Executing "execute"\n')

        #data = [PipelineData()]
        data = []
        for item, kwargs in zip(self.items, self.kwds):
            if hasattr(item, 'execute'):
                self.logger.info(f'Calling "execute" on {item}')
                data = item.execute(data=data, **kwargs)
        self.logger.info(f'Executed "execute" in {time()-t0:.3f} seconds')
        return data


class PipelineData(dict):
    """Wrapper for all results of PipelineItem.execute."""
    def __init__(self, name=None, data=None, schema=None):
        super().__init__()
        self.__setitem__('name', name)
        self.__setitem__('data', data)
        self.__setitem__('schema', schema)


class PipelineItem():
    """An object that can be supplied as one of the items
    `Pipeline.items`.
    """
    def __init__(self):
        """Constructor of PipelineItem class."""
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    @staticmethod
    def get_default_nxentry(nxobject):
        """Given a `nexusformat.nexus.NXroot` or 
        `nexusformat.nexus.NXentry` object, return the default or
        first `nexusformat.nexus.NXentry` match.

        :param nxobject: Input data.
        :type nxobject: nexusformat.nexus.NXroot,
            nexusformat.nexus.NXentry
        :raises ValueError: If unable to retrieve a
            `nexusformat.nexus.NXentry` object.
        :return: The input data if a `nexusformat.nexus.NXentry`
            object or the default or first `nexusformat.nexus.NXentry`
            object if a `nexusformat.nexus.NXroot` object.
        :rtype: nexusformat.nexus.NXentry
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        if isinstance(nxobject, NXroot):
            if 'default' in nxobject.attrs:
                nxentry = nxobject[nxobject.default]
            else:
                nxentries = [
                    v for v in nxobject.values() if isinstance(v, NXentry)]
                if not nxentries:
                    raise ValueError(f'Unable to retrieve a NXentry object')
                elif len(nxentries) != 1:
                    self.logger.warning(
                        f'Found multiple NXentries, returning the first')
                nxentry = nxentries[0]
        elif isinstance(nxobject, NXentry):
            nxentry = nxobject
        else:
            raise ValueError(f'Invalid parameter nxobject ({nxobject})')
        return nxentry

    @staticmethod
    def unwrap_pipelinedata(data):
        """Given a list of PipelineData objects, return a list of
        their `data` values.

        :param data: Input data to read, write, or process that needs
            to be unwrapped from PipelineData before use.
        :type data: list[PipelineData]
        :return: The `'data'` values of the items in the input data.
        :rtype: list[object]
        """
        return [d['data'] for d in data]

    def get_config(self, data, schema, remove=True, **kwargs):
        """Look through `data` for an item whose value for the first
        `'schema'` key matches `schema`. Convert the value for that
        item's `'data'` key into the configuration `BaseModel`
        identified by `schema` and return it.

        :param data: Input data from a previous `PipelineItem`.
        :type data: list[PipelineData].
        :param schema: Name of the `BaseModel` class to match in
            `data` & return.
        :type schema: str
        :param remove: If there is a matching entry in `data`, remove
           it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: If there's no match for `schema` in `data`.
        :return: The first matching configuration model.
        :rtype: BaseModel
        """
        self.logger.debug(f'Getting {schema} configuration')
        t0 = time()

        matching_config = False
        for i, d in enumerate(data):
            if d.get('schema') == schema:
                matching_config = d.get('data')
                if remove:
                    data.pop(i)
                break

        if not matching_config:
            raise ValueError(
                f'Unable to find a configuration for schema `{schema}`')

        mod_name, cls_name = schema.rsplit('.', 1)
        module = __import__(f'CHAP.{mod_name}', fromlist=cls_name)
        matching_config.update(kwargs)
        model_config = getattr(module, cls_name)(**matching_config)

        self.logger.debug(
            f'Got {schema} configuration in {time()-t0:.3f} seconds')

        return model_config

    def get_data(self, data, name=None, remove=True):
        """Look through `data` for an item which is either a
        nexusformat.nexus.NXroot or a nexusformat.nexus.NXentry
        object. Pick the item for which the `'name'` key matches
        `name` if set, pick the first match otherwise.
        Return the default nexusformat.nexus.NXentry object.

        :param data: Input data from a previous `PipelineItem`.
        :type data: list[PipelineData].
        :param name: Name of the data item to match in `data` & return.
        :type name: str
        :param remove: If there is a matching entry in `data`, remove
            it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: If there's no match for `name` in `data`,
            or if the associated object is not of type 
            nexusformat.nexus.NXroot or nexusformat.nexus.NXentry.
        :return: The first matching data item.
        :rtype: Union[nexusformat.nexus.NXroot,
            nexusformat.nexus.NXentry]
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        nxobject = None
        t0 = time()
        if name is None:
            for i, d in enumerate(data):
                if isinstance(d.get('data'), (NXroot, NXentry)):
                    nxobject = d.get('data')
                    name = d.get('name')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(f'No NXroot or NXentry data item found')
        else:
            self.logger.debug(f'Getting {name} data item')
            for i, d in enumerate(data):
                if (d.get('name') == name
                        and isinstance(d.get('data'), (NXroot, NXentry))):
                    nxobject = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(f'No match for {name} data item found')
        self.logger.debug(
           f'Got {name} data in {time()-t0:.3f} seconds')

        return nxobject

    def execute(self, schema=None, **kwargs):
        """Run the appropriate method of the object and return the
        result.

        :param schema: The name of a schema associated with the data
            that will be returned.
        :type schema: str
        :param kwargs: A dictionary of any positional and keyword
            arguments to supply to the read, process, or write method.
        :type kwargs: dict
        :return: The wrapped result of running read, process, or write.
        :rtype: list[PipelineData]
        """
        if hasattr(self, 'read'):
            method_name = 'read'
            inputdir = kwargs.get('inputdir')
            if inputdir is not None and 'filename' in kwargs:
                kwargs['filename'] = os.path.realpath(
                    os.path.join(inputdir, kwargs['filename']))
        elif hasattr(self, 'process'):
            method_name = 'process'
        elif hasattr(self, 'write'):
            method_name = 'write'
            outputdir = kwargs.get('outputdir')
            if outputdir is not None and 'filename' in kwargs:
                kwargs['filename'] = os.path.realpath(
                    os.path.join(outputdir, kwargs['filename']))
        else:
            self.logger.error('No implementation of read, write, or process')
            return None

        method = getattr(self, method_name)
        allowed_args = inspect.getfullargspec(method).args \
                       + inspect.getfullargspec(method).kwonlyargs
        args = {}
        for k, v in kwargs.items():
            if k in allowed_args:
                args[k] = v

        t0 = time()
        self.logger.debug(f'Executing "{method_name}" with {args}')
        self.logger.info(f'Executing "{method_name}"')
        data = method(**args)
        self.logger.info(f'Finished "{method_name}" in '
                         + f'{time()-t0:.0f} seconds\n')

        return [PipelineData(name=self.__name__,
                             data=data,
                             schema=schema)]


class MultiplePipelineItem(PipelineItem):
    """An object to deliver results from multiple `PipelineItem`s to a
    single `PipelineItem` in the `Pipeline.execute()` method.
    """
    def execute(self, items=None, **kwargs):
        """Independently execute all items in `self.items`, then
        return all of their results.

        :param items: PipelineItem configurations.
        :type items: list, optional
        :return: The wrapped result of running multiple read, process,
            or write.
        :rtype: list[PipelineData]
        """
        # System modules
        from tempfile import NamedTemporaryFile

        t0 = time()
        self.logger.info(f'Executing {len(items)} PipelineItems')

        data = kwargs['data']
        if items is None:
            items = []
        for item_config in items:
            if isinstance(item_config, dict):
                item_name = list(item_config.keys())[0]
                item_args = item_config[item_name]
            elif isinstance(item_config, str):
                item_name = item_config
                item_args = {}
            else:
                raise RuntimeError(
                    f'Unknown item config type {type(item_config)}')

            mod_name, cls_name = item_name.rsplit('.', 1)
            module = __import__(f'CHAP.{mod_name}', fromlist=cls_name)
            item = getattr(module, cls_name)()
            # Combine the command line arguments "inputdir",
            # "outputdir" and "interactive" with the item's arguments
            # joining "inputdir" and "outputdir" and giving precedence
            # for "interactive" in the latter
            args = {**kwargs}
            if 'inputdir' in item_args:
                inputdir = os.path.normpath(os.path.join(
                    args['inputdir'], item_args.pop('inputdir')))
                if not os.path.isdir(inputdir):
                    raise OSError(
                        f'input directory does not exist ({inputdir})')
                if not os.access(inputdir, os.R_OK):
                    raise OSError('input directory is not accessible for '
                                  f'reading ({inputdir})')
                args['inputdir'] = inputdir
            # FIX: Right now this can bomb if MultiplePipelineItem
            # is called simultaneously from multiple nodes in MPI
            if 'outputdir' in item_args:
                outputdir = os.path.normpath(os.path.join(
                    args['outputdir'], item_args.pop('outputdir')))
                if not os.path.isdir(outputdir):
                    os.makedirs(outputdir)
                try:
                    NamedTemporaryFile(dir=outputdir)
                except Exception as exc:
                    raise OSError(
                        'output directory is not accessible for writing '
                        f'({outputdir})') from exc
                args['outputdir'] = outputdir
            args = {**args, **item_args}
            if hasattr(item, 'write'):
                item.execute(**args)[0]
            else:
                data.append(item.execute(**args)[0])

        self.logger.info(
            f'Finished executing {len(items)} PipelineItems in {time()-t0:.0f}'
            ' seconds\n')

        return data
