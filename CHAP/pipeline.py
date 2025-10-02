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
    def __init__(self, pipeline_items=None, pipeline_kwargs=None):
        """Pipeline class constructor.

        :param pipeline_items: List of pipeline item objects, optional.
        :type pipeline_items: list[obj]
        :param pipeline_kwargs: List of method keyword arguments for
            the pipeline item objects, optional.
        :type pipeline_kwargs: list[dict]
        """
        self.__name__ = self.__class__.__name__

        self._data = []
        self._items = pipeline_items
        self._kwargs = pipeline_kwargs

        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def validate(self):
        """validate API."""
        t0 = time()
        self.logger.info('Executing "validate"\n')

        for item, kwargs in zip(self._items, self._kwargs):
            if hasattr(item, 'validate'):
                self.logger.info(f'Calling "validate" on {item}')
                item.validate(**kwargs)
        self.logger.info(f'Executed "validate" in {time()-t0:.3f} seconds')

    def execute(self):
        """execute API."""
        t0 = time()
        self.logger.info('Executing "execute"\n')

        for item, kwargs in zip(self._items, self._kwargs):
            if hasattr(item, 'execute'):
                self.logger.info(f'Calling "execute" on {item}')
                data = item.execute(data=self._data, **kwargs)
                name = kwargs.get('name', item.__name__)
                if item.method_type == 'read':
                    self._data.append(PipelineData(
                        name=name, data=data, schema=item.schema))
                elif item.method_type == 'process':
                    if isinstance(data, tuple):
                        self._data.extend(
                            [d if isinstance(d, PipelineData)
                             else PipelineData(
                                 name=name, data=d, schema=item.schema)
                             for d in data])
                    else:
                        self._data.append(PipelineData(
                            name=name, data=data, schema=item.schema))
        self.logger.info(f'Executed "execute" in {time()-t0:.3f} seconds')
        return self._data


class PipelineData(dict):
    """Wrapper for all results of PipelineItem.execute."""
    def __init__(self, name=None, data=None, schema=None):
        super().__init__()
        self.__setitem__('name', name)
        self.__setitem__('data', data)
        self.__setitem__('schema', schema)


class PipelineItem():
    """An object that can be supplied as one of the items
    in `Pipeline.items`.
    """
    def __init__(
            self, inputdir='.', outputdir='.', interactive=False, schema=None):
        """Constructor of PipelineItem class."""
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

        self._inputdir = inputdir
        self._outputdir = outputdir
        self._interactive = interactive
        self._schema = schema

        self._args = {}
        self._method_type = None
        if hasattr(self, 'read'):
            self._method_type = 'read'
        elif hasattr(self, 'process'):
            self._method_type = 'process'
        elif hasattr(self, 'write'):
            self._method_type = 'write'
        else:
            self._method_type = None

    @property
    def method_type(self):
        return self._method_type

    @property
    def schema(self):
        return self._schema

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
        unwrapped_data = []
        if isinstance(data, list):
            for d in data:
                if isinstance(d, PipelineData):
                    unwrapped_data.append(d['data'])
                else:
                    unwrapped_data.append(d)
        else:
            unwrapped_data = [data]
        return unwrapped_data

    def get_config(
            self, data=None, config=None, schema=None, remove=True, **kwargs):
        """Look through `data` for an item whose value for the first
        `'schema'` key matches `schema`. Convert the value for that
        item's `'data'` key into the configuration's Pydantic model
        identified by `schema` and return it. If no item is found and
        config is specified, validate it against the configuration's
        Pydantic model identified by `schema` and return it.

        :param data: Input data from a previous `PipelineItem`.
        :type data: list[PipelineData], optional
        :param config: Initialization parameters for an instance of
            the Pydantic model identified by `schema`, required if
            data is unspecified, invalid or does not contain an item
            that matches the schema.
        :type config: dict, optional
        :param schema: Name of the `BaseModel` class to match in
            `data` & return.
        :type schema: str, optional
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
        if data is not None:
            try:
                for i, d in enumerate(data):
                    if d.get('schema') == schema:
                        matching_config = d.get('data')
                        if remove:
                            data.pop(i)
                        break
            except:
                pass

        if not matching_config:
            if isinstance(config, dict):
                matching_config = config
            else:
                raise ValueError(
                    f'Unable to find a configuration for schema `{schema}`')
        if self._method_type == 'read':
            matching_config['inputdir'] = self._inputdir

        if schema is None:
            raise ValueError(f'Missing schema {type(self)} configuration')
        mod_name, cls_name = schema.rsplit('.', 1)
        module = __import__(f'CHAP.{mod_name}', fromlist=cls_name)
        matching_config.update(kwargs)
        model_config = getattr(module, cls_name)(**matching_config)

        self.logger.debug(
            f'Got {schema} configuration in {time()-t0:.3f} seconds')

        return model_config

    def get_data(self, data, name=None, schema=None, remove=True):
        """Look through `data` for an item whose `'data'` value is
        a nexusformat.nexus.NXobject object or matches a given name or
        schema. Pick the item for which
        the `'name'` key matches `name` if set or the `'schema'` key
        matches `schema` if set, pick the last match for a 
        nexusformat.nexus.NXobject object otherwise.
        Return the data object.

        :param data: Input data from a previous `PipelineItem`.
        :type data: list[PipelineData].
        :param name: Name of the data item to match in `data` & return.
        :type name: str, optional
        :param schema: Name of the `BaseModel` class to match in
            `data` & return.
        :type schema: str, optional
        :param remove: If there is a matching entry in `data`, remove
            it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: If there's no match for `name` or 'schema`
            in `data`, or if there is no object of type
            nexusformat.nexus.NXobject.
        :return: The last matching data item.
        :rtype: obj
        """
        # Third party modules
        from nexusformat.nexus import NXobject

        result = None
        t0 = time()
        if name is None and schema is None:
            for i, d in reversed(list(enumerate(data))):
                if isinstance(d.get('data'), NXobject):
                    result = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(f'No NXobject data item found')
        elif name is not None:
            self.logger.debug(f'Getting data item named "{name}"')
            for i, d in reversed(list(enumerate(data))):
                if d.get('name') == name:
                    result = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(f'No match for data item named "{name}"')
        elif schema is not None:
            self.logger.debug(f'Getting data item with schema "{schema}"')
            for i, d in reversed(list(enumerate(data))):
                if d.get('schema') == schema:
                    result = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(
                    f'No match for data item with schema "{schema}"')
        self.logger.debug(
           f'Obtained pipeline data in {time()-t0:.3f} seconds')

        return result

    def validate(self, **kwargs):
        """Validate the appropriate method of the object."""
        self._method = getattr(self, self._method_type)
        self._allowed_args = inspect.getfullargspec(self._method).args +\
                             inspect.getfullargspec(self._method).kwonlyargs
        if self._method_type == 'read':
            if 'inputdir' in self._allowed_args:
                self._args['inputdir'] = self._inputdir
            if 'filename' in kwargs:
                filename = kwargs.pop('filename')
                newfilename = os.path.normpath(os.path.realpath(
                    os.path.join(self._inputdir, filename)))
                if (not os.path.isfile(newfilename)
                        and not os.path.dirname(filename)):
                    self.logger.warning(
                        f'Unable to find {filename} in {self._inputdir}, '
                        f' looking in {self._outputdir}')
                    newfilename = os.path.normpath(os.path.realpath(
                        os.path.join(self._outputdir, filename)))
                kwargs['filename'] = newfilename
        elif self._method_type == 'write':
            if 'filename' in kwargs:
                filename = os.path.normpath(os.path.realpath(
                    os.path.join(self._outputdir, kwargs['filename'])))
                if (not kwargs.get('force_overwrite', False)
                        and os.path.isfile(filename)):
                    raise ValueError(
                        'Writing to an existing file without overwrite '
                        f'permission. Remove {filename} or set '
                        '"force_overwrite" in the pipeline configuration for '
                        f'{self.__name__}')
                kwargs['filename'] = filename
            elif 'filename' in self._allowed_args:
                raise ValueError(
                    'Missing parameter "filename" in pipeline configuration '
                    f'for {self.__name__}')

        elif self._method_type != 'process':
            self.logger.error('No implementation of read, process, or write')
            return
        if 'schema' in self._allowed_args:
            self._args['schema'] = self._schema
        for k, v in kwargs.items():
            if k in self._allowed_args:
                self._args[k] = v

        #if self._method_type != 'process':
        if self._method_type == 'read':
            self.logger.debug(f'Validating "{self._method_type}" with schema '
                              f'"{self._schema}" and {self._args}')
            self.logger.info(f'Validating "{self._method_type}"')
            self._method(**self._args)


    def execute(self, data, **kwargs):
        """Run the appropriate method of the object and return the
        result.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: The wrapped result of running read, process, or write.
        :rtype: Union[PipelineData, tuple[PipelineData]]
        """
        if 'data' in self._allowed_args:
            self._args['data'] = data

        t0 = time()
        self.logger.debug(f'Executing "{self._method_type}" with schema '
                          f'"{self._schema}" and {self._args}')
        self.logger.info(f'Executing "{self._method_type}"')
        data = self._method(**self._args)
        self.logger.info(
            f'Finished "{self._method}" in {time()-t0:.0f} seconds\n')
        return data


class MultiplePipelineItem(PipelineItem):
    """An object to deliver results from multiple `PipelineItem`s to a
    single `PipelineItem` in the `Pipeline.execute()` method.
    """
    def execute(self, data, items=None, **kwargs):
        """Independently execute all items in `items`, then
        return all of their results.

        :param items: PipelineItem configurations.
        :type items: list, optional
        :return: The wrapped result of running multiple read, process,
            or write.
        :rtype: list[PipelineData]
        """
        # System modules
        from copy import deepcopy
        from tempfile import NamedTemporaryFile

        t0 = time()
        self.logger.info(f'Executing {len(items)} PipelineItems')

        if items is None:
            items = []
        item_list = []
        data_org = None
        for item in items:
            if isinstance(item, dict):
                item_name = list(item.keys())[0]
                item_args = item[item_name]
            elif isinstance(item, str):
                item_name = item
                item_args = {}
            else:
                raise RuntimeError(
                    f'Unknown item config type {type(item)}')
            mod_name, cls_name = item_name.rsplit('.', 1)
            module = __import__(f'CHAP.{mod_name}', fromlist=cls_name)

            current_item = getattr(module, cls_name)()
            item_list.append((current_item, item_args))
            if data_org is None and hasattr(current_item, 'write'):
                data_org = deepcopy(data)

        for (item, item_args) in item_list:
            # Combine the command line arguments "inputdir",
            # "outputdir" and "interactive" with the item's arguments
            # joining "inputdir" and "outputdir" and giving precedence
            # for "interactive" in the latter
            args = {**kwargs}
            if 'inputdir' in item_args:
                inputdir = os.path.normpath(os.path.realpath(os.path.join(
                    args['inputdir'], item_args.pop('inputdir'))))
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
                outputdir = os.path.normpath(os.path.realpath(os.path.join(
                    args['outputdir'], item_args.pop('outputdir'))))
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
                item.execute(data=data, **args)
                data = data_org
            else:
                data = item.execute(data=data, **args)

        self.logger.info(
            f'Finished executing {len(items)} PipelineItems in {time()-t0:.0f}'
            ' seconds\n')

        return data
