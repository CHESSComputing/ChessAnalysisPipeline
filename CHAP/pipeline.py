#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : pipeline.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description:
"""

# System modules
import logging
import os
from time import time
from types import MethodType
from typing import (
    Literal,
    Optional,
)

# Third party modules
from pydantic import (
    ConfigDict,
    Field,
    FilePath,
    PrivateAttr,
    conlist,
    constr,
    model_validator,
)

# Local modules
from CHAP.models import (
    CHAPBaseModel,
    RunConfig,
)


class PipelineData(dict):
    """Wrapper for all results of PipelineItem.execute."""
    def __init__(self, name=None, data=None, schema=None):
        super().__init__()
        self.__setitem__('name', name)
        self.__setitem__('data', data)
        self.__setitem__('schema', schema)


class PipelineItem(RunConfig):
    """Class representing a single item in a `Pipeline` object."""
    logger: Optional[logging.Logger] = None
    name: Optional[constr(strip_whitespace=True, min_length=1)] = None
    schema_: Optional[constr(strip_whitespace=True, min_length=1)] = \
        Field(None, alias='schema')

    _method: MethodType = PrivateAttr(default=None)
    _method_type: Literal[
        'read', 'process', 'write'] = PrivateAttr(default=None)
    _args: dict = PrivateAttr(default={})
    _allowed_args: conlist(item_type=str) = PrivateAttr(default=[])
    _required_args: conlist(item_type=str) = PrivateAttr(default=[])
    _status: Literal[
        'read', 'write_pending', 'written'] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate_config(self):
        """Validate the `PipelineItem` configuration.

        :return: The validated configuration.
        :rtype: PipelineItem
        """
        # System modules
        from inspect import (
            Parameter,
            signature,
        )

        if self.name is None:
            self.__name__ = self.__class__.__name__
        else:
            self.__name__ = self.name
        if self.logger is None:
            self.logger = logging.getLogger(self.__name__)
            self.logger.propagate = False
            log_handler = logging.StreamHandler()
            log_handler.setFormatter(logging.Formatter(
                '{asctime}: {name:20}: {levelname}: {message}',
                datefmt='%Y-%m-%d %H:%M:%S', style='{'))
            self.logger.addHandler(log_handler)

        if hasattr(self, 'read'):
            self._method_type = 'read'
        elif hasattr(self, 'process'):
            self._method_type = 'process'
        elif hasattr(self, 'write'):
            self._method_type = 'write'
        else:
            return self
        self._method = getattr(self, self._method_type)
        sig = signature(self._method)
        self._allowed_args = [k for k, v in sig.parameters.items()
                              if v.kind == v.POSITIONAL_OR_KEYWORD]
        self._required_args = [k for k, v in sig.parameters.items()
                               if (v.kind == v.POSITIONAL_OR_KEYWORD
                                   and v.default is Parameter.empty)]
        return self

    @property
    def method(self):
        return self._method

    @property
    def method_type(self):
        return self._method_type

    @property
    def schema(self):
        return self.schema_

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def get_args(self):
        return self._args

    def set_args(self, **args):
        for k, v in args.items():
            if k in self._allowed_args:
                self._args[k] = v

    def get_required_args(self):
        return self._required_args

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
        """Look through `data` for the first item whose value for the
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
        :param schema: Name of the `PipelineItem` class to match in
            `data` & return, defaults to the internal PipelineItem
            `schema` attribute.
        :type schema: str, optional
        :param remove: If there is a matching entry in `data`, remove
           it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: If there's no match for `schema` in `data`.
        :return: The first matching configuration model.
        :rtype: PipelineItem
        """
        self.logger.debug(f'Getting {schema} configuration')
        t0 = time()

        if schema is None:
            schema = self.schema
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
        if self._method_type == 'read' and 'inputdir' not in matching_config:
            matching_config['inputdir'] = self.inputdir
        if self._method_type == 'write' and 'outputdir' not in matching_config:
            matching_config['outputdir'] = self.outputdir

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
        :param schema: Name of the `PipelineItem` class to match in
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

    def execute(self, data):
        """Run the appropriate method of the object and return the
        result.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: The wrapped result of running read, process, or write.
        :rtype: Union[PipelineData, tuple[PipelineData]]
        """
        self._required_args
        if 'data' in self._allowed_args:
            self._args['data'] = data
        t0 = time()
        self.logger.debug(f'Executing "{self._method_type}" with schema '
                          f'"{self.schema}" and {self._args}')
        self.logger.info(f'Executing "{self._method_type}"')
        data = self._method(**self._args)
        self.logger.info(
            f'Finished "{self._method}" in {time()-t0:.0f} seconds\n')
        return data


class Pipeline(CHAPBaseModel):
    """Class representing a full `Pipeline` object."""
    args: conlist(item_type=dict, min_length=1)
    items: conlist(item_type=PipelineItem, min_length=1)
    logger: Optional[logging.Logger] = None

    _data: conlist(item_type=PipelineData) = PrivateAttr(default=[])
    _output_filenames: conlist(item_type=FilePath) = PrivateAttr(default=[])
    _filename_mapping: dict = PrivateAttr(default={})

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate_config(self):
        """Validate the `Pipeline` configuration and initialize and
        validate the private attributes.

        :return: The validated configuration.
        :rtype: Pipeline
        """
        t0 = time()
        self.__name__ = self.__class__.__name__
        if self.logger is None:
            self.logger = logging.getLogger(self.__name__)
            self.logger.propagate = False

        for item, args in zip(self.items, self.args):
            if hasattr(item, 'filename') and item.filename is not None:
                if item.method_type == 'read':
                    if item._mapping_filename in self._filename_mapping:
                        item.filename = \
                            self._filename_mapping[
                                item._mapping_filename]['path']
                        item.status = \
                            self._filename_mapping[
                                item._mapping_filename]['status']
                    else:
                        if item.filename in self._output_filenames:
                            self._filename_mapping[item._mapping_filename] = {
                                'path': item.filename,
                                'status': 'write_pending'}
                            item.status = 'write_pending'
                        else:
                            self._filename_mapping[item._mapping_filename] = {
                                'path': item.filename, 'status': None}
                elif item.method_type == 'write':
                    if (not item.force_overwrite
                            and self.filename in self._output_filenames):
                        raise ValueError(
                            'Writing to an existing file without overwrite '
                            f'permission. Remove {self.filename} or set '
                            '"force_overwrite" in the pipeline configuration '
                            f'for {item.name}')
            item.set_args(**args)
            if item.method_type == 'read':
                if item.status not in ('read', 'write_pending'):
                    self.logger.debug(
                        f'Validating "{item.method_type}" with schema '
                        f'"{item.schema}" and {item.get_args()}')
                    self.logger.info(f'Validating "{item.method_type}"')
                    data = item.method(**item.get_args())
                    self._data.append(PipelineData(
                        name=item.name, data=data, schema=item.schema))
                    if hasattr(item, 'filename') and item.filename is not None:
                        self._filename_mapping[
                            item._mapping_filename]['status'] = 'read'
            elif item.method_type == 'write':
                if hasattr(item, 'filename') and item.filename is not None:
                    for k, v in self._filename_mapping.items():
                        if v['path'] == item.filename:
                            self._filename_mapping[k]['status'] = \
                                'write_pending'
                    if item.filename not in self._output_filenames:
                        self._output_filenames.append(item.filename)
        self.logger.info(
            f'Executed "validate_config" in {time()-t0:.3f} seconds')

        return self

    def execute(self):
        """Executes the pipeline."""
        t0 = time()
        self.logger.info('Executing "execute"\n')

        for item, args in zip(self.items, self.args):
            if hasattr(item, 'execute'):
                self.logger.info(f'Calling "execute" on {item}')
                if (item.method_type == 'read' and hasattr(item, 'filename')
                        and item.filename is not None):
                    item.status = self._filename_mapping[
                        item._mapping_filename]['status']
                data = item.execute(data=self._data)
                if item.method_type == 'read':
                    self._data.append(PipelineData(
                        name=item.name, data=data, schema=item.schema))
                elif item.method_type == 'process':
                    if isinstance(data, tuple):
                        self._data.extend(
                            [d if isinstance(d, PipelineData)
                             else PipelineData(
                                 name=item.name, data=d, schema=item.schema)
                             for d in data])
                    else:
                        self._data.append(PipelineData(
                            name=item.name, data=data, schema=item.schema))
                elif item.method_type == 'write':
                    if hasattr(item, 'filename') and item.filename is not None:
                        for k, v in self._filename_mapping.items():
                            if v['path'] == item.filename:
                                self._filename_mapping[k]['status'] = 'written'
        self.logger.info(f'Executed "execute" in {time()-t0:.3f} seconds')
        return self._data
