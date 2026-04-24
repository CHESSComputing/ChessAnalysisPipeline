#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Base pipeline `Pydantic <https://github.com/pydantic/pydantic>`__
model classes.
"""

# System modules
import logging
from time import time
from types import MethodType
from typing import (
    Literal,
    Optional,
)

# Third party modules
from pydantic import (
#    ConfigDict,
    Field,
#    FilePath,
    PrivateAttr,
    conlist,
    constr,
    model_validator,
)
from pydantic._internal._model_construction import ModelMetaclass

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
    """Class representing a single item in a `Pipeline` object.

    :ivar logger: CHAP logger.
    :vartype logger: logging.Logger, optional
    :ivar name: `Pipeline` object name.
    :vartype name: str, optional
    :ivar schema: `Pipeline` object schema.
    :vartype schema: str, optional
    """

    logger: Optional[logging.Logger] = None
    name: Optional[constr(strip_whitespace=True, min_length=1)] = None
    schema_: Optional[constr(strip_whitespace=True, min_length=1)] = \
        Field(None, alias='schema')

    _method: MethodType = PrivateAttr(default=None)
    _method_type: Literal[
        'read', 'process', 'write'] = PrivateAttr(default=None)
    _args: dict = PrivateAttr(default={})
    _allowed_args: conlist(item_type=str) = PrivateAttr(default=[])
#    _metadata: dict = PrivateAttr(default=None)
#    _provenance: dict = PrivateAttr(default=None)
    _status: Literal[
        'read', 'write_pending', 'written'] = PrivateAttr(default=None)

    #FIX model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate_pipelineitem_after(self):
        """Validate the `PipelineItem` configuration.

        :return: Validated configuration.
        :rtype: PipelineItem
        """
        # System modules
        from inspect import signature

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
        self.logger.setLevel(self.log_level)
        # Optinal, but it's already available in the 'name' field
        #if self.get_schema() is None:
        #    mod_name = '.'.join(self.__class__.__module__.split('.')[1:])
        #    self.schema_ = f'{mod_name}.{self.__class__.__name__}'

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
        return self

    @property
    def method(self):
        """Return the `PipelineItem`\\s `read`, `process` or `write`
        method.

        :type: types.MethodType
        """
        return self._method

    @property
    def method_type(self):
        """Return the `PipelineItem`\\s execute method type.

        :type: Literal['read', 'process', 'write']
        """
        return self._method_type

    @property
    def run_config(self):
        """Return the `PipelineItem`\\s run configuration.

        :type: RunConfig
        """
        return RunConfig(**self.model_dump()).model_dump()

    @property
    def status(self):
        """Return the `PipelineItem`\\s status.

        :type: Literal['read', 'write_pending', 'written']
        """
        return self._status

    @status.setter
    def status(self, status):
        """Set the `PipelineItem`\\s status.

        :param status: `PipelineItem`\\s status.
        :type: Literal['read', 'write_pending', 'written']
        """
        self._status = status

    def get_args(self):
        """Return the `PipelineItem`\\s execution method run time
        arguments.

        :type: dict
        """
        return self._args

    def set_args(self, **args):
        """Set the `PipelineItem`\\s execution method run time
        arguments that are allowed by its method declaration.

        :param: `PipelineItem`\\s execution method run time arguments.
        :type: dict
        """
        for k, v in args.items():
            if k in self._allowed_args:
                self._args[k] = v

    def has_filename(self):
        """Does the `PipelineItem` has a `filename` class attribute?

        :return: `True` if the `PipelineItem` has a `filename` class
            attribute.
        :rtype: bool
        """
        return hasattr(self, 'filename') and self.filename is not None

    def get_schema(self):
        """Return the `PipelineItem`\\s schema.

        :type: str
        """
        return self.schema_

    def get_config(
            self, data=None, config=None, schema=None, remove=True):
        """Look through `data` for the last item which value for the
        `'schema'` key matches `schema`. Convert the value for that
        item's `'data'` key into the configuration's
        `Pydantic <https://github.com/pydantic/pydantic>`__ model
        identified by `schema` and return it. If no item is found and
        `config` and `schema` are specified, validate `config` against
        the configuration's Pydantic model identified by `schema` and
        return it. Return `config` if no item is found and `config` is
        specified, but `schema` is not.

        :param data: Input data.
        :type data: list[PipelineData], optional
        :param config: Initialization parameters for an instance of the
            `Pydantic <https://github.com/pydantic/pydantic>`__ model
            identified by `schema`, required if data is unspecified,
            invalid or does not contain an item that matches the
            schema, superseeds any equal parameters contained in
            `data`.
        :type config: dict, optional
        :param schema: Schema of the `PipelineItem` class to match in
             `data`, defaults to the internal PipelineItem `schema`
             attribute.
        :type schema: str, optional
        :param remove: If there is a matching entry in `data`, remove
           it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: If there's no match for `schema` in `data`.
        :return: Last matching validated configuration model.
        :rtype: PipelineItem
        """
        self.logger.debug(f'Getting {schema} configuration')
        t0 = time()

        if schema is None:
            schema = self.schema_
        matching_config = False
        if data is not None:
            try:
                for i, d in reversed(list(enumerate(data))):
                    if d.get('schema') == schema:
                        matching_config = d.get('data')
                        if remove:
                            data.pop(i)
                        break
            except Exception:
                pass

        if matching_config:
            if config is not None:
                # Local modules
                from CHAP.utils.general import dictionary_update

                # Update matching_config with config if both exist
                matching_config = dictionary_update(matching_config, config)
        else:
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
        model_config = getattr(module, cls_name)(**matching_config)

        self.logger.debug(
            f'Got {schema} configuration in {time()-t0:.3f} seconds')

        return model_config

    @staticmethod
    def get_data(data, name=None, schema=None, remove=True):
        """Look through `data` for the last item which `'data'` value
        is a NeXus style
        `NXobject <https://manual.nexusformat.org/classes/base_classes/NXobject.html#index-0>`__
        object or matches a given name or schema. Pick the last item for which
        the `'name'` key matches `name` if set or the `'schema'` key matches
        `schema` if set, pick the last match for a `NXobjecta` object
        otherwise. Return the data object.

        :param data: Input data.
        :type data: list[PipelineData].
        :param name: Name of the `PipelineItem` class to match in
            `data`.
        :type name: str, optional
        :param schema: Schema of the `PipelineItem` class to match in
            `data` & return.
        :type schema: str | list[str], optional
        :param remove: If there is a matching entry in `data`, remove
            it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: If there's no match for `name` or 'schema`
            in `data`, or if there is no object of type
            nexusformat.nexus.NXobject.
        :return: Last matching data item.
        :rtype: Any
        """
        # Third party modules
        from nexusformat.nexus import NXobject

        result = None
        if name is None and schema is None:
            for i, d in reversed(list(enumerate(data))):
                if isinstance(d.get('data'), NXobject):
                    result = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError('No NXobject data item found')
        elif name is not None:
            for i, d in reversed(list(enumerate(data))):
                if d.get('name') == name:
                    result = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(f'No match for data item named "{name}"')
        elif schema is not None:
            if isinstance(schema, str):
                schema = [schema]
            for i, d in reversed(list(enumerate(data))):
                if d.get('schema') in schema:
                    result = d.get('data')
                    if remove:
                        data.pop(i)
                    break
            else:
                raise ValueError(
                    f'No match for data item with schema "{schema}"')

        return result

    @staticmethod
    def get_default_nxentry(nxobject):
        """Given a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#index-0>`__ 
        object or a NeXus style
        `NXentry <https://manual.nexusformat.org/classes/base_classes/NXentry.html#index-0>`__
        object, return the default or first `NXentry` match.

        :param nxobject: Input data.
        :type nxobject: nexusformat.nexus.NXroot | nexusformat.nexus.NXentry
        :raises ValueError: If unable to retrieve a `NXentry` object.
        :return: Input data if a `NXentry` object or the default or first
            `NXentry` object if a `NXroot` object.
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
                    raise ValueError('Unable to retrieve a NXentry object')
                if len(nxentries) != 1:
                    print('WARNING: Found multiple NXentries, returning the '
                          'first')
                nxentry = nxentries[0]
        elif isinstance(nxobject, NXentry):
            nxentry = nxobject
        else:
            raise ValueError(f'Invalid parameter nxobject ({nxobject})')
        return nxentry

    @staticmethod
    def get_nxroot(nxobject):
        """Given a NeXus style
        `NXroot <https://manual.nexusformat.org/classes/base_classes/NXroot.html#index-0>`__ 
        object or a NeXus style
        `NXentry <https://manual.nexusformat.org/classes/base_classes/NXentry.html#index-0>`__
        object, return a `NXroot` object with the appropriate default path to
        the `NXentry` object set.

        :param nxobject: Input data.
        :type nxobject: nexusformat.nexus.NXroot | nexusformat.nexus.NXentry
        :raises ValueError: If unable to retrieve a
            `NXroot` or `NXentry` object.
        :return: Input data if a `NXroot` object or a `NXroot` object with the
             input as its default `NXentry` object.
        :return: `NXroot` object.
        :rtype: nexusformat.nexus.NXroot
        """
        # Third party modules
        from nexusformat.nexus import (
            NXentry,
            NXroot,
        )

        if isinstance(nxobject, NXroot):
            nxroot = nxobject
        elif isinstance(nxobject, NXentry):
            nxroot = NXroot()
            nxroot[nxobject.nxname] = nxobject
            nxobject.set_default()
        else:
            raise ValueError(f'Invalid nxobject ({type(nxobject)}')
        return nxroot

    @staticmethod
    def get_pipelinedata_item(data, index=-1, remove=False):
        """If 'data' is a list, then retrieve from `data` the list
        item matching `index` and return it's `data` value, otherwise
        return `data` itself.

        :param data: Input data.
        :type data: Any | list[PipelineData]
        :param index: List index of the item to retrieve from
            `data`, default to -1 or the last item in the list.
        :type index: int, optional
        :param remove: If there is a matching entry in `data`, remove
            it from the list, defaults to `False`.
        :type remove: bool, optional
        :return: Matching data item.
        :rtype: Any
        """
        if isinstance(data, list):
            if remove:
                return data.pop(index)['data']
            return data[index]['data']
        return data

    @staticmethod
    def unwrap_pipelinedata(data):
        """Given a list of PipelineData objects, return a list of
        their `data` values.

        :param data: Input data to read, write, or process that needs
            to be unwrapped from PipelineData before use.
        :type data: list[PipelineData]
        :return: `'data'` values of the items in the input data.
        :rtype: list
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

    def execute(self, data):#, metadata, provenance):
        """Execute the appropriate method of the object and return the
        result.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: Wrapped result of executing read, process, or write.
        :rtype: PipelineData | tuple[PipelineData]
        """
#        self._metadata = metadata
#        self._provenance = provenance

        if 'data' in self._allowed_args:
            self._args['data'] = data
        t0 = time()
        self.logger.debug(f'Executing "{self._method_type}" with schema '
                          f'"{self.get_schema()}" and {self._args}')
        self.logger.info(f'Executing "{self._method_type}"')
        data = self._method(**self._args)
        self.logger.info(
            f'Finished "{self._method}" in {time()-t0:.0f} seconds\n')
        return data


class Pipeline(CHAPBaseModel):
    """Class representing a full `Pipeline` object.

    :ivar args: List of `PipelineItem` arguments for each item in the
        full pipeline.
    :vartype args: list[dict]
    :ivar logger: CHAP logger.
    :vartype logger: logging.Logger, optional
    :ivar mmcs: List of `PipelineItem`\\s classes in the full pipeline.
    :vartype mmcs:
        list[pydantic._internal._model_construction.ModelMetaclass]
    """

    args: conlist(item_type=dict, min_length=1)
    logger: Optional[logging.Logger] = None
    mmcs: conlist(item_type=ModelMetaclass, min_length=1)

    _data: conlist(item_type=PipelineData) = PrivateAttr(default=[])
    _items: conlist(item_type=PipelineItem) = PrivateAttr(default=[])
    #_output_filenames: conlist(item_type=FilePath) = PrivateAttr(default=[])
    _filename_mapping: dict = PrivateAttr(default={})
#    _metadata: dict = PrivateAttr(
#        default={'application': 'CHAP', 'user_metadata': {}})
#    _provenance: dict = PrivateAttr(default={})

    #FIX model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def validate_pipeline_after(self):
        """Validate the `Pipeline` configuration and initialize and
        validate the private attributes.

        :return: Validated configuration.
        :rtype: Pipeline
        """
        t0 = time()
        self.__name__ = self.__class__.__name__
        if self.logger is None:
            self.logger = logging.getLogger(self.__name__)
            self.logger.propagate = False

        output_filenames = []
        for mmc, args in zip(self.mmcs, self.args):
            # FIX add a validation status, so that the validator
            # doesn't get executed twice with the config staying
            # on the pipeline for processors
            self.logger.info(f'Validating {mmc}')
            item = mmc(data=self._data, modelmetaclass=mmc, **args)
            if item.has_filename():
                if item.method_type == 'read':
                    if item._mapping_filename in self._filename_mapping:
                        item.filename = self._filename_mapping[
                            item._mapping_filename]['path']
                        item.status = self._filename_mapping[
                            item._mapping_filename]['status']
                    else:
                        #if item.filename in self._output_filenames:
                        if item.filename in output_filenames:
                            self._filename_mapping[item._mapping_filename] = {
                                'path': item.filename,
                                'status': 'write_pending'}
                            item.status = 'write_pending'
                        else:
                            self._filename_mapping[item._mapping_filename] = {
                                'path': item.filename, 'status': None}
                elif item.method_type == 'write':
                    if (not item.force_overwrite
                            and item.filename in output_filenames):
                            #and self.filename in self._output_filenames):
                        raise ValueError(
                            'Writing to an existing file without overwrite '
                            f'permission. Remove {item.filename} or set '
                            '"force_overwrite" in the pipeline configuration '
                            f'for {item.name}')
            item.set_args(**args)
            if (item.method_type == 'read'
                    and item.status not in ('read', 'write_pending')):
                if item.get_schema() is not None:
                    self.logger.debug(
                        f'Reading "{item.name}" with schema '
                        f'"{item.get_schema()}" and {item.get_args()}')
                    self.logger.info(f'Reading "{item.name}"')
                    data = item.method(**item.get_args())
                    self._data.append(PipelineData(
                        name=item.name, data=data, schema=item.get_schema()))
                    if item.has_filename():
                        self._filename_mapping[
                            item._mapping_filename]['status'] = 'read'
                    # FIX make part of pipelineitem for read
                    item.status = 'read'
            if item.method_type == 'write' and item.has_filename():
                for k, v in self._filename_mapping.items():
                    if v['path'] == item.filename:
                        self._filename_mapping[k]['status'] = \
                            'write_pending'
                #if item.filename not in self._output_filenames:
                #    self._output_filenames.append(item.filename)
                if item.filename not in output_filenames:
                    output_filenames.append(item.filename)
            self._items.append(item)
        self.logger.info(f'Validated pipeline in {time()-t0:.3f} seconds')

        return self

    def execute(self):
        """Executes the pipeline.

        :return: List of `PipelineData` items after pipeline execution.
        :rtype: list[PipelineData]
        """
        t0 = time()
        self.logger.info('Executing "execute"\n')

        for mmc, item, args in zip(self.mmcs, self._items, self.args):
            if hasattr(item, 'execute'):
                current_item = mmc(data=self._data, modelmetaclass=mmc, **args)
                read_status = None
                if item.method_type == 'read' and item.has_filename():
                    read_status = self._filename_mapping[
                        item._mapping_filename]['status']
                    current_item.status = read_status
                    current_item.filename = item.filename
                current_item.set_args(**item.get_args())
                # FIX RV update to only read when not yet read or when
                # written to in the mean time, make this happen for any
                # type of read, from file, url, ...
                if not (item.method_type == 'read' and read_status == 'read'):
                    self.logger.info(
                        f'Calling "execute" on {current_item.name}')
                    data = current_item.execute(self._data)
#                        self._data, self._metadata, self._provenance)
                    if current_item.method_type == 'read':
                        for _, d in reversed(list(enumerate(self._data))):
                            if d == PipelineData(
                                    name=current_item.name, data=data,
                                    schema=current_item.get_schema()):
                                break
                        else:
                            self._data.append(PipelineData(
                                name=current_item.name, data=data,
                                schema=current_item.get_schema()))
                        #FIX RF move to pipelineitem after read
                        current_item.status = 'read'
                    else:
                        if isinstance(data, tuple):
                            self._data.extend(
                                [d if isinstance(d, PipelineData)
                                 else PipelineData(
                                     name=current_item.name, data=d,
                                     schema=current_item.get_schema())
                                 for d in data])
                        elif isinstance(data, PipelineData):
                            self._data.append(data)
                        elif data is not None:
                            self._data.append(PipelineData(
                                name=current_item.name, data=data,
                                schema=current_item.get_schema()))
                    if item.method_type == 'write' and item.has_filename():
                        for k, v in self._filename_mapping.items():
                            if v['path'] == item.filename:
                                self._filename_mapping[k]['status'] = 'written'
        self.logger.info(f'Executed "execute" in {time()-t0:.3f} seconds')
        return self._data
