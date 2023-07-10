#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : pipeline.py
Author     : Valentin Kuznetsov <vkuznet AT gmail dot com>
Description:
"""

# system modules
import inspect
import logging
import os
from time import time


class Pipeline():
    """Pipeline represent generic Pipeline class"""
    def __init__(self, items=None, kwds=None):
        """Pipeline class constructor

        :param items: list of objects
        :param kwds: list of method args for individual objects
        """
        self.__name__ = self.__class__.__name__

        self.items = items
        self.kwds = kwds

        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    def execute(self):
        """execute API"""

        t0 = time()
        self.logger.info('Executing "execute"\n')

        data = PipelineData()
        for item, kwargs in zip(self.items, self.kwds):
            if hasattr(item, 'execute'):
                self.logger.info(f'Calling "execute" on {item}')
                data = item.execute(data=data, **kwargs)

        self.logger.info(f'Executed "execute" in {time()-t0:.3f} seconds')


class PipelineData(dict):
    """Wrapper for all results of PipelineItem.execute"""
    def __init__(self, name=None, data=None, schema=None):
        super().__init__()
        self.__setitem__('name', name)
        self.__setitem__('data', data)
        self.__setitem__('schema', schema)


class PipelineItem():
    """An object that can be supplied as one of the items
    `Pipeline.items`
    """
    def __init__(self):
        """Constructor of PipelineItem class"""
        self.__name__ = self.__class__.__name__
        self.logger = logging.getLogger(self.__name__)
        self.logger.propagate = False

    @staticmethod
    def unwrap_pipelinedata(data):
        """Given a list of PipelineData objects, return a list of
        their "data" values. If there is only one item in ``data``,
        return only its "data" value (not in a list).

        :param data: input data to read, write, or process that needs
            to be unrapped from PipelineData before use
        :type data: list[PipelineData]
        :return: just the "data" values of the items in ``data``
        :rtype: Union[list[object], object]
        """

        values = [d['data'] for d in data]

        if len(values) == 1:
            return values[0]

        return values

    def get_config(self, data, schema, remove=True):
        """Look through `data` for an item whose value for the
        `'schema'` key matches `schema`. Convert the value for that
        item's `'data'` key into the configuration `BaseModel`
        identified by `schema` and return it.

        :param data: input data from a previous `PipelineItem`
        :type data: list[PipelineData]
        :param schema: name of the `BaseModel` class to match in
            `data` & return
        :type schema: str
        :param remove: if there is a matching entry in `data`, remove
           it from the list, defaults to `True`.
        :type remove: bool, optional
        :raises ValueError: if there's no match for `schema` in `data`
        :return: matching configuration model
        :rtype: BaseModel
        """

        self.logger.debug(f'Getting {schema} configuration')
        t0 = time()

        matching_config = False
        for i, d in enumerate(data):
            _schema = d.get('schema')
            if _schema == schema:
                matching_config = d.get('data')
                if remove:
                    data.pop(i)

        if not matching_config:
            raise ValueError(f'No configuration for {schema} found')

        mod_name, cls_name = schema.rsplit('.', 1)
        module = __import__(f'CHAP.{mod_name}', fromlist=cls_name)
        model_config = getattr(module, cls_name)(**matching_config)

        self.logger.debug(
            f'Got {schema} configuration in {time()-t0:.3f} seconds')

        return model_config

    def execute(self, schema=None, **kwargs):
        """Run the appropriate method of the object and return the
        result.

        :param schema: the name of a schema associated with the data
            that will be returned
        :type schema: str
        :param kwargs: a dictionary of any positional and keyword
            arguments to supply to the read, process, or write method.
        :type kwargs: dict
        :return: the wrapped result of running read, process, or write.
        :rtype: list[PipelineData]
        """

        if hasattr(self, 'read'):
            method_name = 'read'
            inputdir = kwargs.get('inputdir')
            if inputdir is not None:
                kwargs['filename'] = os.path.join(inputdir,
                                                  kwargs['filename'])
        elif hasattr(self, 'process'):
            method_name = 'process'
        elif hasattr(self, 'write'):
            method_name = 'write'
            outputdir = kwargs.get('outputdir')
            if outputdir is not None:
                kwargs['filename'] = os.path.join(outputdir,
                                                  kwargs['filename'])

        else:
            self.logger.error('No implementation of read, write, or process')

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

    def execute(self, items=[], **kwargs):
        """Independently execute all items in `self.items`, then
        return all of their results.

        :param items: PipelineItem configurations
        :type items: list
        :rtype: list[PipelineData]
        """

        t0 = time()
        self.logger.info(f'Executing {len(items)} PipelineItems')

        results = []
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
            results.append(item.execute(**item_args, **kwargs)[0])

        self.logger.info(
            f'Finished executing {len(items)} PipelineItems in {time()-t0:.0f}'
            ' seconds\n')
        return results
