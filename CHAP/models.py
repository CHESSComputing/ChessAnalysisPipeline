#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Common `Pydantic <https://github.com/pydantic/pydantic>`__ model
classes.
"""

# System modules
import os
from pathlib import PosixPath
from typing import (
    Literal,
    Optional,
)

# Third party modules
from pydantic import (
    BaseModel,
    DirectoryPath,
    PrivateAttr,
    field_validator,
    model_validator,
)


class CHAPBaseModel(BaseModel):
    """Base CHAP configuration class implementing robust
    serialization tools.
    """

    def dict(self, *args, **kwargs):
        """Dump the class implemention to a dictionary.

        :param **kwargs: Arbitrary keyword arguments.
        :type: dict
        :keyword exclude: Class variable(s) to omit from the output
            dictionary.
        :type exclude: dict or set, optional
        :keyword by_alias: Use aliases as the output dictionary keys for
            class variables that have an alias., defaults to `True`.
        :type by_alias: bool, optional
        :return: Class implementation.
        :rtype: dict
        """
        return self.model_dump(*args, **kwargs)

    def model_dump(self, *args, **kwargs):
        """Dump the class implemention to a dictionary.

        :param **kwargs: Arbitrary keyword arguments.
        :type: dict
        :keyword exclude: Class variable(s) to omit from the output
            dictionary.
        :type exclude: dict or set, optional
        :keyword by_alias: Use aliases as the output dictionary keys for
            class variables that have an alias., defaults to `True`.
        :type by_alias: bool, optional
        :return: Class implementation.
        :rtype: dict
        """
        if hasattr(self, '_exclude'):
            kwargs['exclude'] = self._merge_exclude(
                None if kwargs is None else kwargs.get('exclude'))
        if 'by_alias' not in kwargs:
            kwargs['by_alias'] = True
        return self._serialize(super().model_dump(*args, **kwargs))

    def model_dump_json(self, *args, **kwargs):
        """Dump the class implemention to a JSON string.

        :param **kwargs: Arbitrary keyword arguments.
        :type: dict
        :keyword exclude: Class variable(s) to omit from the output
            dictionary.
        :type exclude: dict or set, optional
        :keyword by_alias: Use aliases as the output dictionary keys for
            class variables that have an alias., defaults to `True`.
        :type by_alias: bool, optional
        :return: Class implementation.
        :rtype: str
        """
        # Third party modules
        from json import dumps

        return dumps(self.model_dump(*args, **kwargs))

    def _merge_exclude(self, exclude):
        if exclude is None:
            exclude = self._exclude
        elif isinstance(exclude, set):
            if isinstance(self._exclude, set):
                exclude |= self._exclude
            elif isinstance(self._exclude, dict):
                exclude = {**{v:True for v in exclude}, **self._exclude}
        elif isinstance(exclude, dict):
            if isinstance(self._exclude, set):
                exclude = {**exclude, **{v:True for v in self._exclude}}
            elif isinstance(self._exclude, dict):
                exclude = {**exclude, **self._exclude}
        return exclude

    def _serialize(self, value):
        if isinstance(value, dict):
            value = {k:self._serialize(v) for k, v in value.items()}
        elif isinstance(value, (tuple, list)):
            value = [self._serialize(v) for v in value]
        elif isinstance(value, PosixPath):
            value = str(value)
        else:
            try:
                # For np.array, np.ndarray, any np scalar, or native types
                value = getattr(value, "tolist", lambda: value)()
            except Exception:
                pass
        return value


class RunConfig(CHAPBaseModel):
    """Pipeline run configuration class.

    :ivar root: Default work directory, defaults to the current run
        directory.
    :vartype root: str, optional
    :ivar inputdir: Input directory, used only if any input file in the
        pipeline is not an absolute path, defaults to `'root'`.
    :vartype inputdir: str, optional
    :ivar outputdir: Output directory, used only if any output file in
        the pipeline is not an absolute path, defaults to `'root'`.
    :vartype outputdir: str, optional
    :ivar interactive: Allows for user interactions,
        defaults to `False`.
    :vartype interactive: bool, optional
    :ivar log_level: Logger level (not case sensitive),
        defaults to `'INFO'`.
    :vartype log_level: Literal[
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], optional
    """

    root: Optional[DirectoryPath] = os.getcwd()
    inputdir: Optional[DirectoryPath] = None
    outputdir: Optional[DirectoryPath] = None
    interactive: Optional[bool] = False
    log_level: Optional[Literal[
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']] = 'INFO'

    # Internal flags, only set them during object construction
    # For code profiling
    _profile: bool = PrivateAttr(default=False)
    # To detemine if a pipeline is executed from a apawned worker
    _spawn: int = PrivateAttr(default=0)

    def __init__(self, **data):
        super().__init__(**data)
        if 'profile' in data:
            self._profile = data.pop('profile')
            if not isinstance(self._profile, bool):
                raise ValueError(
                    f'Invalid private attribute profile {self._profile}')
        if 'spawn' in data:
            self._spawn = data.pop('spawn')
            if not (isinstance(self._spawn, int) and -1 <= self._spawn <= 1):
                raise ValueError(
                    f'Invalid private attribute spawn {self._spawn}')

    @model_validator(mode='before')
    @classmethod
    def validate_runconfig_before(cls, data):
        """Ensure that valid directory paths are provided.

        :param data:
            `Pydantic <https://github.com/pydantic/pydantic>`__
            validator data object.
        :type data: dict
        :return: Currently validated class attributes.
        :rtype: dict
        """
        if isinstance(data, dict):
            # System modules
            from tempfile import NamedTemporaryFile

            # Make sure os.makedirs is only called from the root node
            comm = data.get('comm')
            rank = 0 if comm is None else comm.Get_rank()

            # Check if root exists (create it if not) and is readable
            root = data.get('root')
            if root is None:
                root = os.getcwd()
            if not rank:
                if not os.path.isdir(root):
                    os.makedirs(root)
                if not os.access(root, os.R_OK):
                    raise OSError('root directory is not accessible for '
                                  f'reading ({root})')
            data['root'] = os.path.realpath(root)

            # Check if inputdir exists and is readable
            inputdir = data.get('inputdir', '.')
            if not os.path.isabs(inputdir):
                inputdir = os.path.normpath(os.path.realpath(
                    os.path.join(root, inputdir)))
            if not rank:
                if not os.path.isdir(inputdir):
                    raise OSError(
                        f'input directory does not exist ({inputdir})')
                if not os.access(inputdir, os.R_OK):
                    raise OSError(
                        'input directory is not accessible for reading '
                        f'({inputdir})')
            data['inputdir'] = inputdir

            # Check if outputdir exists (create it if not) and is writable
            outputdir = data.get('outputdir', '.')
            if not os.path.isabs(outputdir):
                outputdir = os.path.normpath(os.path.realpath(
                    os.path.join(root, outputdir)))
            if not rank:
                if not os.path.isdir(outputdir):
                    os.makedirs(outputdir)
                try:
                    NamedTemporaryFile(dir=outputdir)
                except Exception as exc:
                    raise OSError('output directory is not accessible for '
                                  f'writing ({outputdir})') from exc
            data['outputdir'] = outputdir

            # Make sure os.makedirs completes before continuing
            # Make sure barrier() is also called on the main node if
            # this is called from a spawned slave node
            if comm is not None:
                comm.barrier()

        return data

    @field_validator('log_level', mode='before')
    @classmethod
    def validate_log_level(cls, log_level):
        """Capitalize `log_level`.

        param value: Input value for `log_level`.
        :type value: str
        :return: Capitalized `log_level`.
        :rtype: str
        """
        return log_level.upper()

    @property
    def profile(self):
        """Return the profiling flag.

        :type: bool
        """
        #:return: Profiling flag.
        #:rtype: bool
        if hasattr(self, '_profile'):
            return self._profile
        return False

    @property
    def spawn(self):
        """Return the spawned worker flag.

        :type: int
        """
        if hasattr(self, '_spawn'):
            return self._spawn
        return 0
