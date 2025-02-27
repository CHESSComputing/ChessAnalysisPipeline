"""Common Pydantic model classes."""

# System modules
from pathlib import PosixPath

# Third party modules
import numpy as np
from pydantic import BaseModel

class CHAPBaseModel(BaseModel):
    """Base CHAP configuration class implementing robust
    serialization tools.
    """
    def dict(self, *args, **kwargs):
        return self.model_dump(*args, **kwargs)

    def model_dump(self, *args, **kwargs):
        if hasattr(self, '_exclude'):
            kwargs['exclude'] = self._merge_exclude(
                None if kwargs is None else kwargs.get('exclude'))
        return self._serialize(super().model_dump(*args, **kwargs))

    def model_dump_json(self, *args, **kwargs):
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
            except:
                pass
        return value

