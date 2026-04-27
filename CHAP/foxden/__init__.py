"""`PipelineItems` unique to communicating with FOXDEN

This module contains all the `PipelineItems` (Processors, Readers and
Writers) that are unique to communicate with
`FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__. Any of these
`PipelineItems` can be used as items in a :doc:`/pipeline` or
instantiated from a user Python script.

.. note::
    Using the FOXDEN pipeline items in a :doc:`/pipeline` and running
    it, requires a conda environent for the appropriate workflow or
    access to the appropriate CHAP executable.

Submodules summary
------------------

models
    `Pydantic <https://github.com/pydantic/pydantic>`__ model
    configuration classes unique to the FOXDEN pipeline items.
processor
    Processors unique to the FOXDEN pipeline items.
reader
    Readers unique to the FOXDEN pipeline items.
writer
    Writers unique to the FOXDEN pipeline items.
"""

# System modules
import typing

# Local modules
from CHAP.foxden.models import FoxdenRequestConfig

FoxdenRequestConfig.model_rebuild(_types_namespace=vars(typing))
