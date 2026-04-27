"""`PipelineItems` unique to tomography data processing workflows.

This module contains all the `PipelineItems` (Processors, Readers and
Writers) that are unique to the tomography workflow. Any of these
`PipelineItems` can be used as items in a :doc:`/pipeline` or
instantiated from a user Python script.

.. note::
    Using the tomography workflow pipeline items in a :doc:`/pipeline`
    and running it, requires a Tomo conda environent or access to the
    appropriate CHAP Tomo executable, see :doc:`/workflows/Tomo`

Submodules summary
------------------

models
    `Pydantic <https://github.com/pydantic/pydantic>`__ model
    configuration classes unique to the tomography workflow.
processor
    Processors unique to the tomography workflow.
reader
    Readers unique to the tomography workflow.
writer
    Writers unique to the tomography workflow.
"""

# System modules
import typing

# Local modules
from CHAP.tomo.models import (
    TomoCombineConfig,
    TomoFindCenterConfig,
    TomoReconstructConfig,
    TomoReduceConfig,
)

# Avoid Pydantic "Class not fully defined" in sphinx autodoc as a
# result of lazy importing by using any of these within a default
# value of a pydantic instance variable
TomoCombineConfig.model_rebuild(_types_namespace=vars(typing))
TomoFindCenterConfig.model_rebuild(_types_namespace=vars(typing))
TomoReconstructConfig.model_rebuild(_types_namespace=vars(typing))
TomoReduceConfig.model_rebuild(_types_namespace=vars(typing))
