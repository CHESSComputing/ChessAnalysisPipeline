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

from CHAP.tomo.processor import (
    TomoMetadataProcessor,
    TomoCHESSMapConverter,
    TomoReduceProcessor,
    TomoFindCenterProcessor,
    TomoReconstructProcessor,
    TomoCombineProcessor,
    TomoSimFieldProcessor,
    TomoDarkFieldProcessor,
    TomoBrightFieldProcessor,
    TomoSpecProcessor,
)
# from CHAP.tomo.reader import
from CHAP.tomo.writer import TomoWriter
