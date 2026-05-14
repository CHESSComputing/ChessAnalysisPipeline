"""`PipelineItems` unique to GIWAXS data processing workflows.

This module contains all the `PipelineItems` (Processors, Readers and
Writers) that are unique to the GIWAXS workflow. Any of these
`PipelineItems` can be used as items in a :doc:`/pipeline` or
instantiated from a user Python script.

.. note::
    Using the GIWAXS workflow pipeline items in a :doc:`/pipeline`
    and running it, requires a GIWAXS conda environent or access to the
    appropriate CHAP GIWAXS  executable, see :doc:`/workflows/GIWAXS`

Submodules summary
------------------

models
    `Pydantic <https://github.com/pydantic/pydantic>`__ model
    configuration classes unique to the GIWAXS workflow.
processor
    Processors unique to the GIWAXS workflow.
reader
    Readers unique to the GIWAXS workflow.
writer
    Writers unique to the GIWAXS workflow.
"""

