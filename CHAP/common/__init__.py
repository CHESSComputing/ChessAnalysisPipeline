"""`PipelineItems` that are or can be used in workflows for processing
data from multiple different X-ray techniques.

Any of these `PipelineItems` can be used as items in a :doc:`/pipeline`
or instantiated from a user Python script.

Submodules summary
------------------

map_utils
    Common map data model functions and classes.
models
    Subpackage containing `Pydantic <https://github.com/pydantic/pydantic>`__
    model configuration classes for `PipelineItems` that are common to various
    processing workflows.
nexus_utils
    PipelineItems for interacting with `NeXus <https://www.nexusformat.org>`__
    file objects.
processor
    Module for generic Processors used in multiple experiment-specific
    workflows.
reader
    Module for generic Readers used in multiple experiment-specific workflows.
utils
    Some generic utility functions.
writer
    Module for generic Writers used in multiple experiment-specific workflows.
"""

