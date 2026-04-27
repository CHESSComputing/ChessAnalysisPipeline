"""Base `PipelineItems` used in running a ChessAnalysisPipeline (CHAP).

CHAP provides infrastructure to construct and run X-ray data processing
and analysis workflows using a set of modular components. We call these
components `PipelineItem`\\s (subclassed into `Reader`\\s,
`Processor`\\s, and `Writer`\\s). A pipeline uses a sequence of
`PipelineItem`\\s to execute a data processing workflow where the data
returned by one `PipelineItem` becomes input for the following ones.

Many `PipelineItem`\\s can be shared by data processing workflows for
multiple different X-ray techniques, while others may be unique to
just a single technique. The `PipelineItem`\\s that are shared by many
techniques are organized in the `CHAP.common` subpackage.
`PipelineItem`\\s unique to a tomography workflow, for instance, are
organized in the `CHAP.tomo` subpackage.

:mod:`~CHAP.utils contains a broad selection of utilities to assist in
some common tasks that appear in specific `PipelineItem`
implementations.

Submodules summary
------------------

models
    Common `Pydantic <https://github.com/pydantic/pydantic>`__ model
    classes.
pipeline
    Base pipeline Pydantic model classes.
processor
    Module defining the base `Processor` class to derive all others
    from.
reader
    Module defining the base `Reader` class to derive all others from.
runner
    Main functions to execute a ChessAnalysisPipeline (CHAP).
server
    Python server with thread pool and CHAP pipeline.
taskmanager
    Python thread pool.
writer
    Module defining the base `Writer` class to derive all others from.
"""

#from CHAP.models import CHAPBaseModel
#from CHAP.reader import Reader
#from CHAP.processor import Processor
#from CHAP.writer import Writer

version = 'PACKAGE_VERSION'
