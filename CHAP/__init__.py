"""The ChessAnalysisPipeline (CHAP) provides infrastructure to
construct and run X-ray data processing / analysis workflows using a
set of modular components. We call these components `PipelineItem`s
(subclassed into `Reader`s, `Processor`s, and `Writer`s). A `Pipeline`
uses a sequence of `PipelineItem`s to execute a data processing
workflow where the data returned by one `PipelineItem` becomes input
for the next one.

Many `PipelineItem`s can be shared by data processing workflows for
multiple different X-ray techniques, while others may be unique to
just a single technique. The `PipelineItem`s that are shared by many
techniques are organized in the `CHAP.common` subpackage.
`PipelineItem`s unique to a tomography workflow, for instance, are
organized in the `CHAP.tomo` subpackage.

[`CHAP.utils`](CHAP.utils.md) contains a
broad selection of utilities to assist in some common tasks that
appear in specific `Processor` implementations.
"""

from CHAP.reader import Reader
from CHAP.processor import Processor
from CHAP.writer import Writer

version = 'PACKAGE_VERSION'
