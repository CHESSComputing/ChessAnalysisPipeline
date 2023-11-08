"""This subpackage of `CHAP` contains `PipelineItem`s that are or can
be used in workflows for processing data from multiple different X-ray
techniques.

In addition, `CHAP.common` contains a subpackage of its own:
[`CHAP.common.models`](CHAP.common.models.md) contains tools for
validating input data in some `Processor`s.
"""

from CHAP.common.reader import (
    BinaryFileReader,
    H5Reader,
    MapReader,
    NexusReader,
    SpecReader,
    URLReader,
    YAMLReader,
)
from CHAP.common.processor import (
    AnimationProcessor,
    AsyncProcessor,
    ImageProcessor,
    IntegrationProcessor,
    IntegrateMapProcessor,
    MapProcessor,
    NexusToNumpyProcessor,
    NexusToXarrayProcessor,
    PrintProcessor,
    RawDetectorDataMapProcessor,
    StrainAnalysisProcessor,
    XarrayToNexusProcessor,
    XarrayToNumpyProcessor,
)
from CHAP.common.writer import (
    ExtractArchiveWriter,
    FileTreeWriter,
    MatplotlibAnimationWriter,
    MatplotlibFigureWriter,
    NexusWriter,
    YAMLWriter,
    TXTWriter,
)
