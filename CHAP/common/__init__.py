"""This subpackage of `CHAP` contains `PipelineItem`s that are or can
be used in workflows for processing data from multiple different X-ray
techniques.

In addition, `CHAP.common` contains a subpackage of its own:
[`CHAP.common.models`](CHAP.common.models.md) contains tools for
validating input data in some `Processor`s.
"""

from CHAP.common.processor import (
    AnimationProcessor,
    AsyncProcessor,
    BinarizeProcessor,
    ConvertStructuredProcessor,
    ImageProcessor,
    MapProcessor,
    MPICollectProcessor,
    MPIMapProcessor,
    MPISpawnMapProcessor,
    NexusToNumpyProcessor,
    NexusToTiffsprocessor,
    NexusToXarrayProcessor,
    NormalizeNexusProcessor,
    NormalizeMapProcessor,
    PrintProcessor,
    PyfaiAzimuthalIntegrationProcessor,
    RawDetectorDataMapProcessor,
    SetupNXdataProcessor,
    UpdateNXvalueProcessor,
    UpdateNXdataProcessor,
    UnstructuredToStructuredProcessor,
    NXdataToDataPointsProcessor,
    XarrayToNexusProcessor,
    XarrayToNumpyProcessor,
#    SumProcessor,
)
from CHAP.common.reader import (
    BinaryFileReader,
    FabioImageReader,
    H5Reader,
    LinkamReader,
    MapReader,
    NexusReader,
    NXdataReader,
    NXfieldReader,
    SpecReader,
    URLReader,
    YAMLReader,
)
from CHAP.common.writer import (
    ExtractArchiveWriter,
    FileTreeWriter,
    H5Writer,
    ImageWriter,
    MatplotlibAnimationWriter,
    MatplotlibFigureWriter,
    NexusWriter,
    PyfaiResultsWriter,
    YAMLWriter,
    TXTWriter,
)
