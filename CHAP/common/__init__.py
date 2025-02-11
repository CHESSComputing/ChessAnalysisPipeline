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
    ImageProcessor,
    IntegrationProcessor,
    IntegrateMapProcessor,
    MapProcessor,
    MPICollectProcessor,
    MPIMapProcessor,
    MPISpawnMapProcessor,
    NexusToNumpyProcessor,
    NexusToXarrayProcessor,
    PrintProcessor,
    PyfaiAzimuthalIntegrationProcessor,
    RawDetectorDataMapProcessor,
    SetupNXdataProcessor,
    UpdateNXvalueProcessor,
    UpdateNXdataProcessor,
    NXdataToDataPointsProcessor,
    XarrayToNexusProcessor,
    XarrayToNumpyProcessor,
    SumProcessor,
    ZarrToNexusProcessor,
)
from CHAP.common.reader import (
    BinaryFileReader,
    DetectorDataReader,
    FabioImageReader,
    H5Reader,
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
    MatplotlibAnimationWriter,
    MatplotlibFigureWriter,
    NexusWriter,
    PyfaiResultsWriter,
    YAMLWriter,
    TXTWriter,
)
