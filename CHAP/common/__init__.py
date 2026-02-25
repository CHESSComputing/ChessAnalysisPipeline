"""This subpackage of `CHAP` contains `PipelineItem`\ s that are or can
be used in workflows for processing data from multiple different X-ray
techniques.
"""

from CHAP.common.processor import (
#    AnimationProcessor,
    AsyncProcessor,
    BinarizeProcessor,
    ConvertStructuredProcessor,
    ExpressionProcessor,
    ImageProcessor,
    MapProcessor,
    MPICollectProcessor,
    MPIMapProcessor,
    MPISpawnMapProcessor,
    NexusToNumpyProcessor,
#    NexusToTiffsprocessor,
    NexusToXarrayProcessor,
    NexusToZarrProcessor,
    NormalizeNexusProcessor,
    NormalizeMapProcessor,
    PandasToXarrayProcessor,
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
    ZarrToNexusProcessor,
)
from CHAP.common.reader import (
    BinaryFileReader,
    ConfigReader,
    DetectorDataReader,
    FabioImageReader,
    H5Reader,
    LinkamReader,
    MapReader,
    PandasReader,
    NexusReader,
    NXdataReader,
    NXfieldReader,
    SpecReader,
    URLReader,
    YAMLReader,
    ZarrReader,
)
from CHAP.common.writer import (
    ExtractArchiveWriter,
    FileTreeWriter,
    H5Writer,
    ImageWriter,
    MatplotlibAnimationWriter,
    MatplotlibFigureWriter,
    NexusWriter,
    NexusValuesWriter,
    PyfaiResultsWriter,
    YAMLWriter,
    TXTWriter,
    ZarrValuesWriter,
    ZarrWriter,
)

from CHAP.common.map_utils import (
    MapSliceProcessor,
    SpecScanToMapConfigProcessor,
)

from CHAP.common.nexus_utils import (
    NexusMakeLinkProcessor,
)
