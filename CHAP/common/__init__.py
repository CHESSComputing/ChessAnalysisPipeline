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
