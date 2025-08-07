"""This subpackage contains `PipelineItems` to communicate with FOXDEN
services.
"""

from CHAP.foxden.processor import (
#    FoxdenMetadataProcessor,
    FoxdenProvenanceProcessor,
)
from CHAP.foxden.reader import (
    FoxdenDataDiscoveryReader,
    FoxdenMetadataReader,
    FoxdenProvenanceReader,
    FoxdenSpecScansReader,
)
from CHAP.foxden.writer import (
    FoxdenDoiWriter,
    FoxdenMetadataWriter,
    FoxdenProvenanceWriter,
)
