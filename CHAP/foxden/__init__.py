"""This subpackage contains pieces for communication with FOXDEN services.
"""

from CHAP.foxden.processor import \
        FoxdenProvenanceProcessor, FoxdenMetadataProcessor
from CHAP.foxden.reader import \
        FoxdenMetadataReader, FoxdenProvenanceReader, FoxdenSpecScansReader
from CHAP.foxden.writer import FoxdenProvenanceWriter, FoxdenMetadataWriter
