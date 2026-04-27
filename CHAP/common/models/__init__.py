"""This subpackage contains
`Pydantic <https://github.com/pydantic/pydantic>`__ model configuration
classes for `PipelineItems` that are common to various processing
workflows.

common
    Common Pydantic model configuration classes.
integration:
    `pyFAI <https://pyfai.readthedocs.io/en/stable/>`__ integration
    related Pydantic model configuration classes.
map
    Map related Pydantic model configuration classes.
"""

# System modules
import typing

# Local modules
from CHAP.common.models.map import DetectorConfig
from CHAP.common.models.common import (
    ImageProcessorConfig,
    UnstructuredToStructuredConfig,
)

DetectorConfig.model_rebuild(_types_namespace=vars(typing))
