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

# Avoid Pydantic "Class not fully defined" in sphinx autodoc as a
# result of lazy importing by using DetectorConfig within a default
# value of a pydantic instance variable
DetectorConfig.model_rebuild(_types_namespace=vars(typing))
