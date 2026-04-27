"""`PipelineItems` unique to EDD data processing workflows.

This module contains all the `PipelineItems` (Processors, Readers and
Writers) that are unique to the EDD workflow. Any of these
`PipelineItems` can be used as items in a :doc:`/pipeline` or
instantiated from a user Python script.

.. note::
    Using the EDD workflow pipeline items in a :doc:`/pipeline`
    and running it, requires a EDD conda environent or access to the
    appropriate CHAP EDD executable, see :doc:`/workflows/EDD`

Submodules summary
------------------

models
    `Pydantic <https://github.com/pydantic/pydantic>`__ model
    configuration classes unique to the EDD workflow.
processor
    Processors unique to the EDD workflow.
reader
    Readers unique to the EDD workflow.
select_material_params_gui
    Model class and functions to create a GUI to interactively update
    the material properties for an EDD workflow.
utils
    Generic utility functions for EDD workflows.
writer
    Writers unique to the EDD workflow.
"""

# System modules
import typing

# Local modules
from CHAP.edd.models import (
    DiffractionVolumeLengthConfig,
    MCACalibrationConfig,
#    MCADetectorConfig,
#    MCAEnergyCalibrationConfig,
#    MCATthCalibrationConfig,
    StrainAnalysisConfig,
)

DiffractionVolumeLengthConfig.model_rebuild(_types_namespace=vars(typing))
MCACalibrationConfig.model_rebuild(_types_namespace=vars(typing))
MCADetectorConfig.model_rebuild(_types_namespace=vars(typing))
MCAEnergyCalibrationConfig.model_rebuild(_types_namespace=vars(typing))
MCATthCalibrationConfig.model_rebuild(_types_namespace=vars(typing))
StrainAnalysisConfig.model_rebuild(_types_namespace=vars(typing))
#from CHAP.edd.processor import (
#    DiffractionVolumeLengthProcessor,
#    LatticeParameterRefinementProcessor,
#    HKLProcessor,
#    MCAEnergyCalibrationProcessor,
#    MCATthCalibrationProcessor,
#    ReducedDataProcessor,
#    StrainAnalysisProcessor,
#)
#from CHAP.edd.reader import (
#    EddMapReader,
#    EddMPIMapReader,
#    ScanToMapReader,
#    SetupNXdataReader,
#    UpdateNXdataReader,
#    NXdataSliceReader,
#    SliceNXdataReader,
#)
#from CHAP.edd.writer import (
#    StrainAnalysisUpdateWriter,
#)
