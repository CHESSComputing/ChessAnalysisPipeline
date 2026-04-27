"""`PipelineItems` unique to SAXSWAXS data processing workflows.

This module contains all the `PipelineItems` (Processors, Readers and
Writers) that are unique to the SAXSWAXS workflow. Any of these
`PipelineItems` can be used as items in a :doc:`/pipeline` or
instantiated from a user Python script.

.. note::
    Using the SAXSWAXS workflow pipeline items in a :doc:`/pipeline`
    and running it, requires a SAXSWAXS conda environent or access to
    the appropriate CHAP SAXSWAXS  executable,
    see :doc:`/workflows/SAXSWAXS`

Submodules summary
------------------

processor
    Processors unique to the SAXSWAXS workflow.
reader
    Readers unique to the SAXSWAXS workflow.
writer
    Writers unique to the SAXSWAXS workflow.
"""

#from CHAP.saxswaxs.processor import (
#    CfProcessor,
#    FluxCorrectionProcessor,
#    FluxAbsorptionCorrectionProcessor,
#    FluxAbsorptionBackgroundCorrectionProcessor,
#    PyfaiIntegrationProcessor,
#    SetupResultsProcessor,
#    SetupProcessor,
#    UnstructuredToStructuredProcessor,
#    UpdateValuesProcessor,
#)
## from CHAP.saxswaxs.reader import ()
## from CHAP.saxswaxs.writer import ()
