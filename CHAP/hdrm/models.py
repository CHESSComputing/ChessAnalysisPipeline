"""HDRM Pydantic model classes."""

# System modules
from typing import Optional

# Third party modules
from pydantic import (
    confloat,
    conlist,
)

# Local modules
from CHAP import CHAPBaseModel
from CHAP.edd.models import MaterialConfig
from CHAP.giwaxs.models import AzimuthalIntegratorConfig


class HdrmOrmfinderConfig(CHAPBaseModel):
    """Orm finder configuration class.

    :ivar azimuthal_integrators: List of azimuthal integrator
        configurations.
    :type azimuthal_integrators: list[
        CHAP.giwaxs.models.AzimuthalIntegratorConfig]
    :ivar materials: Material parameters configurations.
    :type material: list[CHAP.edd.models.MaterialConfig]
    """
    azimuthal_integrators: conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)
    materials: conlist(item_type=MaterialConfig)


class HdrmPeakfinderConfig(CHAPBaseModel):
    """Peak finder configuration class.

    :param peak_cutoff: Cutoff height for peaks (as a fraction of the
        maximum intensity), defaults to `0.95`.
    :type peak_cutoff: float, optional
    """
    peak_cutoff: Optional[confloat(gt=0.0, le=1.0, allow_inf_nan=False)] = 0.95
