"""HDRM Pydantic model classes."""

# System modules
from typing import Optional

# Third party modules
from pydantic import confloat

# Local modules
from CHAP import CHAPBaseModel


class HdrmPeakfinderConfig(CHAPBaseModel):
    """Peak finder configuration class.

    :param peak_cutoff: Cutoff height for peaks (as a fraction of the
        maximum intensity), defaults to `0.95`.
    :type peak_cutoff: float, optional
    """
    peak_cutoff: Optional[confloat(gt=0.0, le=1.0, allow_inf_nan=False)] = 0.95
