# System modules
from functools import cache
import os
from pathlib import PosixPath
from typing import (
    Optional,
)

# Third party modules
import numpy as np
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)

# Local modules
from CHAP.common.models.map import MapConfig


class Detector(BaseModel):
    """Detector class to represent a single detector used in the
    experiment.

    :param prefix: Prefix of the detector in the SPEC file.
    :type prefix: str
    :param poni_file: Path to the poni file.
    :type poni_file: str
    """
    prefix: constr(strip_whitespace=True, min_length=1)
    poni_file: FilePath

    @field_validator('poni_file')
    @classmethod
    def validate_poni_file(cls, poni_file):
        """Validate the poni file by checking if it's a valid PONI
        file.

        :param poni_file: Path to the poni file.
        :type poni_file: str
        :raises ValueError: If poni_file is not a valid PONI file.
        :returns: Absolute path to the poni file.
        :rtype: str
        """
        # Third party modules
        from pyFAI import load

        poni_file = os.path.abspath(poni_file)
        try:
            load(poni_file)
        except Exception as exc:
            raise ValueError(f'{poni_file} is not a valid PONI file') from exc
        return poni_file


class GiwaxsConversionConfig(BaseModel):
    """Class representing metadata required to locate GIWAXS image
    files for a single scan to convert to q_par/q_perp coordinates.

    :ivar detectors: List of detector configurations.
    :type detectors: list[Detector]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :type scan_step_indices: Union(int, list[int], str), optional
    :ivar save_raw_data: Save the raw data in the NeXus output,
        default to `False`.
    :type save_raw_data: bool, optional
    """
    detectors: conlist(item_type=Detector, min_length=1)
    scan_step_indices: Optional[
        conlist(item_type=conint(ge=0), min_length=1)] = None
    save_raw_data: Optional[bool] = False

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Validate the specified list of scan step indices.

        :param scan_step_indices: List of scan numbers.
        :type scan_step_indices: list of int
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list of int
        """
        if isinstance(scan_step_indices, int):
            scan_step_indices = [scan_step_indices]
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)

        return scan_step_indices

