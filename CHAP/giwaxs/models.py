# System modules
from functools import cache
import os
from pathlib import PosixPath
from typing import (
#    Literal,
    Optional,
#    Union,
)

# Third party modules
import numpy as np
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
#    PrivateAttr,
#    StrictBool,
#    confloat,
    conint,
    conlist,
    constr,
    root_validator,
    validator,
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

    @validator('poni_file', allow_reuse=True)
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

    :ivar inputdir: Input directory, used only if any file in the
            configuration is not an absolute path.
    :type inputdir: str, optional
    :ivar map_config: The map configuration for the GIWAXS data.
    :type map_config: CHAP.common.models.map.MapConfig
    :ivar detectors: List of detector configurations.
    :type detectors: list[Detector]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :type scan_step_indices: list[int], optional
    """
    inputdir: Optional[DirectoryPath]
    map_config: MapConfig
    detectors: conlist(min_items=1, item_type=Detector)
    scan_step_indices: Optional[conlist(min_items=1, item_type=conint(ge=0))]

    @root_validator(pre=True)
    def validate_config(cls, values):
        """Ensure that a valid configuration was provided and finalize
        input filepaths.

        :param values: Dictionary of class field values.
        :type values: dict
        :raises ValueError: Missing par_dims value.
        :return: The validated list of `values`.
        :rtype: dict
        """
        inputdir = values.get('inputdir')
        map_config = values.get('map_config')
        for i, scans in enumerate(map_config.get('spec_scans')):
            spec_file = scans.get('spec_file')
            if inputdir is not None and not os.path.isabs(spec_file):
                values['map_config']['spec_scans'][i]['spec_file'] = \
                    os.path.join(inputdir, spec_file)
        return values

    @validator('scan_step_indices', pre=True, allow_reuse=True)
    def validate_scan_step_indices(cls, scan_step_indices, values):
        """Validate the specified list of scan step indices.

        :param scan_step_indices: List of scan numbers.
        :type scan_step_indices: list of int
        :param values: Dictionary of validated class field values.
        :type values: dict
        :raises ValueError: If a specified scan number is not found in
            the SPEC file.
        :return: List of scan numbers.
        :rtype: list of int
        """
        map_config = values.get('map_config')
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)
            if len(values.get('map_config').shape) != 1:
                raise RuntimeError(f'Illegal use of scan_step_indices')

        return scan_step_indices

    def giwaxs_data(self, detector=None, map_index=None):
        """Get MCA data for a single or multiple detector elements.

        :param detector: Detector(s) for which data is returned,
            defaults to `None`, which return MCA data for all 
            detector elements.
        :type detector: Union[int, MCAElementStrainAnalysisConfig],
            optional
        :param map_index: Index of a single point in the map, defaults
            to `None`, which returns MCA data for each point in the map.
        :type map_index: tuple, optional
        :return: A single MCA spectrum.
        :rtype: np.ndarray
        """
        if detector is None:
            giwaxs_data = []
            for detector in self.detectors:
                giwaxs_data.append(
                    self.giwaxs_data(detector, map_index))
            return np.asarray(giwaxs_data)
        else:
            if isinstance(detector, int):
                detector = self.detectors[detector]
            else:
                if not isinstance(detector, Detector):
                    raise ValueError('Invalid parameter detector ({detector})')
                detector = detector
            if map_index is None:
                giwaxs_data = []
                for map_index in np.ndindex(self.map_config.shape):
                    if self.scan_step_indices is not None:
                        _, _, scan_step_index = \
                            self.map_config.get_scan_step_index(map_index)
                        if scan_step_index not in self.scan_step_indices:
                            continue
                    giwaxs_data.append(self.giwaxs_data(detector, map_index))
                if self.scan_step_indices is None:
                    map_shape = self.map_config.shape
                else:
                    map_shape = (len(self.scan_step_indices), )
                giwaxs_data = np.reshape(
                    giwaxs_data, (*map_shape, *giwaxs_data[0].shape))
                return np.asarray(giwaxs_data)
            else:
                return self.map_config.get_detector_data(
                    detector.prefix, map_index)

    def dict(self, *args, **kwargs):
        """Return a representation of this configuration in a
        dictionary that is suitable for dumping to a YAML file.

        :return: Dictionary representation of the configuration.
        :rtype: dict
        """
        d = super().dict(*args, **kwargs)
        for k, v in d.items():
            if isinstance(v, PosixPath):
                d[k] = str(v)
        return d

