"""GIWAXS Pydantic model classes."""

# System modules
from copy import deepcopy
import os
from pathlib import PosixPath
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
from pydantic import (
    BaseModel,
    FilePath,
    PrivateAttr,
    confloat,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
)
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

# Local modules
from CHAP import CHAPBaseModel
from CHAP.common.models.map import Detector


class AzimuthalIntegratorConfig(Detector, CHAPBaseModel):
    """Azimuthal integrator configuration class to represent a single
    detector used in the experiment.

    :param mask_file: Path to the mask file.
    :type mask_file: FilePath, optional
    :param poni_file: Path to the PONI file, specify either `poni_file`
        or `params`, not both.
    :type poni_file: FilePath, optional
    :param params: Azimuthal integrator configuration parameters,
        specify either `poni_file` or `params`, not both.
    :type params: dict, optional
    """
    mask_file: Optional[FilePath] = None
    params: Optional[dict] = None
    poni_file: Optional[FilePath] = None

    _ai: AzimuthalIntegrator = PrivateAttr()

    @model_validator(mode='before')
    @classmethod
    def validate_root(cls, data):
        if isinstance(data, dict):
            inputdir = data.get('inputdir')
            mask_file = data.get('mask_file')
            params = data.get('params')
            poni_file = data.get('poni_file')
            if mask_file is not None:
                if inputdir is not None and not os.path.isabs(mask_file):
                    data['mask_file'] = mask_file
            if params is not None:
                if poni_file is not None:
                    raise ValueError(
                        'Specify either poni_file or params, not both')
            elif poni_file is not None:
                if inputdir is not None and not os.path.isabs(poni_file):
                    data['poni_file'] = poni_file
            else:
                raise ValueError('Specify either poni_file or params')
        return data

    @model_validator(mode='after')
    def validate_ai(self):
        """Set the default azimuthal integrator.

        :return: Validated configuration class.
        :rtype: AzimuthalIntegratorConfig
        """
        if self.params is not None:
            self.ai = AzimuthalIntegrator(**self.params)
        elif self.poni_file is not None:
            # Third party modules
            from pyFAI import load

            self._ai = load(str(self.poni_file))
        return self

    @property
    def ai(self):
        """Return the azimuthal integrator."""
        return self._ai

class GiwaxsConversionProcessorConfig(CHAPBaseModel):
    """Class representing metadata required to locate GIWAXS image
    files for a single scan to convert to q_par/q_perp coordinates.

    :ivar azimuthal_integrators: List of azimuthal integrator
        configurations.
    :type azimuthal_integrators: list[
        CHAP.giwaxs.models.AzimuthalIntegratorConfig]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :type scan_step_indices: Union(int, list[int], str), optional
    :ivar save_raw_data: Save the raw data in the NeXus output,
        defaults to `False`.
    :type save_raw_data: bool, optional
    """
    azimuthal_integrators: conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)
    scan_step_indices: Optional[
        conlist(min_length=1, item_type=conint(ge=0))] = None
    save_raw_data: Optional[bool] = False

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Ensure that a valid configuration was provided and finalize
        PONI filepaths.

        :param data: Pydantic validator data object.
        :type data: GiwaxsConversionProcessorConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        if isinstance(scan_step_indices, int):
            scan_step_indices = [scan_step_indices]
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)

        return scan_step_indices


class MultiGeometryConfig(CHAPBaseModel):
    """Class representing the configuration for treating simultaneously
    multiple detector configuration within a single integration

    :ivar ais: List of detector IDs of azimuthal integrators
    :type ais: Union[str, list[str]]
    :ivar azimuth_range: Common azimuthal range for integration,
        defaults to `[-180.0, 180.0]`.
    :type azimuth_range: Union(list[float, float],
                                 tuple[float, float]), optional
    :ivar radial_range: Common range for integration, defaults to
        `[0.0, 180.0]`.
    :type radial_range: Union(list[float, float],
                              tuple[float, float]), optional
    """
    ais: conlist(
        min_length=1, item_type=constr(min_length=1, strip_whitespace=True))
    azimuth_range: Optional[
        conlist(
            min_length=2, max_length=2,
            item_type=confloat(ge=-180, le=360, allow_inf_nan=False))
        ] = [-180.0, 180.0]
    radial_range: Optional[
        conlist(
            min_length=2, max_length=2,
            item_type=confloat(ge=0, le=180, allow_inf_nan=False))
        ] = [0.0, 180.0]
    unit: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'q_A^-1'
    chi_disc: Optional[int] = 180
    empty: Optional[confloat(allow_inf_nan=False)] = 0.0
    wavelength: Optional[confloat(allow_inf_nan=False)] = None

    @field_validator('ais', mode='before')
    @classmethod
    def validate_ais(cls, ais):
        """Validate the detector IDs of the azimuthal integrators.

        :param ais: The detector IDs.
        :type ais: str, list[str]
        :return: The detector ais.
        :rtype: list[str]
        """
        if isinstance(ais, str):
            return [ais]
        return ais

#    @field_validator('radial_units')
#    @classmethod
#    def validate_radial_units(cls, radial_units):
#        """Validate the radial units for the integration.
#
#        :param radial_units: Unvalidated radial units for the
#            integration.
#        :type radial_units: str
#        :raises ValueError: If radial units are not one of the
#            recognized radial units.
#        :return: Validated radial units.
#        :rtype: str
#        """
#        # Third party modules
#        from pyFAI.units import RADIAL_UNITS
#
#        if radial_units in RADIAL_UNITS.keys():
#            return radial_units
#        else:
#            raise ValueError(
#                f'Invalid radial units: {radial_units}. Must be one of '
#                ', '.join(RADIAL_UNITS.keys()))

#    @field_validator('azimuthal_units')
#    def validate_azimuthal_units(cls, azimuthal_units):
#        """Validate that `azimuthal_units` is one of the keys in the
#        `pyFAI.units.AZIMUTHAL_UNITS` dictionary.
#
#        :param azimuthal_units: The string representing the unit to be
#            validated.
#        :type azimuthal_units: str
#        :raises ValueError: If `azimuthal_units` is not one of the
#            keys in `pyFAI.units.AZIMUTHAL_UNITS`.
#        :return: The original supplied value, if is one of the keys in
#            `pyFAI.units.AZIMUTHAL_UNITS`.
#        :rtype: str
#        """
#        # Third party modules
#        from pyFAI.units import AZIMUTHAL_UNITS
#
#        if azimuthal_units in AZIMUTHAL_UNITS.keys():
#            return azimuthal_units
#        else:
#            raise ValueError(
#                f'Invalid azimuthal units: {azimuthal_units}. Must be one of '
#                ', '.join(AZIMUTHAL_UNITS.keys()))


class Integrate1dConfig(CHAPBaseModel):
    """Class with the input parameters to performs 1D azimuthal
    integration with `pyFAI`.

    :ivar error_model: When the variance is unknown, an error model
        can be given:
        `poisson` (variance = I) or `azimuthal` (variance = (I-<I>)^2).
    :type error_model: str, optionalw
    :ivar method:  IntegrationMethod instance or 3-tuple with
        (splitting, algorithm, implementation)
    :type method: IntegrationMethod, optional
    :ivar npt: Number of integration points, defaults to 1800.
    :type npt: int, optional
    """
    # correctSolidAngle: true
    # dark: None
    error_model: Optional[constr(strip_whitespace=True, min_length=1)] = None
    # filename: None
    # flat: None
    # mask: None
    # metadata: None
    method: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'csr'
    # normalization_factor: None
    npt: Optional[conint(gt=0)] = 1800
    # polarization_factor: None
    # variance: None


class Integrate2dConfig(CHAPBaseModel):
    """Class with the input parameters to performs 2D azimuthal
    integration with `pyFAI`.

    :ivar error_model: When the variance is unknown, an error model
        can be given:
        `poisson` (variance = I) or `azimuthal` (variance = (I-<I>)^2).
    :type error_model: str, optional
    :ivar method:  IntegrationMethod instance or 3-tuple with
        (splitting, algorithm, implementation)
    :type method: IntegrationMethod, optional
    :ivar npt_azim: Number of points for the integration in the
        azimuthal direction, defaults to 3600.
    :type npt_azim: int, optional
    :ivar npt_rad: Number of points for the integration in the
        radial direction, defaults to 1800.
    :type npt_rad: int, optional
    """
    # correctSolidAngle: true
    # dark: None
    # filename: None
    # flat: None
    error_model: Optional[constr(strip_whitespace=True, min_length=1)] = None
    # mask: None
    # metadata: None
    method: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'bbox'
    # normalization_factor: None
    npt_azim: Optional[conint(gt=0)] = 3600
    npt_rad: Optional[conint(gt=0)] = 1800
    # polarization_factor: None
    # safe: None
    # variance: None


class PyfaiIntegrationConfig(CHAPBaseModel):
    """Class representing the configuration for detector data
    integration with `pyFAI`.

    """
    name: constr(strip_whitespace=True, min_length=1)
    integration_method: Literal[
        'integrate1d', 'integrate2d', 'integrate_radial']
    multi_geometry: MultiGeometryConfig
    integration_params: Optional[
        Union[Integrate1dConfig, Integrate2dConfig]] = None

#    @model_validator('integration_params', mode='before')
#    @classmethod
    @model_validator(mode='after')
#    def validate_integration_params(cls, data):
    def validate_integration_params(self):
        """Choose the integration_params type depending on the
        `integration_method` value.

        :param data: Pydantic validator data object.
        :type data: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid `integration_method`.
        :return: The validated list of class properties.
        :rtype: dict
        """
        if self.integration_method == 'integrate1d':
            self.integration_params = Integrate1dConfig(
                **self.integration_params.model_dump())
        elif self.integration_method == 'integrate2d':
            self.integration_params = Integrate2dConfig(
                **self.integration_params.model_dump())
        else:
            raise ValueError('Invalid parameter integration_params '
                             f'({self.integration_params})')
        return self

    def integrate(self, ais, data):
        #print(f'\nais {type(ais)}:\n{ais}\n')
        #print(f'\ndata {type(data)}:\n{data}\n')
        #print(f'\nmulti_geometry {type(self.multi_geometry)}:\n{self.multi_geometry}\n')
        #print(f'\n\nintegration_params {type(self.integration_params)}:\n{self.integration_params}\n')
        #print(f'\nintegration_method {type(self.integration_method)}:\n{self.integration_method}\n')
        if self.integration_method == 'integrate_radial':
            raise NotImplementedError
        else:
            from pyFAI.multi_geometry import MultiGeometry
            mg = MultiGeometry(
                ais=[ais[ai] for ai in self.multi_geometry.ais],
                **self.multi_geometry.model_dump(exclude={'ais'}))
            #print(f'\nmg {type(mg)}:\n{mg}\n')
            integration_method = getattr(mg, self.integration_method)
            #print(f'\n\nintegration_params:\n{self.integration_params}')
            #print(f'\nintegration_method = {integration_method}')
            npts = [d.shape[0] for d in data.values()]
            if not all(_npts == npts[0] for _npts in npts):
                raise RuntimeError('Different number of detector frames for '
                                   f'each azimuthal integrator ({npts})')
            npts = npts[0]
            #print(f'\nnpts = {npts}')
            #print(f'\nlst_data:')
            #for i in range(npts):
            #    lst_data = [data[ai][i] for ai in self.multi_geometry.ais][0]
            #    print(f'\t{lst_data.shape} {lst_data.sum()}')
            #print(f'\n')
            #return [
            results = [
                integration_method(
                    lst_data=[data[ai][i] for ai in self.multi_geometry.ais],
                    **self.integration_params.model_dump())
                for i in range(npts)
            ]
            #print(f'\nresults {type(results[0])} {results[0][0].shape}:\n{results[0][0].sum()} {results[0][0]}')
            #print(f'\nresults[0].intensity {results[0].intensity.shape} {results[0].intensity.sum()}')
#            print(f'\nresults[0].radial {results[0].radial.shape}\n{np.array(results[0].radial).tolist()}')
            #print(f'azimutal? {hasattr(results[0], "azimuthal")} sigma? {hasattr(results[0], "sigma")}')
            return results


class PyfaiIntegrationProcessorConfig(CHAPBaseModel):
    azimuthal_integrators: Optional[conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)] = None
    integrations: conlist(min_length=1, item_type=PyfaiIntegrationConfig)

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        PONI filepaths.

        :param data: Pydantic validator data object.
        :type data: GiwaxsConversionProcessorConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        if isinstance(data, dict):
            inputdir = data.get('inputdir')
            if inputdir is not None and 'azimuthal_integrators' in data:
                ais = data.get('azimuthal_integrators')
                for i, ai in enumerate(deepcopy(ais)):
                    if isinstance(ai, dict):
                        poni_file = ai['poni_file']
                        if not os.path.isabs(poni_file):
                            ais[i]['poni_file'] = os.path.join(
                                inputdir, poni_file)
                    else:
                        poni_file = ai.poni_file
                        if not os.path.isabs(poni_file):
                            ais[i].poni_file = os.path.join(
                                inputdir, poni_file)
                data['azimuthal_integrators'] = ais
        return data
