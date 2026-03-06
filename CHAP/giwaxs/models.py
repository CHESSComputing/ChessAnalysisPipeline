"""GIWAXS Pydantic model classes."""

# System modules
from copy import deepcopy
import os
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
from pydantic import (
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
from pyFAI.integrator.fiber import FiberIntegrator

# Local modules
from CHAP import CHAPBaseModel
from CHAP.common.models.map import Detector


@model_validator(mode='before')
def validate_azimuthal_integrators_before(cls, data, info):
    ais = data['azimuthal_integrators']
    inputdir = info.data['inputdir']
    for i, ai in enumerate(deepcopy(ais)):
        if isinstance(ai, (AzimuthalIntegratorConfig, FiberIntegratorConfig)):
            ai = ai.model_dump()
        if 'mask_file' in ai:
            mask_file = ai['mask_file']
            if not os.path.isabs(mask_file):
                ai['mask_file'] = os.path.join(inputdir, mask_file)
        if 'poni_file' in ai:
            poni_file = ai['poni_file']
            if not os.path.isabs(poni_file):
                ai['poni_file'] = os.path.join(inputdir, poni_file)
        ais[i] = ai
    data['azimuthal_integrators'] = ais
    return data

class IntegratorConfig(Detector, CHAPBaseModel):
    """Integrator configuration class to represent a single detector
    used in the experiment.

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

    @model_validator(mode='before')
    @classmethod
    def validate_integratorconfig_before(cls, data, info):
        if isinstance(data, dict):
            params = data.get('params')
            poni_file = data.get('poni_file')
            if params is None and poni_file is None:
                raise ValueError('Specify either poni_file or params')
            elif params is not None and poni_file is not None:
                print('Specify either poni_file or params, not both, '
                      'ignoring poni_file')
                data['poni_file'] = None
        return data

    @property
    def ai(self):
        """Return the integrator."""
        return self._ai


class AzimuthalIntegratorConfig(IntegratorConfig):
    """Azimuthal integrator configuration class to represent a single
    detector used in the experiment.
    """
    _ai: AzimuthalIntegrator = PrivateAttr()

    @model_validator(mode='after')
    def validate_azimuthalintegratorconfig_after(self):
        """Set the default azimuthal integrator.

        :return: Validated configuration class.
        :rtype: AzimuthalIntegratorConfig
        """
        if self.params is not None:
            self._ai = AzimuthalIntegrator(**self.params)
        elif self.poni_file is not None:
            # Third party modules
            from pyFAI import load

            self._ai = load(str(self.poni_file))
            self.params = {
                'detector': self._ai.detector.name,
                'dist': self._ai.dist,
                'poni1': self._ai.poni1,
                'poni2': self._ai.poni2,
                'rot1': self._ai.rot1,
                'rot2': self._ai.rot2,
                'rot3': self._ai.rot3,
                'wavelength': self._ai.wavelength,
            }
        return self


class FiberIntegratorConfig(IntegratorConfig):
    """Fiber?grazing incidence integrator configuration class to
    represent a single detector used in the experiment.
    """
    _ai: FiberIntegrator = PrivateAttr()

    @model_validator(mode='after')
    def validate_fiberintegratorconfig_after(self):
        """Set the default fiber/grazing incidence integrator.

        :return: Validated configuration class.
        :rtype: FiberIntegratorConfig
        """
        if self.params is not None:
            self._ai = FiberIntegrator(**self.params)
        elif self.poni_file is not None:
            # Third party modules
            from pyFAI import load

            ai = load(str(self.poni_file))
            self.params = {
                'detector': ai.detector.name,
                'dist': ai.dist,
                'poni1': ai.poni1,
                'poni2': ai.poni2,
                'rot1': ai.rot1,
                'rot2': ai.rot2,
                'rot3': ai.rot3,
                'wavelength': ai.wavelength,
            }
            self._ai = FiberIntegrator(**self.params)
        return self


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
        conlist(
            min_length=3, max_length=3,
            item_type=constr(strip_whitespace=True, min_length=1))
        ] = ['bbox', 'csr', 'cython']
    #normalization_factor: Optional[confloat(allow_inf_nan=False)] = 1.0
    npt: Optional[conint(gt=0)] = 1800
    # polarization_factor: None
    # variance: None
    attrs: Optional[dict] = {}


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
        conlist(
            min_length=3, max_length=3,
            item_type=constr(strip_whitespace=True, min_length=1))
        ] = ['bbox', 'csr', 'cython']
    # normalization_factor: None
    npt_azim: Optional[conint(gt=0)] = 3600
    npt_rad: Optional[conint(gt=0)] = 1800
    # polarization_factor: None
    # safe: None
    # variance: None
    attrs: Optional[dict] = {}


class Integrate2d_GI_Config(CHAPBaseModel):
    """Class with the input parameters to performs 2D grazing incidence
    integration with `pyFAI`.

    :ivar method:  IntegrationMethod instance or 3-tuple with
        (splitting, algorithm, implementation)
    :type method: IntegrationMethod, optional
    :ivar npt_ip: Number of points along the in-plane axis direction,
        defaults to 1000.
    :type npt_ip: int, optional
    :ivar npt_oop: Number of points along the out-of-plane axis
        direction, defaults to 1000.
    :type npt_oop: int, optional
    :ivar sample_orientation: orientation of according to EXIF
        orientation values, defaults to `2`, or `4` for `ais` equal to
        `EIG1` or `PIL5`, and `1` otherwise.
    :type sample_orientation: int, optional
    """
    ais: constr(strip_whitespace=True, min_length=1)
    # correctSolidAngle: true
    # dark: None
    # filename: None
    # flat: None
    # mask: None
    # metadata: None
    method: Optional[
        conlist(
            min_length=3, max_length=3,
            item_type=constr(strip_whitespace=True, min_length=1))
        ] = ['no', 'histogram', 'cython']
    # normalization_factor: None
    npt_ip: Optional[conint(gt=0)] = 1000
    npt_oop: Optional[conint(gt=0)] = 1000
    # polarization_factor: None
    # safe: None
    sample_orientation: Optional[conint(ge=1, le=8)] = None
    unit_ip: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'q_A^-1'
    unit_oop: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'q_A^-1'
    # variance: None
    attrs: Optional[dict] = {}

    @field_validator('sample_orientation', mode='after')
    @classmethod
    def validate_ais(cls, sample_orientation, info):
        """Validate the sample orientation.

        :param sample_orientation: The sample orientation.
        :type sample_orientation: int
        :return: The validated sample orientation.
        :rtype: int
        """
        if sample_orientation is None:
            ais = info.data['ais']
            if ais == 'PIL5':
                sample_orientation = 4
            elif ais == 'EIG1':
                sample_orientation = 2
            else:
                sample_orientation = 1
        return sample_orientation


class PyfaiIntegratorConfig(CHAPBaseModel):
    """Class representing the configuration for detector data
    integrater for `pyFAI`.

    :ivar right_handed: For radial and cake integration, reverse the
        direction of the azimuthal coordinate from pyFAI's convention,
        defaults to True.
    :type right_handed: bool, optional
    """
    name: constr(strip_whitespace=True, min_length=1)
    integration_method: Literal[
        'integrate1d', 'integrate2d', 'integrate_radial',
        'integrate2d_grazing_incidence']
    multi_geometry: Optional[MultiGeometryConfig] = None
    integration_params: Optional[Union[
        Integrate1dConfig, Integrate2dConfig, Integrate2d_GI_Config]] = None
    right_handed: bool = True

    @model_validator(mode='before')
    @classmethod
    def validate_pyfaiintegratorconfig_before(cls, data):
        """Validate the integration parameters.

        :param data: Pydantic validator data object.
        :type data: PyfaiIntegratorConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        integration_method = data['integration_method']
        if integration_method == 'integrate2d_grazing_incidence':
            if 'multi_geometry' in data:
                raise ValueError('Invalid parameter multi_geometry ',
                                 f'(invalid for {integration_method})')
        else:
            if 'multi_geometry' in data:
                return data
            mg = MultiGeometryConfig(**data['integration_params'])
            if len(mg.ais) != 1:
                raise ValueError('Invalid parameter integration_params["ais"]',
                                 f' ({mg.ais}, multiple detectors not allowed')
            data['integration_params']['attrs'] = mg.model_dump(
                include={'azimuth_range', 'radial_range',  'unit'})
        return data

    @model_validator(mode='after')
    def validate_pyfaiintegratorconfig_after(self):
        """Choose the integration_params type depending on the
        `integration_method` value.

        :param data: Pydantic validator data object.
        :type data: pydantic_core._pydantic_core.ValidationInfo
        :raises ValueError: Invalid `integration_method`.
        :return: The validated list of class properties.
        :rtype: dict
        """
        if self.integration_method == 'integrate1d':
            if self.integration_params is None:
                self.integration_params = Integrate1dConfig()
            else:
                self.integration_params = Integrate1dConfig(
                    **self.integration_params.model_dump())
        elif self.integration_method == 'integrate2d':
            if self.integration_params is None:
                self.integration_params = Integrate2dConfig()
            else:
                self.integration_params = Integrate2dConfig(
                    **self.integration_params.model_dump())
        elif self.integration_method == 'integrate2d_grazing_incidence':
            if self.integration_params is None:
                self.integration_params = Integrate2d_GI_Config()
            else:
                self.integration_params = Integrate2d_GI_Config(
                    **self.integration_params.model_dump())
        else:
            raise ValueError('Invalid parameter integration_params '
                             f'({self.integration_params})')
        return self

    def integrate(self, ais, data, masks=None, thetas=None):
        if self.integration_method == 'integrate_radial':
            raise NotImplementedError
        else:
            npts = [d.shape[0] for d in data.values()]
            if not all(_npts == npts[0] for _npts in npts):
                raise RuntimeError('Different number of detector frames for '
                                   f'each azimuthal integrator ({npts})')
            npts = npts[0]
            if self.multi_geometry is None:
                ai_id = list(ais.keys())[0]
                ai = list(ais.values())[0]
                integration_params = self.integration_params.model_dump()
                integration_params = {
                    **integration_params, **integration_params['attrs']}
                del integration_params['attrs']
                if isinstance(self.integration_params, Integrate2d_GI_Config):
                    if thetas is not None:
                        assert len(thetas) == npts
                    # units are currently not working in pyFAI
                    unit_ip = integration_params.pop('unit_ip', None)
                    unit_oop = integration_params.pop('unit_oop', None)
                integration_method = getattr(ai, self.integration_method)
                if thetas is not None:
                    results = [
                        integration_method(
                            data[ai_id][i],
                            #mask=masks[ai_id],
                            incident_angle=theta,
                            tilt_angle=0,
                            **integration_params)
                        for i, theta in enumerate(thetas)
                    ]
                else:
                    results = [
                        integration_method(
                            data[ai_id][i],
                            #mask=masks[ai_id],
                            **integration_params)
                        for i in range(npts)
                    ]
            else:
                # Third party modules
                from pyFAI.multi_geometry import MultiGeometry

                mg = MultiGeometry(
                    [ais[ai] for ai in self.multi_geometry.ais],
                    **self.multi_geometry.model_dump(exclude={'ais'}))
                integration_method = getattr(mg, self.integration_method)
                if masks is None:
                    lst_mask = None
                else:
                    lst_mask = [masks[ai] for ai in self.multi_geometry.ais]
                results = [
                    integration_method(
                        [data[ai][i] for ai in self.multi_geometry.ais],
                        lst_mask=lst_mask,
                        **self.integration_params.model_dump(exclude='attrs'))
#                        normalization_factor=[8177142.28771039],
#                        method=('bbox', 'csr', 'cython'),
#                        **self.integration_params.model_dump(exclude={'normalization_factor', 'method'}))
                    for i in range(npts)
                ]
            if npts == 1:
                intensities = results[0].intensity
            else:
                intensities = [v.intensity for v in results]
            if isinstance(self.integration_params, Integrate1dConfig):
                if self.multi_geometry is None:
                    unit = integration_params['unit']
                else:
                    unit = self.multi_geometry.unit
                results = {
                    'intensities': intensities,
                    'radial': {'coords': results[0].radial, 'unit': unit}}
            elif isinstance(self.integration_params, Integrate2d_GI_Config):
                results = {
                    'intensities': intensities,
                    'inplane': {
                        'coords': results[0].inplane,
                        'unit': results[0].ip_unit.name},
                    'outofplane': {
                        'coords': results[0].outofplane,
                        'unit': results[0].oop_unit.name}}
            else:
                results = {
                    'intensities': intensities,
                    'radial': {
                        'coords': results[0].radial,
                        'unit': results[0].radial_unit.name},
                    'azimuthal': {
                        'coords': results[0].azimuthal,
                        'unit': results[0].azimuthal_unit.name}}
        return results


class GiwaxsConversionConfig(CHAPBaseModel):
    """Class representing metadata required to locate GIWAXS image
    files for a single scan to convert to q_par/q_perp coordinates.

    :ivar azimuthal_integrators: List of azimuthal integrator
        configurations.
    :type azimuthal_integrators: list[
        CHAP.giwaxs.models.FiberIntegratorConfig]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :type scan_step_indices: Union(int, list[int], str), optional
    :ivar save_raw_data: Save the raw data in the NeXus output,
        defaults to `False`.
    :type save_raw_data: bool, optional
    :ivar skip_animation: Skip the animation (subject to `save_figures`
        being `True`), defaults to `False`.
    :type skip_animation: bool, optional

    """
    azimuthal_integrators: conlist(
        min_length=1, item_type=FiberIntegratorConfig)
    integrations: conlist(min_length=1, item_type=PyfaiIntegratorConfig)
    scan_step_indices: Optional[
        conlist(min_length=1, item_type=conint(ge=0))] = None
    save_raw_data: Optional[bool] = False
    skip_animation: Optional[bool] = False

    _validate_filename = validate_azimuthal_integrators_before

    @field_validator('scan_step_indices', mode='before')
    @classmethod
    def validate_scan_step_indices(cls, scan_step_indices):
        """Ensure that a valid configuration was provided and finalize
        PONI filepaths.

        :param data: Pydantic validator data object.
        :type data: GiwaxsConversionConfig,
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


class PyfaiIntegrationConfig(CHAPBaseModel):
    azimuthal_integrators: Optional[conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)] = None
    integrations: conlist(min_length=1, item_type=PyfaiIntegratorConfig)
    sum_axes: Optional[bool] = False
    #sum_axes: Optional[
    #    Union[bool, conlist(min_length=1, item_type=str)]] = False

    _validate_filename = validate_azimuthal_integrators_before
