"""`Pydantic <https://github.com/pydantic/pydantic>`__ model
configuration classes unique to the the tomography workflow.
"""

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
    """Validate an azimuthal integrator model.

    :param data: Input data.
    :vartype data: dict
    :param info: Model parameter validation information.
    :vartype info: pydantic.ValidationInfo
    :return: Validated data.
    :rtype: dict
    """
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

    :ivar mask_file: Path to the mask file.
    :vartype mask_file: FilePath, optional
    :ivar poni_file: Path to the PONI file, specify either `poni_file`
        or `params`, not both.
    :vartype poni_file: FilePath, optional
    :ivar params: Azimuthal integrator configuration parameters,
        specify either `poni_file` or `params`, not both.
    :vartype params: dict, optional
    """

    mask_file: Optional[FilePath] = None
    params: Optional[dict] = None
    poni_file: Optional[FilePath] = None

    @model_validator(mode='before')
    @classmethod
    def validate_integratorconfig_before(cls, data, info):
        """Validate the integrator configuration.

        :param data: Input data.
        :type data: dict
        :param info: Model parameter validation information.
        :type info: pydantic.ValidationInfo
        :return: Validated data.
        :rtype: dict
        """
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
        """Return the integrator.

        :return: The model's integrator.
        :rtype: pyFAI.integrator.azimuthal.AzimuthalIntegrator
        """
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
    """Fiber or grazing incidence integrator configuration class to
    represent a single detector used in the experiment.
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
    :vartype ais: Union[str, list[str]]
    :ivar azimuth_range: Common azimuthal range for integration,
        defaults to `[-180.0, 180.0]`.
    :vartype azimuth_range:
        Union(list[float, float], tuple[float, float]), optional
    :ivar radial_range: Common range for integration, defaults to
        `[0.0, 180.0]`.
    :vartype radial_range:
        Union(list[float, float], tuple[float, float]), optional
    :ivar unit: Output unit, defaults to `q_A^-1`.
    :vartype unit: str, optional
    :ivar chi_disc: chi discontinuity value, defaults to `180`.
    :vartype chi_disc: int, optional
    :ivar empty: Value for empty pixels.
    :vartype empty: float
    :ivar wavelength: Wave length used in meters.
    :vartype wavelength: float, optional
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


class Integrate1dConfig(CHAPBaseModel):
    """Class with the input parameters to performs 1D azimuthal
    integration with
    `pyFAI <https://pyfai.readthedocs.io/en/stable>`__.

    :ivar error_model: When the variance is unknown, an error model
        can be given:
        `poisson` (variance = I) or `azimuthal` (variance = (I-<I>)^2).
    :vartype error_model: str, optional
    :ivar method:  Integration method: 3-tuple with (splitting,
        algorithm, implementation), defaults to [`'bbox'`, `'csr'`,
        `'cython'`]
    :vartype method: list[str, str, str], optional
    :ivar npt: Number of integration points, defaults to 1800.
    :vartype npt: int, optional
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
    integration with
    `pyFAI <https://pyfai.readthedocs.io/en/stable>`__.

    :ivar error_model: When the variance is unknown, an error model
        can be given:
        `poisson` (variance = I) or `azimuthal` (variance = (I-<I>)^2).
    :vartype error_model: str, optional
    :ivar method:  Integration method: 3-tuple with (splitting,
        algorithm, implementation), defaults to [`'bbox'`, `'csr'`,
        `'cython'`]
    :vartype method: list[str, str, str], optional
    :ivar npt_azim: Number of points for the integration in the
        azimuthal direction, defaults to 3600.
    :vartype npt_azim: int, optional
    :ivar npt_rad: Number of points for the integration in the
        radial direction, defaults to 1800.
    :vartype npt_rad: int, optional
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
    integration with
    `pyFAI <https://pyfai.readthedocs.io/en/stable>`__.

    :ivar ais: The detector prefix.
    :vartype ais: str
    :ivar method:  Integration method: 3-tuple with (splitting,
        algorithm, implementation), defaults to [`'no'`, `'histogram'`,
        `'cython'`]
    :vartype method: list[str, str, str], optional
    :ivar npt_ip: Number of points along the in-plane axis direction,
        defaults to 1000.
    :vartype npt_ip: int, optional
    :ivar npt_oop: Number of points along the out-of-plane axis
        direction, defaults to 1000.
    :vartype npt_oop: int, optional
    :ivar sample_orientation: orientation of according to
        `EXIF orientation values <https://pyfai.readthedocs.io/en/stable/usage/tutorial/Orientation.html>`__,
        defaults to `2`, or `4` for `ais` equal to `EIG1` or `PIL5`,
        and `1` otherwise.
    :vartype sample_orientation: int, optional
    :ivar unit_ip: The in-plane unit, defaults to `qip_A^-1`.
    :vartype unit_ip: str, optional
    :ivar unit_oop: The out-of-plane unit, defaults to `qoop_A^-1`.
    :vartype unit_oop: str, optional
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
        constr(strip_whitespace=True, min_length=1)] = 'qip_A^-1'
    unit_oop: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'qoop_A^-1'
    # variance: None
    attrs: Optional[dict] = {}

    @field_validator('unit_ip', mode='after')
    @classmethod
    def validate_unit_ip(cls, unit_ip, info):
        """Validate the sample orientation.

        :param unit_ip: The in-plane unit, defaults to `qip_A^-1`.
        :type unit_ip: str, optional
        :return: The validated unit.
        :rtype: int
        """
        # Third party modules
        from pyFAI import units

        assert unit_ip in units.ANY_FIBER_UNITS
        return unit_ip


    @field_validator('unit_oop', mode='after')
    @classmethod
    def validate_unit_oop(cls, unit_oop, info):
        """Validate the sample orientation.

        :param unit_oop: The out-of-plane unit,
            defaults to `qoop_A^-1`.
        :type unit_oop: str, optional
        :return: The validated unit.
        :rtype: int
        """
        # Third party modules
        from pyFAI import units

        assert unit_oop in units.ANY_FIBER_UNITS
        return unit_oop


    @field_validator('sample_orientation', mode='after')
    @classmethod
    def validate_sample_orientation(cls, sample_orientation, info):
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
    integrater for
    `pyFAI <https://pyfai.readthedocs.io/en/stable>`__.

    :ivar name: Integration type name, e.g. `cake`, or `wedge`.
    :vartype name: str
    :ivar integration_method: Integration method.
    :vartype integration_method: Literal[
        'integrate1d', 'integrate2d', 'integrate_radial',
        'integrate2d_grazing_incidence']
    :ivar multi_geometry: Multiple detector configuration.
    :vartype multi_geometry: MultiGeometryConfig
    :ivar integration_params: Integration parameter configuration.
    :vartype integration_params: Union[
        Integrate1dConfig, Integrate2dConfig, Integrate2d_GI_Config]
    :ivar right_handed: For radial and cake integration, reverse the
        direction of the azimuthal coordinate from pyFAI's convention,
        defaults to True.
    :vartype right_handed: bool, optional
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

        :raises ValueError: Invalid `integration_method`.
        :return: The validated integrater configuration.
        :rtype: PyfaiIntegratorConfig
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
        """Perform the azimuthal integration.

        :param ais: The azimuthal integrators.
        :type ais: dict
        :param data: Detector image(s).
        :type data: dict
        :param masks: Detector mask(s).
        :type masks: numpy.ndarray
        :param thetas: Tilt of the sample stage towards the beam (only
            relevant to wedge or grazing-incidence integration)
        :type thetas: numpy.ndarray
        :return: Integration results.
        :rtype: dict
        """
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
    """Configuration for the wedge correction processor
    :py:class:`~CHAP.giwaxs.processor.GiwaxsConversionProcessor`.

    :ivar azimuthal_integrators: List of azimuthal integrator
        configurations.
    :vartype azimuthal_integrators: list[FiberIntegratorConfig]
    :ivar scan_step_indices: Optional scan step indices to convert.
        If not specified, all images will be converted.
    :vartype scan_step_indices: Union(int, list[int], str), optional
    :ivar save_raw_data: Save the raw data in the NeXus output,
        defaults to `False`.
    :vartype save_raw_data: bool, optional
    :ivar skip_animation: Skip the animation (subject to `save_figures`
        being `True`), defaults to `False`.
    :vartype skip_animation: bool, optional
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
        """Validate the scan step indices.

        :param scan_step_indices: Input scan step indices.
        :vartype scan_step_indices: Union(int, list[int], str), optional
        :return: Validated scan step indices.
        :rtype: list[int]
        """
        if isinstance(scan_step_indices, int):
            scan_step_indices = [scan_step_indices]
        if isinstance(scan_step_indices, str):
            # Local modules
            from CHAP.utils.general import string_to_list

            scan_step_indices = string_to_list(scan_step_indices)

        return scan_step_indices


class PyfaiIntegrationConfig(CHAPBaseModel):
    """Configuration for the azimuthal integrator processor
    :py:class:`~CHAP.giwaxs.processor.PyfaiIntegrationProcessor`.

    :ivar azimuthal_integrators: List of azimuthal integrator
        configurations.
    :vartype azimuthal_integrators: list[AzimuthalIntegratorConfig]
    :ivar integrations: The azimuthal integrator configurations.
    :vartype integrations: list[PyfaiIntegratorConfig]
    :ivar sum_axes: Sum the detector data over the independent
        coordinates before integration, defaults to `False`.
    :vartype sum_axes: bool, optional

    """
    azimuthal_integrators: Optional[conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)] = None
    integrations: conlist(min_length=1, item_type=PyfaiIntegratorConfig)
    sum_axes: Optional[bool] = False
    #sum_axes: Optional[
    #    Union[bool, conlist(min_length=1, item_type=str)]] = False

    _validate_filename = validate_azimuthal_integrators_before
