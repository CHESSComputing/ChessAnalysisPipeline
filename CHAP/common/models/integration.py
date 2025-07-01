"""pyFAI integration related Pydantic model classes."""

# System modules
from copy import deepcopy
import os
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
import numpy as np
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
                    print('Specify either poni_file or params, not both, '
                          'ignoring poni_file')
                    poni_file = None
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

    @property
    def ai(self):
        """Return the azimuthal integrator."""
        return self._ai

    @property
    def mask_data(self):
        """Return the mask array to use for this detector from the
        data in the file specified with the `mask_file` field. Return
        `None` if `mask_file` is `None`.
        """
        if self.mask_file is None:
            return None

        import fabio
        _mask_file = fabio.open(self.mask_file)
        mask_data = _mask_file.data
        _mask_file.close()
        return mask_data


class MultiGeometryConfig(CHAPBaseModel):
    """Class representing the configuration for treating simultaneously
    multiple detector configuration within a single integration

    :ivar ais: List of detector IDs of azimuthal integrators.
    :type ais: Union[str, list[str]]
    :ivar azimuth_range: Common azimuthal range for integration,
        defaults to `[-180.0, 180.0]`.
    :type azimuth_range: Union(
        list[float, float], tuple[float, float]), optional
    :ivar radial_range: Common range for integration,
        defaults to `[0.0, 180.0]`.
    :type radial_range: Union(list[float, float],
                              tuple[float, float]), optional
    :ivar unit: Output unit, defaults to `'q_A^-1'`.
    :type unit: str, optional
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


class IntegrateConfig(CHAPBaseModel):
    """Class with the input parameters to perform various integrations
    with `pyFAI`.

    :ivar error_model: When the variance is unknown, an error model
        can be given (ignored for radial integration):
        `poisson` (variance = I) or `azimuthal` (variance = (I-<I>)^2).
    :type error_model: str, optional
    """
    # correctSolidAngle: true
    # dark: None
    error_model: Optional[constr(strip_whitespace=True, min_length=1)] = None
    # filename: None
    # flat: None
    # mask: None
    # metadata: None
    # normalization_factor: Optional[confloat(allow_inf_nan=False)] = 1.0
    # polarization_factor: None
    # variance: None
    attrs: Optional[dict] = {}


class Integrate1dConfig(IntegrateConfig):
    """Class with the input parameters to perform 1D azimuthal
    integration with `pyFAI`.

    :ivar method: For pyFAI.azimuthalIntegrator.AzimuthalIntegrator
        a registered integration method or a 3-tuple (splitting,
        algorithm, implementation), defaults to `csr`.
        For pyFAI.multi_geometry.MultiGeometry a registered integration
        method, defaults to `splitpixel`.
    :type method: Union[str, tuple], optional
    :ivar npt: Number of integration points, defaults to 1800.
    :type npt: int, optional
    """
    method: Optional[Union[
        str,
        conlist(
            min_length=3, max_length=3,
            item_type=constr(strip_whitespace=True, min_length=1))]] = None
    npt: Optional[conint(gt=0)] = 1800


class Integrate2dConfig(IntegrateConfig):
    """Class with the input parameters to perform 2D azimuthal (cake)
    integration with `pyFAI`.

    :ivar method: Registered integration method, defaults to `bbox`
        for pyFAI.azimuthalIntegrator.AzimuthalIntegrator or
        `splitpixel` for pyFAI.multi_geometry.MultiGeometry.
    :type method: str, optional
    :ivar npt_azim: Number of points for the integration in the
        azimuthal direction, defaults to 3600.
    :type npt_azim: int, optional
    :ivar npt_rad: Number of points for the integration in the
        radial direction, defaults to 1800.
    :type npt_rad: int, optional
    """
    method: Optional[str] = None
    npt_azim: Optional[conint(gt=0)] = 3600
    npt_rad: Optional[conint(gt=0)] = 1800


class IntegrateRadialConfig(IntegrateConfig, MultiGeometryConfig):
    """Class with the input parameters to perform radial integration
    with `pyFAI`.

    :ivar method: Registered integration method, defaults to `csr`.
    :type method: str, optional
    :ivar radial_unit: Unit used for radial representation,
        defaults to `'q_A^-1'`.
    :type radial_unit: str, optional
    :ivar npt: Number of integration points, defaults to 1800.
    :type npt: int, optional
    """
    radial_unit: Optional[
        constr(strip_whitespace=True, min_length=1)] = 'q_A^-1'
    method: Optional[str] = 'csr'
    npt: Optional[conint(gt=0)] = 1800


class PyfaiIntegratorConfig(CHAPBaseModel):
    """Class representing the configuration for detector data
    integrator for `pyFAI`.

    :ivar right_handed: For radial and cake integration, reverse the
        direction of the azimuthal coordinate from pyFAI's convention,
        defaults to True.
    :type right_handed: bool, optional
    """
    name: constr(strip_whitespace=True, min_length=1)
    integration_method: Literal[
        'integrate1d', 'integrate2d', 'integrate_radial']
    multi_geometry: Optional[MultiGeometryConfig] = None
    integration_params: Optional[
            Union[Integrate1dConfig, Integrate2dConfig, IntegrateRadialConfig]
        ] = None
    right_handed: bool = True

    _placeholder_result: PrivateAttr = None

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Validate the input integration configuration.

        :param data: Pydantic validator data object.
        :type data: PyfaiIntegratorConfig,
            pydantic_core._pydantic_core.ValidationInfo
        :return: The currently validated list of class properties.
        :rtype: dict
        """
        integration_method = data.get('integration_method')
        integration_params = data.get('integration_params')
        multi_geometry = data.get('multi_geometry')
        if integration_method == 'integrate1d':
            if multi_geometry is None:
                method = 'csr'
            else:
                method = 'splitpixel'
            if integration_params is None:
                data['integration_params'] = Integrate1dConfig(method=method)
            else:
                if integration_params.get('method') is None:
                    integration_params['method'] = method
                data['integration_params'] = Integrate1dConfig(
                    **integration_params)
        elif integration_method == 'integrate2d':
            if multi_geometry is None:
                method = 'bbox'
            else:
                method = 'splitpixel'
            if integration_params is None:
                data['integration_params'] = Integrate2dConfig(method=method)
            else:
                if integration_params.get('method') is None:
                    integration_params['method'] = method
                data['integration_params'] = Integrate2dConfig(
                    **integration_params)
        elif integration_method == 'integrate_radial':
            data['integration_params'] = IntegrateRadialConfig(
                    **integration_params)
        if (integration_method != 'integrate_radial'
                and 'multi_geometry' not in data):
            mg = MultiGeometryConfig(**integration_params)
            data['multi_geometry'] = mg
            data['integration_params'].attrs.update(mg.model_dump(
                include={'azimuth_range', 'radial_range',  'unit'}))

        return data

    @property
    def result_shape(self):
        if self.integration_method == 'integrate_radial':
            return (self.integration_params.npt, )
        elif self.integration_method == 'integrate1d':
            return (self.integration_params.npt, )
        elif self.integration_method == 'integrate2d':
            return (self.integration_params.npt_azim,
                    self.integration_params.npt_rad)
        else:
            raise NotImplementedError(
                f'Unimplemented integration_method: {self.integration_method}')

    @property
    def result_coords(self):
        # Third party modules
#        from nexusformat.nexus import NXfield
#        from pyFAI.gui.utils.units import Unit

#        print(f'\n\nself {type(self)}:\n{self}')
#        exit(f"\n\ncoords {type(self._placeholder_result['coords'])}: {self._placeholder_result['coords']}\n\n")
        if self._placeholder_result is None:
            raise RuntimeError('Missing placeholder results')
        return self._placeholder_result['coords']
#        if ('azimuthal' in results
#                and results['azimuthal']['unit'] == 'chi_deg'):
#            chi = results['azimuthal']['coords']
#            if self.right_handed:
#                chi = -np.flip(chi)
#                results['intensities'] = np.flip(
#                    results['intensities'], (len(coords)))
#            coords['chi'] = NXfield(chi, 'chi', attrs={'units': 'deg'})
#        if results['radial']['unit'] == 'q_A^-1':
#            unit = Unit.INV_ANGSTROM.symbol
#            coords['q'] = NXfield(results['radial']['coords'], 'q',
#                                  attrs={'units': unit})
#        else:
#            coords['r'] = NXfield(results['radial']['coords'], 'r',
#                                  attrs={'units': '\u212b'})
#        return coords

    def get_axes_indices(self, dataset_ndims):
        return {k: dataset_ndims + i
                for i, k in enumerate(self.result_coords.keys())}

    def get_placeholder_data(self, ais):
        """Return empty input data of the correct shape for use in
        `init_placeholder_data`.
        """
        if self.integration_method == 'integrate_radial':
            return {ai:np.full(ais[ai].ai.detector.shape, 0)
                   for ai in self.integration_params.ais}
        return {ai:np.full(ais[ai].ai.detector.shape, 0)
               for ai in self.multi_geometry.ais}

    def init_placeholder_results(self, ais):
        """Get placeholder results for this integration so we can fill
        in the datasets for results of coordinates when setting up a
        zarr tree for holding results of
        `saxswaxs.PyfaiIntegrationProcessor`.
        """
        self._placeholder_result = self.integrate(
            ais, self.get_placeholder_data(ais))

    def integrate(self, azimuthal_integrators, data):
        """Perform the integration and return the results.

        :param azimuthal_integrators: List of single-detector
            integrator configurations.
        :type azimuthal_integrators: list[AzimuthalIntegratorConfig]
        :param data: Dictionary of 2D detector frames to be
            integrated.
        :type data: dict[str, np.ndarray]
        :return: Integrated intensities and coordinates for every
            frame (or set of frames) in `input_data`.
        :rtype: dict[str, object]
        """
        # Third party modules
        from pyFAI.units import (
            AZIMUTHAL_UNITS,
            RADIAL_UNITS,
            to_unit,
        )

        ais = {name: ai.ai for name, ai in azimuthal_integrators.items()}

        if self.integration_method == 'integrate_radial':
            # Third party modules
            from pyFAI.containers import Integrate1dResult

#            print(f'\n\nais {type(ais)}:\n{ais}\n\n')
#            print(f'\nself.multi_geometry: {self.multi_geometry}\n\n')
#            print(f'\nself.integration_params: {self.integration_params}\n\n')
            results = None
            npts = []
            for k, v in data.items():
#                print(f'\t{v.shape} {v.sum()}')
                if v.ndim == 2:
                    data[k] = np.expand_dims(v, 0)
                elif v.ndim != 3:
                    raise ValueError(
                        f'Illegal dimension for {k} data ({v.ndim})')
                npts.append(data[k].shape[0])
            if not all(n == npts[0] for n in npts):
                raise RuntimeError('Different number of detector frames for '
                                   f'each azimuthal integrator ({npts})')
            npts = npts[0]
            integration_params = self.integration_params.model_dump(
                exclude={'ais', 'attrs', 'error_model', 'chi_disc', 'empty',
                         'wavelength'})
#            print(f'\nintegration_params: {integration_params}\n\n')
            for name in self.integration_params.ais:
                ai = ais[name]
                mask = azimuthal_integrators[name].mask_data
                # if masks is None:
                #     lst_mask = None
                # else:
                #     lst_mask = masks[name]
#                print(f'\n\nai {type(ai)}:\n{ai}\n')
#                print(f'name: {name}\n')
#                print(f'npts: {npts}\n')
                _results = [
                    ai.integrate_radial(
                        data=data[name][i],
                        mask=mask,
                        **integration_params)
                    for i in range(npts)
                ]
#                print(f'\n\n_results {type(_results)}:\n{_results}\n\n')
#                print(f'_results[0] {type(_results[0])}:\n{_results[0]}\n\n')
                if results is None:
                    results = _results
                else:
                    results = [
                        Integrate1dResult(
                            radial=_results[i].radial,
                            intensity=
                                _results[i].intensity + results[i].intensity)
                        for i in range(npts)
                    ]
#                print(f'\n\nresults {type(results)}:\n{results}\n\n')
#                print(f'results[0] {type(results[0])}:\n{results[0]}\n\n')
            if self.right_handed:
                results = [
                    Integrate1dResult(
                        radial=r.radial,
                        intensity=np.flip(r.intensity)
                    )
                    for r in results
                ]
            results = [
                Integrate1dResult(
                    radial=r.radial,
                    intensity=np.where(r.intensity==0, np.nan, r.intensity)
                )
                for r in results
            ]
#            print(f'\n\nresults {type(results)}:\n{results}\n\n')
#            print(f'results[0] {type(results[0])}:\n{results[0]}\n\n')
#            for i, v in enumerate(results):
#                dummy = np.nan_to_num(v[1])
#                print(f'\t{i} {dummy.shape} {dummy.sum()}')
#            print('\n\n')
            # Integrate1dResult's "radial" property is misleadingly
            # named here. When using integrate_radial, the property
            # actually contains azimuthal coordinate values.
            azimuthal_unit = to_unit(
                integration_params['unit'], type_=AZIMUTHAL_UNITS)
            # pyFAI doesn't add the correct unit for chi_deg
            if azimuthal_unit.name[-3:] == 'deg':
                azimuthal_unit.unit_symbol = '$^{o}$'
            coords = {
                azimuthal_unit.name: {
                    'attributes': {
                        'units': azimuthal_unit.unit_symbol,
                        'long_name': azimuthal_unit.label,
                    },
                    'data': results[0].radial,
                    'shape': results[0].radial.shape,
                    'dtype': 'float32',
                },
            }
            results = {'intensities': [v.intensity for v in results],
                       'coords': coords}
#            print(f'\n\ncoords {type(coords)}:\n{coords}\n\n')
#            print(f'results {type(results)}:\n{results}\n\n')
        else:
#            print(f'\n\nais {type(ais)}:\n{ais}\n\n')
#            print(f'data {type(data)}:\n{data}\n\n')
            npts = []
            for k, v in data.items():
#                print(f'\t{v.shape} {v.sum()}')
                if v.ndim == 2:
                    data[k] = np.expand_dims(v, 0)
                elif v.ndim != 3:
                    raise ValueError(
                        f'Illegal dimension for {k} data ({v.ndim})')
                npts.append(data[k].shape[0])
            if not all(n == npts[0] for n in npts):
                raise RuntimeError('Different number of detector frames for '
                                   f'each azimuthal integrator ({npts})')
            npts = npts[0]
#            print(f'\n\ndata {type(data)}:\n{data}\n\n')
#            print(f'\nself.multi_geometry: {self.multi_geometry}\n\n')
#            print(f'\nnpts: {npts}\n\n')
#            exit('Done')
            if self.multi_geometry is None:
                raise RuntimeError('self.multi_geometry is None')
                if len(data) != 1:
                    raise RuntimeError(
                        'Multiple detector not tested without multi_geometry')
                _id = list(ais.keys())[0]
                ai = list(ais.values())[0]
                integration_method = getattr(ai, self.integration_method)
                integration_params = self.integration_params.model_dump()
                integration_params = {
                    **integration_params, **integration_params['attrs']}
                del integration_params['attrs']
                raise RuntimeError(f'Check use of mask in integration_method')
                if masks is None:
                    results = [
                        integration_method(data[_id][i], **integration_params)
                        for i in range(npts)
                    ]
                else:
                    results = [
                        integration_method(
                            np.where(
                                 masks[ai], 0, data[_id][i].astype(np.float64),
                            **integration_params))
                        for i in range(npts)
                    ]
            else:
                # Third party modules
                from pyFAI.multi_geometry import MultiGeometry

                mg = MultiGeometry(
                    [ais[ai] for ai in self.multi_geometry.ais],
                    **self.multi_geometry.model_dump(exclude={'ais'}))
#                print(f'\nmg: {mg}\n\n')
                integration_method = getattr(mg, self.integration_method)
                lst_mask = []
                for name in self.multi_geometry.ais:
                    lst_mask.append(azimuthal_integrators[name].mask_data)
                # if masks is None:
                #     lst_mask = None
                # else:
                #     lst_mask = [masks[ai] for ai in self.multi_geometry.ais]
#                print(f'\nlst_mask: {lst_mask}')
#                print(f'\nself.integration_params {type(self.integration_params)}:\n{self.integration_params.model_dump(exclude="attrs")}')
#                mask = np.asarray(lst_mask[0])
#                print(f'\tmask:{type(mask)} {mask.dtype} {mask.shape} {mask.sum()}')
#                dummy = data['PIL5']
#                print(f'\n\nmasked data {type(dummy)}: {len(dummy)}')
#                for d in dummy:
#                    print(f'\t{d.shape} {d.sum()}')
#                    dd = np.where(masks['PIL5'], 0, d.astype(np.float64))
#                    print(f'\t{dd.shape} {dd.sum()}')
#                print('\n\n')
#                exit('Done')
#                dummy = masks['PIL5']
#                print(f'\n\nmask:\n{dummy}\n\n')
#                masks = None
#                print(f'\n\nmasks: {masks}\n\n')
#                dummy = lst_mask
#                exit(f'\n\ndummy:\n{dummy}')
#                dummy = [data[ai][i].astype(np.float64)
#                          for ai in self.multi_geometry.ais]
#                exit(f'\n\ndummy:\n{dummy}')
                results = [
                    integration_method(
                        [data[ai][i].astype(np.float64)
                         for ai in self.multi_geometry.ais],
                        lst_mask=lst_mask,
                        **self.integration_params.model_dump(
                            exclude='attrs'))
                    for i in range(npts)
                ]
#                if masks is None:
#                    results = [
#                        integration_method(
#                            [data[ai][i].astype(np.float64)
#                             for ai in self.multi_geometry.ais],
#                            lst_mask=lst_mask,
#                            **self.integration_params.model_dump(
#                                exclude='attrs'))
#                        for i in range(npts)
#                    ]
#                else:
#                    results = [
#                        integration_method(
#                            [np.where(
#                                 masks[ai], np.nan, data[ai][i].astype(np.float64))
#                                 masks[ai], 0, data[ai][i].astype(np.float64))
#                             for ai in self.multi_geometry.ais],
#                            [data[ai][i] for ai in self.multi_geometry.ais],
#                            lst_mask=lst_mask,
#                            **self.integration_params.model_dump(
#                                exclude='attrs'))
#                            normalization_factor=[8177142.28771039],
#                            method=('bbox', 'csr', 'cython'),
#                            **self.integration_params.model_dump(exclude={'normalization_factor', 'method'}))
#                        for i in range(npts)
#                    ]
#                dummy = [[np.where(
#                    masks[ai], np.nan, data[ai][i].astype(np.float64))
#                    masks[ai], 0, data[ai][i].astype(np.float64))
#                    for ai in self.multi_geometry.ais] for i in range(npts)]
#                print(f'nan in data? {np.isnan(np.min(dummy))}')
#            print(f'\n\nresults {type(results)}: {results}\n\n')
#            dummy = [v.intensity for v in results]
#            print(f'intensities {type(dummy)}: {len(dummy)}')
#            for d in dummy:
#                print(f'\t{d.shape} {d.sum()}')
#            print('\n\n')
#            exit('Done')
            if isinstance(self.integration_params, Integrate1dConfig):
                if self.multi_geometry is None:
                    unit = integration_params['unit']
                else:
                    unit = self.multi_geometry.unit
                radial_unit = to_unit(unit, type_=RADIAL_UNITS)
                coords = {
                    radial_unit.name: {
                        'attributes': {
                            'units': radial_unit.unit_symbol,
                            'long_name': radial_unit.label,
                        },
                        'data': results[0].radial,
                        'shape': results[0].radial.shape,
                        'dtype': 'float32',
                    },
                }
                results = {'intensities': [v.intensity for v in results],
                           'coords': coords}
            else:
                # pyFAI doesn't add the correct unit for chi_deg
                azimuthal_unit = deepcopy(results[0].azimuthal_unit)
                if azimuthal_unit.name[-3:] == 'deg':
                    azimuthal_unit.unit_symbol = '$^{o}$'
                coords = {
                    azimuthal_unit.name: {
                        'attributes': {
                            'units': azimuthal_unit.unit_symbol,
                            'long_name': azimuthal_unit.label,
                        },
                        'data': results[0].azimuthal,
                        'shape': results[0].azimuthal.shape,
                        'dtype': 'float32',
                    },
                    results[0].radial_unit.name: {
                        'attributes': {
                            'units': results[0].radial_unit.unit_symbol,
                            'long_name': results[0].radial_unit.label,
                        },
                        'data': results[0].radial,
                        'shape': results[0].radial.shape,
                        'dtype': 'float32',
                    },
                }
                if self.right_handed:
                    intensities = [
                        np.flip(v.intensity, axis=0) for v in results]
                else:
                    intensities = [v.intensity for v in results]
                results = {'intensities': intensities, 'coords': coords}
#            print(f'coords {type(coords)}:\n{coords}\n\n')

        return results

    def zarr_tree(self, dataset_shape, dataset_chunks):
        # Third party modules
        import json

        tree = {
            # NXprocess
            'attributes': {
                'default': 'data',
                # 'config': json.dumps(self.dict())
            },
            'children': {
                'data': {
                    # NXdata
                    'attributes': {
                        # 'axes': self.result_axes(),
                        **self.get_axes_indices(len(dataset_shape))
                    },
                    'children': {
                        'I': {
                            # NXfield
                            'attributes': {
                                'long_name': 'Intensity (a.u)',
                                'units': 'a.u'
                            },
                            'dtype': 'float64',
                            'shape': (*dataset_shape, *self.result_shape),
                            'chunks': (*dataset_chunks, *self.result_shape),
                            'compressors': None,
                        },
                        **self.result_coords,
                    }
                }
            }
        }
        return tree


class PyfaiIntegrationConfig(CHAPBaseModel):
    azimuthal_integrators: Optional[conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)] = None
    integrations: conlist(min_length=1, item_type=PyfaiIntegratorConfig)
    sum_axes: Optional[bool] = False
    #sum_axes: Optional[
    #    Union[bool, conlist(min_length=1, item_type=str)]] = False

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, data):
        """Ensure that a valid configuration was provided and finalize
        PONI filepaths.

        :param data: Pydantic validator data object.
        :type data: GiwaxsConversionConfig,
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

    def zarr_tree(self, dataset_shape, dataset_chunks):
        """Return a dictionary representing a `zarr.group` that can be
        used to contain results from `saxswaxs.PyfaiIntegrationProcessor`.
        """
        ais = {ai.id: ai for ai in self.azimuthal_integrators}
        for integration in self.integrations:
            integration.init_placeholder_results(ais)
        tree = {
            'root': {
                'attributes': {
                    'description': 'Container for processed SAXS/WAXS data'
                },
                'children': {
                    integration.name: integration.zarr_tree(
                        dataset_shape, dataset_chunks)
                    for integration in self.integrations
                }
            }
        }
        return tree
