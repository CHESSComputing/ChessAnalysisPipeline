"""Definition and utilities for integration tools"""

# System modules
from typing import Literal, Optional

# Third party modules
from pydantic import (
    ConfigDict,
    FilePath,
    PrivateAttr,
    StringConstraints,
    conlist,
    model_validator,
)
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from typing_extensions import Annotated

# Local modules
from CHAP import CHAPBaseModel


class MyBaseModel(CHAPBaseModel):
    """Subclass `CHAPBaseModel` to allow arbitrary `dicts` in models
    that have fields meant to directly hold some kwargs to `pyFAI`
    methods.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

class AzimuthalIntegratorConfig(MyBaseModel):
    """Configuration for a single detector used in processing
    SAXS/WAXS data.
    """
    name: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1)]
    poni_file: Optional[FilePath] = None
    mask_file: Optional[FilePath] = None
    params: Optional[dict] = None
    ai: AzimuthalIntegrator

    @model_validator(mode='before')
    @classmethod
    def validate_root(cls, data):
        """Make sure `data` contains either `poni_file` _or_ `params`,
        not both, and that the field that is used defines a valid
        `pyFAI.azimuthalIntegrator.AzimuthalIntegrator object`.
        """
        if isinstance(data, dict):
            params = data.get('params')
            poni_file = data.get('poni_file')
            if ((params is None and poni_file is not None) or
                (params is not None and poni_file is None)):
                if poni_file is not None:
                    from pyFAI import load
                    data['ai'] = load(poni_file)
                else:
                    data['ai'] = AzimuthalIntegrator(**params)
            else:
                raise ValueError(
                    'Must specify at exactly one of: poni_file, params')
        return data

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


class PyfaiIntegrationConfig(MyBaseModel):
    """Class for configuring and performing a single type of
    integration on a specific set of detector data.
    """
    name: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1)]
    multi_geometry: dict = {}
    integration_method: Literal[
        'integrate1d', 'integrate2d', 'integrate_radial']
    integration_params: dict
    right_handed: bool = True
    _placeholder_result: PrivateAttr = None

    def integrate(self, azimuthal_integrators, input_data):
        """Perform the integration and return the results.

        :param azimuthal_integrators: List of single-detector
            integrator configurations.
        :type azimuthal_integrators: list[AzimuthalIntegratorConfig]
        :param input_data: Dictionary of 2D detector frames to be
            integrated.
        :type input_data: dict[str, np.ndarray]
        :return: Integrated resulte for every frame (or set of frames)
            in `input_data`.
        :rtype: list[pyFAI.containers.IntegrateResult]
        """
        import numpy as np

        # Confirm same no. of frames for all detectors
        npts = [len(input_data[name])
                for name in self.integration_params['lst_data']]
        if not all([_npts == npts[0] for _npts in npts]):
            raise RuntimeError(
                'Different number of frames of detector data provided')
        npts = npts[0]

        # Get rest of kwargs for the integration method
        integration_params = {k: v
                              for k, v in self.integration_params.items()
                              if k != 'lst_data'}

        # Get dict with values of JUST pyfai azim. int. objects
        ais = {name: ai.ai for name, ai in azimuthal_integrators.items()}

        if self.integration_method == 'integrate_radial':
            from pyFAI.containers import Integrate1dResult
            ais, _, chi_offset = self.adjust_azimuthal_integrators(ais)
            results = None
            for name in self.integration_params.get('lst_data', []):
                ai = ais[name]
                mask = azimuthal_integrators[name].mask_data
                _results = [
                    ai.integrate_radial(
                        data=input_data[name][i],
                        mask=mask,
                        **integration_params)
                    for i in range(npts)
                ]
                if results is None:
                    results = _results
                else:
                    results = [
                        Integrate1dResult(
                            radial=_results[i].radial,
                            intensity=(_results[i].intensity
                                       + results[i].intensity))
                        for i in range(npts)
                    ]
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
                    radial=r.radial + chi_offset,
                    intensity=np.where(r.intensity==0, np.nan, r.intensity)
                )
                for r in results
            ]
            return results
        # Deal with cake / azimuthal integration
        mg = self.get_multi_geometry(ais)
        integration_method = getattr(mg, self.integration_method)
        lst_mask = []
        for name in self.integration_params['lst_data']:
            lst_mask.append(azimuthal_integrators[name].mask_data)
        results = [
            integration_method(
                lst_data=[input_data[name][i]
                          for name in self.integration_params['lst_data']],
                lst_mask=lst_mask,
                **integration_params)
            for i in range(npts)
        ]
        if self.integration_method == 'integrate2d' and self.right_handed:
            # Flip results along azimuthal axis
            from pyFAI.containers import Integrate2dResult
            results = [
                Integrate2dResult(
                    np.flip(result.intensity, axis=0),
                    result.radial, result.azimuthal
                )
                for result in results
            ]

        return results

    def init_placeholder_results(self, ais):
        """Get placeholder results for this integration so we can fill
        in the datasets for results of coordinates when setting up a
        zarr tree for holding results of
        `saxswaxs.PyfaiIntegrationProcessor`.
        """
        placeholder_result = None
        placeholder_data = self.get_placeholder_data(ais)
        if self.integration_method == 'integrate_radial':
            # Radial integration happens detector-by-detector with the
            # same aprameters for integration range and number of
            # points. So, for getting radially integrated placeholder
            # results, only need to perform integration for at most
            # one detector since it's all 0s anyways.
            det_name = self.integration_params.get('lst_data', [])[0]
            ai = ais[det_name]
            integration_params = {k: v
                                  for k, v in self.integration_params.items()
                                  if k != 'lst_data'}
            placeholder_result = ai.integrate_radial(
                data=placeholder_data[0], **integration_params)
        else:
            mg = self.get_multi_geometry(ais)
            integration_method = getattr(mg, self.integration_method)
            integration_params = {k: v
                                  for k, v in self.integration_params.items()
                                  if k != 'lst_data'}
            placeholder_result = integration_method(
                lst_data=placeholder_data, **integration_params)
            # right handed coorinate system acieved by flipping
            # intensity results, not the coordinate axes, so no need
            # to do this with placeholder results since they're all 0s
            # anyways.

        self._placeholder_result = placeholder_result

    def adjust_azimuthal_integrators(self, ais):
        """Artificially rotate the integrators in the azimuthal
        direction so none of the input data crosses a discontinuity
        when performing the integration.
        """
        import numpy as np

        # Adjust azimuthal angles used (since pyfai has some odd conventions)
        chi_min, chi_max = self.multi_geometry.get('azimuth_range',
                                                   (-180.0, 180.0))
        # Force a right-handed coordinate system
        chi_min, chi_max = 360 - chi_max, 360 - chi_min

        # If the discontinuity is crossed, artificially rotate the
        # detectors to achieve a continuous azimuthal integration range
        chi_disc = self.multi_geometry.get('chi_disc', 180)
        if chi_min < chi_disc and chi_max > chi_disc:
            chi_offset = chi_max - chi_disc
        else:
            chi_offset = 0
        chi_min -= chi_offset
        chi_max -= chi_offset
        for ai in ais.values():
            ai.rot3 += chi_offset * np.pi/180.0
        return ais, (chi_min, chi_max), chi_offset

    def get_multi_geometry(self, ais):
        """Return a `pyFAI.multi_geometry.MultiGeometry` object for
        use with azimuthal or cake integrations.
        """
        from pyFAI.multi_geometry import MultiGeometry

        ais, azimuth_range, _ = \
            self.adjust_azimuthal_integrators(ais)
        ais = [ais[name]
               for name in self.multi_geometry['ais']]
        kwargs = {k: v for k, v in self.multi_geometry.items()
                  if k not in ('ais', 'azimuth_range')}
        return MultiGeometry(ais, azimuth_range=azimuth_range, **kwargs)

    def get_placeholder_data(self, ais):
        """Return empty input data of the correct shape for use in
        `init_placeholder_data`.
        """
        import numpy as np

        data = [np.full(ais[name].detector.shape, 0)
                for name in self.integration_params.get('lst_data', [])]
        return data

    # def result_axes(self):
    #     return []

    @property
    def result_shape(self):
        """Return shape of one frame of results from this integration."""
        if self.integration_method == 'integrate_radial':
            return (self.integration_params['npt'], )
        if self.integration_method == 'integrate1d':
            return (self.integration_params['npt'], )
        if self.integration_method == 'integrate2d':
            return (self.integration_params['npt_azim'],
                    self.integration_params['npt_rad'])
        raise NotImplementedError(
            f'Unimplemented integration_method: {self.integration_method}')

    @property
    def result_coords(self):
        """Return a dictionary representing the `zarr.array` objects
        for the coordinates of a single frame of results from this
        integration.
        """
        import pyFAI.containers
        import pyFAI.units

        coords = {}
        if self._placeholder_result is None:
            raise RuntimeError
        elif isinstance(self._placeholder_result,
                        pyFAI.containers.Integrate2dResult):
            radial_unit = pyFAI.units.to_unit(
                self._placeholder_result.radial_unit,
                type_=pyFAI.units.RADIAL_UNITS)
            coords[radial_unit.name] = {
                'attributes': {
                    'units': radial_unit.unit_symbol,
                    'long_name': radial_unit.label,
                },
                'data': self._placeholder_result.radial.tolist(),
                'shape': self._placeholder_result.radial.shape,
                'dtype': 'float32',
            }
            azimuthal_unit = pyFAI.units.to_unit(
                self._placeholder_result.azimuthal_unit,
                type_=pyFAI.units.AZIMUTHAL_UNITS)
            coords[azimuthal_unit.name] = {
                'attributes': {
                    'units': azimuthal_unit.name.split('_')[-1],
		    'long_name': azimuthal_unit.label,
                },
                'data': self._placeholder_result.azimuthal.tolist(),
                'shape': self._placeholder_result.azimuthal.shape,
                'dtype': 'float32',
            }
        elif isinstance(self._placeholder_result,
                        pyFAI.containers.Integrate1dResult):
            # Integrate1dResult's "radial" property is misleadingly
            # named here. When using integrate_radial, the property
            # actually contains azimuthal coordinate values.
            if self.integration_method == 'integrate_radial':
                azimuthal_unit = pyFAI.units.to_unit(
                    self._placeholder_result.unit,
                    type_=pyFAI.units.AZIMUTHAL_UNITS)
                coords[azimuthal_unit.name] = {
                    'attributes': {
                        'units': azimuthal_unit.name.split('_')[-1],
                        'long_name': azimuthal_unit.label,
                    },
                    'data': self._placeholder_result.radial.tolist(),
                    'shape': self._placeholder_result.radial.shape,
                    'dtype': 'float32',
                }
            elif self.integration_method == 'integrate1d':
                radial_unit = pyFAI.units.to_unit(
                    self.multi_geometry.get('unit', '2th_deg'),
                    type_=pyFAI.units.RADIAL_UNITS)
                coords[radial_unit.name] = {
                    'attributes': {
                        'units': radial_unit.unit_symbol,
                        'long_name': radial_unit.label,
                    },
                    'data': self._placeholder_result.radial.tolist(),
                    'shape': self._placeholder_result.radial.shape,
                    'dtype': 'float32',
                }
            else:
                raise ValueError
        else:
            raise TypeError
        return coords

    def get_axes_indices(self, dataset_ndims):
        """Return the index of each coordinate orienting a single
        frame of results from this integration.
        """
        return {k: dataset_ndims + i
                for i, k in enumerate(self.result_coords.keys())}

    def zarr_tree(self, dataset_shape, dataset_chunks):
        """Return a dictionary representing a `zarr.group` that can be
        used to contain results from this integration.
        """
        # import json
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


class PyfaiIntegrationProcessorConfig(MyBaseModel):
    """Class defining components needed for performing one or more
    integrations on the same set of 2D input data with
    `saxswaxs.PyfaiIntegrationProcessor`.
    """
    azimuthal_integrators: conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)
    integrations: conlist(min_length=1, item_type=PyfaiIntegrationConfig)

    def zarr_tree(self, dataset_shape, dataset_chunks):
        """Return a dictionary representing a `zarr.group` that can be
        used to contain results from `saxswaxs.PyfaiIntegrationProcessor`.
        """
        ais = {ai.name: ai.ai for ai in self.azimuthal_integrators}
        for integration in self.integrations:
            integration.init_placeholder_results(ais)
        tree = {
            'root': {
                'attributes': {
                    'description': 'Container for processed SAXS/WAXS data'
                },
                'children': {
                    'entry': {
                        # NXentry
                        'attributes': {},
                        'children': {
                            'data': {
                                # NXdata
                                'attributes': {
                                    'axes': []
                                },
                                'children': {
                                    'spec_file': {
                                        'shape': dataset_shape,
                                        'dtype': 'b',
                                        'chunks': dataset_chunks,
                                    },
                                    'scan_number': {
                                        'shape': dataset_shape,
                                        'dtype': 'uint8',
                                        'chunks': dataset_chunks,
                                    },
                                    'scan_step_index': {
                                        'shape': dataset_shape,
                                        'dtype': 'uint64',
                                        'chunks': dataset_chunks,
                                    },
                                }
                            }
                        }
                    },
                    **{integration.name: integration.zarr_tree(
                        dataset_shape, dataset_chunks)
                       for integration in self.integrations},
                }
            }
        }
        return tree
