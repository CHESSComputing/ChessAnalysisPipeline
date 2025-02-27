'''Definition and utilities for integration tools'''

from pydantic import (
    ConfigDict,
    field_validator, Field, StringConstraints, BaseModel,
    FilePath,
    model_validator,
    validator,
    conlist,
    PrivateAttr,
)
from typing import Literal, Optional
from typing_extensions import Annotated


from pyFAI.integrator.azimuthal import AzimuthalIntegrator

class MyBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class AzimuthalIntegratorConfig(MyBaseModel):
    name: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1)]
    poni_file: Optional[FilePath] = None
    params: Optional[dict] = None
    ai: AzimuthalIntegrator

    @model_validator(mode='before')
    @classmethod
    def validate_root(cls, data):
        if isinstance(data, dict):
            params = data.get('params')
            poni_file = data.get('poni_file')
            if ((params is None and poni_file is not None) or
                (params is not None and poni_file is None)):
                if poni_file is not None:
                    from pyFAI import load
                    data['ai'] = load(poni_file)
                else:
                    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
                    data['ai'] = AzimuthalIntegrator(**params)
            else:
                raise ValueError(
                    'Must specify at exactly one of: poni_file, params')
        return data

    
class PyfaiIntegrationConfig(MyBaseModel):
    name: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1)]
    multi_geometry: dict
    integration_method: Literal[
        'integrate1d', 'integrate2d', 'integrate_radial']
    integration_params: dict
    _placeholder_result: PrivateAttr = None

    def integrate(self, ais, input_data):
        from copy import deepcopy
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
        # Use adjusted azimuthal integration range
        self.multi_geometry['azimuth_range'] = (chi_min, chi_max)

        if self.integration_method == 'integrate_radial':
            raise NotImplementedError
        else:
            mg = self.get_multi_geometry(ais)
            integration_method = getattr(mg, self.integration_method)
            integration_params = {k: v
                                  for k, v in self.integration_params.items()
                                  if k != 'lst_data'}
            npts = [len(input_data[name])
                        for name in self.integration_params['lst_data']]
            if not all([_npts == npts[0] for _npts in npts]):
                raise RuntimeError(
                    'Different number of frames of detector data provided')
            npts = npts[0]
            results = [
                integration_method(
                    lst_data=[input_data[name][i]
                              for name in self.integration_params['lst_data']],
                    **integration_params)
                for i in range(npts)
            ]
            return results
            # Integrate first frame, extract objects to perform
            # integrations quicker from the result.
            results = [None] * npts
            results[0] = integration_method(
                lst_data=[input_data[name][0]
                          for name in self.integration_params['lst_data']],
                **integration_params)
            engine = mg.engines[results[0].method].engine
            omega = mg.solidAngleArray()
            for i in range(1, npts):
                results[i] = engine.integrate_ng(
                    lst_data=[input_data[name][i]
                              for name in self.integration_params['lst_data']],
                    **integration+params, solidangle=omega
                )
            return results

    def init_placeholder_results(self, azimuthal_integrators):
        placeholder_result = None
        placeholder_data = self.get_placeholder_data(azimuthal_integrators)
        if self.integration_method == 'integrate_radial':
            raise NotImplementedError
        else:
            mg = self.get_multi_geometry(azimuthal_integrators)
            integration_method = getattr(mg, self.integration_method)
            integration_params = {k: v
                                  for k, v in self.integration_params.items()
                                  if k != 'lst_data'}
            placeholder_result = integration_method(
                lst_data=placeholder_data, **integration_params)
        self._placeholder_result = placeholder_result

    def get_multi_geometry(self, azimuthal_integrators):
        from copy import deepcopy
        from pyFAI.multi_geometry import MultiGeometry

        ais = [azimuthal_integrators[name]
               for name in self.multi_geometry['ais']]
        kwargs = deepcopy(self.multi_geometry)
        del kwargs['ais']
        return MultiGeometry(ais, **kwargs)

    def get_placeholder_data(self, azimuthal_integrators):
        import numpy as np

        data = []
        for ai in azimuthal_integrators.values():
            data.append(np.full(ai.detector.shape, 0))
        return data

    def result_axes(self):
        return []

    @property
    def result_shape(self):
        if self.integration_method == 'integrate_radial':
            return (self.integration_params['npt'], )
        elif self.integration_method == 'integrate1d':
            return (self.integration_params['npt'], )
        elif self.integration_method == 'integrate2d':
            return (self.integration_params['npt_azim'],
                    self.integration_params['npt_rad'])
        else:
            raise NotImplementedError(
                f'Unimplemented integration_method: {self.integration_method}')

    @property
    def result_coords(self):
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
            if self.integration_method == 'integrate_radial':
                raise NotImplementedError
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
        return {k: dataset_ndims + i
                for i, k in enumerate(self.result_coords.keys())}

    def zarr_tree(self, dataset_shape, dataset_chunks):
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
                            'dtype': 'int64',
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
    azimuthal_integrators: conlist(
        min_length=1, item_type=AzimuthalIntegratorConfig)
    integrations: conlist(min_length=1, item_type=PyfaiIntegrationConfig)

    def zarr_tree(self, dataset_shape, dataset_chunks):
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
                        dataset_shape, dataset_chunks)},
                }
            }
        }
        return tree
