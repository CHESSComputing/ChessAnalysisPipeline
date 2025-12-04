"""Functions for converting between some commonly used data
formats."""


def convert_sparse_dense(data):
    """Converts between dense and sparse representations for NumPy
    arrays, xarray DataArrays, and xarray Datasets.

    - If input is a NumPy array, converts to a SciPy sparse CSR matrix.
    - If input is a SciPy sparse matrix, converts to a dense NumPy array.
    - If input is an xarray DataArray or Dataset containing sparse
      arrays, converts to dense.
    - If input is an xarray DataArray or Dataset containing dense
      arrays, converts to sparse.

    :param data: The imput data.
    :type data: Union[numpy.ndarray, scipy.sparse.spmatrix,
        xarray.DataArray, xarray.Dataset]
    :return: Converted object -- sparse if input is dense, dense if
        input is sparse.
    """
    import numpy as np
    import scipy.sparse as sp
    import xarray as xr

    if isinstance(data, np.ndarray):
        return sp.csr_matrix(data)  # Convert dense NumPy array to sparse

    if sp.issparse(data):
        return data.toarray()  # Convert sparse matrix to dense NumPy array

    if isinstance(data, xr.DataArray):
        # Convert DataArray values while preserving metadata
        if sp.issparse(data.data):
            return xr.DataArray(
                data.data.toarray(), dims=data.dims,
                coords=data.coords, attrs=data.attrs)
        return xr.DataArray(
            sp.csr_matrix(data.data), dims=data.dims,
            coords=data.coords, attrs=data.attrs)

    if isinstance(data, xr.Dataset):
        # Convert each variable in the Dataset
        def convert_var(var):
            if sp.issparse(var.data):
                return xr.DataArray(var.data.toarray(), dims=var.dims,
                                    coords=var.coords, attrs=var.attrs)
            return xr.DataArray(sp.csr_matrix(var.data), dims=var.dims,
                                coords=var.coords, attrs=var.attrs)

        return data.map(convert_var)

    raise TypeError(f'Unsupported data type: {type(data)}. '
                    'Input must be a NumPy array, SciPy sparse matrix, '
                    'xarray DataArray, or xarray Dataset.')


def convert_xarray_nexus(data):
    """Convert an `xarray.DataArray` or `xarray.Dataset` into an
    `nexusformat.nexus.NXdata` or vice versa.

    :param data: Input data.
    :type data: Union[xarray.DataArray, xarray.Dataset,
        nexusformat.nexus.NXdata]
    :return: Conveted data.
    :rtype:  Union[xarray.DataArray, xarray.Dataset,
        nexusformat.nexus.NXdata]
    """
    import xarray as xr
    from nexusformat.nexus import NXdata, NXfield

    if isinstance(data, xr.DataArray):
        return NXdata(
            value=data.values,
            name=data.name,
            attrs=data.attrs,
            axes=tuple(
                NXfield(
                    value=data[dim].values,
                    name=dim,
                    attrs=data[dim].attrs,
                )
                for dim in data.dims),
        )
    if isinstance(data, xr.Dataset):
        return NXdata(
            **{var:
               NXfield(
                   value=data[var].values,
                   name=var,
                   attrs=data[var].attrs,
               )
               for var in data.data_vars},
            name=data.name,
            attrs=data.attrs,
            axes=tuple(
                NXfield(
                    value=data[dim].values,
                    name=dim,
                    attrs=data[dim].attrs,
                )
                for dim in data.dims),
        )
    if isinstance(data, NXdata):
        nxaxes = data.nxaxes
        if nxaxes is None:
            if 'unstructured_axes' in data.attrs:
                nxaxes = data.unstructured_axes
        if isinstance(nxaxes, str):
            nxaxes = [nxaxes]
        return xr.Dataset(
            data_vars={
                name: xr.DataArray(
                    data=field.nxdata,
                    name=name,
                    attrs=field.attrs,
                )
                for name, field in data.items()
                if isinstance(field, NXfield)
                and name not in nxaxes
            },
            coords={
                axis: xr.DataArray(
                    data=data[axis].nxdata,
                    name=data[axis].nxname,
                    attrs=data[axis].attrs,
                )
                for axis in nxaxes
            },
            attrs=data.attrs,
        )
    raise TypeError(f'Unsupported data type: {type(data)}. '
                    'Must be xarray.DataArray, xarray.Dataset, or NXdata.')


def convert_structured_unstructured(data):
    from copy import deepcopy
    from nexusformat.nexus import NXdata, NXfield
    import numpy as np

    if isinstance(data, NXdata):
        if 'unstructured_axes' in data.attrs:
            # Convert unstructured to structured
            nxaxes = data.attrs['unstructured_axes']
            attrs = deepcopy(data.attrs)
            attrs['axes'] = attrs['unstructured_axes']
            attrs.pop('unstructured_axes')
            signals = [name for name, child in data.items()
                       if name not in nxaxes]
            structured_axes = {a: np.unique(data[a].nxdata) for a in nxaxes}
            dataset_shape = tuple(len(v) for a, v in structured_axes.items())
            structured_signals = {s: np.empty(
                (*dataset_shape, *data[s].shape[1:]),
                dtype=data[s].nxdata.dtype,
            ) for s in signals}
            npts = len(data[signals[0]].nxdata.tolist())
            print(f'converting {npts} data points')
            indices = {
                a: np.searchsorted(structured_axes[a], data[a].nxdata)
                for a in nxaxes
            }
            for s, value in structured_signals.items():
                value[tuple(indices[a] for a in nxaxes)] = data[s]
            structured_data = NXdata(
                **{s: NXfield(
                    value=structured_signals[s],
                    name=s,
                    attrs=data[s].attrs,
                ) for s in signals},
                name=data.nxname,
                attrs=attrs,
                axes=tuple(
                    NXfield(
                        value=value,
                        name=a,
                        attrs={k: v for k, v in data[a].attrs.items()
                               if not k == 'target'},
                    )
                    for a, value in structured_axes.items()
                )
            )
            return structured_data

        if 'axes' in data.attrs:
            # Convert structued to unstructured
            raise NotImplementedError(
                'Conversion from structured to unstructured not implemented.')
    raise TypeError(f'Unsupported data type: {type(data)}')
