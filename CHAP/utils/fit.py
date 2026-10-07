#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Generic curve fitting module."""

# System modules
from collections import Counter
from copy import deepcopy
from functools import cached_property
from os import (
    cpu_count,
    mkdir,
    path,
)
from shutil import rmtree
from sys import float_info
#from time import time
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party modules
try:
    from joblib import (
        Parallel,
        delayed,
    )
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False
import numpy as np
from pydantic import (
    conint,
    conlist,
    Field,
)

# Local modules
from CHAP.processor import Processor
from CHAP.utils.general import quick_plot
from CHAP.utils.models import FitConfig

FLOAT_MIN = float_info.min
FLOAT_MAX = float_info.max
FLOAT_EPS = float_info.epsilon

# sigma = fwhm_factor*fwhm
fwhm_factor = {
    'gaussian': 'fwhm/(2*sqrt(2*log(2)))',
    'lorentzian': '0.5*fwhm',
    'splitlorentzian': '0.5*fwhm',  # sigma = sigma_r
    'voigt': '0.2777*fwhm',         # sigma = gamma
    'pvoigt': '0.5*fwhm',           # fraction = 0.5
}
# fwhm = sigma_factor*sigma
sigma_factor = {
    'gaussian': '2*sigma*sqrt(2*log(2))',
    'lorentzian': '2*sigma',
    'splitlorentzian': '2*sigma',   # sigma = sigma_r
    'voigt': '3.6009*sigma',        # sigma = gamma
    'pvoigt': '2*sigma',            # fraction = 0.5
}

# amplitude = height_factor*height*fwhm
height_factor = {
    'gaussian': 'height*fwhm*0.5*sqrt(pi/log(2))',
    'lorentzian': 'height*fwhm*0.5*pi',
    'splitlorentzian': 'height*fwhm*0.5*pi',   # sigma = sigma_r
    'voigt': '1.3306*height*fwhm',             # sigma = gamma
    'pvoigt': '1.2690*height*fwhm',            # fraction = 0.5
}
# height = amplitude_factor*amplitude/sigma
amplitude_factor = {
    'gaussian': 'amplitude/(sigma*sqrt(2*pi))',
    'lorentzian': 'amplitude/(sigma*pi)',
    'splitlorentzian': 'amplitude/(sigma*pi)', # sigma = sigma_r
    'voigt': '0.2087*amplitude/sigma',         # sigma = gamma
    'pvoigt': '0.3940*amplitude/sigma',        # fraction = 0.5
}
# height = amplitude_factor*amplitude/sigma
amplitude_factor = {
    'gaussian': 'amplitude/(sigma*sqrt(2*pi))',
}


class FitProcessor(Processor):
    """A processor to perform a fit on a data set or data map.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.utils.models.FitConfig`.
    :vartype config: dict, optional
    """

    pipeline_fields: dict = Field(
        default = {
            'config': 'CHAP.utils.models.FitConfig'}, init_var=True)
    config: Optional[FitConfig] = None

    def _get_pipelinedata_item(self, data, remove=True):
        """Retrieve the input data to :meth:`process` from the list of
        PipelineData items.

        :param data: Input data.
        :type data: list[PipelineData]
        :param remove: If there is a matching entry in `data`, remove
            it from the list, defaults to `True`.
        :type remove: bool, optional
        :return: Matching data item(s).
        :rtype: Any
        """
        # Retrieve the data to (re)fit from the pipeline
        for i, d in reversed(list(enumerate(data))):
            ddata = d.get('data')
            if isinstance(ddata, Fit):
                if remove:
                    data.pop(i)
                return ddata
            if d.get('name') == 'signal':
                if remove:
                    data.pop(i)
                break
        else:
            raise ValueError(
                f'Unable to extract suitable fit input data from {data}')

        # Retrieve the optional coordinates and mask from the pipeline
        x = None
        mask = None
        try:
            y = np.asarray(ddata)
            for i, d in reversed(list(enumerate(data))):
                name = d.get('name')
                if name == 'coordinates':
                    x = np.asarray(d.get('data'))
                    if remove:
                        data.pop(i)
                    assert x.size == y.shape[-1]
                elif name == 'mask':
                    mask = np.asarray(d.get('data'))
                    if remove:
                        data.pop(i)
                    assert mask.size == y.shape[-1]
                if x is not None and mask is not None:
                    break
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f'Unable to extract suitable fit input data from {data}')
        return x, y, mask

    def process(self, data):
        """Fit the data and return a :class:`~CHAP.utils.fit.Fitmap`
        The input data should be a list of PipelineData items
        containing either one with a `data` field of type
        :class:`~CHAP.utils.fit.Fit` (to refit or continue a
        previous fit), or one with the `name` of `signal` and an
        array-like `data` field. In the latter case, optional
        x-coordinates or a mask can be supplied by additional
        PipelineData item with the `name` of `coordinates` or `mask`
        and again an array-like `data` field.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: The fitted data object.
        :rtype: Fit
        """
        # Local modules
        from CHAP.utils.models import MultipeakModel

        # Unwrap the PipelineData
        data = self._get_pipelinedata_item(data)

        if isinstance(data, Fit):
            # Refit/continue the fit with possibly updated parameters
            data.fit(config=self.config, max_nfev=self.config.max_nfev)
            return data

        # Expand multipeak model if present
        found_multipeak = False
        multipeak_info = None
        for i, model in enumerate(deepcopy(self.config.models)):
            if isinstance(model, MultipeakModel):
                if found_multipeak:
                    raise ValueError(
                        f'Invalid parameter models ({self.config.models}) '
                        '(multiple instances of multipeak not allowed)')
                parameters, models = self.create_multipeak_model(model)
                if parameters:
                    self.config.parameters += parameters
                self.config.models += models
                self.config.models.pop(i)
                found_multipeak = True
                multipeak_info = model.model_dump()

        # Instantiate the Fit object and fit the data
        y = np.squeeze(data[1])[None,:] \
            if np.squeeze(data[1]).ndim == 1 else data[1]
        fit = Fit(y, self.config, self.logger, x=data[0], mask=data[2])
        fit.fit(
            abs_height_cutoff=self.config.abs_height_cutoff,
            max_nfev=self.config.max_nfev,
            multipeak_info=multipeak_info,
            num_proc=self.config.num_proc,
            plot=self.config.plot,
            print_report=self.config.print_report,
            rel_height_cutoff=self.config.rel_height_cutoff)

        return fit

    @staticmethod
    def create_multipeak_model(model):
        """Create a multipeak model.

        :param model: A Multipeak fit model class.
        :type model: :class:`~CHAP.utils.models.MultipeakModel`
        :return: The fit parameters and fit model classes.
        :rtype: list[:attr:`~CHAP.utils.models.FitParameter`],
            list[:attr:`~CHAP.utils.models.FitConfig.models`]
        """
        # Local modules
        from CHAP.utils.models import PEAK_LIKE_MODELS

        def _uniform_model(
                model, num_peak, peak_model_class, sig_min, sig_max):
            """Create an uniform multipeak model."""
            # Local modules
            from CHAP.utils.models import FitParameter

            if not model.centers_range_fraction:
                parameters = [FitParameter(
                    name='scale_factor', value=1.0, min=FLOAT_MIN)]
            else:
                parameters = [FitParameter(
                    name='scale_factor', value=1.0,
                    min=1.0-model.centers_range_fraction,
                    max=1.0+model.centers_range_fraction)]
            peak_models = []
            for i, cen in enumerate(model.centers):
                peak_models.append(peak_model_class(
                    model_type=model.peak_models,
                    prefix=f'peak{i+1}_',
                    parameters=[
                         {'name': 'amplitude', 'min': FLOAT_MIN},
                         {'name': 'center', 'expr': f'scale_factor*{cen}'},
                         {'name': 'sigma', 'min': sig_min, 'max': sig_max}]))
            return parameters, peak_models

        def _unconstrained_model(
                model, num_peak, peak_model_class, sig_min, sig_max):
            """Create an unconstrained multipeak model."""
            peak_models = []
            for i, cen in enumerate(model.centers):
                if not (model.centers_range and model.centers_range_fraction):
                    peak_models.append(peak_model_class(
                        model_type=model.peak_models,
                        prefix=f'peak{i+1}_',
                        parameters=[
                             {'name': 'amplitude', 'min': FLOAT_MIN},
                             {'name': 'center', 'value': cen},
                             {'name': 'sigma', 'min': sig_min, 'max': sig_max}
                        ]))
                else:
                    delta = max(
                        model.centers_range, cen*model.centers_range_fraction)
                    peak_models.append(peak_model_class(
                        model_type=model.peak_models,
                        prefix=f'peak{i+1}_',
                        parameters=[
                             {'name': 'amplitude', 'min': FLOAT_MIN},
                             {'name': 'center', 'value': cen,
                              'min': max(0.0, cen-delta), 'max': cen+delta},
                             {'name': 'sigma', 'min': sig_min, 'max': sig_max}
                        ]))
            return [], peak_models

        peak_model_class = PEAK_LIKE_MODELS[model.peak_models]
        num_peak = len(model.centers)
        if num_peak == 1 and model.fit_type == 'uniform':
            model.fit_type = 'unconstrained'

        sig_min = FLOAT_MIN
        sig_max = np.inf
        if (model.fwhm_min is not None
                or model.fwhm_max is not None):
            # Third party modules
            from asteval import Interpreter

            ast = Interpreter()
            if model.fwhm_min is not None:
                ast(f'fwhm = {model.fwhm_min}')
                sig_min = ast(fwhm_factor[model.peak_models])
            if model.fwhm_max is not None:
                ast(f'fwhm = {model.fwhm_max}')
                sig_max = ast(fwhm_factor[model.peak_models])

        if model.fit_type == 'uniform':
            return _uniform_model(
                model, num_peak, peak_model_class, sig_min, sig_max)
        return _unconstrained_model(
            model, num_peak, peak_model_class, sig_min, sig_max)

class SetupProcessor(Processor):
    """Processor to set up an empty results container for
    :class:`~CHAP.utils.fit.FitProcessor`.

    Creates a `Zarr <https://zarr.readthedocs.io/en/stable/>`__ store
    whose structure is determined by the fit configuration and the
    scan / signal dimensions.

    :ivar config: Initialization parameters for an instance of
        :class:`~CHAP.utils.models.FitConfig`.
    :vartype config: dict, optional
    :ivar dataset_shape: Shape of the map (scan) dimensions of the
        output datasets, defaults to `[0]`.
    :vartype dataset_shape: list[int], optional
    :ivar dataset_chunks: Chunk shape along the scan dimensions, or
        ``'auto'`` to let Zarr choose, defaults to ``'auto'``.
    :vartype dataset_chunks: list[int] or 'auto', optional
    :ivar signal_shape: Shape of one frame of the 1-D signal being
        fit (the signal dimension).
    :vartype signal_shape: list[int]
    """

    pipeline_fields: dict = Field(
        default = {
            'config': 'CHAP.utils.models.FitConfig'
        },
        init_var=True
    )
    config: Optional[FitConfig] = None
    dataset_shape: Optional[
        conlist(item_type=conint(ge=0), min_length=1)] = [0]
    dataset_chunks: Optional[
        Union[
            Literal['auto'],
            conlist(item_type=conint(gt=0), min_length=1)
        ]] = 'auto'
    signal_shape: conlist(item_type=conint(ge=0), min_length=1)

    def process(self, data):
        """Create and return an empty
        `Zarr <https://zarr.readthedocs.io/en/stable/>`__ store whose
        group hierarchy mirrors the output of
        :class:`~CHAP.utils.fit.FitProcessor` for the configured fit
        model.

        The store structure is produced by
        :meth:`~CHAP.utils.models.FitConfig.zarr_tree` and populated
        into an in-memory ``zarr.Group`` so that a downstream writer
        can write fit results into it incrementally.

        :param data: Input pipeline data; may supply a
            :class:`~CHAP.utils.models.FitConfig` to populate
            :attr:`config` when one is not already set.
        :type data: list[PipelineData]
        :return: In-memory Zarr root group pre-structured for fit
            results.
        :rtype: zarr.Group
        """
        # System modules
        import asyncio
        import zarr

        # Third party modules
        from zarr.core.buffer import default_buffer_prototype
        from zarr.storage import MemoryStore

        # Local modules
        from CHAP.saxswaxs.utils import dict_to_zarr

        zarr_fit = dict_to_zarr(
            self.config.zarr_tree(
                self.dataset_shape, self.dataset_chunks,
                self.signal_shape,
            ),
            logger=self.logger,
        )
        zarr_root = zarr.create_group(store=MemoryStore({}))
        async def copy_zarr_store(source_store, dest_store):
            """Copy a Zarr <https://zarr.readthedocs.io/en/stable/>`__
            store.

            :param source_store: Source Zarr group.
            :type: zarr.Group
            :param dest_store: Destination Zarr group.
            :type: zarr.Group
            """
            async for k in source_store.list():
                self.logger.info(f'Copying {k}')
                buf = await source_store.get(
                    k, prototype=default_buffer_prototype())
                await dest_store.set(k, buf)
        asyncio.run(copy_zarr_store(zarr_fit.store, zarr_root.store))
        return zarr_root


class Component():
    """A model fit component."""

    def __init__(self, model):
        """Initialize Component.

        :param model: A fit model class (make sure its prefix is
            specified in `model.prefix` for duplicative model names).
        :type model: :attr:`~CHAP.utils.models.FitConfig.models`
        """
        names = [f'{par.name}' for par in model.parameters]
        self.func = model.func
        self.func_args = model.func_args
        self.func_args_indices = [names.index(arg) for arg in self.func_args]
        self.model_identifiers = {
            k:getattr(model, k) for k in model.MODEL_IDENTIFIERS}
        self.param_names = [model.prefix + name for name in names]
        self.prefix = model.prefix
        self._name = model.model_type

    def eval(self, params=None, x=None):
        if x is None:
            return None
        if params is None:
            return self.func(x, **self.model_identifiers)
        par_values = tuple(params[par].value for par in self.param_names)
        ppar_values = tuple(par_values[i] for i in self.func_args_indices)
        return self.func(x, *ppar_values, **self.model_identifiers)


class Components(dict):
    """The dictionary of model fit components."""

    def __init__(self):
        """Initialize Components."""
        super().__init__(self)

    def __setitem__(self, key, value):
        if key not in self and not isinstance(key, str):
            raise KeyError(f'Invalid component name ({key})')
        if not isinstance(value, Component):
            raise ValueError(f'Invalid component ({value})')
        dict.__setitem__(self, key, value)
        value.name = key

    @property
    def components(self):
        """Return the model fit components.

        :type: list[:attr:`~CHAP.utils.models.FitConfig.models`]
        """
        return self.values()


class Parameters(dict):
    """A dictionary of FitParameter objects, mimicking the
    functionality of a similarly named
    `class in the lmfit library <https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameters>`__.
    """

    def __init__(self):
        """Initialize Parameters."""
        super().__init__(self)

    def __setitem__(self, key, value):
        # Local modules
        from CHAP.utils.models import FitParameter

        if key in self:
            raise KeyError(f'Duplicate name for FitParameter ({key})')
        if key not in self and not isinstance(key, str):
            raise KeyError(f'Invalid FitParameter name ({key})')
        if value is not None and not isinstance(value, FitParameter):
            raise ValueError(f'Invalid FitParameter ({value})')
        dict.__setitem__(self, key, value)
        value.name = key

    def add(self, parameter, prefix=''):
        """Add a fit parameter.

        :param parameter: The fit parameter to add to the dictionary.
        :type parameter: FitParameter
        :param prefix: Prefix of the component to which this parameter
            belongs, defaults to `''`.
        :type prefix: str, optional
        """
        # Local modules
        from CHAP.utils.models import FitParameter

        if isinstance(parameter, FitParameter):
            name = prefix + parameter.name
            self.__setitem__(name, parameter)
        else:
            raise RuntimeError('Must test')
            parameter = prefix + parameter
            self.__setitem__(parameter, FitParameter(name=parameter))
#        setattr(self[parameter.name], '_prefix', prefix)


class ModelResult():
    """The result of a model fit, mimicking the functionality of a
    similarly named
    `class in the lmfit library <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult>`__.
    """

    def __init__(
            self, model, y, parameters, *, x=None, method=None, ast=None,
            res_par_exprs=None, res_par_indices=None, res_par_names=None,
            result=None, best_pars=None):
        """Initialize ModelResult.

        :param model: Fit model.
        :type model: Components or lmfit.model.Model
        :param parameters: Fit parameters.
        :type parameters: Parameters or lmfit.parameter.Parameter
        :param x: x-coordinates.
        :type x: array-like, optional
        :param y: y-coordinates.
        :type y: array-like, optional
        :param method: Minimization method name.
        :type method: str, optional
        :param ast:
            `Asteval <https://lmfit.github.io/asteval/>`
            `Interpreter <https://lmfit.github.io/asteval/api.html#the-interpreter-class>`__.
        :type ast: asteval.Interpreter, optional
        :param res_par_exprs: The expression parameter expressions.
        :type res_par_exprs: list[dict], optional
        :param res_par_indices: The parameter indices of all free fit
            parameters in the list of fit parameters.
        :type res_par_indices: list[int], optional
        :param res_par_names: The parameter names of all free fit
            parameters in the list of fit parameters.
        :type res_par_names: list[str], optional
        """
        self.components = model.components
        self.init_params = deepcopy(parameters)
        self.method = method
        self.params = deepcopy(parameters)
        self.x = x
        if result is None:
            self.ier = -1
            self.nfev = 0
            self.residual = np.zeros(self.x.shape)
            self.success = False
        elif method == 'leastsq':
            best_pars = result[0]
            self.ier = result[4]
            self.message = result[3]
            self.nfev = result[2]['nfev']
            self.residual = result[2]['fvec']
            self.success = 1 <= result[4] <= 4
        else:
            best_pars = result.x
            self.ier = result.status
            self.message = result.message
            self.nfev = result.nfev
            self.residual = result.fun
            self.success = result.success

        # Get the covarience matrix
        self.ndata = len(self.residual)
        self.nvarys = len(res_par_indices)
        self.chisqr = (self.residual**2).sum()
        self.redchi = self.chisqr / (self.ndata-self.nvarys)
        self.covar = None
        if result is not None:
            if method == 'leastsq':
                if result[1] is not None:
                    self.covar = result[1]*self.redchi
            else:
                try:
                    self.covar = self.redchi * np.linalg.inv(
                        np.dot(result.jac.T, result.jac))
                except Exception:
                    self.covar = None
        # Update the fit parameters with the fit result
        self._ast = ast
        self._expr_pars = {}
        par_names = list(self.params.keys())
        self.var_names = []
        if res_par_indices:
            def _get_stderr(i):
                if self.covar is not None:
                    stderr = self.covar[i,i]
                    if stderr is not None:
                        stderr = None if stderr < 0.0 else np.sqrt(stderr)
                    return stderr
                return None

            assert len(best_pars) == len(res_par_indices)
            for i, (value, index) in enumerate(zip(best_pars, res_par_indices)):
                par = self.params[par_names[index]]
                par.set(value=value)
                setattr(par, '_stderr', _get_stderr(i))
                self.var_names.append(par.name)
        init_params = deepcopy(self.init_params)
        if res_par_exprs:
            # Third party modules
            from sympy import diff

            def _get_stderr_expr(expr):
                stderr = 0
                for i, name in enumerate(self.var_names):
                    d = diff(expr, name)
                    if not d:
                        continue
                    for ii, nname in enumerate(self.var_names):
                        dd = diff(expr, nname)
                        if not dd:
                            continue
                        stderr += (self._ast.eval(str(d))
                                   * self._ast.eval(str(dd))
                                   * self.covar[i,ii])
                return np.sqrt(stderr)

            for value, name in zip(best_pars, res_par_names):
                self._ast.symtable[name] = value
            for par_expr in res_par_exprs:
                name = par_names[par_expr['index']]
                expr = par_expr['expr']
                value = self._ast.eval(expr)
                par = self.params[name]
                par.set(value=value)
                par_init = init_params[name]
                par_init.set(value=value)
                self._expr_pars[name] = expr
                setattr(
                    par, '_stderr',
                    None if self.covar is None else _get_stderr_expr(expr))

        # Evaluate the initial fit and the residual if needed
        if result is None:
            self.residual = self.eval(self.params) - y
        self.best_fit = y + self.residual
        self.init_fit = self.eval(init_params)


    def eval(self, params=None, x=None):
        """Evaluate the model function.

        :param params: Model parameters, defaults to
            `None`, in which case the class variable params is used.
        :type params: Parameters, optional
        :param x: Independent variable, defaults to `None`, in which
            case the class variable x is used.
        :type x: array-like, optional
        :return: Evaluated model function values.
        :rtype: numpy.ndarray
        """
        component_results = self.eval_components(params, x)
        result = None
        for component_result in component_results.values():
            if result is None:
                result = component_result
            else:
                result += component_result
        return result

    def eval_components(self, params=None, x=None):
        """Evaluate each component of a composite model function.

        :param params: Composite model parameters, defaults to
            `None`, in which case the class variable params is used.
        :type params: Parameters, optional
        :param x: Independent variable, defaults to `None`, in which 
            case the class variable x is used.
        :type x: array-like, optional
        :return: Component names and evaluated function values.
        :rtype: dict[str, numpy.ndarray]
        """
        if x is None:
            x = self.x
        if params is None:
            params = self.params
        result = {}
        for component in self.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            name = component.prefix if component.prefix else component._name
            result[name] = component.eval(params=params, x=x)
        return result

    def fit_report(self, show_correl=False):
        """Generates a report of the fitting results with their best
        parameter values and uncertainties.

        :param show_correl: Whether to show list of correlations,
            defaults to `False`.
        :type show_correl: bool, optional
        """
        # FIX add show_correl option
        # Local modules
        from CHAP.utils.general import (
            getfloat_attr,
            gformat,
        )

        buff = []
        add = buff.append
        parnames = list(self.params.keys())
        namelen = max(len(n) for n in parnames)

        add("[[Fit Statistics]]")
        add(f"    # fitting method   = {self.method}")
        add(f"    # function evals   = {getfloat_attr(self, 'nfev')}")
        add(f"    # data points      = {getfloat_attr(self, 'ndata')}")
        add(f"    # variables        = {getfloat_attr(self, 'nvarys')}")
        add(f"    chi-square         = {getfloat_attr(self, 'chisqr')}")
        add(f"    reduced chi-square = {getfloat_attr(self, 'redchi')}")
#        add(f"    Akaike info crit   = {getfloat_attr(self, 'aic')}")
#        add(f"    Bayesian info crit = {getfloat_attr(self, 'bic')}")
#        if hasattr(self, 'rsquared'):
#            add(f"    R-squared          = {getfloat_attr(self, 'rsquared')}")

        add("[[Variables]]")
        for name in parnames:
            par = self.params[name]
            space = ' '*(namelen-len(name))
            nout = f'{name}:{space}'
            inval = '(init = ?)'
            if par.init_value is not None:
                inval = f'(init = {par.init_value:.7g})'
            if hasattr(self, '_expr_pars'):
                expr = self._expr_pars.get(name, par.expr)
            else:
                expr = None
            val = par.value if expr is None else self._ast.eval(expr)
            try:
                val = gformat(par.value)
            except (TypeError, ValueError):
                val = ' Non Numeric Value?'
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = f'({abs(par.stderr/par.value):.2%})'
                except ZeroDivisionError:
                    spercent = ''
                val = f'{val} +/-{serr} {spercent}'
            if par.vary:
                add(f'    {nout} {val} {inval}')
            elif expr is not None:
                add(f"    {nout} {val} == '{expr}'")
            else:
                add(f'    {nout} {par.value:.7g} (fixed)')

        return '\n'.join(buff)


class UpdateValuesProcessor(Processor):
    """Processor to extract fit results from a
    :class:`~CHAP.utils.fit.Fit` (or :class:`~CHAP.utils.fit.Fit`)
    object and format them as a list of path-keyed value dicts
    suitable for a downstream
    :class:`~CHAP.common.processor.NexusValuesWriter`.

    This processor is the write-side complement of
    :class:`~CHAP.utils.fit.SetupProcessor`: the paths it emits
    correspond to the Zarr dataset layout defined by
    :meth:`~CHAP.utils.models.FitConfig.zarr_tree`.
    """

    def process(self, data):
        """Extract fit results from a :class:`~CHAP.utils.fit.Fit`
        object and return them as a flat list of path-keyed value dicts.

        The paths in the returned dicts match the dataset layout
        defined by :meth:`~CHAP.utils.models.FitConfig.zarr_tree` and
        the container created by
        :class:`~CHAP.utils.fit.SetupProcessor`.

        :param data: Input pipeline data; must include a
            ``'FitProcessor'``-tagged item whose ``data`` field is a
            :class:`~CHAP.utils.fit.Fit` instance.
        :type data: list[PipelineData]
        :return: List of dicts, each with keys ``'path'`` (str) and
            ``'data'`` (scalar or array), for use with a downstream
            :class:`~CHAP.common.processor.NexusValuesWriter`.
        :rtype: list[dict]
        """
        fit = self.get_data(data, name='FitProcessor')
        values = [
            {'path': 'data/best_fit', 'data': fit.best_fit},
            {'path': 'data/num_func_eval', 'data': fit.num_func_eval},
            {'path': 'data/redchi', 'data': fit.redchi},
            {'path': 'data/residual', 'data': fit.residual},
            {'path': 'data/success', 'data': fit.success},
        ]
        for (comp_name, comp_info), (_, comp_evals) in zip(
                fit.components_info.items(), fit.components_evals.items()):
            comp_path = f'components/{comp_name}'
            values.append({
                'path': f'{comp_path}/data/best_fit',
                'data': comp_evals,
            })
            for name in comp_info['parameters']:
                param_path = f'{comp_path}/parameters/{name}'
                # FIXME param_value for any parameters that are part
                # of a component that was excluded from the fit?
                init_params = fit.init_parameters[name]
                if name in fit.best_parameters:
                    param_values = fit.best_values[name]
                    param_errors = fit.best_errors[name]
                else:
                    param_values = init_params['values']
                    param_errors = None
                values.extend([
                    {'path': f'{param_path}/value',
                     'data': param_values},
                    {'path': f'{param_path}/error',
                     'data': param_errors},
                    {'path': f'{param_path}/initial',
                     'data': init_params['values']},
                    {'path': f'{param_path}/min',
                     'data': init_params['min'] * np.ones(param_values.shape)},
                    {'path': f'{param_path}/max',
                     'data': init_params['max'] * np.ones(param_values.shape)},
                    {'path': f'{param_path}/vary',
                     'data': init_params['vary']},
                    {'path': f'{param_path}/expression',
                     'data': init_params['expr']},
                ])
        return values


class Fit:
    """Wrapper class for scipy/lmfit to fit data on a N-dimensional
    map (can be a single data point)."""

    def __init__(self, y, config, logger, x=None, mask=None):
        """Initialize Fit.

        :param y: Input signal data.
        :type y: array-like
        :param config: Fit configuration.
        :type config: CHAP.utils.models.FitConfig
        :param logger: A python Logger object.
        :type logger: logging.Logger
        :param x: Input coordinate data.
        :type x: array-like, optional
        :param mask: Input mask data.
        :type mask: array-like, optional
        """
        self._code = config.code
        for model in config.models:
            if model.model_type == 'expression' and self._code != 'lmfit':
                self._code = 'lmfit'
                logger.warning('Using lmfit instead of scipy with an '
                               'expression model')
        if self._code == 'scipy':
            # Local modules
            from CHAP.utils.fit import Parameters
        else:
            # Third party modules
            from lmfit import Parameters

        self._abs_height_cutoff = None
        self._best_errors = None
        self._best_fit = None
        self._best_parameters = None
        self._best_values = None
        self._best_vary = None
        self._init_values = None
        self._inv_transpose = None
        self._logger = logger
        self._mask = mask
        self._max_nfev = None
        self._memfolder = config.memfolder
        self._method = config.method
        self._model = None
        self._multipeak_info = None
        self._new_parameters = None
        self._num_func_eval = None
        self._out_of_bounds = None
        self._plot = False
        self._plot_init = False
        self._print_report = False
        self._redchi = None
        self._redchi_cutoff = 0.1
        self._rel_height_cutoff = None
        self._success = None
        self._try_no_bounds = True

        self._free_parameters = []
        self._parameters = Parameters()
        if self._code == 'scipy':
            self._ast = None
            self._res_num_pars = []
            self._res_par_exprs = []
            self._res_par_indices = []
            self._res_par_names = []
            self._res_par_values = []
        self._parameter_bounds = None
        self._linear_parameters = []
        self._nonlinear_parameters = []
        self._model_parameters = []
        self._result = None
#        self._try_linear_fit = True
#        self._fwhm_min = None
#        self._fwhm_max = None
#        self._sigma_min = None
#        self._sigma_max = None
#        if 'try_linear_fit' in kwargs:
#            self._try_linear_fit = kwargs.pop('try_linear_fit')
#            if not isinstance(self._try_linear_fit, bool):
#                raise ValueError(
#                    'Invalid value of keyword argument try_linear_fit '
#                    f'({self._try_linear_fit})')

        self._x = np.arange(self._y.shape[-1]) if x is None else x
        self._y = y

        # Flatten and normalize the map and store in self._y_norm
        # At this point the fastest index should always be the signal
        # dimension so that the slowest ndim-1 dimensions are the
        # map dimensions
        self._map_dim = int(self._y.size / self._x.size)
        self._map_shape = self._y.shape[:-1]
        y_min = self._y.min()
        y_range = self._y.max() - y_min
        if y_range <= FLOAT_MIN:
            y_range = 0.
        self._norm = (y_min, y_range)
        self._y_norm = np.reshape(
            self._y, (self._map_dim, self._x.size)).copy()
        if self._norm[0]:
            self._y_norm -= self._norm[0]
        if self._norm[1]:
            self._y_norm /= self._norm[1]

        # Setup fit model
        self._setup_fit_model(config.models, config.parameters)

    @cached_property
    def best_errors(self):
        """Return errors in the best fit parameters.

        :type: dict[str, numpy.ndarray]
        """
        return {name:par['errors']
                for name, par in self.best_parameters.items()}

    @property
    def best_fit(self):
        """Return the best fits.

        :type: numpy.ndarray
        """
        return self._best_fit

    @cached_property
    def best_parameters(self):
        """Return the best fit parameters.

        :type: dict[str, dict]
        """
        return {
            name:{'errors': self._best_errors[i],
                'init_values': self._init_values[i],
                'values': self._best_values[i],
                'vary': self._best_vary[i],
            } for i, name in enumerate(self._best_parameters)
        }

    @cached_property
    def best_values(self):
        """Return values for the best fit parameters.

        :type: dict[str, numpy.ndarray]
        """
        return {name:par['values']
                for name, par in self.best_parameters.items()}

    @cached_property
    def best_vary(self):
        """Return vary parameters for the best fit parameters.

        :type: dict[str, numpy.ndarray]
        """
        return {name:par['vary']
                for name, par in self.best_parameters.items()}

    @cached_property
    def components_evals(self):
        """Return the fit model components evaluations.

        :type: dict[str, numpy.ndarray]
        """
        # Third party modules
        from lmfit.models import ExpressionModel

        components_evals = {}
        for component, (comp_name, _) in zip(
                self._result.components, self.components_info.items()):
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            evals = np.zeros(self._best_fit.shape)
            for index in np.ndindex(self._map_shape):
                params = deepcopy(self._parameters)
                for name in component.param_names:
                    if name in self._best_parameters:
                        params[name].set(value=self.best_values[name][index])
                evals[index] = component.eval(params=params, x=self._x)
            components_evals[comp_name] = evals
        return components_evals

    @cached_property
    def components_info(self):
        """Return the fit model components info.

        :type: dict[str, dict]
        """
        # Third party modules
        from lmfit.models import ExpressionModel

        components_info = {}
        for component in self._result.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            parameters = {}
            for name in component.param_names:
                if self._parameters[name].vary:
                    parameters[name] = {'free': True}
                elif self._parameters[name].expr is not None:
                    parameters[name] = {
                        'free': False,
                        'expr': self._parameters[name].expr,
                    }
                else:
                    parameters[name] = {
                        'free': False,
                        'value': self.init_values[name],
                    }
            expr = None
            if isinstance(component, ExpressionModel):
                comp_name = component._name.rstrip('_')
                expr = component.expr
            else:
                prefix = component.prefix.rstrip('_')
                comp_name  = prefix + f' ({component._name})' if prefix \
                    else component._name
            if expr is None:
                components_info[comp_name] = {'parameters': parameters}
            else:
                components_info[comp_name] = {
                    'expr': expr, 'parameters': parameters}
        return components_info

    @property
    def coordinates(self):
        """Return the x-coordinates.

        :type: numpy.ndarray
        """
        return self._x

    @cached_property
    def init_parameters(self):
        """Return the initial parameters for the fit model.

        :type: dict[str, dict]
        """
        parameters = {}
        for name in sorted(self._result.init_params):
            if name != 'tmp_normalization_offset_c':
                par = self._result.init_params[name]
                if name in self._best_parameters:
                    init_values = self.best_parameters[name]['init_values']
                    vary = self.best_vary[name]
                else:
                    init_values = self._map_dim*[par.value]
                    vary = self._map_dim*[par.vary]
                parameters[name] = {
                    'expr': par.expr,
                    'min': par.min,
                    'max': par.max,
                    'values': init_values,
                    'vary': vary,
                }
        return parameters

    @cached_property
    def init_values(self):
        """Return initial values for the fit parameters.

        :type: dict[str, numpy.ndarray]
        """
        return {name:par['values']
                for name, par in self.init_parameters.items()}

    @property
    def mask(self):
        """Return the mask

        :type: numpy.ndarray
        """
        return self._mask

    @property
    def max_nfev(self):
        """Return if the maximum number of function evaluations is
        reached for each fit.

        :type: numpy.ndarray
        """
        return self._max_nfev

    @property
    def num_func_eval(self):
        """Return the number of function evaluations for each best fit.

        :type: numpy.ndarray
        """
        return self._num_func_eval

    @property
    def out_of_bounds(self):
        """Return the out_of_bounds flag values for each best fit.

        :type: numpy.ndarray
        """
        return self._out_of_bounds

    @property
    def parameters(self):
        """Return the fit parameters info.

        :type: dict
        """
        return {name:{'min': par.min, 'max': par.max, 'vary': par.vary,
                'expr': par.expr} for name, par in self._parameters.items()
                if name != 'tmp_normalization_offset_c'}

    @property
    def redchi(self):
        """Return the redchi value for each best fit.

        :type: numpy.ndarray
        """
        return self._redchi

    @cached_property
    def residual(self):
        """Return the residual in each best fit.

        :type: numpy.ndarray
        """
        if self.best_fit is None:
            return None
        if self._mask is None:
            residual = self._y - self.best_fit
        else:
            y_flat = np.reshape(self._y, (self._map_dim, self._x.size))
            y_flat_masked = y_flat[:,~self._mask]
            y_masked = np.reshape(
                y_flat_masked,
                list(self._map_shape) + [y_flat_masked.shape[-1]])
            residual = y_masked - self.best_fit
        return residual

    @property
    def signal(self):
        """Return the signal (or y-coordinates).

        :type: numpy.ndarray
        """
        return self._y

    @property
    def success(self):
        """Return the success value for each fit.

        :type: bool
        """
        return self._success

    @staticmethod
    def guess_init_peak(
            x, y, target_centers, centers_range, centers_range_fraction,
            min_height=None, min_width=None):
        """Return guesses for the initial height, center and fwhm for
        peak-like models.
        """
        # Third party modules
        from scipy.signal import find_peaks as find_peaks_scipy

        x = np.asarray(x)
        y = np.asarray(y)
        target_centers = np.asarray(target_centers)
        assert x.ndim == 1 and x.shape == y.shape
        assert target_centers.ndim == 1
        assert isinstance(centers_range, (int, float))
        assert isinstance(centers_range_fraction, (int, float))
        peaks = find_peaks_scipy(y, height=min_height, width=min_width)
        centers = [x[v] for v in peaks[0]]
        if 'peak_heights' in peaks[1]:
            heights = peaks[1]['peak_heights']
        else:
            heights = [y[v] for v in peaks[0]]
        widths = peaks[1]['widths']

        num_peak = target_centers.size
        use_peaks = num_peak*[False]
        peak_centers = num_peak*[None]
        peak_heights = num_peak*[None]
        peak_widths = num_peak*[None]
        delta_x = x[1] - x[0]
        for n, target_center in enumerate(target_centers):
            if centers:
                index = np.abs(centers - target_center).argmin()
                delta = max(centers_range, target_center*centers_range_fraction)
                if np.abs(target_center - centers[index]) < delta:
                    use_peaks[n] = True
                    peak_centers[n] = centers[index]
                    peak_heights[n] = heights[index]
                    peak_widths[n] = widths[index]*delta_x
        return use_peaks, peak_centers, peak_heights, peak_widths

    def add_model(self, model):
        """Add a model component to the fit model.

        :param model: A fit model class (make sure its prefix is
            specified in `model.prefix` for duplicative model names).
        :type model: :attr:`~CHAP.utils.models.FitConfig.models`
        """
        # Local modules
        from CHAP.utils.models import MODEL_CLASSES

        assert isinstance(model, tuple(MODEL_CLASSES))

        def _set_parameter_group(model, name, long_name):
            if name in model.LINEAR_PARAMETERS:
                self._linear_parameters.append(long_name)
            elif name in model.MODEL_PARAMETERS:
                self._model_parameters.append(long_name)
            elif name not in model.MODEL_IDENTIFIERS:
                self._nonlinear_parameters.append(long_name)

        def _set_parameter_info_scipy(model):
            new_parameters = []
            for par in deepcopy(model.parameters):
                name = par.name
                self._parameters.add(par, model.prefix)
                if self._parameters[par.name].expr is None:
                    self._parameters[par.name].set(value=par.default)
                new_parameters.append(par.name)
                _set_parameter_group(model, name, par.name)
            self._res_num_pars += [len(model.parameters)]
            return new_parameters

        def _set_parameter_info_lmfit(model):
            if model.model_type == 'expression':
                # Third party modules
                from sympy import diff

                newmodel = model.lmfit_model(
                    prefix=model.prefix, parameters=self._parameters)
                # Remove already existing names
                for name in newmodel.param_names.copy():
                    if name not in model.expr_parameters:
                        newmodel._func_allargs.remove(name)
                        newmodel._param_names.remove(name)
                # Check linearity of expression model parameters
                for name in newmodel.param_names:
                    if not diff(newmodel.expr, name, name):
                        if name not in self._linear_parameters:
                            self._linear_parameters.append(name)
                    else:
                        if name not in self._nonlinear_parameters:
                            self._nonlinear_parameters.append(name)
            else:
                kwargs = {}
                if model.model_type == 'rectangle':
                    kwargs['form'] = model.form
                newmodel = model.LMFITMODEL(prefix=model.prefix, **kwargs)
                for par in model.parameters:
                    _set_parameter_group(
                        model, par.name, model.prefix + par.name)
            return newmodel

        def _set_default_initial_parameters(new_parameters):
            for name in new_parameters:
                par = self._parameters[name]
                if name in self._linear_parameters:
                    if par.expr is None:
                        value = par.default if self._code == 'scipy' else None
                        if value is None:
                            value = par.value
                        _min = par.min
                        _max = par.max
                        if self._norm[1]:
                            if value is not None:
                                value *= self._norm[1]
                            if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                                _min *= self._norm[1]
                            if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                                _max *= self._norm[1]
                        par.set(value=value, min=_min, max=_max)
                elif par.expr is None:
                    par.set(value=par.value)


        def _initialize_model_parameters(model, new_parameters):
            for par in deepcopy(model.parameters):
                name = par.name
                if name not in new_parameters:
                    name = model.prefix + name
                    if name not in new_parameters:
                        raise ValueError(f'Unable to match parameter {name}')
                if par.expr is None:
                    self._parameters[name].set(
                        value=par.value, min=par.min, max=par.max,
                        vary=par.vary)
                else:
                    if par.value is not None:
                        self._logger.warning(
                            'Ignoring input "value" for expression parameter'
                            f'{name} = {par.expr}')
                    if not np.isinf(par.min):
                        self._logger.warning(
                            'Ignoring input "min" for expression parameter'
                            f'{name} = {par.expr}')
                    if not np.isinf(par.max):
                        self._logger.warning(
                            'Ignoring input "max" for expression parameter'
                            f'{name} = {par.expr}')
                    self._parameters[name].set(
                        value=None, min=-np.inf, max=np.inf, expr=par.expr)

        # Set model parameter info
        if self._code == 'scipy':
            new_parameters = _set_parameter_info_scipy(model)
            if self._model is None:
                self._model = Components()
            self._model |= {
                model.long_name: Component(model)}
        else:
            newmodel = _set_parameter_info_lmfit(model)
            if self._model is None:
                self._model = newmodel
            else:
                self._model += newmodel
            new_parameters = newmodel.make_params()
            self._parameters += new_parameters

        # Scale default initial model parameters
        _set_default_initial_parameters(new_parameters)

        # Initialize the model parameters
        _initialize_model_parameters(model, new_parameters)

    def add_parameter(self, parameter):
        """Add a fit parameter to the fit model.

        :param parameter: A new parameter to be added to the fit model.
        :type parameter: dict
        """
        # Local modules
        from CHAP.utils.models import FitParameter

        assert isinstance(parameter, dict)
        if parameter.get('expr') is not None:
            raise KeyError(f'Invalid "expr" key in parameter {parameter}')
        name = parameter['name']
        if not parameter['vary']:
            self._logger.warning(
                f'Ignoring min in parameter {name} in '
                f'Fit.add_parameter (vary = {parameter["vary"]})')
            parameter['min'] = -np.inf
            self._logger.warning(
                f'Ignoring max in parameter {name} in '
                f'Fit.add_parameter (vary = {parameter["vary"]})')
            parameter['max'] = np.inf
        if self._code == 'scipy':
            self._parameters.add(FitParameter(**parameter))
        else:
            self._parameters.add(**parameter)
        self._free_parameters.append(name)

    def fit(self, config=None, **kwargs):
        """Fit the model to the input data.

        :param config: Fit configuration.
        :type config: CHAP.utils.models.FitConfig, optional
        :param **kwargs: Additional key, value pairs to pass on
            directly to the core fit routine.
        """
        # Check input parameters
        if self._model is None:
            self._logger.error('Undefined fit model')
        num_proc_max = max(1, cpu_count())
        if config is None:
            num_proc = kwargs.pop('num_proc', num_proc_max)
            self._abs_height_cutoff = kwargs.pop('abs_height_cutoff')
            self._multipeak_info = kwargs.pop('multipeak_info', None)
            self._plot = kwargs.pop('plot', False)
            self._plot_init = kwargs.pop('plot_init', False)
            self._print_report = kwargs.pop('print_report', False)
            self._redchi_cutoff = kwargs.pop('redchi_cutoff', 0.1)
            self._rel_height_cutoff = kwargs.pop('rel_height_cutoff')
            self._try_no_bounds = kwargs.pop('try_no_bounds', False)
        else:
            num_proc = config.num_proc
            self._abs_height_cutoff = config.abs_height_cutoff
            self._plot = config.plot
#            self._plot_init = config.plot_init
            self._print_report = config.print_report
            self._rel_height_cutoff = config.rel_height_cutoff
#            self._try_no_bounds = config.try_no_bounds
        if num_proc > 1 and not HAVE_JOBLIB:
            self._logger.warning(
                'Missing joblib in the conda environment, running serially')
            num_proc = 1
        if num_proc > num_proc_max:
            self._logger.info(
                f'The requested number of processors ({num_proc}) exceeds the '
                'maximum allowed number of processors, num_proc reduced to '
                f'{num_proc_max}')
            num_proc = num_proc_max
        self._logger.info(f'Using {num_proc} processors to fit the data')
        if self._abs_height_cutoff is not None:
            self._abs_height_cutoff -= self._norm[0]
            if self._norm[1]:
                self._abs_height_cutoff /= self._norm[1]

        # Setup the fit
        self._setup_fit(config)

        # Create the best parameter list, consisting of all varying
        # parameters plus the expression parameters in order to collect
        # their errors
        if self._result is None:
            # Initial fit
            assert self._best_parameters is None
            self._best_parameters = [
                name for name, par in self._parameters.items()
                if par.vary or par.expr is not None]
            num_new_parameters = 0
        else:
            # Refit
            assert self._best_parameters
            self._new_parameters = [
                name for name, par in self._parameters.items()
                if name != 'tmp_normalization_offset_c'
                    and name not in self._best_parameters
                    and (par.vary or par.expr is not None)]
            num_new_parameters = len(self._new_parameters)
        num_best_parameters = len(self._best_parameters)

        # Flatten and normalize the best values of the previous fit,
        # remove the remaining results of the previous fit
        if self._result is not None:
            self._out_of_bounds = None
            self._max_nfev = None
            self._num_func_eval = None
            self._redchi = None
            self._success = None
            self._best_errors = None
            self._best_fit = None
            self._best_vary = None
            self._init_values = None
            assert self._best_values is not None
            assert self._best_values.shape[0] == num_best_parameters
            assert self._best_values.shape[1:] == self._map_shape
            self._best_values = [
                np.reshape(self._best_values[i], self._map_dim)
                for i in range(num_best_parameters)]
            if self._norm[1]:
                for i, name in enumerate(self._best_parameters):
                    if name in self._linear_parameters:
                        self._best_values[i] /= self._norm[1]

        # Normalize the initial parameters
        # (and best values for a refit)
        self._normalize()

        # Initialize parameter bounds and check to prevent initial
        # values at boundaries
        self._parameter_bounds = {
            name:{'min': par.min, 'max': par.max}
            for name, par in self._parameters.items() if par.vary}
        self._reset_par_at_boundary()

        # Set parameter bounds to unbound
        #     (only use bounds when fit fails)
        if 'fraction' in self._parameters and self._try_no_bounds:
            self._logger.warning(
                'Setting self._try_no_bounds to False for PseudoVoigt model')
            self._try_no_bounds = False
        if self._try_no_bounds:
            for name in self._parameter_bounds.keys():
                self._parameters[name].set(min=-np.inf, max=np.inf)

        # Allocate memory to store fit results
        if self._mask is None:
            x_size = self._x.size
        else:
            x_size = self._x[~self._mask].size
        if num_proc == 1:
            self._out_of_bounds_flat = np.zeros(self._map_dim, dtype=bool)
            self._max_nfev_flat = np.zeros(self._map_dim, dtype=bool)
            self._num_func_eval_flat = np.zeros(self._map_dim, dtype=np.intc)
            self._redchi_flat = np.zeros(self._map_dim, dtype=np.float64)
            self._success_flat = np.zeros(self._map_dim, dtype=bool)
            self._best_fit_flat = np.zeros(
                (self._map_dim, x_size), dtype=self._y.dtype)
            self._best_errors_flat = [
                np.zeros(self._map_dim, dtype=np.float64)
                for _ in range(num_best_parameters+num_new_parameters)]
            if self._result is None:
                self._best_values_flat = [
                    np.zeros(self._map_dim, dtype=np.float64)
                    for _ in range(num_best_parameters)]
            else:
                self._best_values_flat = self._best_values
                self._best_values_flat += [
                    np.zeros(self._map_dim, dtype=np.float64)
                    for _ in range(num_new_parameters)]
            self._best_vary_flat = [
                np.zeros(self._map_dim, dtype=bool)
                for _ in range(num_best_parameters+num_new_parameters)]
            self._init_values_flat = [
                np.zeros(self._map_dim, dtype=np.float64)
                for _ in range(num_best_parameters+num_new_parameters)]
        else:
            try:
                mkdir(self._memfolder)
            except FileExistsError:
                pass
            filename_memmap = path.join(
                self._memfolder, 'out_of_bounds_memmap')
            self._out_of_bounds_flat = np.memmap(
                filename_memmap, dtype=bool, shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(self._memfolder, 'max_nfev_memmap')
            self._max_nfev_flat = np.memmap(
                filename_memmap, dtype=bool, shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(
                self._memfolder, 'num_func_eval_memmap')
            self._num_func_eval_flat = np.memmap(
                filename_memmap, dtype=np.intc, shape=(self._map_dim),
                mode='w+')
            filename_memmap = path.join(self._memfolder, 'redchi_memmap')
            self._redchi_flat = np.memmap(
                filename_memmap, dtype=np.float64, shape=(self._map_dim),
                mode='w+')
            filename_memmap = path.join(self._memfolder, 'success_memmap')
            self._success_flat = np.memmap(
                filename_memmap, dtype=bool, shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(self._memfolder, 'best_fit_memmap')
            self._best_fit_flat = np.memmap(
                filename_memmap, dtype=self._y.dtype,
                shape=(self._map_dim, x_size), mode='w+')
            self._best_errors_flat = []
            for i in range(num_best_parameters+num_new_parameters):
                filename_memmap = path.join(
                    self._memfolder, f'best_errors_memmap_{i}')
                self._best_errors_flat.append(
                    np.memmap(filename_memmap, dtype=np.float64,
                              shape=self._map_dim, mode='w+'))
            self._best_values_flat = []
            for i in range(num_best_parameters):
                filename_memmap = path.join(
                    self._memfolder, f'best_values_memmap_{i}')
                self._best_values_flat.append(
                    np.memmap(filename_memmap, dtype=np.float64,
                              shape=self._map_dim, mode='w+'))
                if self._result is not None:
                    self._best_values_flat[i][:] = self._best_values[i][:]
            for i in range(num_new_parameters):
                filename_memmap = path.join(
                    self._memfolder,
                    f'best_values_memmap_{i+num_best_parameters}')
                self._best_values_flat.append(
                    np.memmap(filename_memmap, dtype=np.float64,
                              shape=self._map_dim, mode='w+'))
            self._best_vary_flat = []
            for i in range(num_best_parameters+num_new_parameters):
                filename_memmap = path.join(
                    self._memfolder, f'best_vary_memmap_{i}')
                self._best_vary_flat.append(
                    np.memmap(filename_memmap, dtype=bool,
                              shape=self._map_dim, mode='w+'))
            self._init_values_flat = []
            for i in range(num_best_parameters+num_new_parameters):
                filename_memmap = path.join(
                    self._memfolder, f'init_values_memmap_{i}')
                self._init_values_flat.append(
                    np.memmap(filename_memmap, dtype=np.float64,
                              shape=self._map_dim, mode='w+'))

        # Update the best parameter list
        if num_new_parameters:
            self._best_parameters += self._new_parameters

        # Perform the first fit to get model component info and
        # initial parameters
        current_best_values = {}
        self._result = self._fit(
            0, current_best_values, return_result=True, **kwargs)

        # Remove all irrelevant content from self._result
        for attr in (
                '_abort', 'aborted', 'aic', 'best_fit', 'best_values', 'bic',
                'calc_covar', 'call_kws', 'chisqr', 'ci_out', 'col_deriv',
                'covar', 'data', 'errorbars', 'flatchain', 'ier', 'init_vals',
                'init_fit', 'iter_cb', 'jacfcn', 'kws', 'last_internal_values',
                'lmdif_message', 'message', 'method', 'nan_policy', 'ndata',
                'nfev', 'nfree', 'params', 'redchi', 'reduce_fcn', 'residual',
                'result', 'scale_covar', 'show_candidates', 'calc_covar',
                'success', 'userargs', 'userfcn', 'userkws', 'values',
                'var_names', 'weights', 'user_options'):
            try:
                delattr(self._result, attr)
            except AttributeError:
                pass

        if self._map_dim > 1:
            if num_proc == 1:
                # Perform the remaining fits serially
                for n in range(1, self._map_dim):
                    self._fit(n, current_best_values, **kwargs)
            else:
                # Perform the remaining fits in parallel
                num_fit = self._map_dim-1
                if num_proc > num_fit:
                    self._logger.info(
                        f'The requested number of processors ({num_proc}) '
                        'exceeds the number of fits, num_proc reduced to '
                        f'{num_fit}')
                    num_proc = num_fit
                    num_fit_per_proc = 1
                else:
                    num_fit_per_proc = round((num_fit)/num_proc)
                    if num_proc*num_fit_per_proc < num_fit:
                        num_fit_per_proc += 1
                num_fit_batch = min(num_fit_per_proc, 40)
                with Parallel(n_jobs=num_proc) as parallel:
                    parallel(
                        delayed(self._fit_parallel)
                            (current_best_values, num_fit_batch, n_start,
                             **kwargs)
                        for n_start in range(1, self._map_dim, num_fit_batch))

        # Remap the best results
        self._out_of_bounds = np.copy(np.reshape(
            self._out_of_bounds_flat, self._map_shape))
        self._max_nfev = np.copy(np.reshape(
            self._max_nfev_flat, self._map_shape))
        self._num_func_eval = np.copy(np.reshape(
            self._num_func_eval_flat, self._map_shape))
        self._redchi = np.copy(np.reshape(self._redchi_flat, self._map_shape))
        self._success = np.copy(np.reshape(
            self._success_flat, self._map_shape))
        self._best_fit = np.copy(np.reshape(
            self._best_fit_flat, list(self._map_shape)+[x_size]))
        self._best_errors = np.asarray([np.reshape(
            par, list(self._map_shape)) for par in self._best_errors_flat])
        self._best_values = np.asarray([np.reshape(
            par, list(self._map_shape)) for par in self._best_values_flat])
        self._best_vary = np.asarray([np.reshape(
            par, list(self._map_shape)) for par in self._best_vary_flat])
        self._init_values = np.asarray([np.reshape(
            par, list(self._map_shape)) for par in self._init_values_flat])
        if self._inv_transpose is not None:
            self._out_of_bounds = np.transpose(
                self._out_of_bounds, self._inv_transpose)
            self._max_nfev = np.transpose(self._max_nfev, self._inv_transpose)
            self._num_func_eval = np.transpose(
                self._num_func_eval, self._inv_transpose)
            self._redchi = np.transpose(self._redchi, self._inv_transpose)
            self._success = np.transpose(self._success, self._inv_transpose)
            self._best_fit = np.transpose(
                self._best_fit,
                list(self._inv_transpose) + [len(self._inv_transpose)])
            self._best_errors = np.transpose(
                self._best_errors, [0] + [i+1 for i in self._inv_transpose])
            self._best_values = np.transpose(
                self._best_values, [0] + [i+1 for i in self._inv_transpose])
            self._best_vary = np.transpose(
                self._best_vary, [0] + [i+1 for i in self._inv_transpose])
            self._init_values = np.transpose(
                self._init_values, [0] + [i+1 for i in self._inv_transpose])
        del self._out_of_bounds_flat
        del self._max_nfev_flat
        del self._num_func_eval_flat
        del self._redchi_flat
        del self._success_flat
        del self._best_fit_flat
        del self._best_errors_flat
        del self._best_values_flat
        del self._best_vary_flat
        del self._init_values_flat

        # Restore parameter bounds and renormalize the parameters
        for name, par in self._parameter_bounds.items():
            self._parameters[name].set(min=par['min'], max=par['max'])
        if self._norm[1]:
            for name in self._linear_parameters:
                par = self._parameters[name]
                if par.expr is None:
                    value = par.value*self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)

        # Renormalize the initial parameters
        if self._norm[1]:
            for name, par in self._result.init_params.items():
                if par.expr is None and name in self._linear_parameters:
                    value = par.value*self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
                par.init_value = par.value

        if num_proc > 1:
            # Free the shared memory
            self.freemem()

    def freemem(self):
        """Free memory allocated for parallel processing."""
        if self._memfolder is None:
            return
        try:
            rmtree(self._memfolder)
        except Exception:
            self._logger.warning('Could not clean-up automatically.')

    def _create_prefixes(self, models):
        """Check for duplicate model names and create prefixes."""
        names = []
        for model in models:
            names.append(model.long_name)
        counts = Counter(names)
        for model, count in counts.items():
            if count > 1:
                n = 0
                for i, name in enumerate(names):
                    if name == model:
                        n += 1
                        models[i].prefix = f'{name}{n}_'

    def _fit(self, n, current_best_values, return_result=False, **kwargs):
        # Do not attempt a fit if the normalized data is zero or
        # entirely below the cutoff
        y_min = self._y_norm[n].min()
        y_max = self._y_norm[n].max()
        y_range = y_max - y_min
        if y_range <= FLOAT_MIN:
            y_range = 0.
        if (not y_range
                or (self._abs_height_cutoff is not None
                    and (y_range < self._abs_height_cutoff))):
            if self._abs_height_cutoff is not None:
                if self._norm[1]:
                    y_max *= self._norm[1]
                y_max += self._norm[0]
                self._logger.debug(
                    f'Skipping fit {n} (height = {y_max:.5f})')
            parameters = deepcopy(self._parameters)
            for name in self._linear_parameters:
                if name != 'tmp_normalization_offset_c':
                    parameters[name].set(value=0)
            if self._code == 'scipy':
                # Third party modules
                from asteval import Interpreter

                # Local modules
                from CHAP.utils.fit import ModelResult

                ast = Interpreter()
                ast.basesymtable = dict(ast.symtable.items())
                best_pars = []
                res_par_indices = []
                for i, (name, par) in enumerate(parameters.items()):
                    value = par.value
                    if par.expr is None:
                        ast.symtable[name] = value
                        if par.vary:
                            best_pars.append(value)
                            res_par_indices.append(
                                self._res_par_indices[
                                    self._res_par_names.index(name)])

                result = ModelResult(
                   # self._model, self._y[n], deepcopy(self._parameters),
                   # x=self._x, method=self._method)
                    self._model, self._y_norm[n], parameters, x=self._x,
                    method=self._method,
                    ast=ast, res_par_exprs=self._res_par_exprs,
                    res_par_indices=res_par_indices,
                    res_par_names=self._res_par_names, best_pars=best_pars)
            else:
                # Third party modules
                from lmfit.model import ModelResult

                result = self._model.fit(
                    self._y_norm[n], parameters, x=self._x,
                    method=self._method, max_nfev=0)
                if 'tmp_normalization_offset_c' in parameters:
                    result.init_fit -= parameters[
                        'tmp_normalization_offset_c'].value
                    result.best_fit -= parameters[
                        'tmp_normalization_offset_c'].value
                result.aic = 0
                result.bic = 0
                result.chisqr = 0
                result.redchi = 0
            # Renormalize the data and results
            result.success = True
            for name in self._linear_parameters:
                if (name != 'tmp_normalization_offset_c'
                        and name in ('c', 'intercept')):
                    result.init_params[name].set(value=self._norm[0])
                    result.params[name].set(value=self._norm[0])
                    break
            self._renormalize(n, result)
            if y_range:
                self._success_flat[n] = False
            # Print output or plot
            if self._print_report:
                print(result.fit_report(show_correl=False))
            if self._plot:
                self._plot_result(
                    n, result, plot_comp_legends=True,
                    plot_init=self._plot_init)
            return result

        parameters_save = deepcopy(self._parameters)
        parameters_bounds_save = deepcopy(self._parameter_bounds)
        if self._multipeak_info is not None:
            # Third party modules
            from asteval import Interpreter

            centers = self._multipeak_info.get('centers')
            centers_range = self._multipeak_info.get('centers_range')
            centers_range_fraction = \
                self._multipeak_info.get('centers_range_fraction')
            model_type = self._multipeak_info.get('peak_models')
            min_height = None if self._rel_height_cutoff is None \
                else y_max*self._rel_height_cutoff
            use_peaks, _, peak_heights, peak_widths = \
                self.guess_init_peak(
                    self._x, self._y_norm[n], centers, centers_range,
                    centers_range_fraction, min_height=min_height, min_width=5)

            ast = Interpreter()
            for i, (use_peak, height, width) in enumerate(zip(
                    use_peaks, peak_heights, peak_widths)):
                name = f'peak{i+1}_amplitude'
                ast(f'fwhm = {width}')
                ast(f'height = {height}')
                sigma = ast(fwhm_factor[model_type])
                amplitude = ast(height_factor[model_type])
                if use_peak:
                    self._parameters[name].set(value=amplitude)
                    self._parameters[name.replace('amplitude', 'sigma')].set(
                        value=sigma)
                else:
                    self._parameters[name].set(
                        value=0.0, min=0.0, vary=False)
                    self._parameters[
                        name.replace('amplitude', 'center')].set(vary=False)
                    self._parameters[name.replace('amplitude', 'sigma')].set(
                        value=0.0, min=0.0, vary=False)

        # Regular full fit
        result = self._fit_with_bounds_check(n, current_best_values, **kwargs)
        if result.nfev == kwargs.get('max_nfev'):
            self._logger.info(
                f'Hit max_nfev limit for n={n}\n\tnfev: {result.nfev}')

        if result.redchi >= self._redchi_cutoff:
            result.success = False
        self._num_func_eval_flat[n] = result.nfev
        if result.nfev == result.max_nfev:
            if result.redchi < self._redchi_cutoff:
                result.success = True
            self._max_nfev_flat[n] = True
        if result.success:
            assert all(
                True for par in current_best_values
                if par in result.params.values())
            # FIX made a flag to propagete best values to the next fit
            # do not do it by default (add a kwarg to Fit.fit())
            #for par in result.params.values():
            #    if par.vary:
            #        current_best_values[par.name] = par.value
        else:
            errortxt = f'Fit for n = {n} failed'
            if hasattr(result, 'lmdif_message'):
                errortxt += f'\n\t{result.lmdif_message}'
            if hasattr(result, 'message'):
                errortxt += f'\n\t{result.message}'
            self._logger.warning(f'{errortxt}')

        # Reset parameters to defaults
        self._parameters = deepcopy(parameters_save)
        self._parameter_bounds = deepcopy(parameters_bounds_save)

        # Renormalize the data and results
        self._renormalize(n, result)

        # Print output or plot
        if self._print_report:
            print(result.fit_report(show_correl=False))
        if self._plot:
            self._plot_result(
                n, result, plot_comp_legends=True, plot_init=self._plot_init)

        if return_result:
            return result
        return None

    def _fit_nonlinear_model(self, x, y, **kwargs):
        """Perform a nonlinear fit with spipy or lmfit."""
        def _fit_scipy(x, y, have_bounds, **kwargs):
            # Third party modules
            from asteval import Interpreter
            from scipy.optimize import (
                leastsq,
                least_squares,
            )

            self._ast = Interpreter()
            self._ast.basesymtable = dict(self._ast.symtable.items())
            pars_init = []
            res_par_indices = []
            for i, (name, par) in enumerate(self._parameters.items()):
                value = par.value
                self._res_par_values[i] = value
                if par.expr is None:
                    self._ast.symtable[name] = value
                    if par.vary:
                        pars_init.append(value)
                        res_par_indices.append(
                            self._res_par_indices[
                                self._res_par_names.index(name)])
            if have_bounds:
                bounds = (
                    [v['min'] for v in self._parameter_bounds.values()],
                    [v['max'] for v in self._parameter_bounds.values()])
                if self._method in ('lm', 'leastsq'):
                    self._method = 'trf'
                    self._logger.debug(
                        f'Fit method changed to {self._method} for fit with '
                        'bounds')
            else:
                bounds = (-np.inf, np.inf)
            lskws = {
                'ftol': 1.49012e-08,
                'xtol': 1.49012e-08,
                'gtol': 10*FLOAT_EPS,
            }
            max_nfev = kwargs.get('max_nfev')
            if self._method == 'leastsq':
                if max_nfev is not None:
                    lskws['maxfev'] = max_nfev
                result = leastsq(
                    self._residual, pars_init, args=(x, y, res_par_indices),
                    full_output=True, **lskws)
            else:
                if max_nfev is not None:
                    lskws['max_nfev'] = max_nfev
                result = least_squares(
                    self._residual, pars_init, bounds=bounds,
                    method=self._method, args=(x, y, res_par_indices), **lskws)
            model_result = ModelResult(
                self._model, y, self._parameters, x=x, method=self._method,
                ast=self._ast, res_par_exprs=self._res_par_exprs,
                res_par_indices=res_par_indices,
                res_par_names=self._res_par_names, result=result)
            model_result.max_nfev = lskws.get('maxfev')
            return model_result

        # Check bounds and prevent initial values at boundaries
        have_bounds = False
        self._parameter_bounds = {}
        for name, par in self._parameters.items():
            if par.vary:
                self._parameter_bounds[name] = {
                    'min': par.min, 'max': par.max}
                if not have_bounds and (
                        not np.isinf(par.min) or not np.isinf(par.max)):
                    have_bounds = True
        if have_bounds:
            self._reset_par_at_boundary()

        # Perform the fit
        if self._mask is not None:
            x = x[~self._mask]
            y = np.asarray(y)[~self._mask]
        if self._code == 'scipy':
            return _fit_scipy(x, y, have_bounds, **kwargs)
#        fit_kws = {}
#        if 'Dfun' in kwargs:
#            fit_kws['Dfun'] = kwargs.pop('Dfun')
        model_result  = self._model.fit(
            y, self._parameters, x=x, method=self._method, #fit_kws=fit_kws,
            **kwargs)
        if 'tmp_normalization_offset_c' in self._parameters:
            model_result.init_fit -= self._parameters[
                'tmp_normalization_offset_c'].value
        return model_result

    def _fit_parallel(self, current_best_values, num, n_start, **kwargs):
        num = min(num, self._map_dim-n_start)
        for n in range(num):
            self._fit(n_start+n, current_best_values, **kwargs)

    def _fit_with_bounds_check(self, n, current_best_values, **kwargs):
        # Set parameters to current best values, but prevent them from
        #     sitting at boundaries
        if self._new_parameters is None:
            # Initial fit
            for name, value in current_best_values.items():
                par = self._parameters[name]
                if par.vary:
                    par.set(value=value)
        else:
            # Refit
            for i, name in enumerate(self._best_parameters):
                par = self._parameters[name]
                if par.vary:
                    if name in self._new_parameters:
                        if name in current_best_values:
                            par.set(value=current_best_values[name])
                    elif par.expr is None:
                        par.set(value=self._best_values[i][n])
        self._reset_par_at_boundary()
        result = self._fit_nonlinear_model(self._x, self._y_norm[n], **kwargs)
        out_of_bounds = False
        for name, par in self._parameter_bounds.items():
            if self._parameters[name].vary:
                value = result.params[name].value
                if not np.isinf(par['min']) and value < par['min']:
                    out_of_bounds = True
                    break
                if not np.isinf(par['max']) and value > par['max']:
                    out_of_bounds = True
                    break
        self._out_of_bounds_flat[n] = out_of_bounds
        if self._try_no_bounds and out_of_bounds:
            # Rerun fit with parameter bounds in place
            for name, par in self._parameter_bounds.items():
                if self._parameters[name].vary:
                    self._parameters[name].set(min=par['min'], max=par['max'])
            # Set parameters to current best values, but prevent them
            #     from sitting at boundaries
            if self._new_parameters is None:
                # Initial fit
                for name, value in current_best_values.items():
                    par = self._parameters[name]
                    if par.vary:
                        par.set(value=value)
            else:
                # Refit
                for i, name in enumerate(self._best_parameters):
                    par = self._parameters[name]
                    if par.vary:
                        if name in self._new_parameters:
                            if name in current_best_values:
                                par.set(value=current_best_values[name])
                        elif par.expr is None:
                            par.set(value=self._best_values[i][n])
            self._reset_par_at_boundary()
            result = self._fit_nonlinear_model(
                self._x, self._y_norm[n], **kwargs)
            out_of_bounds = False
            for name, par in self._parameter_bounds.items():
                if self._parameters[name].vary:
                    value = result.params[name].value
                    if not np.isinf(par['min']) and value < par['min']:
                        out_of_bounds = True
                        break
                    if not np.isinf(par['max']) and value > par['max']:
                        out_of_bounds = True
                        break
                    # Reset parameters back to unbound
                    self._parameters[name].set(min=-np.inf, max=np.inf)
        assert not out_of_bounds
        return result

    def _normalize(self):
        """Normalize the data and initial parameters."""
        if self._norm[1]:
            for name in self._linear_parameters:
                par = self._parameters[name]
                if par.expr is None:
                    value = par.value/self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min /= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max /= self._norm[1]
                    par.set(value=value, min=_min, max=_max)

    def _plot_result(
            self, n, result, plot_comp_legends=False, plot_masked_data=True,
            plot_init=False, **kwargs):
        """Plot the best fits.

        :param n: Index of flattened map point to plot.
        :type n: int
        :param result: Fit result
        :type result: class:`~CHAP.utils.fit.ModelResult` or
            lmfit.model.ModelResult.
        :param plot_comp_legends: Add a legend for the individual
            model components, defaults to `False`.
        :type plot_comp_legends: bool, optional
        :param plot_masked_data: Visually distinguish the masked from
            the unmasked data, defaults to `True`.
        :type plot_masked_data: bool, optional
        :param plot_init:Plot the initial guess, defaults to `True`.
        :type plot_init: bool, optional
        :param **kwargs: Additional key, value pairs to pass on
            directly to the Matplotlib plot function.
        """
        # Third party modules
        from lmfit.models import ExpressionModel

        if self._mask is None:
            mask = np.zeros(self._x.size).astype(bool)
            plot_masked_data = False
        else:
            mask = self._mask
        x = self._x[~mask]
        y = self._y[n][~mask]
        x_masked = self._x[mask]
        y_masked = self._y[n][mask]
        if plot_masked_data:
            plots = [(x, y, 'b.')]
            legend = ['data']
            plots += [(x_masked, y_masked, 'bx')]
            legend += ['masked data']
        else:
            plots = [(x, y, 'b.')]
            legend = ['data']
        plots += [(x, self._best_fit_flat[n], 'k-')]
        legend += ['best fit']
        if plot_init:
            plots += [(x[~mask], result.init_fit, 'g-')]
            legend += ['init']
        # Create current parameters
        parameters = deepcopy(self._parameters)
        for i, name in enumerate(self._best_parameters):
            parameters[name].set(value=self._best_values_flat[i][n])
        for component in result.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            if isinstance(component, ExpressionModel):
                modelname = component._name.rstrip('_')+f' ({component.expr})'
            else:
                prefix = component.prefix.rstrip('_')
                modelname = prefix + f' ({component._name})' if prefix \
                    else component._name
            if len(modelname) > 20:
                modelname = f'{modelname[0:16]} ...'
            y = component.eval(params=parameters, x=x)
            if y is not None:
                if isinstance(y, (int, float)):
                    y *= np.ones(x.size)
                plots += [(x, y, '--')]
            if plot_comp_legends:
                legend.append(modelname)
        quick_plot(
            tuple(plots), legend=legend, title=f'point {n}', block=True,
            **kwargs)

    def _renormalize(self, n, result):
        self._success_flat[n] = result.success
        if result.success:
            self._redchi_flat[n] = np.float64(result.redchi)
        for name, par in result.params.items():
            par.init_value = result.init_params[name].value
            if self._norm[1]:
                if name in self._linear_parameters:
                    if par.init_value is not None:
                        par.init_value *= self._norm[1]
                    if par.stderr is not None:
                        par.stderr *= self._norm[1]
                    if par.expr is None:
                        par.value *= self._norm[1]
                        if self._print_report:
                            if (not np.isinf(par.min)
                                    and abs(par.min) != FLOAT_MIN):
                                par.min *= self._norm[1]
                            if (not np.isinf(par.max)
                                    and abs(par.max) != FLOAT_MIN):
                                par.max *= self._norm[1]
        for i, name in enumerate(self._best_parameters):
            self._best_errors_flat[i][n] = np.float64(
                result.params[name].stderr)
            self._best_values_flat[i][n] = np.float64(
                result.params[name].value)
            self._best_vary_flat[i][n] = (
                result.params[name].stderr and result.success)
            self._init_values_flat[i][n] = np.float64(
                result.params[name].init_value)
        if result.success:
            if self._norm[1]:
                result.best_fit = (
                    result.best_fit*self._norm[1] + self._norm[0])
            else:
                result.best_fit += self._norm[0]
            self._best_fit_flat[n] = result.best_fit
        if self._plot and self._plot_init:
            if self._norm[1]:
                result.init_fit = (
                    result.init_fit*self._norm[1] + self._norm[0])
            else:
                result.init_fit += self._norm[0]

    def _reset_par_at_boundary(self):
        fraction = 0.02
        y_range = self._norm[1] if self._norm[1] else 1
        for name, par in self._parameters.items():
            if par.vary:
                value = par.value
                _min = self._parameter_bounds[name]['min']
                _max = self._parameter_bounds[name]['max']
                if np.isinf(_min):
                    if not np.isinf(_max):
                        if name in self._linear_parameters:
                            upp = _max - fraction*y_range
                        elif not _max:
                            upp = _max - fraction
                        else:
                            upp = _max - fraction*abs(_max)
                        if value >= upp:
                            par.set(value=upp)
                else:
                    if np.isinf(_max):
                        if name in self._linear_parameters:
                            low = _min + fraction*y_range
                        elif not _min:
                            low = _min + fraction
                        else:
                            low = _min + fraction*abs(_min)
                        if value <= low:
                            par.set(value=low)
                    else:
                        low = (1.-fraction)*_min + fraction*_max
                        upp = fraction*_min + (1.-fraction)*_max
                        if value <= low:
                            par.set(value=low)
                        if value >= upp:
                            par.set(value=upp)

    def _residual(self, pars, x, y, res_par_indices):
        res = np.zeros((x.size))
        n_par = len(self._free_parameters)
        for par, index in zip(pars, res_par_indices):
            self._res_par_values[index] = par
        if self._res_par_exprs:
            for par, name in zip(pars, self._res_par_names):
                self._ast.symtable[name] = par
            for expr in self._res_par_exprs:
                self._res_par_values[expr['index']] = \
                    self._ast.eval(expr['expr'])
        for component, num_par in zip(
                self._model.components, self._res_num_pars):
            parvalues = self._res_par_values[n_par:n_par+num_par]
            res += component.func(
                x, *tuple([parvalues[i] for i in component.func_args_indices]),
                **component.model_identifiers)
            n_par += num_par
        return res - y

    def _setup_fit(self, config):
        """Setup the fit."""
        def _setup_parameters_refit(config):
            # Local modules
            from CHAP.utils.models import (
                FitConfig,
                MultipeakModel,
            )

            # Expand multipeak model if present
            found_multipeak = False
            scale_factor = None
            # RV FIX do I need multipeak_info here too?
            for i, model in enumerate(deepcopy(config.models)):
                if isinstance(model, MultipeakModel):
                    if found_multipeak:
                        raise ValueError(
                            f'Invalid parameter models ({config.models}) '
                            '(multiple instances of multipeak not allowed)')
                    if (model.fit_type == 'uniform'
                            and 'scale_factor' not in self._free_parameters):
                        raise ValueError(
                            f'Invalid parameter models ({config.models}) '
                            '(uniform multipeak fit after unconstrained fit)')
                    parameters, models = FitProcessor.create_multipeak_model(
                        model)
                    if (model.fit_type == 'unconstrained'
                            and 'scale_factor' in self._free_parameters):
                        # Third party modules
                        from asteval import Interpreter

                        scale_factor = self._parameters['scale_factor'].value
                        self._parameters.pop('scale_factor')
                        self._free_parameters.remove('scale_factor')
                        ast = Interpreter()
                        ast(f'scale_factor = {scale_factor}')
                    if parameters:
                        config.parameters += parameters
                    config.models += models
                    config.models.pop(i)
                    found_multipeak = True

            # Check for duplicate model names and create prefixes
            self._create_prefixes(config.models)
            parameters = config.parameters
            for model in config.models:
                for par in model.parameters:
                    par.name = model.prefix + par.name
                parameters += model.parameters

            # Adjust parameters for refit as needed
            scale_factor_index = \
                self._best_parameters.index('scale_factor')
            self._best_errors = np.delete(
                self._best_errors, scale_factor_index, 0)
            self._best_parameters.pop(scale_factor_index)
            self._best_values = np.delete(
                self._best_values, scale_factor_index, 0)
            self._best_vary = np.delete(
                self._best_vary, scale_factor_index, 0)
            self._init_values = np.delete(
                self._init_values, scale_factor_index, 0)
            for par in parameters:
                name = par.name
                if name not in self._parameters:
                    raise ValueError(
                        f'Unable to match {name} parameter {par} to an '
                        'existing one')
                ppar = self._parameters[name]
                if ppar.expr is not None:
                    if (scale_factor is not None and 'center' in name
                            and 'scale_factor' in ppar.expr):
                        ppar.set(value=ast(ppar.expr), expr='')
                        value = ppar.value
                    else:
                        raise ValueError(
                            f'Unable to modify {name} parameter {par} '
                            '(currently an expression)')
                else:
                    value = par.value
                if par.expr is not None:
                    raise KeyError(
                        f'Invalid "expr" key in {name} parameter {par}')
                ppar.set(
                    value=value, min=par.min, max=par.max, vary=par.vary)

        # Add constant offset for a normalized model
        if self._result is None and self._norm[0]:
            # Local modules
            from CHAP.utils.models import ConstantModel

            model = ConstantModel(
                model_type='constant',
                parameters=[{
                    'name': 'c',
                    'value': -self._norm[0],
                    'vary': False,
                }],
                prefix='tmp_normalization_offset_'
            )
            self.add_model(model)

        # Adjust existing parameters for refit:
        if config is not None:
            _setup_parameters_refit(config)

        # Set scipy parameters configuration
        if self._code == 'scipy':
            self._res_par_exprs = []
            self._res_par_indices = []
            self._res_par_names = []
            self._res_par_values = []
            for i, (name, par) in enumerate(self._parameters.items()):
                self._res_par_values.append(par.value)
                if par.expr:
                    self._res_par_exprs.append(
                        {'expr': par.expr, 'index': i})
                elif par.vary:
                    self._res_par_indices.append(i)
                    self._res_par_names.append(name)

        # Check for uninitialized parameters
        for name, par in self._parameters.items():
            if par.expr is None:
                value = par.value
                if value is None or np.isinf(value) or np.isnan(value):
                    if self._norm[1] or name in self._nonlinear_parameters:
                        self._parameters[name].set(value=1.0)
                    elif name not in self._model_parameters:
                        self._parameters[name].set(value=self._norm[1])

    def _setup_fit_model(self, models, parameters):
        """Setup the fit model."""
        # Third party modules
        from sympy import diff

        # Local modules
        from CHAP.utils.models import PEAK_LIKE_MODELS

        # Check for duplicate model names and create prefixes
        self._create_prefixes(models)

        # Add the free fit parameters
        for par in parameters:
            self.add_parameter(
                par.model_dump(exclude=('description', 'units')))

        # Add the model functions
        for model in models:
            self.add_model(model)

        # Check linearity of free fit parameters
        for name in reversed(self._parameters):
            if (name not in (self._linear_parameters +
                             self._nonlinear_parameters +
                             self._model_parameters)
                    and not (model.model_type in PEAK_LIKE_MODELS
                        and ('height' in name or 'fwhm' in name))):
                for nname, par in self._parameters.items():
                    if par.expr is not None:
                        expr = par.expr.replace('fraction', 'fraction_') \
                            if 'fraction' in par.expr else par.expr
                        nnname = 'fraction_' \
                            if name == 'fraction' else name
                        if nnname in expr:
                            if nname in self._nonlinear_parameters:
                                self._nonlinear_parameters.insert(0, name)
                                break
                            else:
                                raise RuntimeError('not updated and tested')
                                if diff(expr, nnname, nnname):
                                    if name not in self._nonlinear_parameters:
                                        self._nonlinear_parameters.insert(
                                            0, name)
                                elif name not in self._linear_parameters:
                                    self._linear_parameters.insert(0, name)
