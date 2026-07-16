#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""Generic curve fitting module."""

# System modules
from collections import Counter
from copy import deepcopy
from os import (
    cpu_count,
    mkdir,
    path,
)
from re import sub
from shutil import rmtree
from sys import float_info
#from time import time
from typing import Optional

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
from pydantic import Field

# Local modules
from CHAP.processor import Processor
from CHAP.utils.general import (
#    is_int,
    is_index,
    index_nearest,
    quick_plot,
)
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
            if isinstance(ddata, (Fit, FitMap)):
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

        # Retrieve the optional coordinates from the pipeline
        try:
            y = np.asarray(ddata)
            for i, d in reversed(list(enumerate(data))):
                if d.get('name') == 'coordinates':
                    x = np.asarray(d.get('data'))
                    if remove:
                        data.pop(i)
                    assert x.size == y.shape[-1]
                    break
            else:
                x = None
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f'Unable to extract suitable fit input data from {data}')
        return x, y

    def process(self, data):
        """Fit the data and return a :class:`~CHAP.utils.fit.Fit` or
        :class:`~CHAP.utils.fit.FitMap` object depending on the
        dimensionality of the input data. The input data should be a
        list of PipelineData items containing either one with a `data`
        field of type :class:`~CHAP.utils.fit.Fit` or
        :class:`~CHAP.utils.fit.FitMap` (to refit or continue a
        previous fit), or one with the `name` of `signal` and an
        array-like `data` field. In the latter case, optional
        x-coordinates can be supplied by a second PipelineData item
        with the `name` of `coordinates` and again an array-like
        `data` field.

        :param data: Input data.
        :type data: list[PipelineData]
        :return: The fitted data object.
        :rtype: Fit or FitMap
        """
        # Unwrap the PipelineData
        data = self._get_pipelinedata_item(data)

        if isinstance(data, (Fit, FitMap)):
            # Refit/continue the fit with possibly updated parameters
            fit = data
            fit.fit(config=self.config, max_nfev=self.config.max_nfev)
            if self.config is not None and not isinstance(data, FitMap):
                if self.config.print_report:
                    fit.print_fit_report()
                if self.config.plot:
                    fit.plot(skip_init=True)
        else:
            # Local modules
            from CHAP.utils.models import MultipeakModel

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

            # Instantiate the Fit or FitMap object and fit the data
            if np.squeeze(data[1]).ndim == 1:
                fit = Fit(data[1], self.config, self.logger, x=data[0])
                fit.fit(max_nfev=self.config.max_nfev)
                if self.config.print_report:
                    fit.print_fit_report()
                if self.config.plot:
                    fit.plot(skip_init=True)
            else:
                fit = FitMap(data[1], self.config, self.logger, x=data[0])
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


class Component():
    """A model fit component."""

    def __init__(self, model, prefix=''):
        """Initialize Component.

        :param model: A fit model class.
        :type model: :attr:`~CHAP.utils.models.FitConfig.models`
        :param prefix: Model prefix, defaults to `''`.
        :type prefix: str, optional
        """
        self.func = model.func
        self.func_args = model.func_args
        self.model_identifiers = {
            k:getattr(model, k) for k in model.MODEL_IDENTIFIERS}
        self.param_names = [f'{prefix}{par.name}' for par in model.parameters]
        self.prefix = prefix
        self._name = model.model_type
        names = [f'{par.name}' for par in model.parameters]
        self.func_args_indices = [names.index(arg) for arg in self.func_args]


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
        :type parameter: str or FitParameter
        :param prefix: Prefix of the model to which this parameter
            belongs, defaults to `''`.
        :type prefix: str, optional
        """
        # Local modules
        from CHAP.utils.models import FitParameter

        if isinstance(parameter, FitParameter):
            name = f'{prefix}{parameter.name}'
            self.__setitem__(name, parameter)
        else:
            raise RuntimeError('Must test')
            parameter = f'{prefix}{parameter}'
            self.__setitem__(
                parameter,
                FitParameter(name=parameter))
        setattr(self[parameter.name], '_prefix', prefix)


class ModelResult():
    """The result of a model fit, mimicking the functionality of a
    similarly named
    `class in the lmfit library <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult>`__.
    """

    def __init__(
            self, model, parameters, *, x=None, y=None, method=None, ast=None,
            res_par_exprs=None, res_par_indices=None, res_par_names=None,
            result=None):
        """Initialize SetNumexprThreads.

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
        self.params = deepcopy(parameters)
        if x is None:
            self.success = False
            return
        if method == 'leastsq':
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
        self.best_fit = y + self.residual
        self.method = method
        self.ndata = len(self.residual)
        self.nvarys = len(res_par_indices)
        self.x = x
        self._ast = ast
        self._expr_pars = {}

        # Get the covarience matrix
        self.chisqr = (self.residual**2).sum()
        self.redchi = self.chisqr / (self.ndata-self.nvarys)
        self.covar = None
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
                par = self.params[name]
                par.set(value=self._ast.eval(expr))
                self._expr_pars[name] = expr
                setattr(
                    par, '_stderr',
                    None if self.covar is None else _get_stderr_expr(expr))

    def eval_components(self, x=None, parameters=None):
        """Evaluate each component of a composite model function.

        :param x: Independent variable, defaults to `None`, in which 
            case the class variable x is used.
        :type x: array-like, optional
        :param parameters: Composite model parameters, defaults to
            `None`, in which case the class variable params is used.
        :type parameters: Parameters or lmfit.parameter.Parameter,
            optional
        :return: A dictionary with component name and evaluated
            function values key, value pairs.
        :rtype: dict
        """
        if x is None:
            x = self.x
        if parameters is None:
            parameters = self.params
        result = {}
        for component in self.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            par_values = tuple(
                parameters[par].value for par in component.param_names)
            name = component._name if component.prefix == '' \
                else component.prefix
            ppar_values = tuple(
                par_values[i] for i in component.func_args_indices)
            result[name] = component.func(
                x, *ppar_values, **component.model_identifiers)
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
            expr = self._expr_pars.get(name, par.expr)
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


class Fit:
    """Wrapper class for scipy/lmfit."""

    def __init__(self, y, config, logger, x=None):
        """Initialize Fit.

        :param y: Input signal data.
        :type y: array-like
        :param config: Fit configuration.
        :type config: CHAP.utils.models.FitConfig
        :param logger: A python Logger object.
        :type logger: logging.Logger
        :param x: Input coordinate data.
        :type x: array-like, optional
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
        self._logger = logger
        self._mask = None
        self._method = config.method
        self._model = None
        self._norm = None
        self._normalized = False
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
        self._x = None
        self._y = None
        self._y_norm = None
        self._y_range = None
#        if 'try_linear_fit' in kwargs:
#            self._try_linear_fit = kwargs.pop('try_linear_fit')
#            if not isinstance(self._try_linear_fit, bool):
#                raise ValueError(
#                    'Invalid value of keyword argument try_linear_fit '
#                    f'({self._try_linear_fit})')
        if y is not None:
            self._y = np.squeeze(y)
            if self._y.ndim != 1:
                raise ValueError(
                    f'Invalid input signal dimension ({self._y.ndim})')
            if x is None:
                self._x = np.arange(self._y.size)
            else:
                self._x = x
                assert self._x.size == self._y.size
#            if 'mask' in kwargs:
#                self._mask = kwargs.pop('mask')
            if True: #self._mask is None:
                y_min = float(self._y.min())
                self._y_range = float(self._y.max())-y_min
                if self._y_range > 0.0:
                    self._norm = (y_min, self._y_range)
#            else:
#                self._mask = np.asarray(self._mask).astype(bool)
#                if self._x.size != self._mask.size:
#                    raise ValueError(
#                        f'Inconsistent x and mask dimensions ({self._x.size} '
#                        f'vs {self._mask.size})')
#                y_masked = np.asarray(self._y)[~self._mask]
#                y_min = float(y_masked.min())
#                self._y_range = float(y_masked.max())-y_min
#                if self._y_range > 0.0:
#                    self._norm = (y_min, self._y_range)

            # Setup fit model
            self._setup_fit_model(config.parameters, config.models)

    @property
    def best_errors(self):
        """Return errors in the best fit parameters.

        :type: dict
        """
        if self._result is None:
            return {}
        return {name:self._result.params[name].stderr
                for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def best_fit(self):
        """Return the best fit.

        :type: numpy.ndarray
        """
        if self._result is None:
            return None
        return self._result.best_fit

    @property
    def best_parameters(self):
        """Return the best fit parameters.

        :type: dict
        """
        if self._result is None:
            return {}
        parameters = {}
        for name in sorted(self._result.params):
            if name != 'tmp_normalization_offset_c':
                par = self._result.params[name]
                parameters[name] = {
                    'value': par.value,
                    'error': par.stderr,
                    'init_value': par.init_value,
                    'min': par.min,
                    'max': par.max,
                    'vary': par.vary, 'expr': par.expr
                }
        return parameters

    @property
    def best_values(self):
        """Return values for the best fit parameters.

        :type: dict
        """
        if self._result is None:
            return {}
        return {name:self._result.params[name].value
                for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def best_vary(self):
        """Return vary parameters for the best fit parameters.

        :type: dict
        """
        if self._result is None:
            return {}
        return {name:self._result.params[name].vary
                for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def chisqr(self):
        """Return chisqr values for the best fit parameters.

        :type: numpy.ndarray
        """
        if self._result is None:
            return None
        return self._result.chisqr

    @property
    def components(self):
        """Return fit model components info.

        :type: dict
        """
        # Third party modules
        from lmfit.models import ExpressionModel

        components = {}
        if self._result is None:
            self._logger.warning(
                'Unable to collect components in Fit.components')
            return components
        for component in self._result.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            parameters = {}
            for name in component.param_names:
                par = self._parameters[name]
                parameters[name] = {
                    'free': par.vary,
                    'value': self._result.params[name].value,
                }
                if par.expr is not None:
                    parameters[name]['expr'] = par.expr
            expr = None
            if isinstance(component, ExpressionModel):
                name = component._name
                if name[-1] == '_':
                    name = name[:-1]
                expr = component.expr
            else:
                if prefix := component.prefix:
                    if prefix[-1] == '_':
                        prefix = prefix[:-1]
                    name = f'{prefix} ({component._name})'
                else:
                    name = f'{component._name}'
            if expr is None:
                components[name] = {
                    'parameters': parameters,
                }
            else:
                components[name] = {
                    'expr': expr,
                    'parameters': parameters,
                }
        return components

    @property
    def covar(self):
        """Return the covarience matrix for the best fit parameters.

        :type: numpy.ndarray
        """
        if self._result is None:
            return None
        return self._result.covar

    @property
    def init_parameters(self):
        """Return the initial parameters for the fit model.

        :type: dict
        """
        if self._result is None or self._result.init_params is None:
            return {}
        parameters = {}
        for name in sorted(self._result.init_params):
            if name != 'tmp_normalization_offset_c':
                par = self._result.init_params[name]
                parameters[name] = {
                    'expr': par.expr,
                    'min': par.min,
                    'max': par.max,
                    'value': par.value,
                    'vary': par.vary,
                }
        return parameters

    @property
    def init_values(self):
        """Return the initial values for the fit parameters.

        :type: dict
        """
        if self._result is None or self._result.init_params is None:
            return {}
        return {name:self._result.init_params[name].value
                for name in sorted(self._result.init_params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def normalization_offset(self):
        """Return the `normalization_offset` value for the fit model.

        :type: float
        """
        if self._result is None:
            return None
        if self._norm is None:
            return 0.0
        if self._result.init_params is not None:
            normalization_offset = float(
                self._result.init_params['tmp_normalization_offset_c'].value)
        else:
            normalization_offset = float(
                self._result.params['tmp_normalization_offset_c'].value)
        return normalization_offset

    @property
    def num_func_eval(self):
        """Return the number of function evaluations for the best fit.

        :type: int
        """
        if self._result is None:
            return None
        return self._result.nfev

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
        """Return redchi for the best fit.

        :type: dict
        """
        if self._result is None:
            return None
        return self._result.redchi

    @property
    def residual(self):
        """Return the residual in the best fit.

        :type: numpy.ndarray
        """
        if self._result is None:
            return None
        # lmfit return the negative of the residual in its common
        # definition as (data - fit)
        return -self._result.residual

    @property
    def success(self):
        """Return the success value for the fit.

        :type: bool
        """
        if self._result is None:
            return None
        if not self._result.success:
            self._logger.warning(
                f'ier = {self._result.ier}: {self._result.message}')
            if (self._code == 'lmfit' and self._result.ier
                    and self._result.ier != 5):
                return True
        return self._result.success

    @property
    def var_names(self):
        """Return the variable names for the covarience matrix
        property.

        :type: list[str]
        """
        if self._result is None:
            return None
        return getattr(self._result, 'var_names', None)

    @property
    def y(self):
        """Return the input y-coordinates.

        :type: numpy.ndarray
        """
        return self._y

    def print_fit_report(self, result=None, show_correl=False):
        """Print a fit report.

        :param result: Model result, defaults to the `self._result`
            class attribute.
        :type result: ModelResult or lmfit.model.ModelResult, optional
        :param show_correl: Whether to show list of correlations,
            defaults to `False`.
        :type show_correl: bool, optional
        """
        if result is None:
            result = self._result
        if result is not None:
            print(result.fit_report(show_correl=show_correl))

    def add_parameter(self, parameter):
        """Add a fit parameter to the fit model.

        :param parameter: A new parameter to be added to the fit model.
        :type parameter: FitParameter
        """
        # Local modules
        from CHAP.utils.models import FitParameter

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

    def add_model(self, model, prefix):
        """Add a model component to the fit model.

        :param model: A fit model class.
        :type model: :attr:`~CHAP.utils.models.FitConfig.models`
        :param prefix: Model prefix.
        :type prefix: str
        """
        def _set_parameter_info_scipy(model, prefix):
            new_parameters = []
            for par in deepcopy(model.parameters):
                name = par.name
                self._parameters.add(par, prefix)
                if self._parameters[par.name].expr is None:
                    self._parameters[par.name].set(value=par.default)
                new_parameters.append(par.name)
                if name in model.LINEAR_PARAMETERS:
                    self._linear_parameters.append(par.name)
                elif name in model.MODEL_PARAMETERS:
                    self._model_parameters.append(par.name)
                elif name not in model.MODEL_IDENTIFIERS:
                    self._nonlinear_parameters.append(par.name)
            self._res_num_pars += [len(model.parameters)]
            return new_parameters

        def _set_parameter_info_lmfit(model, prefix):
            if model.model_type == 'expression':
                newmodel = model.lmfit_model(
                    prefix=prefix, parameters=self._parameters)
                # Remove already existing names
                for name in newmodel.param_names.copy():
                    if name not in expr_parameters:
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
                    kwargs['form'] = 'atan'
                newmodel = model.LMFITMODEL(prefix=prefix, **kwargs)
                for par in model.parameters:
                    name = par.name
                    nname = f'{prefix}{name}'
                    if name in model.LINEAR_PARAMETERS:
                        self._linear_parameters.append(nname)
                    elif name in model.MODEL_PARAMETERS:
                        self._model_parameters.append(nname)
                    elif name not in model.MODEL_IDENTIFIERS:
                        self._nonlinear_parameters.append(nname)
#                self._linear_parameters.extend(
#                        model.linear_parameters(prefix))
#                self._nonlinear_parameters.extend(
#                    model.nonlinear_parameters(prefix))
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
                        if value is not None:
                            value *= self._norm[1]
                        if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                            _min *= self._norm[1]
                        if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                            _max *= self._norm[1]
                        par.set(value=value, min=_min, max=_max)
                elif par.expr is None:
                    par.set(value=par.value)


        def _initialize_model_parameters(model, new_parameters, prefix):
            for parameter in deepcopy(model.parameters):
                name = parameter.name
                if name not in new_parameters:
                    name = prefix+name
                    if name not in new_parameters:
                        raise ValueError(f'Unable to match parameter {name}')
                if parameter.expr is None:
                    self._parameters[name].set(
                        value=parameter.value, min=parameter.min,
                        max=parameter.max, vary=parameter.vary)
                else:
                    if parameter.value is not None:
                        self._logger.warning(
                            'Ignoring input "value" for expression parameter'
                            f'{name} = {parameter.expr}')
                    if not np.isinf(parameter.min):
                        self._logger.warning(
                            'Ignoring input "min" for expression parameter'
                            f'{name} = {parameter.expr}')
                    if not np.isinf(parameter.max):
                        self._logger.warning(
                            'Ignoring input "max" for expression parameter'
                            f'{name} = {parameter.expr}')
                    self._parameters[name].set(
                        value=None, min=-np.inf, max=np.inf,
                        expr=parameter.expr)
                par = self._parameters[name]

        # Set model parameter info
        assert prefix is not None
        if self._code == 'scipy':
            new_parameters = _set_parameter_info_scipy(model, prefix)
            if self._model is None:
                self._model = Components()
            self._model |= {
                f'{prefix}{model.model_type}': Component(model, prefix)}
        else:
            newmodel = _set_parameter_info_lmfit(model, prefix)
            if self._model is None:
                self._model = newmodel
            else:
                self._model += newmodel
            new_parameters = newmodel.make_params()
            self._parameters += new_parameters

        # Scale default initial model parameters
        if self._norm is not None:
            _set_default_initial_parameters(new_parameters)

        # Initialize the model parameters
        _initialize_model_parameters(model, new_parameters, prefix)

    def eval(self, x, result=None):
        """Evaluate the best fit.

        :param x: x-coordinates.
        :type x: array-like, optional
        :param result: Model result, defaults to the `self._result`
            class attribute.
        :type result: ModelResult or lmfit.model.ModelResult, optional
        """
        if result is None:
            result = self._result
        if result is None:
            return None
        return result.eval(x=np.asarray(x))-self.normalization_offset

    def fit(self, config=None, **kwargs):
        """Fit the model to the input data.

        :param config: Fit configuration.
        :type config: CHAP.utils.models.FitConfig, optional
        :param \*\*kwargs: Additional key, value pairs to pass on
            directly to the core fit routine.
        """
        # Check input parameters
        if self._model is None:
            self._logger.error('Undefined fit model')
            return None
        self._mask = kwargs.pop('mask', None)
#        if 'try_linear_fit' in kwargs:
#            raise RuntimeError('try_linear_fit needs testing')
#            try_linear_fit = kwargs.pop('try_linear_fit')
#            if not isinstance(try_linear_fit, bool):
#                raise ValueError(
#                    'Invalid value of keyword argument try_linear_fit '
#                    f'({try_linear_fit})')
#            if not self._try_linear_fit:
#                self._logger.warning(
#                    'Ignore superfluous keyword argument "try_linear_fit" '
#                    '(not yet supported for callable models)')
#            else:
#                self._try_linear_fit = try_linear_fit

        # Setup the fit
        self._setup_fit(config)

        # Check if model is linear
        try:
            linear_model = self._check_linearity_model()
        except Exception:
            linear_model = False
        if kwargs.get('check_only_linearity') is not None:
            return linear_model

        # Normalize the data and initial parameters
        self._normalize()

        if linear_model:
            raise RuntimeError('linear solver needs testing')
            # Perform a linear fit by direct matrix solution with numpy
            try:
                if self._mask is None:
                    self._fit_linear_model(self._x, self._y_norm)
                else:
                    self._fit_linear_model(
                        self._x[~self._mask],
                        np.asarray(self._y_norm)[~self._mask])
            except Exception:
                linear_model = False
        if not linear_model:
            self._result = self._fit_nonlinear_model(
                self._x, self._y_norm, **kwargs)

        # Set internal parameter values to fit results upon success
        if self.success:
            for name, par in self._parameters.items():
                if par.expr is None and par.vary:
                    par.set(
                        value=self._result.params[name].value,
                        is_init_value=False)

        # Renormalize the data and results
        self._renormalize()

        return None

    def plot(
            self, x=None, y=None, *, y_title=None, title=None, result=None,
            skip_init=False, plot_comp=True, plot_comp_legends=False,
            plot_residual=False, plot_masked_data=True, **kwargs):
        """Plot the best fit.

        :param x: x-coordinates.
        :type x: array-like, optional
        :param y: y-coordinates.
        :type y: array-like, optional
        :param y_title: y-axis label.
        :type y_title: str, optional
        :param title: Graph title.
        :type title: str, optional
        :param result: Model result, defaults to the `self._result`
            class attribute.
        :type result: ModelResult or lmfit.model.ModelResult, optional
        :param skip_init: Skip plotting the initial guess, defaults
            to `False`.
        :type skip_init: bool, optional
        :param plot_comp: Plot the individual model components,
            defaults to `True`.
        :type plot_comp: bool, optional
        :param plot_comp_legends: Add a legend for the individual
            model components, defaults to `False`.
        :type plot_comp_legends: bool, optional
        :param plot_residual: Plot the residual, defaults to `False`.
        :type plot_residual: bool, optional
        :param plot_masked_data:
        :type plot_masked_data: bool, optional
        :param \*\*kwargs: Additional key, value pairs to pass on
            directly to the Matplotlib plot function.
        """
        if result is None:
            result = self._result
        if result is None:
            return
        plots = []
        legend = []
        if self._mask is None:
            mask = np.zeros(self._x.size).astype(bool)
            plot_masked_data = False
        else:
            mask = self._mask
        if x is not None:
            if not isinstance(x, (tuple, list, np.ndarray)):
                self._logger.warning(
                    'Ignoring invalid parameter x ({type(x)})')
            if len(x) != len(self._x):
                self._logger.warning(
                    'Ignoring parameter x in plot (wrong dimension)')
                x = None
        if x is None:
            x = self._x
        if y is not None:
            if not isinstance(y, (tuple, list, np.ndarray)):
                self._logger.warning(
                    'Ignoring invalid parameter y ({type(y)})')
            if len(y) != len(x):
                self._logger.warning(
                    'Ignoring parameter y in plot (wrong dimension)')
                y = None
        if y is not None:
            if y_title is None or not isinstance(y_title, str):
                y_title = 'data'
            plots += [(x, y, '.')]
            legend += [y_title]
        if self._y is not None:
            plots += [(x, np.asarray(self._y), 'b.')]
            legend += ['data']
            if plot_masked_data:
                plots += [(x[mask], np.asarray(self._y)[mask], 'bx')]
                legend += ['masked data']
        if isinstance(plot_residual, bool) and plot_residual:
            plots += [(x[~mask], result.residual, 'r-')]
            legend += ['residual']
        plots += [(x[~mask], result.best_fit, 'k-')]
        legend += ['best fit']
        if not skip_init and hasattr(result, 'init_fit'):
            plots += [(x[~mask], result.init_fit, 'g-')]
            legend += ['init']
        if plot_comp:
            components = result.eval_components(x=x[~mask])
            num_components = len(components)
            if 'tmp_normalization_offset_' in components:
                num_components -= 1
            if num_components > 1:
                eval_index = 0
                for modelname, y_comp in components.items():
                    if modelname == 'tmp_normalization_offset_':
                        continue
                    if modelname == '_eval':
                        modelname = f'eval{eval_index}'
                    if len(modelname) > 20:
                        modelname = f'{modelname[0:16]} ...'
                    if isinstance(y_comp, (int, float)):
                        y_comp *= np.ones(x[~mask].size)
                    plots += [(x[~mask], y_comp, '--')]
                    if plot_comp_legends:
                        if modelname[-1] == '_':
                            legend.append(modelname[:-1])
                        else:
                            legend.append(modelname)
        quick_plot(
            tuple(plots), legend=legend, title=title, block=True, **kwargs)

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

    def _create_prefixes(self, models):
        """Check for duplicate model names and create prefixes."""
        names = []
        prefixes = []
        for model in models:
            names.append(f'{model.prefix}{model.model_type}')
            prefixes.append(model.prefix)
        counts = Counter(names)
        for model, count in counts.items():
            if count > 1:
                n = 0
                for i, name in enumerate(names):
                    if name == model:
                        n += 1
                        prefixes[i] = f'{name}{n}_'

        return prefixes

    def _setup_fit_model(self, parameters, models):
        """Setup the fit model."""
        # Local modules
        from CHAP.utils.models import PEAK_LIKE_MODELS

        # Check for duplicate model names and create prefixes
        prefixes = self._create_prefixes(models)

        # Add the free fit parameters
        for par in parameters:
            self.add_parameter(par.model_dump())

        # Add the model functions
        for prefix, model in zip(prefixes, models):
            self.add_model(model, prefix)

        # Check linearity of free fit parameters
        for name in reversed(self._parameters):
            if (name not in (self._linear_parameters +
                             self._nonlinear_parameters +
                             self._model_parameters)
                    and not (model.model_type in PEAK_LIKE_MODELS
                        and ('height' in name or 'fwhm' in name))):
                for nname, par in self._parameters.items():
                    if par.expr is not None:
                        # Third party modules
                        from sympy import diff

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

    def _setup_fit(self, config):
        """Setup the fit."""
        def _setup_parameters_refit(config):
            # Local modules
            from CHAP.utils.models import (
                FitConfig,
                MultipeakModel,
            )

            # Expand multipeak model if present
            scale_factor = None
            for i, model in enumerate(deepcopy(config.models)):
                found_multipeak = False
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
                    config.models.remove(model)
                    found_multipeak = True

            # Check for duplicate model names and create prefixes
            prefixes = self._create_prefixes(config.models)
            if not isinstance(config, FitConfig):
                raise ValueError(f'Invalid parameter config ({config})')
            parameters = config.parameters
            for prefix, model in zip(prefixes, config.models):
                for par in model.parameters:
                    par.name = f'{prefix}{par.name}'
                parameters += model.parameters

            # Adjust parameters for refit as needed
            if isinstance(self, FitMap):
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

        # Apply mask if supplied:
        if self._mask is not None:
            raise RuntimeError('mask needs testing')
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(
                    f'Inconsistent x and mask dimensions ({self._x.size} vs '
                    f'{self._mask.size})')

        # Add constant offset for a normalized model
        if self._result is None and self._norm is not None and self._norm[0]:
            # Local modules
            from CHAP.utils.models import ConstantModel

            model = ConstantModel(
                model_type='constant',
                parameters=[{
                    'name': 'c',
                    'value': -self._norm[0],
                    'vary': False,
                }])
            self.add_model(model, 'tmp_normalization_offset_')

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
                    if (self._norm is None
                            or name in self._nonlinear_parameters):
                        self._parameters[name].set(value=1.0)
                    elif name not in self._model_parameters:
                        self._parameters[name].set(value=self._norm[1])

    def _check_linearity_model(self):
        """Identify the linearity of all model parameters and check if
        the model is linear or not.
        """
        # Third party modules
        from lmfit.models import ExpressionModel
        from sympy import diff

        raise RuntimeError('linear solver needs testing')
#        if not self._try_linear_fit:
#            self._logger.info(
#                'Skip linearity check (not yet supported for callable models)')
#            return False
        free_parameters = \
            [name for name, par in self._parameters.items() if par.vary]
        for component in self._model.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            if isinstance(component, ExpressionModel):
                for name in free_parameters:
                    if diff(component.expr, name, name):
                        self._nonlinear_parameters.append(name)
                        if name in self._linear_parameters:
                            self._linear_parameters.remove(name)
            else:
                model_parameters = component.param_names.copy()
                for basename, hint in component.param_hints.items():
                    name = f'{component.prefix}{basename}'
                    if hint.get('expr') is not None:
                        model_parameters.remove(name)
                for name in model_parameters:
                    expr = self._parameters[name].expr
                    if expr is not None:
                        for nname in free_parameters:
                            if name in self._nonlinear_parameters:
                                if diff(expr, nname):
                                    self._nonlinear_parameters.append(nname)
                                    if nname in self._linear_parameters:
                                        self._linear_parameters.remove(nname)
                            else:
                                assert name in self._linear_parameters
                                if diff(expr, nname, nname):
                                    self._nonlinear_parameters.append(nname)
                                    if nname in self._linear_parameters:
                                        self._linear_parameters.remove(nname)
        if any(True for name in self._nonlinear_parameters
                if self._parameters[name].vary):
            return False
        return True

    def _fit_linear_model(self, x, y):
        """Perform a linear fit by direct matrix solution with numpy.
        """
        # Third party modules
        from asteval import Interpreter
        from lmfit.model import ModelResult
        from lmfit.models import (
            ConstantModel,
            LinearModel,
            QuadraticModel,
            ExpressionModel,
        )
        # Third party modules
        from sympy import (
            diff,
            simplify,
        )

        # FIX self._parameter_norms
        # pylint: disable=no-member
        raise RuntimeError
        # Construct the matrix and the free parameter vector
        free_parameters = \
            [name for name, par in self._parameters.items() if par.vary]
        expr_parameters = {
            name:par.expr for name, par in self._parameters.items()
            if par.expr is not None}
        model_parameters = []
        for component in self._model.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            model_parameters += component.param_names
            for basename, hint in component.param_hints.items():
                name = f'{component.prefix}{basename}'
                if hint.get('expr') is not None:
                    expr_parameters.pop(name)
                    model_parameters.remove(name)
        norm = 1.0
        if self._normalized:
            norm = self._norm[1]
        # Add expression parameters to asteval
        ast = Interpreter()
        for name, expr in expr_parameters.items():
            ast.symtable[name] = expr
        # Add constant parameters to asteval
        # (renormalize to use correctly in evaluation of expression
        #     models)
        for name, par in self._parameters.items():
            if par.expr is None and not par.vary:
                if self._parameter_norms[name]:
                    ast.symtable[name] = par.value*norm
                else:
                    ast.symtable[name] = par.value
        mat_a = np.zeros((len(x), len(free_parameters)), dtype='float64')
        y_const = np.zeros(len(x), dtype='float64')
        have_expression_model = False
        for component in self._model.components:
            if isinstance(component, ConstantModel):
                name = component.param_names[0]
                if name in free_parameters:
                    mat_a[:,free_parameters.index(name)] = 1.0
                else:
                    if self._parameter_norms[name]:
                        delta_y_const = \
                            self._parameters[name] * np.ones(len(x))
                    else:
                        delta_y_const = \
                            (self._parameters[name]*norm) * np.ones(len(x))
                    y_const += delta_y_const
            elif isinstance(component, ExpressionModel):
                have_expression_model = True
                const_expr = component.expr
                for name in free_parameters:
                    dexpr_dname = diff(component.expr, name)
                    if dexpr_dname:
                        const_expr = \
                            f'{const_expr}-({str(dexpr_dname)})*{name}'
                        if not self._parameter_norms[name]:
                            dexpr_dname = f'({dexpr_dname})/{norm}'
                        y_expr = [(lambda _: ast.eval(str(dexpr_dname)))
                                  (ast(f'x={v}')) for v in x]
                        if ast.error:
                            raise ValueError(
                                f'Unable to evaluate {dexpr_dname}')
                        mat_a[:,free_parameters.index(name)] += y_expr
                const_expr = str(simplify(f'({const_expr})/{norm}'))
                delta_y_const = [(lambda _: ast.eval(const_expr))
                                 (ast(f'x = {v}')) for v in x]
                y_const += delta_y_const
                if ast.error:
                    raise ValueError(f'Unable to evaluate {const_expr}')
            else:
                free_model_parameters = [
                    name for name in component.param_names
                    if name in free_parameters or name in expr_parameters]
                if not free_model_parameters:
                    y_const += component.eval(params=self._parameters, x=x)
                elif isinstance(component, LinearModel):
                    name = f'{component.prefix}slope'
                    if name in free_model_parameters:
                        mat_a[:,free_parameters.index(name)] = x
                    else:
                        y_const += self._parameters[name].value * x
                    name = f'{component.prefix}intercept'
                    if name in free_model_parameters:
                        mat_a[:,free_parameters.index(name)] = 1.0
                    else:
                        y_const += self._parameters[name].value \
                            * np.ones(len(x))
                elif isinstance(component, QuadraticModel):
                    name = f'{component.prefix}a'
                    if name in free_model_parameters:
                        mat_a[:,free_parameters.index(name)] = x**2
                    else:
                        y_const += self._parameters[name].value * x**2
                    name = f'{component.prefix}b'
                    if name in free_model_parameters:
                        mat_a[:,free_parameters.index(name)] = x
                    else:
                        y_const += self._parameters[name].value * x
                    name = f'{component.prefix}c'
                    if name in free_model_parameters:
                        mat_a[:,free_parameters.index(name)] = 1.0
                    else:
                        y_const += self._parameters[name].value \
                            * np.ones(len(x))
                else:
                    # At this point each build-in model must be
                    #     strictly proportional to each linear model
                    #     parameter. Without this assumption, the model
                    #     equation is needed
                    # For the current build-in lmfit models, this can
                    #     only ever be the amplitude
                    assert len(free_model_parameters) == 1
                    name = f'{component.prefix}amplitude'
                    assert free_model_parameters[0] == name
                    assert self._parameter_norms[name]
                    expr = self._parameters[name].expr
                    if expr is None:
                        parameters = deepcopy(self._parameters)
                        parameters[name].set(value=1.0)
                        mat_a[:,free_parameters.index(name)] += component.eval(
                            params=parameters, x=x)
                    else:
                        const_expr = expr
                        parameters = deepcopy(self._parameters)
                        parameters[name].set(value=1.0)
                        dcomp_dname = component.eval(params=parameters, x=x)
                        for nname in free_parameters:
                            dexpr_dnname = diff(expr, nname)
                            if dexpr_dnname:
                                assert self._parameter_norms[name]
                                y_expr = np.asarray(
                                    dexpr_dnname*dcomp_dname, dtype='float64')
                                if self._parameter_norms[nname]:
                                    mat_a[:,free_parameters.index(nname)] += \
                                        y_expr
                                else:
                                    mat_a[:,free_parameters.index(nname)] += \
                                        y_expr/norm
                                const_expr = \
                                    f'{const_expr}-({dexpr_dnname})*{nname}'
                        const_expr = str(simplify(f'({const_expr})/{norm}'))
                        y_expr = [
                            (lambda _: ast.eval(const_expr))(ast(f'x = {v}'))
                            for v in x]
                        delta_y_const = np.multiply(y_expr, dcomp_dname)
                        y_const += delta_y_const
        solution, _, _, _ = np.linalg.lstsq(
            mat_a, y-y_const, rcond=None)

        # Assemble result
        # (compensate for normalization in expression models)
        for name, value in zip(free_parameters, solution):
            self._parameters[name].set(value=value)
        if (self._normalized
                and (have_expression_model or expr_parameters)):
            for name, norm in self._parameter_norms.items():
                par = self._parameters[name]
                if par.expr is None and norm and 'fraction' not in name:
                    self._parameters[name].set(value=par.value*self._norm[1])
        #RV FIX
        self._result = ModelResult(
            self._model, deepcopy(self._parameters), 'linear')
        self._result.best_fit = self._model.eval(params=self._parameters, x=x)
        if (self._normalized
                and (have_expression_model or expr_parameters)):
            if 'tmp_normalization_offset_c' in self._parameters:
                offset = self._parameters['tmp_normalization_offset_c']
            else:
                offset = 0.0
            self._result.best_fit = \
                (self._result.best_fit-offset-self._norm[0]) / self._norm[1]
            if self._normalized:
                for name, norm in self._parameter_norms.items():
                    par = self._parameters[name]
                    if par.expr is None and norm and 'fraction' not in name:
                        value = par.value/self._norm[1]
                        self._parameters[name].set(value=value)
                        self._result.params[name].set(value=value)
        self._result.residual = y-self._result.best_fit
        self._result.components = self._model.components
        self._result.init_params = None

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
            init_params = deepcopy(self._parameters)
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
                self._model, self._parameters, x=x, y=y, method=self._method,
                ast=self._ast, res_par_exprs=self._res_par_exprs,
                res_par_indices=res_par_indices,
                res_par_names=self._res_par_names, result=result)
            model_result.init_params = init_params
            model_result.init_values = {}
            for name, par in init_params.items():
                model_result.init_values[name] = par.value
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
            #FIX mask not implemented and tested
            assert self._mask is None
            return _fit_scipy(x, y, have_bounds, **kwargs)
#        fit_kws = {}
#        if 'Dfun' in kwargs:
#            fit_kws['Dfun'] = kwargs.pop('Dfun')
        return self._model.fit(
            y, self._parameters, x=x, method=self._method, #fit_kws=fit_kws,
            **kwargs)

    def _normalize(self):
        """Normalize the data and initial parameters."""
        if self._normalized:
            return
        if self._norm is None:
            if self._y is not None and self._y_norm is None:
                self._y_norm = np.asarray(self._y)
        else:
            if self._y is not None and self._y_norm is None:
                self._y_norm = \
                    (np.asarray(self._y)-self._norm[0]) / self._norm[1]
            self._y_range = 1.0
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
            self._normalized = True

    def _renormalize(self):
        """Renormalize the data and results."""
        if self._norm is None or not self._normalized:
            return
        self._normalized = False
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
                par.init_value *= self._norm[1]
                par.set(value=value, min=_min, max=_max, is_init_value=False)
        if self._result is None:
            return
        self._result.best_fit = (
            self._result.best_fit*self._norm[1] + self._norm[0])
        if hasattr(self._result, 'init_fit'):
            self._result.init_fit = (
                self._result.init_fit*self._norm[1] + self._norm[0])
        init_values = {}
        if hasattr(self._result, 'init_values'):
            for name, value in self._result.init_values.items():
                if name in self._linear_parameters:
                    init_values[name] = value*self._norm[1]
                else:
                    init_values[name] = value
            self._result.init_values = init_values
        if (hasattr(self._result, 'init_params')
                and self._result.init_params is not None):
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
        for name, par in self._result.params.items():
            par.init_value = init_values.get(name)
            if name in self._linear_parameters:
                if par.stderr is not None:
                    if self._code == 'scipy':
                        setattr(
                            par, '_stderr', par.stderr*self._norm[1])
                    else:
                        par.stderr *= self._norm[1]
                if par.expr is None:
                    value = par.value*self._norm[1]
#                    if par.init_value is not None:
#                        if self._code == 'scipy':
#                            setattr(par, '_init_value',
#                                    par.init_value*self._norm[1])
#                        else:
#                            par.init_value *= self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(
                        value=value, min=_min, max=_max, is_init_value=False)
        # Don't renormalize chisqr, it has no useful meaning in
        #     physical units
#        self._result.chisqr *= self._norm[1]*self._norm[1]
        if self._result.covar is not None:
            for i, name in enumerate(self._result.var_names):
                if name in self._linear_parameters:
                    norm_sq = self._norm[1]**2
                    for j in range(len(self._result.var_names)):
                        if self._result.covar[i,j] is not None:
                            self._result.covar[i,j] *= norm_sq
                        if self._result.covar[j,i] is not None:
                            self._result.covar[j,i] *= norm_sq
        # Don't renormalize redchi, it has no useful meaning in
        #     physical units
#        self._result.redchi *= self._norm[1]*self._norm[1]
        if self._result.residual is not None:
            self._result.residual *= self._norm[1]

    def _reset_par_at_boundary(self):
        fraction = 0.02
        for name, par in self._parameters.items():
            if par.vary:
                value = par.value
                _min = self._parameter_bounds[name]['min']
                _max = self._parameter_bounds[name]['max']
                if np.isinf(_min):
                    if not np.isinf(_max):
                        if name in self._linear_parameters:
                            upp = _max - fraction*self._y_range
                        elif _max == 0.0:
                            upp = _max - fraction
                        else:
                            upp = _max - fraction*abs(_max)
                        if value >= upp:
                            par.set(value=upp)
                else:
                    if np.isinf(_max):
                        if name in self._linear_parameters:
                            low = _min + fraction*self._y_range
                        elif _min == 0.0:
                            low = _min + fraction
                        else:
                            low = _min + fraction*abs(_min)
                        if value <= low:
                            par.set(value=low)
                    else:
                        low = (1.0-fraction)*_min + fraction*_max
                        upp = fraction*_min + (1.0-fraction)*_max
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


class FitMap(Fit):
    """Wrapper to the Fit class to fit data on a N-dimensional map."""

    def __init__(self, y, config, logger, x=None):
        """Initialize FitMap.

        :param y: Input signal data.
        :type y: array-like
        :param config: Fit configuration.
        :type config: CHAP.utils.models.FitConfig
        :param logger: A python Logger object.
        :type logger: logging.Logger
        :param x: Input coordinate data.
        :type x: array-like, optional
        """
        super().__init__(None, config, logger)
        self._abs_height_cutoff = None
        self._best_errors = None
        self._best_fit = None
        self._best_parameters = None
        self._best_values = None
        self._best_vary = None
        self._init_values = None
        self._inv_transpose = None
        self._max_nfev = None
        self._memfolder = config.memfolder
        self._multipeak_info = None
        self._new_parameters = None
        self._num_func_eval = None
        self._out_of_bounds = None
        self._plot = False
        self._print_report = False
        self._redchi = None
        self._redchi_cutoff = 0.1
        self._rel_height_cutoff = None
        self._skip_init = True
        self._success = None
        self._try_no_bounds = True

        # At this point the fastest index should always be the signal
        #     dimension so that the slowest ndim-1 dimensions are the
        #     map dimensions
        self._ymap = y
        if x is None:
            self._x = np.arange(self._ymap.shape[-1])
        else:
            self._x = x
            assert self._x.size == self._ymap.shape[-1]

        # Flatten the map
        # Store the flattened map in self._ymap_norm
        self._map_dim = int(self._ymap.size/self._x.size)
        self._map_shape = self._ymap.shape[:-1]
        self._ymap_norm = np.reshape(
            self._ymap, (self._map_dim, self._x.size))

        # Check if a mask is provided
#        if 'mask' in kwargs:
#            self._mask = kwargs.pop('mask')
        if True: #self._mask is None:
            ymap_min = float(self._ymap_norm.min())
            ymap_max = float(self._ymap_norm.max())
        else:
            ymap_min = None
            ymap_max = None
#            self._mask = np.asarray(self._mask).astype(bool)
#            if self._x.size != self._mask.size:
#                raise ValueError(
#                    f'Inconsistent mask dimension ({self._x.size} vs '
#                    f'{self._mask.size})')
#            ymap_masked = np.asarray(self._ymap_norm)[:,~self._mask]
#            ymap_min = float(ymap_masked.min())
#            ymap_max = float(ymap_masked.max())

        # Normalize the data
        self._y_range = ymap_max-ymap_min
        if self._y_range > 0.0:
            self._norm = (ymap_min, self._y_range)
            self._ymap_norm = (self._ymap_norm-self._norm[0])/self._norm[1]
        else:
            self._redchi_cutoff *= self._y_range**2

        # Setup fit model
        self._setup_fit_model(config.parameters, config.models)

    @property
    def best_errors(self):
        """Return errors in the best fit parameters.

        :type: numpy.ndarray
        """
        return self._best_errors

    @property
    def best_fit(self):
        """Return the best fits.

        :type: numpy.ndarray
        """
        return self._best_fit

    @property
    def best_values(self):
        """Return values for the best fit parameters.

        :type: numpy.ndarray
        """
        return self._best_values

    @property
    def best_vary(self):
        """Return vary parameters for the best fit parameters.

        :type: numpy.ndarray
        """
        return self._best_vary

    @property
    def chisqr(self):
        """Return the chisqr value for each best fit.

        :type: numpy.ndarray
        """
        self._logger.warning('Undefined property chisqr')

    @property
    def components(self):
        """Return the fit model components info.

        :type: dict
        """
        # Third party modules
        from lmfit.models import ExpressionModel

        components = {}
        if self._result is None:
            self._logger.warning(
                'Unable to collect components in FitMap.components')
            return components
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
                        'value': self.init_parameters[name]['value'],
                    }
            expr = None
            if isinstance(component, ExpressionModel):
                name = component._name
                if name[-1] == '_':
                    name = name[:-1]
                expr = component.expr
            else:
                if prefix := component.prefix:
                    if prefix[-1] == '_':
                        prefix = prefix[:-1]
                    name = f'{prefix} ({component._name})'
                else:
                    name = f'{component._name}'
            if expr is None:
                components[name] = {'parameters': parameters}
            else:
                components[name] = {'expr': expr, 'parameters': parameters}
        return components

    @property
    def covar(self):
        """Return the covarience matrices for the best fit parameters.

        :type: numpy.ndarray
        """
        self._logger.warning('Undefined property covar')

    @property
    def init_values(self):
        """Return init values of the fit parameters.

        :type: numpy.ndarray
        """
        return self._init_values

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
    def redchi(self):
        """Return the redchi value for each best fit.

        :type: numpy.ndarray
        """
        return self._redchi

    @property
    def residual(self):
        """Return the residual in each best fit.

        :type: numpy.ndarray
        """
        if self.best_fit is None:
            return None
        if self._mask is None:
            residual = np.asarray(self._ymap)-self.best_fit
        else:
            ymap_flat = np.reshape(
                np.asarray(self._ymap), (self._map_dim, self._x.size))
            ymap_flat_masked = ymap_flat[:,~self._mask]
            ymap_masked = np.reshape(
                ymap_flat_masked,
                list(self._map_shape) + [ymap_flat_masked.shape[-1]])
            residual = ymap_masked-self.best_fit
        return residual

    @property
    def success(self):
        """Return the success value for each fit.

        :type: bool
        """
        return self._success

    @property
    def var_names(self):
        """Return the variable names for the covarience matrix
        property.

        :type: list[str]
        """
        self._logger.warning('Undefined property var_names')

    @property
    def y(self):
        """Return the input y-coordinates.

        :type: numpy.ndarray
        """
        self._logger.warning('Undefined property y')

    @property
    def ymap(self):
        """Return the input y-coordinates map.

        :type: numpy.ndarray
        """
        return self._ymap

    def best_parameters(self, dims=None):
        """Return the best fit parameters.

        :param dims: Map indices of the best fit parameters to return,
            defaults to `None` which will return the list of the best
            parameter names in the correct order for the results.
        :type dims: int or list or tuple, optional
        :return: Best fit parameters.
        :rtype: list[str] or dict
        """
        if dims is None:
            return self._best_parameters
# FIX use something else, self._best_parameters is "reserved" to get the
# parameters in the EDD strain analysis and must return the order of the
# parameters in self.best_values, self.best_errors, and self_init_values
#            parameters_dict = {}
#            for i, name in enumerate(self._best_parameters):
#                parameters_dict[name] = {
#                    'errors': self._best_errors[i],
#                    'init_values': self._init_values[i],
#                    'values': self._best_values[i],
#                }
#            return parameters_dict
        if (self.best_errors is None or self.best_values is None
                or self.init_values is None):
            self._logger.warning('No data for best parameter values')
            return {}
        if isinstance(dims, int):
            dims = (dims,)
        elif isinstance(dims, (list, tuple)):
            dims = tuple(dims)
        else:
            raise ValueError(f'Invalid parameter dims ({dims})')
        # Create current parameters
        parameters = deepcopy(self._parameters)
        for n, name in enumerate(self._best_parameters):
            if self._parameters[name].vary:
                parameters[name].set(
                    value=self.best_values[n][dims], is_init_value=False)
            parameters[name].init_value = self.init_values[n][dims]
            parameters[name].stderr = self.best_errors[n][dims]
        parameters_dict = {}
        for name in sorted(parameters):
            if name != 'tmp_normalization_offset_c':
                par = parameters[name]
                parameters_dict[name] = {
                    'error': par.stderr,
                    'expr': par.expr,
                    'init_value': par.init_value,
                    'min': par.min,
                    'max': par.max,
                    'value': par.value,
                    'vary': par.vary,
                }
        return parameters_dict

    def init_parameters(self, dims=None):
        """Return the initial fit parameters.

        :param dims: Map indices of the initial fit parameters to
            return, defaults to `None` which will return the list of
            the best parameter names in the correct order for the
            results.
        :type dims: int or list or tuple, optional
        :return: Initial fit parameters.
        :rtype: list[str] or dict
        """
        parameters_dict = self.best_parameters(dims)
        if dims is None:
            return parameters_dict
        for name, par in parameters_dict.items():
            par.pop('error')
            par['value'] = par.pop('init_value')
        return parameters_dict

    def freemem(self):
        """Free memory allocated for parallel processing."""
        if self._memfolder is None:
            return
        try:
            rmtree(self._memfolder)
        except Exception:
            self._logger.warning('Could not clean-up automatically.')

    def plot(
            self, x=None, dims=None, *, y_title=None, plot_comp_legends=False,
            plot_residual=False, plot_masked_data=True, **kwargs):
        """Plot the best fits.

        :param x: x-coordinates.
        :type x: array-like, optional
        :param dims: Map indices of the data point to plot,
            defaults to `None` which will plot the first data point.
        :type dims: list or tuple, optional
        :param y_title: y-axis label.
        :type y_title: str, optional
        :param plot_comp_legends: Add a legend for the individual
            model components, defaults to `False`.
        :type plot_comp_legends: bool, optional
        :param plot_residual: Plot the residual, defaults to `False`.
        :type plot_residual: bool, optional
        :param plot_masked_data:
        :type plot_masked_data: bool, optional
        :param \*\*kwargs: Additional key, value pairs to pass on
            directly to the Matplotlib plot function.
        """
        # Third party modules
        from lmfit.models import ExpressionModel

        if x is not None:
            if not isinstance(x, (tuple, list, np.ndarray)):
                self._logger.warning(
                    'Ignoring invalid parameter x ({type(x)})')
            if len(x) != len(self._x):
                self._logger.warning(
                    'Ignoring parameter x in plot (wrong dimension)')
                x = None
        if x is None:
            x = self._x
        if dims is None:
            dims = [0]*len(self._map_shape)
        if (not isinstance(dims, (list, tuple))
                or len(dims) != len(self._map_shape)):
            raise ValueError('Invalid parameter dims ({dims})')
        dims = tuple(dims)
        if (self._result is None or self.best_fit is None
                or self.best_values is None):
            self._logger.warning(
                f'Unable to plot fit for dims = {dims}')
            return
        if y_title is None or not isinstance(y_title, str):
            y_title = 'data'
        if self._mask is None:
            mask = np.zeros(x.size).astype(bool)
            plot_masked_data = False
        else:
            mask = self._mask
        plots = [(x, np.asarray(self._ymap[dims]), 'b.')]
        legend = [y_title]
        if plot_masked_data:
            plots += \
                [(x[mask], np.asarray(self._ymap)[(*dims,mask)], 'bx')]
            legend += ['masked data']
        plots += [(x[~mask], self.best_fit[dims], 'k-')]
        legend += ['best fit']
        if plot_residual:
            plots += [(x[~mask], self.residual[dims], 'r--')]
            legend += ['residual']
        # Create current parameters
        parameters = deepcopy(self._parameters)
        for name in self._best_parameters:
            if self._parameters[name].vary:
                parameters[name].set(
                    value=self.best_values[self._best_parameters.index(name)]
                    [dims])
        for component in self._result.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            if isinstance(component, ExpressionModel):
                prefix = component._name
                if prefix[-1] == '_':
                    prefix = prefix[:-1]
                modelname = f'{prefix}: {component.expr}'
            else:
                if prefix := component.prefix:
                    if prefix[-1] == '_':
                        prefix = prefix[:-1]
                    modelname = f'{prefix} ({component._name})'
                else:
                    modelname = f'{component._name}'
            if len(modelname) > 20:
                modelname = f'{modelname[0:16]} ...'
            y = component.eval(params=parameters, x=x[~mask])
            if isinstance(y, (int, float)):
                y *= np.ones(x[~mask].size)
            plots += [(x[~mask], y, '--')]
            if plot_comp_legends:
                legend.append(modelname)
        quick_plot(
            tuple(plots), legend=legend, title=str(dims), block=True, **kwargs)

    def fit(self, config=None, **kwargs):
        """Fit the model to the input data.

        :param config: Fit configuration.
        :type config: CHAP.utils.models.FitConfig, optional
        :param \*\*kwargs: Additional key, value pairs to pass on
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
            self._print_report = kwargs.pop('print_report', False)
            self._redchi_cutoff = kwargs.pop('redchi_cutoff', 0.1)
            self._rel_height_cutoff = kwargs.pop('rel_height_cutoff')
            self._skip_init = kwargs.pop('skip_init', True)
            self._try_no_bounds = kwargs.pop('try_no_bounds', False)
        else:
            num_proc = config.num_proc
            self._abs_height_cutoff = config.abs_height_cutoff
            self._plot = config.plot
            self._print_report = config.print_report
#            self._redchi_cutoff = config.redchi_cutoff
            self._rel_height_cutoff = config.rel_height_cutoff
#            self._skip_init = config.skip_init
#            self._try_no_bounds = config.try_no_bounds
        if num_proc > 1 and not HAVE_JOBLIB:
            self._logger.warning(
                'Missing joblib in the conda environment, running serially')
            num_proc = 1
        if num_proc > num_proc_max:
            self._logger.warning(
                f'The requested number of processors ({num_proc}) exceeds the '
                'maximum allowed number of processors, num_proc reduced to '
                f'{num_proc_max}')
            num_proc = num_proc_max
        self._logger.info(f'Using {num_proc} processors to fit the data')
        self._redchi_cutoff *= self._y_range**2

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
            if self._norm is not None:
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
                (self._map_dim, x_size), dtype=self._ymap_norm.dtype)
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
                filename_memmap, dtype=self._ymap_norm.dtype,
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
                    self._logger.warning(
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

        # Renormalize the initial parameters for external use
        if self._norm is not None and self._normalized:
            if hasattr(self._result, 'init_values'):
                init_values = {}
                for name, value in self._result.init_values.items():
                    if (name in self._nonlinear_parameters
                            or self._parameters[name].expr is not None):
                        init_values[name] = value
                    elif 'fraction' not in name:
                        init_values[name] = value*self._norm[1]
                    else:
                        raise RuntimeError('must check, should not be here')
                self._result.init_values = init_values
            if (hasattr(self._result, 'init_params')
                    and self._result.init_params is not None):
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
                    if self._code == 'scipy':
                        setattr(par, '_init_value', par.value)
                    else:
                        par.init_value = par.value

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
        self._normalized = False
        if self._norm is not None:
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
                # RV FIX
                #if self._code == 'scipy':
                #    setattr(par, '_init_value', par.init_value*self._norm[1])

        if num_proc > 1:
            # Free the shared memory
            self.freemem()

    def _fit_parallel(self, current_best_values, num, n_start, **kwargs):
        num = min(num, self._map_dim-n_start)
        for n in range(num):
            self._fit(n_start+n, current_best_values, **kwargs)

    def _fit(self, n, current_best_values, return_result=False, **kwargs):
        # Do not attempt a fit if the data is zero or entirely below
        # the cutoff
        y_max = self._ymap_norm[n].max()
        if (y_max == 0.0
                or (self._normalized and self._abs_height_cutoff is not None
                    and (y_max*self._norm[1] + self._norm[0]
                         < self._abs_height_cutoff))):
#            print(f'\t------->skipping n={n}!!!!!!!!')
            self._logger.debug(f'Skipping fit {n} (rel norm = {y_max:.5f})')
            if self._code == 'scipy':
                # Local modules
                from CHAP.utils.fit import ModelResult

                result = ModelResult(self._model, deepcopy(self._parameters))
            else:
                # Third party modules
                from lmfit.model import ModelResult

                result = ModelResult(self._model, deepcopy(self._parameters))
            result.success = False
            # Renormalize the data and results
            self._renormalize(n, result)
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
            use_peaks, _, peak_heights, peak_widths = \
                self.guess_init_peak(
                    self._x, self._ymap_norm[n], centers, centers_range,
                    centers_range_fraction,
                    min_height=y_max*self._rel_height_cutoff,
                    min_width=5)

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
            # do not do it by default (add a kwarg to FitMap.fit())
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
            dims = np.unravel_index(n, self._map_shape)
            if self._inv_transpose is not None:
                dims = tuple(
                    dims[self._inv_transpose[i]] for i in range(len(dims)))
            super().plot(
                result=result, y=np.asarray(self._ymap[dims]),
                plot_comp_legends=True, skip_init=self._skip_init,
                title=str(dims))

        if return_result:
            return result
        return None

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
        result = self._fit_nonlinear_model(
            self._x, self._ymap_norm[n], **kwargs)
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
#            print(f'\t------->refitting n={n} after out_of_bounds!!!!!!!!')
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
                self._x, self._ymap_norm[n], **kwargs)
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

    def _renormalize(self, n, result):
        self._success_flat[n] = result.success
        if result.success:
            self._redchi_flat[n] = np.float64(result.redchi)
        if self._norm is None or not self._normalized:
            for i, name in enumerate(self._best_parameters):
                self._best_errors_flat[i][n] = np.float64(
                    result.params[name].stderr)
                self._best_values_flat[i][n] = np.float64(
                    result.params[name].value)
                self._best_vary_flat[i][n] = (
                    result.params[name].vary and result.success)
                self._init_values_flat[i][n] = np.float64(
                    result.init_params[name].value)
            if result.success:
                self._best_fit_flat[n] = result.best_fit
        else:
            for name, par in result.params.items():
                init_value = result.init_params[name].value
                if init_value is not None:
                    par.init_value = init_value*self._norm[1]
                if name in self._linear_parameters:
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
                self._best_fit_flat[n] = (
                    result.best_fit*self._norm[1] + self._norm[0])
                if self._plot:
                    if not self._skip_init:
                        result.init_fit = (
                            result.init_fit*self._norm[1] + self._norm[0])
                    result.best_fit = np.copy(self._best_fit_flat[n])
