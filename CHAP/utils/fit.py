#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : fit.py
Author     : Rolf Verberg <rolfverberg AT gmail dot com>
Description: General curve fitting module
"""

# System modules
from collections import Counter
from copy import deepcopy
from logging import getLogger
from os import (
    cpu_count,
    mkdir,
    path,
)
from re import sub
from shutil import rmtree
from sys import float_info
#from time import time

# Third party modules
try:
    from joblib import (
        Parallel,
        delayed,
    )
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False
from nexusformat.nexus import NXdata
import numpy as np

# Local modules
from CHAP.processor import Processor
from CHAP.utils.general import (
#    is_int,
    is_index,
    index_nearest,
    quick_plot,
)

logger = getLogger(__name__)
FLOAT_MIN = float_info.min
FLOAT_MAX = float_info.max
FLOAT_EPS = float_info.epsilon

# sigma = fwhm_factor*fwhm
fwhm_factor = {
    'gaussian': 'fwhm/(2*sqrt(2*log(2)))',
    'lorentzian': '0.5*fwhm',
    'splitlorentzian': '0.5*fwhm',  # sigma = sigma_r
    'voight': '0.2776*fwhm',        # sigma = gamma
    'pseudovoight': '0.5*fwhm',     # fraction = 0.5
}

# amplitude = height_factor*height*fwhm
height_factor = {
    'gaussian': 'height*fwhm*0.5*sqrt(pi/log(2))',
    'lorentzian': 'height*fwhm*0.5*pi',
    'splitlorentzian': 'height*fwhm*0.5*pi',  # sigma = sigma_r
    'voight': '3.334*height*fwhm',            # sigma = gamma
    'pseudovoight': '1.268*height*fwhm',      # fraction = 0.5
}


class FitProcessor(Processor):
    """A processor to perform a fit on a data set or data map. """
    def process(self, data, config=None):
        """Fit the data and return a CHAP.utils.fit.Fit or
        CHAP.utils.fit.FitMap object depending on the dimensionality
        of the input data. The input data should contain a NeXus NXdata 
        object, with properly defined signal and axis.

        :param data: Input data containing the
            nexusformat.nexus.NXdata object to fit.
        :type data: list[PipelineData]
        :param config: Fit configuration.
        :type config: dict, optional
        :raises ValueError: Invalid input or configuration parameter.
        :return: The fitted data object.
        :rtype: Union[CHAP.utils.fit.Fit, CHAP.utils.fit.FitMap]
        """
        # Local modules
        from CHAP.utils.models import (
            FitConfig,
            Multipeak,
        )

        # Unwrap the PipelineData if called as a Pipeline Processor
        if not isinstance(data, (Fit, FitMap)) and not isinstance(data, NXdata):
            data = self.unwrap_pipelinedata(data)[0]

        if isinstance(data, (Fit, FitMap)):

            # Refit/continue the fit with possibly updated parameters
            fit = data
            fit_config = None
            if config is not None:
                try:
                    fit_config = FitConfig(**config)
                except Exception as exc:
                    raise RuntimeError from exc

            if isinstance(data, FitMap):
                fit.fit(config=fit_config)
            else:
                fit.fit(config=fit_config)
                if fit_config is not None:
                    if fit_config.print_report:
                        fit.print_fit_report()
                    if fit_config.plot:
                        fit.plot(skip_init=True)

        else:

            # Get the default NXdata object
            try:
                nxdata = data.get_default()
                assert nxdata is not None
            except Exception as exc:
                if nxdata is None or nxdata.nxclass != 'NXdata':
                    raise ValueError(
                        'Invalid default pathway to an NXdata '
                        f'object in ({data})') from exc

            # Get the validated fit configuration
            fit_config = self.get_config(
                data=data, config=config, schema='utils.models.FitConfig')

            # Expand multipeak model if present
            found_multipeak = False
            for i, model in enumerate(deepcopy(fit_config.models)):
                if isinstance(model, Multipeak):
                    if found_multipeak:
                        raise ValueError(
                            f'Invalid parameter models ({fit_config.models}) '
                            '(multiple instances of multipeak not allowed)')
                    parameters, models = self.create_multipeak_model(model)
                    if parameters:
                        fit_config.parameters += parameters
                    fit_config.models += models
                    fit_config.models.pop(i)
                    found_multipeak = True

            # Instantiate the Fit or FitMap object and fit the data
            if np.squeeze(nxdata.nxsignal).ndim == 1:
                fit = Fit(nxdata, fit_config)
                fit.fit()
                if fit_config.print_report:
                    fit.print_fit_report()
                if fit_config.plot:
                    fit.plot(skip_init=True)
            else:
                fit = FitMap(nxdata, fit_config)
                fit.fit(
                    rel_height_cutoff=fit_config.rel_height_cutoff,
                    num_proc=fit_config.num_proc, plot=fit_config.plot,
                    print_report=fit_config.print_report)

        return fit

    @staticmethod
    def create_multipeak_model(model_config):
        """Create a multipeak model."""
        # Local modules
        from CHAP.utils.models import (
            FitParameter,
            Gaussian,
        )

        parameters = []
        models = []
        num_peak = len(model_config.centers)
        if num_peak == 1 and model_config.fit_type == 'uniform':
            logger.debug('Ignoring fit_type input for fitting one peak')
            model_config.fit_type = 'unconstrained'

        sig_min = FLOAT_MIN
        sig_max = np.inf
        if (model_config.fwhm_min is not None
                or model_config.fwhm_max is not None):
            # Third party modules
            from asteval import Interpreter
            ast = Interpreter()

            if model_config.fwhm_min is not None:
                ast(f'fwhm = {model_config.fwhm_min}')
                sig_min = ast(fwhm_factor[model_config.peak_models])
            if model_config.fwhm_max is not None:
                ast(f'fwhm = {model_config.fwhm_max}')
                sig_max = ast(fwhm_factor[model_config.peak_models])

        if model_config.fit_type == 'uniform':
            parameters.append(FitParameter(
                name='scale_factor', value=1.0, min=FLOAT_MIN))
            if num_peak == 1:
                prefix = ''
            for i, cen in enumerate(model_config.centers):
                if num_peak > 1:
                    prefix = f'peak{i+1}_'
                models.append(Gaussian(
                    model='gaussian',
                    prefix=prefix,
                    parameters=[
                         {'name': 'amplitude', 'min': FLOAT_MIN},
                         {'name': 'center', 'expr': f'scale_factor*{cen}'},
                         {'name': 'sigma', 'min': sig_min, 'max': sig_max}]))
        else:
            if num_peak == 1:
                prefix = ''
            for i, cen in enumerate(model_config.centers):
                if num_peak > 1:
                    prefix = f'peak{i+1}_'
                if model_config.centers_range == 0:
                    models.append(Gaussian(
                        model='gaussian',
                        prefix=prefix,
                        parameters=[
                             {'name': 'amplitude', 'min': FLOAT_MIN},
                             {'name': 'center', 'value': cen, 'vary': False},
                             {'name': 'sigma', 'min': sig_min, 'max': sig_max}
                        ]))
                else:
                    if model_config.centers_range is None:
                        cen_min = None
                        cen_max = None
                    else:
                        cen_min = cen - model_config.centers_range
                        cen_max = cen + model_config.centers_range
                    models.append(Gaussian(
                        model='gaussian',
                        prefix=prefix,
                        parameters=[
                             {'name': 'amplitude', 'min': FLOAT_MIN},
                             {'name': 'center', 'value': cen, 'min': cen_min,
                              'max': cen_max},
                             {'name': 'sigma', 'min': sig_min, 'max': sig_max}
                        ]))

        return parameters, models


class Component():
    """A model fit component."""
    def __init__(self, model, prefix=''):
        # Local modules
        from CHAP.utils.models import models

        self.func = models[model.model]
        self.param_names = [f'{prefix}{par.name}' for par in model.parameters]
        self.prefix = prefix
        self._name = model.model


class Components(dict):
    """The dictionary of model fit components."""
    def __init__(self):
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
        """Return the model fit component dictionary."""
        return self.values()

    def add(self, model, prefix=''):
        """Add a model to the model fit component dictionary."""
        # Local modules
        from CHAP.utils.models import model_classes

        if not isinstance(model, model_classes):
            raise ValueError(f'Invalid parameter model ({model})')
        if not isinstance(prefix, str):
            raise ValueError(f'Invalid parameter prefix ({prefix})')
        name = f'{prefix}{model.model}'
        self.__setitem__(name, Component(model, prefix))


class Parameters(dict):
    """A dictionary of FitParameter objects, mimicking the
    functionality of a similarly named class in the lmfit library.
    """
    def __init__(self):
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
        :type parameter: Union[str, FitParameter]
        :param prefix: The prefix for the model to which this
             parameter belongs, defaults to `''`.
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
    similarly named class in the lmfit library.
    """
    def __init__(
            self, model, parameters, x=None, y=None, method=None, ast=None,
            res_par_exprs=None, res_par_indices=None, res_par_names=None,
            result=None):
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
            except:
                self.covar = None

        # Update the fit parameters with the fit result
        par_names = list(self.params.keys())
        self.var_names = []
        for i, (value, index) in enumerate(zip(best_pars, res_par_indices)):
            par = self.params[par_names[index]]
            par.set(value=value)
            stderr = None
            if self.covar is not None:
                stderr = self.covar[i,i]
                if stderr is not None:
                    if stderr < 0.0:
                        stderr = None
                    else:
                        stderr = np.sqrt(stderr)
            self.var_names.append(par.name)
        if res_par_exprs:
            # Third party modules
            from sympy import diff
            for value, name in zip(best_pars, res_par_names):
                self._ast.symtable[name] = value
            for par_expr in res_par_exprs:
                name = par_names[par_expr['index']]
                expr = par_expr['expr']
                par = self.params[name]
                par.set(value=self._ast.eval(expr))
                self._expr_pars[name] = expr
                stderr = None
                if self.covar is not None:
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
                    stderr = np.sqrt(stderr)
                setattr(par, '_stderr', stderr)

    def eval_components(self, x=None, parameters=None):
        """Evaluate each component of a composite model function.

        :param x: Independent variable, defaults to `None`, in which 
            case the class variable x is used.
        :type x: Union[list, np.ndarray], optional
        :param parameters: Composite model parameters, defaults to
            None, in which case the class variable params is used.
        :type parameters: Parameters, optional
        :return: A dictionary with component name and evealuated
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
            if component.prefix == '':
                name = component._name
            else:
                name = component.prefix
            result[name] = component.func(x, *par_values)
        return result

    def fit_report(self, show_correl=False):
        """Generates a report of the fitting results with their best
        parameter values and uncertainties.

        :param show_correl: Whether to show list of correlations,
            defaults to `False`.
        :type show_correl: bool, optional
        """
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
            if expr is not None:
                val = self._ast.eval(expr)
            else:
                val = par.value
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
    """
    Wrapper class for scipy/lmfit.
    """
    def __init__(self, nxdata, config):
        """Initialize Fit."""
        self._code = config.code
        for model in config.models:
            if model.model == 'expression' and self._code != 'lmfit':
                self._code = 'lmfit'
                logger.warning('Using lmfit instead of scipy with '
                                    'an expression model')
        if self._code == 'scipy':
            # Local modules
            from CHAP.utils.fit import Parameters
        else:
            # Third party modules
            from lmfit import Parameters
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
        if nxdata is not None:
            if isinstance(nxdata.attrs['axes'], str):
                dim_x = nxdata.attrs['axes']
            else:
                dim_x = nxdata.attrs['axes'][-1]
            self._x = np.asarray(nxdata[dim_x])
            self._y = np.squeeze(nxdata.nxsignal)
            if self._x.ndim != 1:
                raise ValueError(
                    f'Invalid x dimension ({self._x.ndim})')
            if self._x.size != self._y.size:
                raise ValueError(
                    f'Inconsistent x and y dimensions ({self._x.size} vs '
                    f'{self._y.size})')
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
        """Return errors in the best fit parameters."""
        if self._result is None:
            return None
        return {name:self._result.params[name].stderr
                for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def best_fit(self):
        """Return the best fit."""
        if self._result is None:
            return None
        return self._result.best_fit

    @property
    def best_parameters(self):
        """Return the best fit parameters."""
        if self._result is None:
            return None
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
        """Return values of the best fit parameters."""
        if self._result is None:
            return None
        return {name:self._result.params[name].value
                for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def chisqr(self):
        """Return the chisqr value of the best fit."""
        if self._result is None:
            return None
        return self._result.chisqr

    @property
    def components(self):
        """Return the fit model components info."""
        # Third party modules
        from lmfit.models import ExpressionModel

        components = {}
        if self._result is None:
            logger.warning('Unable to collect components in Fit.components')
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
                prefix = component.prefix
                if prefix:
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
        """Return the covarience matrix of the best fit parameters."""
        if self._result is None:
            return None
        return self._result.covar

    @property
    def init_parameters(self):
        """Return the initial parameters for the fit model."""
        if self._result is None or self._result.init_params is None:
            return None
        parameters = {}
        for name in sorted(self._result.init_params):
            if name != 'tmp_normalization_offset_c':
                par = self._result.init_params[name]
                parameters[name] = {
                    'value': par.value,
                    'min': par.min,
                    'max': par.max,
                    'vary': par.vary,
                    'expr': par.expr,
                }
        return parameters

    @property
    def init_values(self):
        """Return the initial values for the fit parameters."""
        if self._result is None or self._result.init_params is None:
            return None
        return {name:self._result.init_params[name].value
                for name in sorted(self._result.init_params)
                if name != 'tmp_normalization_offset_c'}

    @property
    def normalization_offset(self):
        """Return the normalization_offset for the fit model."""
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
        """
        if self._result is None:
            return None
        return self._result.nfev

    @property
    def parameters(self):
        """Return the fit parameter info."""
        return {name:{'min': par.min, 'max': par.max, 'vary': par.vary,
                'expr': par.expr} for name, par in self._parameters.items()
                if name != 'tmp_normalization_offset_c'}

    @property
    def redchi(self):
        """Return the redchi value of the best fit."""
        if self._result is None:
            return None
        return self._result.redchi

    @property
    def residual(self):
        """Return the residual in the best fit."""
        if self._result is None:
            return None
        # lmfit return the negative of the residual in its common
        # definition as (data - fit)
        return -self._result.residual

    @property
    def success(self):
        """Return the success value for the fit."""
        if self._result is None:
            return None
        if not self._result.success:
            logger.warning(
                f'ier = {self._result.ier}: {self._result.message}')
            if (self._code == 'lmfit' and self._result.ier
                    and self._result.ier != 5):
                return True
        return self._result.success

    @property
    def var_names(self):
        """Return the variable names for the covarience matrix
        property.
        """
        if self._result is None:
            return None
        return getattr(self._result, 'var_names', None)

    @property
    def x(self):
        """Return the input x-array."""
        return self._x

    @property
    def y(self):
        """Return the input y-array."""
        return self._y

    def print_fit_report(self, result=None, show_correl=False):
        """Print a fit report."""
        if result is None:
            result = self._result
        if result is not None:
            print(result.fit_report(show_correl=show_correl))

    def add_parameter(self, parameter):
        """Add a fit parameter to the fit model."""
        # Local modules
        from CHAP.utils.models import FitParameter

        if parameter.get('expr') is not None:
            raise KeyError(f'Invalid "expr" key in parameter {parameter}')
        name = parameter['name']
        if not parameter['vary']:
            logger.warning(
                f'Ignoring min in parameter {name} in '
                f'Fit.add_parameter (vary = {parameter["vary"]})')
            parameter['min'] = -np.inf
            logger.warning(
                f'Ignoring max in parameter {name} in '
                f'Fit.add_parameter (vary = {parameter["vary"]})')
            parameter['max'] = np.inf
        if self._code == 'scipy':
            self._parameters.add(FitParameter(**parameter))
        else:
            self._parameters.add(**parameter)
        self._free_parameters.append(name)

    def add_model(self, model, prefix):
        """Add a model component to the fit model."""
        if self._code == 'lmfit':
            from lmfit.models import (
                ConstantModel,
                LinearModel,
                QuadraticModel,
#                PolynomialModel,
                ExponentialModel,
                GaussianModel,
                LorentzianModel,
                ExpressionModel,
#                StepModel,
                RectangleModel,
        )

        if model.model == 'expression':
            expr = model.expr
        else:
            expr = None
        parameters = model.parameters
        model_name = model.model

        if prefix is None:
            pprefix = ''
        else:
            pprefix = prefix
        if self._code == 'scipy':
            new_parameters = []
            for par in deepcopy(parameters):
                self._parameters.add(par, pprefix)
                if self._parameters[par.name].expr is None:
                    self._parameters[par.name].set(value=par.default)
                new_parameters.append(par.name)
            self._res_num_pars += [len(parameters)]

        if model_name == 'constant':
            # Par: c
            if self._code == 'lmfit':
                newmodel = ConstantModel(prefix=prefix)
            self._linear_parameters.append(f'{pprefix}c')
        elif model_name == 'linear':
            # Par: slope, intercept
            if self._code == 'lmfit':
                newmodel = LinearModel(prefix=prefix)
            self._linear_parameters.append(f'{pprefix}slope')
            self._linear_parameters.append(f'{pprefix}intercept')
        elif model_name == 'quadratic':
            # Par: a, b, c
            if self._code == 'lmfit':
                newmodel = QuadraticModel(prefix=prefix)
            self._linear_parameters.append(f'{pprefix}a')
            self._linear_parameters.append(f'{pprefix}b')
            self._linear_parameters.append(f'{pprefix}c')
#        elif model_name == 'polynomial':
#            # Par: c0, c1,..., c7
#            degree = kwargs.get('degree')
#            if degree is not None:
#                kwargs.pop('degree')
#            if degree is None or not is_int(degree, ge=0, le=7):
#                raise ValueError(
#                    'Invalid parameter degree for build-in step model '
#                    f'({degree})')
#            if self._code == 'lmfit':
#                newmodel = PolynomialModel(degree=degree, prefix=prefix)
#            for i in range(degree+1):
#                self._linear_parameters.append(f'{pprefix}c{i}')
        elif model_name == 'exponential':
            # Par: amplitude, decay
            if self._code == 'lmfit':
                newmodel = ExponentialModel(prefix=prefix)
            self._linear_parameters.append(f'{pprefix}amplitude')
            self._nonlinear_parameters.append(f'{pprefix}decay')
        elif model_name == 'gaussian':
            # Par: amplitude, center, sigma (fwhm, height)
            if self._code == 'lmfit':
                newmodel = GaussianModel(prefix=prefix)
                # parameter norms for height and fwhm are needed to
                #   get correct errors
            self._linear_parameters.append(f'{pprefix}amplitude')
            self._nonlinear_parameters.append(f'{pprefix}center')
            self._nonlinear_parameters.append(f'{pprefix}sigma')
        elif model_name == 'lorentzian':
            # Par: amplitude, center, sigma (fwhm, height)
            if self._code == 'lmfit':
                newmodel = LorentzianModel(prefix=prefix)
                # parameter norms for height and fwhm are needed to
                #   get correct errors
            self._linear_parameters.append(f'{pprefix}amplitude')
            self._nonlinear_parameters.append(f'{pprefix}center')
            self._nonlinear_parameters.append(f'{pprefix}sigma')
#        elif model_name == 'step':
#            # Par: amplitude, center, sigma
#            form = kwargs.get('form')
#            if form is not None:
#                kwargs.pop('form')
#            if (form is None or form not in
#                    ('linear', 'atan', 'arctan', 'erf', 'logistic')):
#                raise ValueError(
#                    'Invalid parameter form for build-in step model '
#                    f'({form})')
#            if self._code == 'lmfit':
#                newmodel = StepModel(prefix=prefix, form=form)
#            self._linear_parameters.append(f'{pprefix}amplitude')
#            self._nonlinear_parameters.append(f'{pprefix}center')
#            self._nonlinear_parameters.append(f'{pprefix}sigma')
        elif model_name == 'rectangle':
            # Par: amplitude, center1, center2, sigma1, sigma2
            form = 'atan' #kwargs.get('form')
            #if form is not None:
            #    kwargs.pop('form')
            # RV: Implement and test other forms when needed
            if (form is None or form not in
                    ('linear', 'atan', 'arctan', 'erf', 'logistic')):
                raise ValueError(
                    'Invalid parameter form for build-in rectangle model '
                    f'({form})')
            if self._code == 'lmfit':
                newmodel = RectangleModel(prefix=prefix, form=form)
            self._linear_parameters.append(f'{pprefix}amplitude')
            self._nonlinear_parameters.append(f'{pprefix}center1')
            self._nonlinear_parameters.append(f'{pprefix}center2')
            self._nonlinear_parameters.append(f'{pprefix}sigma1')
            self._nonlinear_parameters.append(f'{pprefix}sigma2')
        elif model_name == 'expression' and self._code == 'lmfit':
            # Third party modules
            from asteval import (
                Interpreter,
                get_ast_names,
            )
            for par in parameters:
                if par.expr is not None:
                    raise KeyError(
                        f'Invalid "expr" key ({par.expr}) in '
                        f'parameter ({par}) for an expression model')
            ast = Interpreter()
            expr_parameters = [
                name for name in get_ast_names(ast.parse(expr))
                if (name != 'x' and name not in self._parameters
                    and name not in ast.symtable)]
            if prefix is None:
                newmodel = ExpressionModel(expr=expr)
            else:
                for name in expr_parameters:
                    expr = sub(rf'\b{name}\b', f'{prefix}{name}', expr)
                expr_parameters = [
                    f'{prefix}{name}' for name in expr_parameters]
            newmodel = ExpressionModel(expr=expr, name=model_name)
            # Remove already existing names
            for name in newmodel.param_names.copy():
                if name not in expr_parameters:
                    newmodel._func_allargs.remove(name)
                    newmodel._param_names.remove(name)
        else:
            raise ValueError(f'Unknown fit model ({model_name})')

        # Add the new model to the current one
        if self._code == 'scipy':
            if self._model is None:
                self._model = Components()
            self._model.add(model, prefix)
        else:
            if self._model is None:
                self._model = newmodel
            else:
                self._model += newmodel
            new_parameters = newmodel.make_params()
            self._parameters += new_parameters

        # Check linearity of expression model parameters
        if self._code == 'lmfit' and isinstance(newmodel, ExpressionModel):
            # Third party modules
            from sympy import diff
            for name in newmodel.param_names:
                if not diff(newmodel.expr, name, name):
                    if name not in self._linear_parameters:
                        self._linear_parameters.append(name)
                else:
                    if name not in self._nonlinear_parameters:
                        self._nonlinear_parameters.append(name)

        # Scale the default initial model parameters
        if self._norm is not None:
            for name in new_parameters:
                if name in self._linear_parameters:
                    par = self._parameters.get(name)
                    if par.expr is None:
                        if self._code == 'scipy':
                            value = par.default
                        else:
                            value = None
                        if value is None:
                            value = par.value
                        if value is not None:
                            value *= self._norm[1]
                        _min = par.min
                        _max = par.max
                        if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                            _min *= self._norm[1]
                        if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                            _max *= self._norm[1]
                        par.set(value=value, min=_min, max=_max)

        # Initialize the model parameters
        for parameter in deepcopy(parameters):
            name = parameter.name
            if name not in new_parameters:
                name = pprefix+name
                if name not in new_parameters:
                    raise ValueError(
                        f'Unable to match parameter {name}')
            if parameter.expr is None:
                self._parameters[name].set(
                    value=parameter.value, min=parameter.min,
                    max=parameter.max, vary=parameter.vary)
            else:
                if parameter.value is not None:
                    logger.warning(
                        'Ignoring input "value" for expression parameter'
                        f'{name} = {parameter.expr}')
                if not np.isinf(parameter.min):
                    logger.warning(
                        'Ignoring input "min" for expression parameter'
                        f'{name} = {parameter.expr}')
                if not np.isinf(parameter.max):
                    logger.warning(
                        'Ignoring input "max" for expression parameter'
                        f'{name} = {parameter.expr}')
                self._parameters[name].set(
                    value=None, min=-np.inf, max=np.inf, expr=parameter.expr)

    def eval(self, x, result=None):
        """Evaluate the best fit."""
        if result is None:
            result = self._result
        if result is None:
            return None
        return result.eval(x=np.asarray(x))-self.normalization_offset

    def fit(self, config=None, **kwargs):
        """Fit the model to the input data."""

        # Check input parameters
        if self._model is None:
            logger.error('Undefined fit model')
            return None
        self._mask = kwargs.pop('mask', None)
        guess = kwargs.pop('guess', False)
        if not isinstance(guess, bool):
            raise ValueError(
                f'Invalid value of keyword argument guess ({guess})')
        if self._result is not None:
            if guess:
                logger.warning(
                    'Ignoring input parameter guess during refitting')
                guess = False
#        if 'try_linear_fit' in kwargs:
#            raise RuntimeError('try_linear_fit needs testing')
#            try_linear_fit = kwargs.pop('try_linear_fit')
#            if not isinstance(try_linear_fit, bool):
#                raise ValueError(
#                    'Invalid value of keyword argument try_linear_fit '
#                    f'({try_linear_fit})')
#            if not self._try_linear_fit:
#                logger.warning(
#                    'Ignore superfluous keyword argument "try_linear_fit" '
#                    '(not yet supported for callable models)')
#            else:
#                self._try_linear_fit = try_linear_fit

        # Setup the fit
        self._setup_fit(config, guess)

        # Check if model is linear
        try:
            linear_model = self._check_linearity_model()
        except:
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
            except:
                linear_model = False
        if not linear_model:
            self._result = self._fit_nonlinear_model(
                self._x, self._y_norm, **kwargs)

        # Set internal parameter values to fit results upon success
        if self.success:
            for name, par in self._parameters.items():
                if par.expr is None and par.vary:
                    par.set(value=self._result.params[name].value)

        # Renormalize the data and results
        self._renormalize()

        return None

    def plot(
            self, y=None, y_title=None, title=None, result=None,
            skip_init=False, plot_comp=True, plot_comp_legends=False,
            plot_residual=False, plot_masked_data=True, **kwargs):
        """Plot the best fit."""
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
        if y is not None:
            if not isinstance(y, (tuple, list, np.ndarray)):
                logger.warning('Ignorint invalid parameter y ({y}')
            if len(y) != len(self._x):
                logger.warning(
                    'Ignoring parameter y in plot (wrong dimension)')
                y = None
        if y is not None:
            if y_title is None or not isinstance(y_title, str):
                y_title = 'data'
            plots += [(self._x, y, '.')]
            legend += [y_title]
        if self._y is not None:
            plots += [(self._x, np.asarray(self._y), 'b.')]
            legend += ['data']
            if plot_masked_data:
                plots += [(self._x[mask], np.asarray(self._y)[mask], 'bx')]
                legend += ['masked data']
        if isinstance(plot_residual, bool) and plot_residual:
            plots += [(self._x[~mask], result.residual, 'r-')]
            legend += ['residual']
        plots += [(self._x[~mask], result.best_fit, 'k-')]
        legend += ['best fit']
        if not skip_init and hasattr(result, 'init_fit'):
            plots += [(self._x[~mask], result.init_fit, 'g-')]
            legend += ['init']
        if plot_comp:
            components = result.eval_components(x=self._x[~mask])
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
                        y_comp *= np.ones(self._x[~mask].size)
                    plots += [(self._x[~mask], y_comp, '--')]
                    if plot_comp_legends:
                        if modelname[-1] == '_':
                            legend.append(modelname[:-1])
                        else:
                            legend.append(modelname)
        quick_plot(
            tuple(plots), legend=legend, title=title, block=True, **kwargs)

    @staticmethod
    def guess_init_peak(
            x, y, *args, center_guess=None, use_max_for_center=True):
        """Return a guess for the initial height, center and fwhm for a
        single peak.
        """
        center_guesses = None
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            logger.error(
                f'Invalid x and y lengths ({len(x)}, {len(y)}), '
                'skip initial guess')
            return None, None, None
        if isinstance(center_guess, (int, float)):
            if args:
                logger.warning(
                    'Ignoring additional arguments for single center_guess '
                    'value')
        elif isinstance(center_guess, (tuple, list, np.ndarray)):
            if len(center_guess) == 1:
                logger.warning(
                    'Ignoring additional arguments for single center_guess '
                    'value')
                if not isinstance(center_guess[0], (int, float)):
                    raise ValueError(
                        'Invalid parameter center_guess '
                        f'({type(center_guess[0])})')
                center_guess = center_guess[0]
            else:
                if len(args) != 1:
                    raise ValueError(
                        f'Invalid number of arguments ({len(args)})')
                n = args[0]
                if not is_index(n, 0, len(center_guess)):
                    raise ValueError('Invalid argument')
                center_guesses = center_guess
                center_guess = center_guesses[n]
        elif center_guess is not None:
            raise ValueError(
                f'Invalid center_guess type ({type(center_guess)})')

        # Sort the inputs
        index = np.argsort(x)
        x = x[index]
        y = y[index]
        miny = y.min()

        # Set range for current peak
        if center_guesses is not None:
            if len(center_guesses) > 1:
                index = np.argsort(center_guesses)
                n = list(index).index(n)
                center_guesses = np.asarray(center_guesses)[index]
            if n == 0:
                low = 0
                upp = index_nearest(
                    x, (center_guesses[0]+center_guesses[1]) / 2)
            elif n == len(center_guesses)-1:
                low = index_nearest(
                    x, (center_guesses[n-1]+center_guesses[n]) / 2)
                upp = len(x)
            else:
                low = index_nearest(
                    x, (center_guesses[n-1]+center_guesses[n]) / 2)
                upp = index_nearest(
                    x, (center_guesses[n]+center_guesses[n+1]) / 2)
            x = x[low:upp]
            y = y[low:upp]

        # Estimate FWHM
        maxy = y.max()
        if center_guess is None:
            center_index = np.argmax(y)
            center = x[center_index]
            height = maxy-miny
        else:
            if use_max_for_center:
                center_index = np.argmax(y)
                center = x[center_index]
                if center_index < 0.1*len(x) or center_index > 0.9*len(x):
                    center_index = index_nearest(x, center_guess)
                    center = center_guess
            else:
                center_index = index_nearest(x, center_guess)
                center = center_guess
            height = y[center_index]-miny
        half_height = miny + 0.5*height
        fwhm_index1 = 0
        for i in range(center_index, fwhm_index1, -1):
            if y[i] < half_height:
                fwhm_index1 = i
                break
        fwhm_index2 = len(x)-1
        for i in range(center_index, fwhm_index2):
            if y[i] < half_height:
                fwhm_index2 = i
                break
        if fwhm_index1 == 0 and fwhm_index2 < len(x)-1:
            fwhm = 2 * (x[fwhm_index2]-center)
        elif fwhm_index1 > 0 and fwhm_index2 == len(x)-1:
            fwhm = 2 * (center-x[fwhm_index1])
        else:
            fwhm = x[fwhm_index2]-x[fwhm_index1]

        if center_guess is not None and not use_max_for_center:
            index = fwhm_index1+np.argmax(y[fwhm_index1:fwhm_index2])
            center = x[index]
            height = y[index]-miny

        return height, center, fwhm

    def _create_prefixes(self, models):
        """Create model prefixes."""
        # Check for duplicate model names and create prefixes
        names = []
        prefixes = []
        for model in models:
            names.append(f'{model.prefix}{model.model}')
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
        # Check for duplicate model names and create prefixes
        prefixes = self._create_prefixes(models)

        # Add the free fit parameters
        for par in parameters:
            self.add_parameter(par.model_dump())

        # Add the model functions
        for prefix, model in zip(prefixes, models):
            self.add_model(model, prefix)

        # Check linearity of free fit parameters:
        known_parameters = (
            self._linear_parameters + self._nonlinear_parameters)
        for name in reversed(self._parameters):
            if name not in known_parameters:
                for nname, par in self._parameters.items():
                    if par.expr is not None:
                        # Third party modules
                        from sympy import diff

                        if nname in self._nonlinear_parameters:
                            self._nonlinear_parameters.insert(0, name)
                        elif diff(par.expr, name, name):
                            self._nonlinear_parameters.insert(0, name)
                        else:
                            self._linear_parameters.insert(0, name)

    def _setup_fit(self, config, guess=False):
        """Setup the fit."""
        # Apply mask if supplied:
        if self._mask is not None:
            raise RuntimeError('mask needs testing')
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(
                    f'Inconsistent x and mask dimensions ({self._x.size} vs '
                    f'{self._mask.size})')

        # Estimate initial parameters
        if guess and not isinstance(self, FitMap):
            raise RuntimeError('Estimate initial parameters needs testing')
            if self._mask is None:
                xx = self._x
                yy = self._y
            else:
                xx = self._x[~self._mask]
                yy = np.asarray(self._y)[~self._mask]
            try:
                # Try with the build-in lmfit guess method
                # (only implemented for a single model)
                self._parameters = self._model.guess(yy, x=xx)
            except:
                # Third party modules
                from asteval import Interpreter
                from lmfit.models import GaussianModel

                ast = Interpreter()
                # Should work for other peak-like models,
                #   but will need tests first
                for component in self._model.components:
                    if isinstance(component, GaussianModel):
                        center = self._parameters[
                            f"{component.prefix}center"].value
                        height_init, cen_init, fwhm_init = \
                            self.guess_init_peak(
                                xx, yy, center_guess=center,
                                use_max_for_center=False)
#                        if (self._fwhm_min is not None
#                                and fwhm_init < self._fwhm_min):
#                            fwhm_init = self._fwhm_min
#                        elif (self._fwhm_max is not None
#                                and fwhm_init > self._fwhm_max):
#                            fwhm_init = self._fwhm_max
                        ast(f'fwhm = {fwhm_init}')
                        ast(f'height = {height_init}')
                        sig_init = ast(fwhm_factor[component._name])
                        amp_init = ast(height_factor[component._name])
                        par = self._parameters[
                            f"{component.prefix}amplitude"]
                        if par.vary:
                            par.set(value=amp_init)
                        par = self._parameters[
                            f"{component.prefix}center"]
                        if par.vary:
                            par.set(value=cen_init)
                        par = self._parameters[
                            f"{component.prefix}sigma"]
                        if par.vary:
                            par.set(value=sig_init)

        # Add constant offset for a normalized model
        if self._result is None and self._norm is not None and self._norm[0]:
            from CHAP.utils.models import Constant
            model = Constant(
                model='constant',
                parameters=[{
                    'name': 'c',
                    'value': -self._norm[0],
                    'vary': False,
                }])
            self.add_model(model, 'tmp_normalization_offset_')

        # Adjust existing parameters for refit:
        if config is not None:
            # Local modules
            from CHAP.utils.models import (
                FitConfig,
                Multipeak,
            )

            # Expand multipeak model if present
            scale_factor = None
            for i, model in enumerate(deepcopy(config.models)):
                found_multipeak = False
                if isinstance(model, Multipeak):
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
                self._best_parameters.pop(scale_factor_index)
                self._best_values = np.delete(
                    self._best_values, scale_factor_index, 0)
                self._best_errors = np.delete(
                    self._best_errors, scale_factor_index, 0)
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

        # Set parameters configuration
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
                else:
                    if par.vary:
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
                    else:
                        self._parameters[name].set(value=self._norm[1])

    def _check_linearity_model(self):
        """Identify the linearity of all model parameters and check if
        the model is linear or not.
        """
        # Third party modules
        from lmfit.models import ExpressionModel
        from sympy import diff

#        if not self._try_linear_fit:
#            logger.info(
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
                if par.expr is None and norm:
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
                    if par.expr is None and norm:
                        value = par.value/self._norm[1]
                        self._parameters[name].set(value=value)
                        self._result.params[name].set(value=value)
        self._result.residual = y-self._result.best_fit
        self._result.components = self._model.components
        self._result.init_params = None

    def _fit_nonlinear_model(self, x, y, **kwargs):
        """Perform a nonlinear fit with spipy or lmfit."""
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
            # Third party modules
            from asteval import Interpreter
            from scipy.optimize import (
                leastsq,
                least_squares,
            )

            assert self._mask is None
            self._ast = Interpreter()
            self._ast.basesymtable = dict(self._ast.symtable.items())
            pars_init = []
            for i, (name, par) in enumerate(self._parameters.items()):
                setattr(par, '_init_value', par.value)
                self._res_par_values[i] = par.value
                if par.expr is None:
                    self._ast.symtable[name] = par.value
                    if par.vary:
                        pars_init.append(par.value)
            if have_bounds:
                bounds = (
                    [v['min'] for v in self._parameter_bounds.values()],
                    [v['max'] for v in self._parameter_bounds.values()])
                if self._method in ('lm', 'leastsq'):
                    self._method = 'trf'
                    logger.debug(
                        f'Fit method changed to {self._method} for fit with '
                        'bounds')
            else:
                bounds = (-np.inf, np.inf)
            init_params = deepcopy(self._parameters)
#            t0 = time()
            lskws = {
                'ftol': 1.49012e-08,
                'xtol': 1.49012e-08,
                'gtol': 10*FLOAT_EPS,
            }
            if self._method == 'leastsq':
                lskws['maxfev'] = 64000
                result = leastsq(
                    self._residual, pars_init, args=(x, y), full_output=True,
                    **lskws)
            else:
                lskws['max_nfev'] = 64000
                result = least_squares(
                    self._residual, pars_init, bounds=bounds,
                    method=self._method, args=(x, y), **lskws)
#            t1 = time()
#            print(f'\n\nFitting took {1000*(t1-t0):.3f} ms\n\n')
            model_result = ModelResult(
                self._model, self._parameters, x, y, self._method, self._ast,
                self._res_par_exprs, self._res_par_indices,
                self._res_par_names, result)
            model_result.init_params = init_params
            model_result.init_values = {}
            for name, par in init_params.items():
                model_result.init_values[name] = par.value
            model_result.max_nfev = lskws.get('maxfev')
        else:
            fit_kws = {}
#            if 'Dfun' in kwargs:
#                fit_kws['Dfun'] = kwargs.pop('Dfun')
#            t0 = time()
            model_result = self._model.fit(
                y, self._parameters, x=x, method=self._method, fit_kws=fit_kws,
                **kwargs)
#            t1 = time()
#            print(f'\n\nFitting took {1000*(t1-t0):.3f} ms\n\n')

        return model_result

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
                par.set(value=value, min=_min, max=_max)
        if self._result is None:
            return
        self._result.best_fit = (
            self._result.best_fit*self._norm[1] + self._norm[0])
        for name, par in self._result.params.items():
            if name in self._linear_parameters:
                if par.stderr is not None:
                    if self._code == 'scipy':
                        setattr(par, '_stderr', par.stderr*self._norm[1])
                    else:
                        par.stderr *= self._norm[1]
                if par.expr is None:
                    _min = par.min
                    _max = par.max
                    value = par.value*self._norm[1]
                    if par.init_value is not None:
                        if self._code == 'scipy':
                            setattr(par, '_init_value',
                                    par.init_value*self._norm[1])
                        else:
                            par.init_value *= self._norm[1]
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
        if hasattr(self._result, 'init_fit'):
            self._result.init_fit = (
                self._result.init_fit*self._norm[1] + self._norm[0])
        if hasattr(self._result, 'init_values'):
            init_values = {}
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
                    value = par.value
                    _min = par.min
                    _max = par.max
                    value *= self._norm[1]
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
                if self._code == 'scipy':
                    setattr(par, '_init_value', par.value)
                else:
                    par.init_value = par.value
        # Don't renormalize chisqr, it has no useful meaning in
        #     physical units
#        self._result.chisqr *= self._norm[1]*self._norm[1]
        if self._result.covar is not None:
            norm_sq = self._norm[1]*self._norm[1]
            for i, name in enumerate(self._result.var_names):
                if name in self._linear_parameters:
                    for j in range(len(self._result.var_names)):
                        if self._result.covar[i,j] is not None:
                            #self._result.covar[i,j] *= self._norm[1]
                            self._result.covar[i,j] *= norm_sq
                        if self._result.covar[j,i] is not None:
                            #self._result.covar[j,i] *= self._norm[1]
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

    def _residual(self, pars, x, y):
        res = np.zeros((x.size))
        n_par = len(self._free_parameters)
        for par, index in zip(pars, self._res_par_indices):
            self._res_par_values[index] = par
        if self._res_par_exprs:
            for par, name in zip(pars, self._res_par_names):
                self._ast.symtable[name] = par
            for expr in self._res_par_exprs:
                self._res_par_values[expr['index']] = \
                    self._ast.eval(expr['expr'])
        for component, num_par in zip(
                self._model.components, self._res_num_pars):
            res += component.func(
                x, *tuple(self._res_par_values[n_par:n_par+num_par]))
            n_par += num_par
        return res - y


class FitMap(Fit):
    """Wrapper to the Fit class to fit data on a N-dimensional map."""
    def __init__(self, nxdata, config):
        """Initialize FitMap."""
        super().__init__(None, config)
        self._best_errors = None
        self._best_fit = None
        self._best_parameters = None
        self._best_values = None
        self._inv_transpose = None
        self._max_nfev = None
        self._memfolder = config.memfolder
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
        self._x = np.asarray(nxdata[nxdata.attrs['axes'][-1]])
        self._ymap = np.asarray(nxdata.nxsignal)

        # Check input parameters
        if self._x.ndim != 1:
            raise ValueError(f'Invalid x dimension ({self._x.ndim})')
        if self._x.size != self._ymap.shape[-1]:
            raise ValueError(
                f'Inconsistent x and y dimensions ({self._x.size} vs '
                f'{self._ymap.shape[-1]})')

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
#        else:
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
            self._ymap_norm = (self._ymap_norm-self._norm[0]) / self._norm[1]
        else:
            self._redchi_cutoff *= self._y_range**2

        # Setup fit model
        self._setup_fit_model(config.parameters, config.models)

    @property
    def best_errors(self):
        """Return errors in the best fit parameters."""
        return self._best_errors

    @property
    def best_fit(self):
        """Return the best fits."""
        return self._best_fit

    @property
    def best_values(self):
        """Return values of the best fit parameters."""
        return self._best_values

    @property
    def chisqr(self):
        """Return the chisqr value of each best fit."""
        logger.warning('Undefined property chisqr')

    @property
    def components(self):
        """Return the fit model components info."""
        # Third party modules
        from lmfit.models import ExpressionModel

        components = {}
        if self._result is None:
            logger.warning(
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
                prefix = component.prefix
                if prefix:
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
        """Return the covarience matrices of the best fit parameters.
        """
        logger.warning('Undefined property covar')

    @property
    def max_nfev(self):
        """Return if the maximum number of function evaluations is
        reached for each fit.
        """
        return self._max_nfev

    @property
    def num_func_eval(self):
        """Return the number of function evaluations for each best fit.
        """
        return self._num_func_eval

    @property
    def out_of_bounds(self):
        """Return the out_of_bounds value of each best fit."""
        return self._out_of_bounds

    @property
    def redchi(self):
        """Return the redchi value of each best fit."""
        return self._redchi

    @property
    def residual(self):
        """Return the residual in each best fit."""
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
        """Return the success value for each fit."""
        return self._success

    @property
    def var_names(self):
        """Return the variable names for the covarience matrix
        property.
        """
        logger.warning('Undefined property var_names')

    @property
    def y(self):
        """Return the input y-array."""
        logger.warning('Undefined property y')

    @property
    def ymap(self):
        """Return the input y-array map."""
        return self._ymap

    def best_parameters(self, dims=None):
        """Return the best fit parameters."""
        if dims is None:
            return self._best_parameters
        if (not isinstance(dims, (list, tuple))
                or len(dims) != len(self._map_shape)):
            raise ValueError('Invalid parameter dims ({dims})')
        if self.best_values is None or self.best_errors is None:
            logger.warning(
                f'Unable to obtain best parameter values for dims = {dims}')
            return {}
        # Create current parameters
        parameters = deepcopy(self._parameters)
        for n, name in enumerate(self._best_parameters):
            if self._parameters[name].vary:
                parameters[name].set(value=self.best_values[n][dims])
            parameters[name].stderr = self.best_errors[n][dims]
        parameters_dict = {}
        for name in sorted(parameters):
            if name != 'tmp_normalization_offset_c':
                par = parameters[name]
                parameters_dict[name] = {
                    'value': par.value,
                    'error': par.stderr,
                    'init_value': self.init_parameters[name]['value'],
                    'min': par.min,
                    'max': par.max,
                    'vary': par.vary,
                    'expr': par.expr,
                }
        return parameters_dict

    def freemem(self):
        """Free memory allocated for parallel processing."""
        if self._memfolder is None:
            return
        try:
            rmtree(self._memfolder)
        except:
            logger.warning('Could not clean-up automatically.')

    def plot(
            self, dims=None, y_title=None, plot_residual=False,
            plot_comp_legends=False, plot_masked_data=True, **kwargs):
        """Plot the best fits."""
        # Third party modules
        from lmfit.models import ExpressionModel

        if dims is None:
            dims = [0]*len(self._map_shape)
        if (not isinstance(dims, (list, tuple))
                or len(dims) != len(self._map_shape)):
            raise ValueError('Invalid parameter dims ({dims})')
        dims = tuple(dims)
        if (self._result is None or self.best_fit is None
                or self.best_values is None):
            logger.warning(
                f'Unable to plot fit for dims = {dims}')
            return
        if y_title is None or not isinstance(y_title, str):
            y_title = 'data'
        if self._mask is None:
            mask = np.zeros(self._x.size).astype(bool)
            plot_masked_data = False
        else:
            mask = self._mask
        plots = [(self._x, np.asarray(self._ymap[dims]), 'b.')]
        legend = [y_title]
        if plot_masked_data:
            plots += \
                [(self._x[mask], np.asarray(self._ymap)[(*dims,mask)], 'bx')]
            legend += ['masked data']
        plots += [(self._x[~mask], self.best_fit[dims], 'k-')]
        legend += ['best fit']
        if plot_residual:
            plots += [(self._x[~mask], self.residual[dims], 'r--')]
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
                prefix = component.prefix
                if prefix:
                    if prefix[-1] == '_':
                        prefix = prefix[:-1]
                    modelname = f'{prefix} ({component._name})'
                else:
                    modelname = f'{component._name}'
            if len(modelname) > 20:
                modelname = f'{modelname[0:16]} ...'
            y = component.eval(params=parameters, x=self._x[~mask])
            if isinstance(y, (int, float)):
                y *= np.ones(self._x[~mask].size)
            plots += [(self._x[~mask], y, '--')]
            if plot_comp_legends:
                legend.append(modelname)
        quick_plot(
            tuple(plots), legend=legend, title=str(dims), block=True, **kwargs)

    def fit(self, config=None, **kwargs):
        """Fit the model to the input data."""
        # Check input parameters
        if self._model is None:
            logger.error('Undefined fit model')
        num_proc_max = max(1, cpu_count())
        if config is None:
            num_proc = kwargs.pop('num_proc', num_proc_max)
            self._rel_height_cutoff = kwargs.pop('rel_height_cutoff')
            self._try_no_bounds = kwargs.pop('try_no_bounds', False)
            self._redchi_cutoff = kwargs.pop('redchi_cutoff', 0.1)
            self._print_report = kwargs.pop('print_report', False)
            self._plot = kwargs.pop('plot', False)
            self._skip_init = kwargs.pop('skip_init', True)
        else:
            num_proc = config.num_proc
            self._rel_height_cutoff = config.rel_height_cutoff
#            self._try_no_bounds = config.try_no_bounds
#            self._redchi_cutoff = config.redchi_cutoff
            self._print_report = config.print_report
            self._plot = config.plot
#            self._skip_init = config.skip_init
        if num_proc > 1 and not HAVE_JOBLIB:
            logger.warning(
                'Missing joblib in the conda environment, running serially')
            num_proc = 1
        if num_proc > num_proc_max:
            logger.warning(
                f'The requested number of processors ({num_proc}) exceeds the '
                'maximum allowed number of processors, num_proc reduced to '
                f'{num_proc_max}')
            num_proc = num_proc_max
        logger.debug(f'Using {num_proc} processors to fit the data')
        self._redchi_cutoff *= self._y_range**2

        # Setup the fit
        self._setup_fit(config)

        # Create the best parameter list, consisting of all varying
        #     parameters plus the expression parameters in order to
        #     collect their errors
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
        #     remove the remaining results of the previous fit
        if self._result is not None:
            self._out_of_bounds = None
            self._max_nfev = None
            self._num_func_eval = None
            self._redchi = None
            self._success = None
            self._best_fit = None
            self._best_errors = None
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
        #     (and best values for a refit)
        self._normalize()

        # Prevent initial values from sitting at boundaries
        self._parameter_bounds = {
            name:{'min': par.min, 'max': par.max}
            for name, par in self._parameters.items() if par.vary}
        self._reset_par_at_boundary()

        # Set parameter bounds to unbound
        #     (only use bounds when fit fails)
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

        # Update the best parameter list
        if num_new_parameters:
            self._best_parameters += self._new_parameters

        # Perform the first fit to get model component info and
        #     initial parameters
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
                    logger.warning(
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
                    else:
                        init_values[name] = value*self._norm[1]
                self._result.init_values = init_values
            if (hasattr(self._result, 'init_params')
                    and self._result.init_params is not None):
                for name, par in self._result.init_params.items():
                    if par.expr is None and name in self._linear_parameters:
                        _min = par.min
                        _max = par.max
                        value = par.value*self._norm[1]
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
        self._best_values = np.asarray([np.reshape(
            par, list(self._map_shape)) for par in self._best_values_flat])
        self._best_errors = np.asarray([np.reshape(
            par, list(self._map_shape)) for par in self._best_errors_flat])
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
            self._best_values = np.transpose(
                self._best_values, [0] + [i+1 for i in self._inv_transpose])
            self._best_errors = np.transpose(
                self._best_errors, [0] + [i+1 for i in self._inv_transpose])
        del self._out_of_bounds_flat
        del self._max_nfev_flat
        del self._num_func_eval_flat
        del self._redchi_flat
        del self._success_flat
        del self._best_fit_flat
        del self._best_values_flat
        del self._best_errors_flat

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
                or (self._rel_height_cutoff is not None
                    and y_max < self._rel_height_cutoff)):
            logger.debug(f'Skipping fit for n = {n} (rel norm = {y_max:.5f})')
            if self._code == 'scipy':
                from CHAP.utils.fit import ModelResult

                result = ModelResult(self._model, deepcopy(self._parameters))
            else:
                from lmfit.model import ModelResult

                result = ModelResult(self._model, deepcopy(self._parameters))
            result.success = False
            # Renormalize the data and results
            self._renormalize(n, result)
            return result

        # Regular full fit
        result = self._fit_with_bounds_check(n, current_best_values, **kwargs)

        if self._rel_height_cutoff is not None:
            # Third party modules
            from lmfit.models import (
                GaussianModel,
                LorentzianModel,
            )

            # Check for low heights peaks and refit without them
            heights = []
            names = []
            for component in result.components:
                if isinstance(component, (GaussianModel, LorentzianModel)):
                    for name in component.param_names:
                        if 'height' in name:
                            heights.append(result.params[name].value)
                            names.append(name)
            if heights:
                refit = False
                max_height = max(heights)
                parameters_save = deepcopy(self._parameters)
                for i, (name, height) in enumerate(zip(names, heights)):
                    if height < self._rel_height_cutoff*max_height:
                        self._parameters[
                            name.replace('height', 'amplitude')].set(
                               value=0.0, min=0.0, vary=False)
                        self._parameters[
                            name.replace('height', 'center')].set(
                               vary=False)
                        self._parameters[
                            name.replace('height', 'sigma')].set(
                               value=0.0, min=0.0, vary=False)
                        refit = True
                if refit:
                    result = self._fit_with_bounds_check(
                        n, current_best_values, **kwargs)
                    # Reset fixed amplitudes back to default
                    self._parameters = deepcopy(parameters_save)

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
            for par in result.params.values():
                if par.vary:
                    current_best_values[par.name] = par.value
        else:
            errortxt = f'Fit for n = {n} failed'
            if hasattr(result, 'lmdif_message'):
                errortxt += f'\n\t{result.lmdif_message}'
            if hasattr(result, 'message'):
                errortxt += f'\n\t{result.message}'
            logger.warning(f'{errortxt}')

        # Renormalize the data and results
        self._renormalize(n, result)

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
                self._best_values_flat[i][n] = np.float64(
                    result.params[name].value)
                self._best_errors_flat[i][n] = np.float64(
                    result.params[name].stderr)
            if result.success:
                self._best_fit_flat[n] = result.best_fit
        else:
            for name, par in result.params.items():
                if name in self._linear_parameters:
                    if par.stderr is not None:
                        if self._code == 'scipy':
                            setattr(par, '_stderr', par.stderr*self._norm[1])
                        else:
                            par.stderr *= self._norm[1]
                    if par.expr is None:
                        par.value *= self._norm[1]
                        if self._print_report:
                            if par.init_value is not None:
                                if self._code == 'scipy':
                                    setattr(par, '_init_value',
                                            par.init_value*self._norm[1])
                                else:
                                    par.init_value *= self._norm[1]
                            if (not np.isinf(par.min)
                                    and abs(par.min) != FLOAT_MIN):
                                par.min *= self._norm[1]
                            if (not np.isinf(par.max)
                                    and abs(par.max) != FLOAT_MIN):
                                par.max *= self._norm[1]
            for i, name in enumerate(self._best_parameters):
                self._best_values_flat[i][n] = np.float64(
                    result.params[name].value)
                self._best_errors_flat[i][n] = np.float64(
                    result.params[name].stderr)
            if result.success:
                self._best_fit_flat[n] = (
                    result.best_fit*self._norm[1] + self._norm[0])
                if self._plot:
                    if not self._skip_init:
                        result.init_fit = (
                            result.init_fit*self._norm[1] + self._norm[0])
                    result.best_fit = np.copy(self._best_fit_flat[n])
