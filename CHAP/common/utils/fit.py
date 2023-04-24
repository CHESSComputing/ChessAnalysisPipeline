#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=
"""
File       : fit.py
Author     : Rolf Verberg <rolfverberg AT gmail dot com>
Description: General curve fitting module
"""

# System modules
from copy import deepcopy
from logging import getLogger
from os import (
    cpu_count,
    mkdir,
    path,
)
from re import compile as re_compile
from re import sub
from shutil import rmtree
from sys import float_info

# Third party modules
try:
    from joblib import (
        Parallel,
        delayed,
    )
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False
from lmfit import (
    Parameters,
    Model,
)
from lmfit.model import ModelResult
from lmfit.models import (
    ConstantModel,
    LinearModel,
    QuadraticModel,
    PolynomialModel,
    ExponentialModel,
    StepModel,
    RectangleModel,
    ExpressionModel,
    GaussianModel,
    LorentzianModel,
)
import numpy as np
try:
    from sympy import (
        diff,
        simplify,
    )
except ImportError:
    pass
try:
    import xarray as xr
    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False

# Local modules
from CHAP.common.utils.general import (
    is_int,
    is_num,
    is_dict_series,
    is_index,
    index_nearest,
    input_num,
    quick_plot,
)
#    eval_expr,

logger = getLogger(__name__)
FLOAT_MIN = float_info.min
FLOAT_MAX = float_info.max

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


class Fit:
    """
    Wrapper class for lmfit.
    """
    def __init__(self, y, x=None, models=None, normalize=True, **kwargs):
        """Initialize Fit."""
        # Third party modules
        if not isinstance(normalize, bool):
            raise ValueError(f'Invalid parameter normalize ({normalize})')
        self._mask = None
        self._model = None
        self._norm = None
        self._normalized = False
        self._parameters = Parameters()
        self._parameter_bounds = None
        self._parameter_norms = {}
        self._linear_parameters = []
        self._nonlinear_parameters = []
        self._result = None
        self._try_linear_fit = True
        self._y = None
        self._y_norm = None
        self._y_range = None
        if 'try_linear_fit' in kwargs:
            self._try_linear_fit = kwargs.pop('try_linear_fit')
            if not isinstance(self._try_linear_fit, bool):
                raise ValueError(
                    'Invalid value of keyword argument try_linear_fit '
                    f'({self._try_linear_fit})')
        if y is not None:
            if isinstance(y, (tuple, list, np.ndarray)):
                self._x = np.asarray(x)
                self._y = np.asarray(y)
            elif HAVE_XARRAY and isinstance(y, xr.DataArray):
                if x is not None:
                    logger.warning('Ignoring superfluous input x ({x})')
                if y.ndim != 1:
                    raise ValueError(
                        'Invalid DataArray dimensions for parameter y '
                        f'({y.ndim})')
                self._x = np.asarray(y[y.dims[0]])
                self._y = y
            else:
                raise ValueError(f'Invalid parameter y ({y})')
            if self._x.ndim != 1:
                raise ValueError(
                    f'Invalid dimension for input x ({self._x.ndim})')
            if self._x.size != self._y.size:
                raise ValueError(
                    f'Inconsistent x and y dimensions ({self._x.size} vs '
                    f'{self._y.size})')
            if 'mask' in kwargs:
                self._mask = kwargs.pop('mask')
            if self._mask is None:
                y_min = float(self._y.min())
                self._y_range = float(self._y.max())-y_min
                if normalize and self._y_range > 0.0:
                    self._norm = (y_min, self._y_range)
            else:
                self._mask = np.asarray(self._mask).astype(bool)
                if self._x.size != self._mask.size:
                    raise ValueError(
                        f'Inconsistent x and mask dimensions ({self._x.size} '
                        f'vs {self._mask.size})')
                y_masked = np.asarray(self._y)[~self._mask]
                y_min = float(y_masked.min())
                self._y_range = float(y_masked.max())-y_min
                if normalize and self._y_range > 0.0:
                    if normalize and self._y_range > 0.0:
                        self._norm = (y_min, self._y_range)
        if models is not None:
            if callable(models) or isinstance(models, str):
                kwargs = self.add_model(models, **kwargs)
            elif isinstance(models, (tuple, list)):
                for model in models:
                    kwargs = self.add_model(model, **kwargs)
            self.fit(**kwargs)

    @classmethod
    def fit_data(cls, y, models, x=None, normalize=True, **kwargs):
        """Class method for Fit."""
        return cls(y, x=x, models=models, normalize=normalize, **kwargs)

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
    def best_results(self):
        """
        Convert the input DataArray to a data set and add the fit
        results.
        """
        if self._result is None:
            return None
        if not HAVE_XARRAY:
            logger.warning(
                'fit.best_results requires xarray in the conda environment')
            return None
        if isinstance(self._y, xr.DataArray):
            best_results = self._y.to_dataset()
            dims = self._y.dims
            fit_name = f'{self._y.name}_fit'
        else:
            coords = {'x': (['x'], self._x)}
            dims = ('x',)
            best_results = xr.Dataset(coords=coords)
            best_results['y'] = (dims, self._y)
            fit_name = 'y_fit'
        best_results[fit_name] = (dims, self.best_fit)
        if self._mask is not None:
            best_results['mask'] = self._mask
        best_results.coords['par_names'] = ('peak', self.best_values.keys())
        best_results['best_values'] = \
            (['par_names'], self.best_values.values())
        best_results['best_errors'] = \
            (['par_names'], self.best_errors.values())
        best_results.attrs['components'] = self.components
        return best_results

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
                self._result.init_params['tmp_normalization_offset_c'])
        else:
            normalization_offset = float(
                self._result.params['tmp_normalization_offset_c'])
        return normalization_offset

    @property
    def num_func_eval(self):
        """
        Return the number of function evaluations for the best fit.
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
        return self._result.residual

    @property
    def success(self):
        """Return the success value for the fit."""
        if self._result is None:
            return None
        if not self._result.success:
            logger.warning(
                f'ier = {self._result.ier}: {self._result.message}')
            if self._result.ier and self._result.ier != 5:
                return True
        return self._result.success

    @property
    def var_names(self):
        """
        Return the variable names for the covarience matrix property.
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

    def add_parameter(self, **parameter):
        """Add a fit fit parameter to the fit model."""
        if not isinstance(parameter, dict):
            raise ValueError(f'Invalid parameter ({parameter})')
        if parameter.get('expr') is not None:
            raise KeyError(f'Invalid "expr" key in parameter {parameter}')
        name = parameter['name']
        if not isinstance(name, str):
            raise ValueError(
                f'Invalid "name" value ({name}) in parameter {parameter}')
        if parameter.get('norm') is None:
            self._parameter_norms[name] = False
        else:
            norm = parameter.pop('norm')
            if self._norm is None:
                logger.warning(
                    f'Ignoring norm in parameter {name} in Fit.add_parameter '
                    '(normalization is turned off)')
                self._parameter_norms[name] = False
            else:
                if not isinstance(norm, bool):
                    raise ValueError(
                        f'Invalid "norm" value ({norm}) in parameter '
                        f'{parameter}')
                self._parameter_norms[name] = norm
        vary = parameter.get('vary')
        if vary is not None:
            if not isinstance(vary, bool):
                raise ValueError(
                    f'Invalid "vary" value ({vary}) in parameter {parameter}')
            if not vary:
                if 'min' in parameter:
                    logger.warning(
                        f'Ignoring min in parameter {name} in '
                        f'Fit.add_parameter (vary = {vary})')
                    parameter.pop('min')
                if 'max' in parameter:
                    logger.warning(
                        f'Ignoring max in parameter {name} in '
                        f'Fit.add_parameter (vary = {vary})')
                    parameter.pop('max')
        if self._norm is not None and name not in self._parameter_norms:
            raise ValueError(
                f'Missing parameter normalization type for parameter {name}')
        self._parameters.add(**parameter)

    def add_model(
            self, model, prefix=None, parameters=None, parameter_norms=None,
            **kwargs):
        """Add a model component to the fit model."""
        # Third party modules
        from asteval import (
            Interpreter,
            get_ast_names,
        )

        if prefix is not None and not isinstance(prefix, str):
            logger.warning('Ignoring illegal prefix: {model} {type(model)}')
            prefix = None
        if prefix is None:
            pprefix = ''
        else:
            pprefix = prefix
        if parameters is not None:
            if isinstance(parameters, dict):
                parameters = (parameters, )
            elif not is_dict_series(parameters):
                raise ValueError('Invalid parameter parameters ({parameters})')
            parameters = deepcopy(parameters)
        if parameter_norms is not None:
            if isinstance(parameter_norms, dict):
                parameter_norms = (parameter_norms, )
            if not is_dict_series(parameter_norms):
                raise ValueError(
                    'Invalid parameter parameters_norms ({parameters_norms})')
        new_parameter_norms = {}
        if callable(model):
            # Linear fit not yet implemented for callable models
            self._try_linear_fit = False
            if parameter_norms is None:
                if parameters is None:
                    raise ValueError(
                        'Either parameters or parameter_norms is required in '
                        f'{model}')
                for par in parameters:
                    name = par['name']
                    if not isinstance(name, str):
                        raise ValueError(
                            f'Invalid "name" value ({name}) in input '
                            'parameters')
                    if par.get('norm') is not None:
                        norm = par.pop('norm')
                        if not isinstance(norm, bool):
                            raise ValueError(
                                f'Invalid "norm" value ({norm}) in input '
                                'parameters')
                        new_parameter_norms[f'{pprefix}{name}'] = norm
            else:
                for par in parameter_norms:
                    name = par['name']
                    if not isinstance(name, str):
                        raise ValueError(
                            f'Invalid "name" value ({name}) in input '
                            'parameters')
                    norm = par.get('norm')
                    if norm is None or not isinstance(norm, bool):
                        raise ValueError(
                            f'Invalid "norm" value ({norm}) in input '
                            'parameters')
                    new_parameter_norms[f'{pprefix}{name}'] = norm
            if parameters is not None:
                for par in parameters:
                    if par.get('expr') is not None:
                        raise KeyError(
                            f'Invalid "expr" key ({par.get("expr")}) in '
                            f'parameter {name} for a callable model {model}')
                    name = par['name']
                    if not isinstance(name, str):
                        raise ValueError(
                            f'Invalid "name" value ({name}) in input '
                            'parameters')
# RV callable model will need partial deriv functions for any linear
#     parameter to get the linearized matrix, so for now skip linear
#     solution option
            newmodel = Model(model, prefix=prefix)
        elif isinstance(model, str):
            if model == 'constant':
                # Par: c
                newmodel = ConstantModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}c'] = True
                self._linear_parameters.append(f'{pprefix}c')
            elif model == 'linear':
                # Par: slope, intercept
                newmodel = LinearModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}slope'] = True
                new_parameter_norms[f'{pprefix}intercept'] = True
                self._linear_parameters.append(f'{pprefix}slope')
                self._linear_parameters.append(f'{pprefix}intercept')
            elif model == 'quadratic':
                # Par: a, b, c
                newmodel = QuadraticModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}a'] = True
                new_parameter_norms[f'{pprefix}b'] = True
                new_parameter_norms[f'{pprefix}c'] = True
                self._linear_parameters.append(f'{pprefix}a')
                self._linear_parameters.append(f'{pprefix}b')
                self._linear_parameters.append(f'{pprefix}c')
            elif model == 'polynomial':
                # Par: c0, c1,..., c7
                degree = kwargs.get('degree')
                if degree is not None:
                    kwargs.pop('degree')
                if degree is None or not is_int(degree, ge=0, le=7):
                    raise ValueError(
                        'Invalid parameter degree for build-in step model '
                        f'({degree})')
                newmodel = PolynomialModel(degree=degree, prefix=prefix)
                for i in range(degree+1):
                    new_parameter_norms[f'{pprefix}c{i}'] = True
                    self._linear_parameters.append(f'{pprefix}c{i}')
            elif model == 'gaussian':
                # Par: amplitude, center, sigma (fwhm, height)
                newmodel = GaussianModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center'] = False
                new_parameter_norms[f'{pprefix}sigma'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center')
                self._nonlinear_parameters.append(f'{pprefix}sigma')
                # parameter norms for height and fwhm are needed to
                #   get correct errors
                new_parameter_norms[f'{pprefix}height'] = True
                new_parameter_norms[f'{pprefix}fwhm'] = False
            elif model == 'lorentzian':
                # Par: amplitude, center, sigma (fwhm, height)
                newmodel = LorentzianModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center'] = False
                new_parameter_norms[f'{pprefix}sigma'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center')
                self._nonlinear_parameters.append(f'{pprefix}sigma')
                # parameter norms for height and fwhm are needed to
                #   get correct errors
                new_parameter_norms[f'{pprefix}height'] = True
                new_parameter_norms[f'{pprefix}fwhm'] = False
            elif model == 'exponential':
                # Par: amplitude, decay
                newmodel = ExponentialModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}decay'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}decay')
            elif model == 'step':
                # Par: amplitude, center, sigma
                form = kwargs.get('form')
                if form is not None:
                    kwargs.pop('form')
                if (form is None or form not in
                        ('linear', 'atan', 'arctan', 'erf', 'logistic')):
                    raise ValueError(
                        'Invalid parameter form for build-in step model '
                        f'({form})')
                newmodel = StepModel(prefix=prefix, form=form)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center'] = False
                new_parameter_norms[f'{pprefix}sigma'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center')
                self._nonlinear_parameters.append(f'{pprefix}sigma')
            elif model == 'rectangle':
                # Par: amplitude, center1, center2, sigma1, sigma2
                form = kwargs.get('form')
                if form is not None:
                    kwargs.pop('form')
                if (form is None or form not in
                        ('linear', 'atan', 'arctan', 'erf', 'logistic')):
                    raise ValueError(
                        'Invalid parameter form for build-in rectangle model '
                        f'({form})')
                newmodel = RectangleModel(prefix=prefix, form=form)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center1'] = False
                new_parameter_norms[f'{pprefix}center2'] = False
                new_parameter_norms[f'{pprefix}sigma1'] = False
                new_parameter_norms[f'{pprefix}sigma2'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center1')
                self._nonlinear_parameters.append(f'{pprefix}center2')
                self._nonlinear_parameters.append(f'{pprefix}sigma1')
                self._nonlinear_parameters.append(f'{pprefix}sigma2')
            elif model == 'expression':
                # Par: by expression
                expr = kwargs['expr']
                if not isinstance(expr, str):
                    raise ValueError(
                        f'Invalid "expr" value ({expr}) in {model}')
                kwargs.pop('expr')
                if parameter_norms is not None:
                    logger.warning(
                        'Ignoring parameter_norms (normalization '
                        'determined from linearity)}')
                if parameters is not None:
                    for par in parameters:
                        if par.get('expr') is not None:
                            raise KeyError(
                                f'Invalid "expr" key ({par.get("expr")}) in '
                                f'parameter ({par}) for an expression model')
                        if par.get('norm') is not None:
                            logger.warning(
                                f'Ignoring "norm" key in parameter ({par}) '
                                '(normalization determined from linearity)')
                            par.pop('norm')
                        name = par['name']
                        if not isinstance(name, str):
                            raise ValueError(
                                f'Invalid "name" value ({name}) in input '
                                'parameters')
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
                    newmodel = ExpressionModel(expr=expr, name=name)
                # Remove already existing names
                for name in newmodel.param_names.copy():
                    if name not in expr_parameters:
                        newmodel._func_allargs.remove(name)
                        newmodel._param_names.remove(name)
            else:
                raise ValueError(f'Unknown build-in fit model ({model})')
        else:
            raise ValueError('Invalid parameter model ({model})')

        # Add the new model to the current one
        if self._model is None:
            self._model = newmodel
        else:
            self._model += newmodel
        new_parameters = newmodel.make_params()
        self._parameters += new_parameters

        # Check linearity of expression model parameters
        if isinstance(newmodel, ExpressionModel):
            for name in newmodel.param_names:
                if not diff(newmodel.expr, name, name):
                    if name not in self._linear_parameters:
                        self._linear_parameters.append(name)
                        new_parameter_norms[name] = True
                else:
                    if name not in self._nonlinear_parameters:
                        self._nonlinear_parameters.append(name)
                        new_parameter_norms[name] = False

        # Scale the default initial model parameters
        if self._norm is not None:
            for name, norm in new_parameter_norms.copy().items():
                par = self._parameters.get(name)
                if par is None:
                    new_parameter_norms.pop(name)
                    continue
                if par.expr is None and norm:
                    value = par.value*self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)

        # Initialize the model parameters from parameters
        if prefix is None:
            prefix = ''
        if parameters is not None:
            for parameter in parameters:
                name = parameter['name']
                if not isinstance(name, str):
                    raise ValueError(
                        f'Invalid "name" value ({name}) in input parameters')
                if name not in new_parameters:
                    name = prefix+name
                    parameter['name'] = name
                if name not in new_parameters:
                    logger.warning(
                        f'Ignoring superfluous parameter info for {name}')
                    continue
                if name in self._parameters:
                    parameter.pop('name')
                    if 'norm' in parameter:
                        if not isinstance(parameter['norm'], bool):
                            raise ValueError(
                                f'Invalid "norm" value ({norm}) in the '
                                f'input parameter {name}')
                        new_parameter_norms[name] = parameter['norm']
                        parameter.pop('norm')
                    if parameter.get('expr') is not None:
                        if 'value' in parameter:
                            logger.warning(
                                f'Ignoring value in parameter {name} '
                                f'(set by expression: {parameter["expr"]})')
                            parameter.pop('value')
                        if 'vary' in parameter:
                            logger.warning(
                                f'Ignoring vary in parameter {name} '
                                f'(set by expression: {parameter["expr"]})')
                            parameter.pop('vary')
                        if 'min' in parameter:
                            logger.warning(
                                f'Ignoring min in parameter {name} '
                                f'(set by expression: {parameter["expr"]})')
                            parameter.pop('min')
                        if 'max' in parameter:
                            logger.warning(
                                f'Ignoring max in parameter {name} '
                                f'(set by expression: {parameter["expr"]})')
                            parameter.pop('max')
                    if 'vary' in parameter:
                        if not isinstance(parameter['vary'], bool):
                            raise ValueError(
                                f'Invalid "vary" value ({parameter["vary"]}) '
                                f'in the input parameter {name}')
                        if not parameter['vary']:
                            if 'min' in parameter:
                                logger.warning(
                                    f'Ignoring min in parameter {name} '
                                    f'(vary = {parameter["vary"]})')
                                parameter.pop('min')
                            if 'max' in parameter:
                                logger.warning(
                                    f'Ignoring max in parameter {name} '
                                    f'(vary = {parameter["vary"]})')
                                parameter.pop('max')
                    self._parameters[name].set(**parameter)
                    parameter['name'] = name
                else:
                    raise ValueError(
                        'Invalid parameter name in parameters ({name})')
        self._parameter_norms = {
            **self._parameter_norms,
            **new_parameter_norms,
        }

        # Initialize the model parameters from kwargs
        for name, value in {**kwargs}.items():
            full_name = f'{pprefix}{name}'
            if (full_name in new_parameter_norms
                    and isinstance(value, (int, float))):
                kwargs.pop(name)
                if self._parameters[full_name].expr is None:
                    self._parameters[full_name].set(value=value)
                else:
                    logger.warning(
                        f'Ignoring parameter {name} (set by expression: '
                        f'{self._parameters[full_name].expr})')

        # Check parameter norms
        # (also need it for expressions to renormalize the errors)
        if (self._norm is not None
                and (callable(model) or model == 'expression')):
            missing_norm = False
            for name in new_parameters.valuesdict():
                if name not in self._parameter_norms:
                    print(f'new_parameters:\n{new_parameters.valuesdict()}')
                    print(f'self._parameter_norms:\n{self._parameter_norms}')
                    logger.error(
                        f'Missing parameter normalization type for {name} in '
                        f'{model}')
                    missing_norm = True
            if missing_norm:
                raise ValueError

        return kwargs

    def eval(self, x, result=None):
        """Evaluate the best fit."""
        if result is None:
            result = self._result
        if result is None:
            return None
        return result.eval(x=np.asarray(x))-self.normalization_offset

    def fit(self, **kwargs):
        """Fit the model to the input data."""
        # Check inputs
        if self._model is None:
            logger.error('Undefined fit model')
            return None
        if 'interactive' in kwargs:
            interactive = kwargs.pop('interactive')
            if not isinstance(interactive, bool):
                raise ValueError(
                    'Invalid value of keyword argument interactive '
                    f'({interactive})')
        else:
            interactive = False
        if 'guess' in kwargs:
            guess = kwargs.pop('guess')
            if not isinstance(guess, bool):
                raise ValueError(
                    f'Invalid value of keyword argument guess ({guess})')
        else:
            guess = False
        if 'try_linear_fit' in kwargs:
            try_linear_fit = kwargs.pop('try_linear_fit')
            if not isinstance(try_linear_fit, bool):
                raise ValueError(
                    'Invalid value of keyword argument try_linear_fit '
                    f'({try_linear_fit})')
            if not self._try_linear_fit:
                logger.warning(
                    'Ignore superfluous keyword argument "try_linear_fit" '
                    '(not yet supported for callable models)')
            else:
                self._try_linear_fit = try_linear_fit
        if self._result is not None:
            if guess:
                logger.warning(
                    'Ignoring input parameter guess during refitting')
                guess = False

        # Check for circular expressions
        # RV
#        for name1, par1 in self._parameters.items():
#            if par1.expr is not None:

        # Apply mask if supplied:
        if 'mask' in kwargs:
            self._mask = kwargs.pop('mask')
        if self._mask is not None:
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(
                    f'Inconsistent x and mask dimensions ({self._x.size} vs '
                    f'{self._mask.size})')

        # Estimate initial parameters with build-in lmfit guess method
        # (only mplemented for a single model)
        if guess:
            if self._mask is None:
                self._parameters = self._model.guess(self._y, x=self._x)
            else:
                self._parameters = self._model.guess(
                    np.asarray(self._y)[~self._mask], x=self._x[~self._mask])

        # Add constant offset for a normalized model
        if self._result is None and self._norm is not None and self._norm[0]:
            self.add_model(
                'constant', prefix='tmp_normalization_offset_',
                parameters={
                    'name': 'c',
                    'value': -self._norm[0],
                    'vary': False,
                    'norm': True,
                })
#                    'value': -self._norm[0]/self._norm[1],
#                    'vary': False,
#                    'norm': False,

        # Adjust existing parameters for refit:
        if 'parameters' in kwargs:
            parameters = kwargs.pop('parameters')
            if isinstance(parameters, dict):
                parameters = (parameters, )
            elif not is_dict_series(parameters):
                raise ValueError(
                    'Invalid value of keyword argument parameters '
                    f'({parameters})')
            for par in parameters:
                name = par['name']
                if name not in self._parameters:
                    raise ValueError(
                        f'Unable to match {name} parameter {par} to an '
                        'existing one')
                if self._parameters[name].expr is not None:
                    raise ValueError(
                        f'Unable to modify {name} parameter {par} '
                        '(currently an expression)')
                if par.get('expr') is not None:
                    raise KeyError(
                        f'Invalid "expr" key in {name} parameter {par}')
                self._parameters[name].set(vary=par.get('vary'))
                self._parameters[name].set(min=par.get('min'))
                self._parameters[name].set(max=par.get('max'))
                self._parameters[name].set(value=par.get('value'))

        # Apply parameter updates through keyword arguments
        for name in set(self._parameters) & set(kwargs):
            value = kwargs.pop(name)
            if self._parameters[name].expr is None:
                self._parameters[name].set(value=value)
            else:
                logger.warning(
                    f'Ignoring parameter {name} (set by expression: '
                    f'{self._parameters[name].expr})')

        # Check for uninitialized parameters
        for name, par in self._parameters.items():
            if par.expr is None:
                value = par.value
                if value is None or np.isinf(value) or np.isnan(value):
                    if interactive:
                        value = input_num(
                            f'Enter an initial value for {name}', default=1.0)
                    else:
                        value = 1.0
                    if self._norm is None or name not in self._parameter_norms:
                        self._parameters[name].set(value=value)
                    elif self._parameter_norms[name]:
                        self._parameters[name].set(value=value*self._norm[1])

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
            # Perform a non-linear fit with lmfit
            # Prevent initial values from sitting at boundaries
            self._parameter_bounds = {
                name:{'min': par.min, 'max': par.max}
                for name, par in self._parameters.items() if par.vary}
            for par in self._parameters.values():
                if par.vary:
                    par.set(value=self._reset_par_at_boundary(par, par.value))

            # Perform the fit
#            fit_kws = None
#            if 'Dfun' in kwargs:
#                fit_kws = {'Dfun': kwargs.pop('Dfun')}
#            self._result = self._model.fit(
#                self._y_norm, self._parameters, x=self._x, fit_kws=fit_kws,
#                **kwargs)
            if self._mask is None:
                self._result = self._model.fit(
                    self._y_norm, self._parameters, x=self._x, **kwargs)
            else:
                self._result = self._model.fit(
                    np.asarray(self._y_norm)[~self._mask], self._parameters,
                    x=self._x[~self._mask], **kwargs)

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
        """
        Return a guess for the initial height, center and fwhm for a
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
            center_guesses = [center_guess]
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

        return height, center, fwhm

    def _check_linearity_model(self):
        """
        Identify the linearity of all model parameters and check if
        the model is linear or not.
        """
        if not self._try_linear_fit:
            logger.info(
                'Skip linearity check (not yet supported for callable models)')
            return False
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
        """
        Perform a linear fit by direct matrix solution with numpy.
        """
        # Third party modules
        from asteval import Interpreter

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
                # RV find another solution if expr not supported by
                #     simplify
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
        self._result = ModelResult(self._model, deepcopy(self._parameters))
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
        self._result.residual = self._result.best_fit-y
        self._result.components = self._model.components
        self._result.init_params = None

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
            for name, norm in self._parameter_norms.items():
                par = self._parameters[name]
                if par.expr is None and norm:
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
        for name, norm in self._parameter_norms.items():
            par = self._parameters[name]
            if par.expr is None and norm:
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
            if self._parameter_norms.get(name, False):
                if par.stderr is not None:
                    par.stderr *= self._norm[1]
                if par.expr is None:
                    _min = par.min
                    _max = par.max
                    value = par.value*self._norm[1]
                    if par.init_value is not None:
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
                if (name not in self._parameter_norms
                        or self._parameters[name].expr is not None):
                    init_values[name] = value
                elif self._parameter_norms[name]:
                    init_values[name] = value*self._norm[1]
            self._result.init_values = init_values
            for name, par in self._result.init_params.items():
                if par.expr is None and self._parameter_norms.get(name, False):
                    value = par.value
                    _min = par.min
                    _max = par.max
                    value *= self._norm[1]
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
                par.init_value = par.value
        # Don't renormalize chisqr, it has no useful meaning in
        #     physical units
#        self._result.chisqr *= self._norm[1]*self._norm[1]
        if self._result.covar is not None:
            for i, name in enumerate(self._result.var_names):
                if self._parameter_norms.get(name, False):
                    for j in range(len(self._result.var_names)):
                        if self._result.covar[i,j] is not None:
                            self._result.covar[i,j] *= self._norm[1]
                        if self._result.covar[j,i] is not None:
                            self._result.covar[j,i] *= self._norm[1]
        # Don't renormalize redchi, it has no useful meaning in
        #     physical units
#        self._result.redchi *= self._norm[1]*self._norm[1]
        if self._result.residual is not None:
            self._result.residual *= self._norm[1]

    def _reset_par_at_boundary(self, par, value):
        assert par.vary
        name = par.name
        _min = self._parameter_bounds[name]['min']
        _max = self._parameter_bounds[name]['max']
        if np.isinf(_min):
            if not np.isinf(_max):
                if self._parameter_norms.get(name, False):
                    upp = _max-0.1*self._y_range
                elif _max == 0.0:
                    upp = _max-0.1
                else:
                    upp = _max-0.1*abs(_max)
                if value >= upp:
                    return upp
        else:
            if np.isinf(_max):
                if self._parameter_norms.get(name, False):
                    low = _min + 0.1*self._y_range
                elif _min == 0.0:
                    low = _min+0.1
                else:
                    low = _min + 0.1*abs(_min)
                if value <= low:
                    return low
            else:
                low = 0.9*_min + 0.1*_max
                upp = 0.1*_min + 0.9*_max
                if value <= low:
                    return low
                if value >= upp:
                    return upp
        return value


class FitMultipeak(Fit):
    """
    Wrapper to the Fit class to fit data with multiple peaks
    """
    def __init__(self, y, x=None, normalize=True):
        """Initialize FitMultipeak."""
        super().__init__(y, x=x, normalize=normalize)
        self._fwhm_max = None
        self._sigma_max = None

    @classmethod
    def fit_multipeak(
            cls, y, centers, x=None, normalize=True, peak_models='gaussian',
            center_exprs=None, fit_type=None, background=None, fwhm_max=None,
            print_report=False, plot=False, x_eval=None):
        """Class method for FitMultipeak.

        Make sure that centers and fwhm_max are in the correct units
        and consistent with expr for a uniform fit (fit_type ==
        'uniform').
        """
        if (x_eval is not None
                and not isinstance(x_eval, (tuple, list, np.ndarray))):
            raise ValueError(f'Invalid parameter x_eval ({x_eval})')
        fit = cls(y, x=x, normalize=normalize)
        success = fit.fit(
            centers=centers, fit_type=fit_type, peak_models=peak_models,
            fwhm_max=fwhm_max, center_exprs=center_exprs,
            background=background, print_report=print_report, plot=plot)
        if x_eval is None:
            best_fit = fit.best_fit
        else:
            best_fit = fit.eval(x_eval)
        if success:
            return (
                best_fit, fit.residual, fit.best_values, fit.best_errors,
                fit.redchi, fit.success)
        return np.array([]), np.array([]), {}, {}, FLOAT_MAX, False

    def fit(
            self, centers=None, fit_type=None, peak_models=None,
            center_exprs=None, fwhm_max=None, background=None,
            print_report=False, plot=True, param_constraint=False, **kwargs):
        """Fit the model to the input data."""
        if centers is None:
            raise ValueError('Missing required parameter centers')
        if not isinstance(centers, (int, float, tuple, list, np.ndarray)):
            raise ValueError(f'Invalid parameter centers ({centers})')
        self._fwhm_max = fwhm_max
        self._create_model(
            centers, fit_type, peak_models, center_exprs, background,
            param_constraint)

        # Perform the fit
        try:
            if param_constraint:
                super().fit(
                    fit_kws={'xtol': 1.e-5, 'ftol': 1.e-5, 'gtol': 1.e-5})
            else:
                super().fit()
        except:
            return False

        # Check for valid fit parameter results
        fit_failure = self._check_validity()
        success = True
        if fit_failure:
            if param_constraint:
                logger.warning(
                    '  -> Should not happen with param_constraint set, '
                    'fail the fit')
                success = False
            else:
                logger.info('  -> Retry fitting with constraints')
                self.fit(
                    centers, fit_type, peak_models, center_exprs,
                    fwhm_max=fwhm_max, background=background,
                    print_report=print_report, plot=plot,
                    param_constraint=True)
        else:
            # Print report and plot components if requested
            if print_report:
                self.print_fit_report()
            if plot:
                self.plot(
                    skip_init=True, plot_comp=True, plot_comp_legends=True,
                    plot_residual=True)

        return success

    def _create_model(
            self, centers, fit_type=None, peak_models=None, center_exprs=None,
            background=None, param_constraint=False):
        """Create the multipeak model."""
        # Third party modules
        from asteval import Interpreter

        if isinstance(centers, (int, float)):
            centers = [centers]
        num_peaks = len(centers)
        if peak_models is None:
            peak_models = num_peaks*['gaussian']
        elif (isinstance(peak_models, str)
                and peak_models in ('gaussian', 'lorentzian')):
            peak_models = num_peaks*[peak_models]
        else:
            raise ValueError(f'Invalid parameter peak model ({peak_models})')
        if len(peak_models) != num_peaks:
            raise ValueError(
                'Inconsistent number of peaks in peak_models '
                f'({len(peak_models)} vs {num_peaks})')
        if num_peaks == 1:
            if fit_type is not None:
                logger.debug('Ignoring fit_type input for fitting one peak')
            fit_type = None
            if center_exprs is not None:
                logger.debug(
                    'Ignoring center_exprs input for fitting one peak')
                center_exprs = None
        else:
            if fit_type == 'uniform':
                if center_exprs is None:
                    center_exprs = [f'scale_factor*{cen}' for cen in centers]
                if len(center_exprs) != num_peaks:
                    raise ValueError(
                        'Inconsistent number of peaks in center_exprs '
                        f'({len(center_exprs)} vs {num_peaks})')
            elif fit_type == 'unconstrained' or fit_type is None:
                if center_exprs is not None:
                    logger.warning(
                        'Ignoring center_exprs input for unconstrained fit')
                    center_exprs = None
            else:
                raise ValueError(
                    f'Invalid parameter fit_type ({fit_type})')
        self._sigma_max = None
        if param_constraint:
            min_value = FLOAT_MIN
            if self._fwhm_max is not None:
                self._sigma_max = np.zeros(num_peaks)
        else:
            min_value = None

        # Reset the fit
        self._model = None
        self._parameters = Parameters()
        self._result = None

        # Add background model(s)
        if background is not None:
            if isinstance(background, dict):
                background = [background]
            if isinstance(background, str):
                self.add_model(background, prefix='bkgd_')
            elif is_dict_series(background):
                for model in deepcopy(background):
                    if 'model' not in model:
                        raise KeyError(
                            'Missing keyword "model" in model in background '
                            f'({model})')
                    name = model.pop('model')
                    parameters = model.pop('parameters', None)
                    self.add_model(
                        name, prefix=f'bkgd_{name}_', parameters=parameters,
                        **model)
            else:
                raise ValueError(
                    f'Invalid parameter background ({background})')

        # Add peaks and guess initial fit parameters
        ast = Interpreter()
        if num_peaks == 1:
            height_init, cen_init, fwhm_init = self.guess_init_peak(
                self._x, self._y)
            if self._fwhm_max is not None and fwhm_init > self._fwhm_max:
                fwhm_init = self._fwhm_max
            ast(f'fwhm = {fwhm_init}')
            ast(f'height = {height_init}')
            sig_init = ast(fwhm_factor[peak_models[0]])
            amp_init = ast(height_factor[peak_models[0]])
            sig_max = None
            if self._sigma_max is not None:
                ast(f'fwhm = {self._fwhm_max}')
                sig_max = ast(fwhm_factor[peak_models[0]])
                self._sigma_max[0] = sig_max
            self.add_model(
                peak_models[0],
                parameters=(
                    {'name': 'amplitude', 'value': amp_init, 'min': min_value},
                    {'name': 'center', 'value': cen_init, 'min': min_value},
                    {'name': 'sigma', 'value': sig_init, 'min': min_value,
                     'max': sig_max},
                ))
        else:
            if fit_type == 'uniform':
                self.add_parameter(name='scale_factor', value=1.0)
            for i in range(num_peaks):
                height_init, cen_init, fwhm_init = self.guess_init_peak(
                    self._x, self._y, i, center_guess=centers)
                if self._fwhm_max is not None and fwhm_init > self._fwhm_max:
                    fwhm_init = self._fwhm_max
                ast(f'fwhm = {fwhm_init}')
                ast(f'height = {height_init}')
                sig_init = ast(fwhm_factor[peak_models[i]])
                amp_init = ast(height_factor[peak_models[i]])
                sig_max = None
                if self._sigma_max is not None:
                    ast(f'fwhm = {self._fwhm_max}')
                    sig_max = ast(fwhm_factor[peak_models[i]])
                    self._sigma_max[i] = sig_max
                if fit_type == 'uniform':
                    self.add_model(
                        peak_models[i], prefix=f'peak{i+1}_',
                        parameters=(
                            {'name': 'amplitude', 'value': amp_init,
                             'min': min_value},
                            {'name': 'center', 'expr': center_exprs[i]},
                            {'name': 'sigma', 'value': sig_init,
                             'min': min_value, 'max': sig_max},
                        ))
                else:
                    self.add_model(
                        'gaussian',
                        prefix=f'peak{i+1}_',
                        parameters=(
                            {'name': 'amplitude', 'value': amp_init,
                             'min': min_value},
                            {'name': 'center', 'value': cen_init,
                             'min': min_value},
                            {'name': 'sigma', 'value': sig_init,
                             'min': min_value, 'max': sig_max},
                        ))

    def _check_validity(self):
        """Check for valid fit parameter results."""
        fit_failure = False
        index = re_compile(r'\d+')
        for name, par in self.best_parameters().items():
            if 'bkgd' in name:
                if ((name.endswith('amplitude') and par['value'] <= 0.0)
                        or (name.endswith('decay') and par['value'] <= 0.0)):
                    logger.info(
                        f'Invalid fit result for {name} ({par["value"]})')
                    fit_failure = True
            elif (((name.endswith('amplitude') or name.endswith('height'))
                    and par['value'] <= 0.0)
                    or ((name.endswith('sigma') or name.endswith('fwhm'))
                        and par['value'] <= 0.0)
                    or (name.endswith('center') and par['value'] <= 0.0)
                    or (name == 'scale_factor' and par['value'] <= 0.0)):
                logger.info(f'Invalid fit result for {name} ({par["value"]})')
                fit_failure = True
            if ('bkgd' not in name and name.endswith('sigma')
                    and self._sigma_max is not None):
                if name == 'sigma':
                    sigma_max = self._sigma_max[0]
                else:
                    sigma_max = self._sigma_max[
                        int(index.search(name).group())-1]
                if par['value'] > sigma_max:
                    logger.info(
                        f'Invalid fit result for {name} ({par["value"]})')
                    fit_failure = True
                elif par['value'] == sigma_max:
                    logger.warning(
                        f'Edge result on for {name} ({par["value"]})')
            if ('bkgd' not in name and name.endswith('fwhm')
                    and self._fwhm_max is not None):
                if par['value'] > self._fwhm_max:
                    logger.info(
                        f'Invalid fit result for {name} ({par["value"]})')
                    fit_failure = True
                elif par['value'] == self._fwhm_max:
                    logger.warning(
                        f'Edge result on for {name} ({par["value"]})')
        return fit_failure


class FitMap(Fit):
    """
    Wrapper to the Fit class to fit dat on a N-dimensional map
    """
    def __init__(
            self, ymap, x=None, models=None, normalize=True, transpose=None,
            **kwargs):
        """Initialize FitMap."""
        super().__init__(None)
        self._best_errors = None
        self._best_fit = None
        self._best_parameters = None
        self._best_values = None
        self._inv_transpose = None
        self._max_nfev = None
        self._memfolder = None
        self._new_parameters = None
        self._out_of_bounds = None
        self._plot = False
        self._print_report = False
        self._redchi = None
        self._redchi_cutoff = 0.1
        self._skip_init = True
        self._success = None
        self._transpose = None
        self._try_no_bounds = True

        # At this point the fastest index should always be the signal
        #     dimension so that the slowest ndim-1 dimensions are the
        #     map dimensions
        if isinstance(ymap, (tuple, list, np.ndarray)):
            self._x = np.asarray(x)
        elif HAVE_XARRAY and isinstance(ymap, xr.DataArray):
            if x is not None:
                logger.warning('Ignoring superfluous input x ({x})')
            self._x = np.asarray(ymap[ymap.dims[-1]])
        else:
            raise ValueError('Invalid parameter ymap ({ymap})')
        self._ymap = ymap

        # Verify the input parameters
        if self._x.ndim != 1:
            raise ValueError(f'Invalid dimension for input x {self._x.ndim}')
        if self._ymap.ndim < 2:
            raise ValueError(
                'Invalid number of dimension of the input dataset '
                f'{self._ymap.ndim}')
        if self._x.size != self._ymap.shape[-1]:
            raise ValueError(
                f'Inconsistent x and y dimensions ({self._x.size} vs '
                f'{self._ymap.shape[-1]})')
        if not isinstance(normalize, bool):
            logger.warning(
                f'Invalid value for normalize ({normalize}) in Fit.__init__: '
                'setting normalize to True')
            normalize = True
        if isinstance(transpose, bool) and not transpose:
            transpose = None
        if transpose is not None and self._ymap.ndim < 3:
            logger.warning(
                f'Transpose meaningless for {self._ymap.ndim-1}D data maps: '
                'ignoring transpose')
        if transpose is not None:
            if (self._ymap.ndim == 3 and isinstance(transpose, bool)
                    and transpose):
                self._transpose = (1, 0)
            elif not isinstance(transpose, (tuple, list)):
                logger.warning(
                    f'Invalid data type for transpose ({transpose}, '
                    f'{type(transpose)}): setting transpose to False')
            elif transpose != self._ymap.ndim-1:
                logger.warning(
                    f'Invalid dimension for transpose ({transpose}, must be '
                    f'equal to {self._ymap.ndim-1}): '
                    'setting transpose to False')
            elif any(i not in transpose for i in range(len(transpose))):
                logger.warning(
                    f'Invalid index in transpose ({transpose}): '
                    'setting transpose to False')
            elif not all(i == transpose[i] for i in range(self._ymap.ndim-1)):
                self._transpose = transpose
            if self._transpose is not None:
                self._inv_transpose = tuple(
                    self._transpose.index(i)
                    for i in range(len(self._transpose)))

        # Flatten the map (transpose if requested)
        # Store the flattened map in self._ymap_norm, whether
        #     normalized or not
        if self._transpose is not None:
            self._ymap_norm = np.transpose(
                np.asarray(self._ymap),
                list(self._transpose) + [len(self._transpose)])
        else:
            self._ymap_norm = np.asarray(self._ymap)
        self._map_dim = int(self._ymap_norm.size/self._x.size)
        self._map_shape = self._ymap_norm.shape[:-1]
        self._ymap_norm = np.reshape(
            self._ymap_norm, (self._map_dim, self._x.size))

        # Check if a mask is provided
        if 'mask' in kwargs:
            self._mask = kwargs.pop('mask')
        if self._mask is None:
            ymap_min = float(self._ymap_norm.min())
            ymap_max = float(self._ymap_norm.max())
        else:
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(
                    f'Inconsistent mask dimension ({self._x.size} vs '
                    f'{self._mask.size})')
            ymap_masked = np.asarray(self._ymap_norm)[:,~self._mask]
            ymap_min = float(ymap_masked.min())
            ymap_max = float(ymap_masked.max())

        # Normalize the data
        self._y_range = ymap_max-ymap_min
        if normalize and self._y_range > 0.0:
            self._norm = (ymap_min, self._y_range)
            self._ymap_norm = (self._ymap_norm-self._norm[0]) / self._norm[1]
        else:
            self._redchi_cutoff *= self._y_range**2
        if models is not None:
            if callable(models) or isinstance(models, str):
                kwargs = self.add_model(models, **kwargs)
            elif isinstance(models, (tuple, list)):
                for model in models:
                    kwargs = self.add_model(model, **kwargs)
            self.fit(**kwargs)

    @classmethod
    def fit_map(cls, ymap, models, x=None, normalize=True, **kwargs):
        """Class method for FitMap."""
        return cls(ymap, x=x, models=models, normalize=normalize, **kwargs)

    @property
    def best_errors(self):
        """Return errors in the best fit parameters."""
        return self._best_errors

    @property
    def best_fit(self):
        """Return the best fits."""
        return self._best_fit

    @property
    def best_results(self):
        """
        Convert the input DataArray to a data set and add the fit
        results.
        """
        if (self.best_values is None or self.best_errors is None
                or self.best_fit is None):
            return None
        if not HAVE_XARRAY:
            logger.warning('Unable to load xarray module')
            return None
        best_values = self.best_values
        best_errors = self.best_errors
        if isinstance(self._ymap, xr.DataArray):
            best_results = self._ymap.to_dataset()
            dims = self._ymap.dims
            fit_name = f'{self._ymap.name}_fit'
        else:
            coords = {
                f'dim{n}_index':([f'dim{n}_index'], range(self._ymap.shape[n]))
                for n in range(self._ymap.ndim-1)}
            coords['x'] = (['x'], self._x)
            dims = list(coords.keys())
            best_results = xr.Dataset(coords=coords)
            best_results['y'] = (dims, self._ymap)
            fit_name = 'y_fit'
        best_results[fit_name] = (dims, self.best_fit)
        if self._mask is not None:
            best_results['mask'] = self._mask
        for n in range(best_values.shape[0]):
            best_results[f'{self._best_parameters[n]}_values'] = \
                (dims[:-1], best_values[n])
            best_results[f'{self._best_parameters[n]}_errors'] = \
                (dims[:-1], best_errors[n])
        best_results.attrs['components'] = self.components
        return best_results

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
        """
        Return the covarience matrices of the best fit parameters.
        """
        logger.warning('Undefined property covar')

    @property
    def max_nfev(self):
        """
        Return the maximum number of function evaluations for each fit.
        """
        return self._max_nfev

    @property
    def num_func_eval(self):
        """
        Return the number of function evaluations for each best fit.
        """
        logger.warning('Undefined property num_func_eval')

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
        """
        Return the variable names for the covarience matrix property.
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
            self._memfolder = None
        except:
            logger.warning('Could not clean-up automatically.')

    def plot(
            self, dims=None, y_title=None, plot_residual=False,
            plot_comp_legends=False, plot_masked_data=True, **kwargs):
        """Plot the best fits."""
        if dims is None:
            dims = [0]*len(self._map_shape)
        if (not isinstance(dims, (list, tuple))
                or len(dims) != len(self._map_shape)):
            raise ValueError('Invalid parameter dims ({dims})')
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

    def fit(self, **kwargs):
        """Fit the model to the input data."""
        # Check input parameters
        if self._model is None:
            logger.error('Undefined fit model')
        if 'num_proc' in kwargs:
            num_proc = kwargs.pop('num_proc')
            if not is_int(num_proc, ge=1):
                raise ValueError(
                    'Invalid value for keyword argument num_proc ({num_proc})')
        else:
            num_proc = cpu_count()
        if num_proc > 1 and not HAVE_JOBLIB:
            logger.warning(
                'Missing joblib in the conda environment, running serially')
            num_proc = 1
        if num_proc > cpu_count():
            logger.warning(
                f'The requested number of processors ({num_proc}) exceeds the '
                'maximum number of processors, num_proc reduced to '
                f'({cpu_count()})')
            num_proc = cpu_count()
        if 'try_no_bounds' in kwargs:
            self._try_no_bounds = kwargs.pop('try_no_bounds')
            if not isinstance(self._try_no_bounds, bool):
                raise ValueError(
                    'Invalid value for keyword argument try_no_bounds '
                    f'({self._try_no_bounds})')
        if 'redchi_cutoff' in kwargs:
            self._redchi_cutoff = kwargs.pop('redchi_cutoff')
            if not is_num(self._redchi_cutoff, gt=0):
                raise ValueError(
                    'Invalid value for keyword argument redchi_cutoff'
                    f'({self._redchi_cutoff})')
        if 'print_report' in kwargs:
            self._print_report = kwargs.pop('print_report')
            if not isinstance(self._print_report, bool):
                raise ValueError(
                    'Invalid value for keyword argument print_report'
                    f'({self._print_report})')
        if 'plot' in kwargs:
            self._plot = kwargs.pop('plot')
            if not isinstance(self._plot, bool):
                raise ValueError(
                    'Invalid value for keyword argument plot'
                    f'({self._plot})')
        if 'skip_init' in kwargs:
            self._skip_init = kwargs.pop('skip_init')
            if not isinstance(self._skip_init, bool):
                raise ValueError(
                    'Invalid value for keyword argument skip_init'
                    f'({self._skip_init})')

        # Apply mask if supplied:
        if 'mask' in kwargs:
            self._mask = kwargs.pop('mask')
        if self._mask is not None:
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(
                    f'Inconsistent x and mask dimensions ({self._x.size} vs '
                    f'{self._mask.size})')

        # Add constant offset for a normalized single component model
        if self._result is None and self._norm is not None and self._norm[0]:
            self.add_model(
                'constant',
                prefix='tmp_normalization_offset_',
                parameters={
                    'name': 'c',
                    'value': -self._norm[0],
                    'vary': False,
                    'norm': True,
                })
#                    'value': -self._norm[0]/self._norm[1],
#                    'vary': False,
#                    'norm': False,

        # Adjust existing parameters for refit:
        if 'parameters' in kwargs:
            parameters = kwargs.pop('parameters')
            if isinstance(parameters, dict):
                parameters = (parameters, )
            elif not is_dict_series(parameters):
                raise ValueError(
                    'Invalid value for keyword argument parameters'
                    f'({parameters})')
            for par in parameters:
                name = par['name']
                if name not in self._parameters:
                    raise ValueError(
                        f'Unable to match {name} parameter {par} to an '
                        'existing one')
                if self._parameters[name].expr is not None:
                    raise ValueError(
                        f'Unable to modify {name} parameter {par} '
                        '(currently an expression)')
                value = par.get('value')
                vary = par.get('vary')
                if par.get('expr') is not None:
                    raise KeyError(
                        f'Invalid "expr" key in {name} parameter {par}')
                self._parameters[name].set(
                    value=value, vary=vary, min=par.get('min'),
                    max=par.get('max'))
                # Overwrite existing best values for fixed parameters
                #     when a value is specified
                if isinstance(value, (int, float)) and vary is False:
                    for i, nname in enumerate(self._best_parameters):
                        if nname == name:
                            self._best_values[i] = value

        # Check for uninitialized parameters
        for name, par in self._parameters.items():
            if par.expr is None:
                value = par.value
                if value is None or np.isinf(value) or np.isnan(value):
                    value = 1.0
                    if self._norm is None or name not in self._parameter_norms:
                        self._parameters[name].set(value=value)
                    elif self._parameter_norms[name]:
                        self._parameters[name].set(value=value*self._norm[1])

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
            self._redchi = None
            self._success = None
            self._best_fit = None
            self._best_errors = None
            assert self._best_values is not None
            assert self._best_values.shape[0] == num_best_parameters
            assert self._best_values.shape[1:] == self._map_shape
            if self._transpose is not None:
                self._best_values = np.transpose(
                    self._best_values, [0]+[i+1 for i in self._transpose])
            self._best_values = [
                np.reshape(self._best_values[i], self._map_dim)
                for i in range(num_best_parameters)]
            if self._norm is not None:
                for i, name in enumerate(self._best_parameters):
                    if self._parameter_norms.get(name, False):
                        self._best_values[i] /= self._norm[1]

        # Normalize the initial parameters
        #     (and best values for a refit)
        self._normalize()

        # Prevent initial values from sitting at boundaries
        self._parameter_bounds = {
            name:{'min': par.min, 'max': par.max}
            for name, par in self._parameters.items() if par.vary}
        for name, par in self._parameters.items():
            if par.vary:
                par.set(value=self._reset_par_at_boundary(par, par.value))

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
            self._memfolder = './joblib_memmap'
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

        if num_proc == 1:
            # Perform the remaining fits serially
            for n in range(1, self._map_dim):
                self._fit(n, current_best_values, **kwargs)
        else:
            # Perform the remaining fits in parallel
            num_fit = self._map_dim-1
            if num_proc > num_fit:
                logger.warning(
                    f'The requested number of processors ({num_proc}) exceeds '
                    f'the number of fits, num_proc reduced to ({num_fit})')
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
                        (current_best_values, num_fit_batch, n_start, **kwargs)
                    for n_start in range(1, self._map_dim, num_fit_batch))

        # Renormalize the initial parameters for external use
        if self._norm is not None and self._normalized:
            init_values = {}
            for name, value in self._result.init_values.items():
                if (name not in self._parameter_norms
                        or self._parameters[name].expr is not None):
                    init_values[name] = value
                elif self._parameter_norms[name]:
                    init_values[name] = value*self._norm[1]
            self._result.init_values = init_values
            for name, par in self._result.init_params.items():
                if par.expr is None and self._parameter_norms.get(name, False):
                    _min = par.min
                    _max = par.max
                    value = par.value*self._norm[1]
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
                par.init_value = par.value

        # Remap the best results
        self._out_of_bounds = np.copy(np.reshape(
            self._out_of_bounds_flat, self._map_shape))
        self._max_nfev = np.copy(np.reshape(
            self._max_nfev_flat, self._map_shape))
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
            for name, norm in self._parameter_norms.items():
                par = self._parameters[name]
                if par.expr is None and norm:
                    value = par.value*self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != FLOAT_MIN:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != FLOAT_MIN:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)

        # Free the shared memory
        self.freemem()

    def _fit_parallel(self, current_best_values, num, n_start, **kwargs):
        num = min(num, self._map_dim-n_start)
        for n in range(num):
            self._fit(n_start+n, current_best_values, **kwargs)

    def _fit(self, n, current_best_values, return_result=False, **kwargs):
        # Set parameters to current best values, but prevent them from
        #     sitting at boundaries
        if self._new_parameters is None:
            # Initial fit
            for name, value in current_best_values.items():
                par = self._parameters[name]
                par.set(value=self._reset_par_at_boundary(par, value))
        else:
            # Refit
            for i, name in enumerate(self._best_parameters):
                par = self._parameters[name]
                if name in self._new_parameters:
                    if name in current_best_values:
                        par.set(value=self._reset_par_at_boundary(
                            par, current_best_values[name]))
                elif par.expr is None:
                    par.set(value=self._best_values[i][n])
        if self._mask is None:
            result = self._model.fit(
                self._ymap_norm[n], self._parameters, x=self._x, **kwargs)
        else:
            result = self._model.fit(
                self._ymap_norm[n][~self._mask], self._parameters,
                x=self._x[~self._mask], **kwargs)
        out_of_bounds = False
        for name, par in self._parameter_bounds.items():
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
                self._parameters[name].set(min=par['min'], max=par['max'])
            # Set parameters to current best values, but prevent them
            #     from sitting at boundaries
            if self._new_parameters is None:
                # Initial fit
                for name, value in current_best_values.items():
                    par = self._parameters[name]
                    par.set(value=self._reset_par_at_boundary(par, value))
            else:
                # Refit
                for i, name in enumerate(self._best_parameters):
                    par = self._parameters[name]
                    if name in self._new_parameters:
                        if name in current_best_values:
                            par.set(value=self._reset_par_at_boundary(par,
                                    current_best_values[name]))
                    elif par.expr is None:
                        par.set(value=self._best_values[i][n])
            if self._mask is None:
                result = self._model.fit(
                    self._ymap_norm[n], self._parameters, x=self._x, **kwargs)
            else:
                result = self._model.fit(
                    self._ymap_norm[n][~self._mask], self._parameters,
                    x=self._x[~self._mask], **kwargs)
            out_of_bounds = False
            for name, par in self._parameter_bounds.items():
                value = result.params[name].value
                if not np.isinf(par['min']) and value < par['min']:
                    out_of_bounds = True
                    break
                if not np.isinf(par['max']) and value > par['max']:
                    out_of_bounds = True
                    break
            # Reset parameters back to unbound
            for name in self._parameter_bounds.keys():
                self._parameters[name].set(min=-np.inf, max=np.inf)
        assert not out_of_bounds
        if result.redchi >= self._redchi_cutoff:
            result.success = False
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
            logger.warning(f'Fit for n = {n} failed: {result.lmdif_message}')
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

    def _renormalize(self, n, result):
        self._redchi_flat[n] = np.float64(result.redchi)
        self._success_flat[n] = result.success
        if self._norm is None or not self._normalized:
            self._best_fit_flat[n] = result.best_fit
            for i, name in enumerate(self._best_parameters):
                self._best_values_flat[i][n] = np.float64(
                    result.params[name].value)
                self._best_errors_flat[i][n] = np.float64(
                    result.params[name].stderr)
        else:
            pars = set(self._parameter_norms) & set(self._best_parameters)
            for name, par in result.params.items():
                if name in pars and self._parameter_norms[name]:
                    if par.stderr is not None:
                        par.stderr *= self._norm[1]
                    if par.expr is None:
                        par.value *= self._norm[1]
                        if self._print_report:
                            if par.init_value is not None:
                                par.init_value *= self._norm[1]
                            if (not np.isinf(par.min)
                                    and abs(par.min) != FLOAT_MIN):
                                par.min *= self._norm[1]
                            if (not np.isinf(par.max)
                                    and abs(par.max) != FLOAT_MIN):
                                par.max *= self._norm[1]
            self._best_fit_flat[n] = (
                result.best_fit*self._norm[1] + self._norm[0])
            for i, name in enumerate(self._best_parameters):
                self._best_values_flat[i][n] = np.float64(
                    result.params[name].value)
                self._best_errors_flat[i][n] = np.float64(
                    result.params[name].stderr)
            if self._plot:
                if not self._skip_init:
                    result.init_fit = (
                        result.init_fit*self._norm[1] + self._norm[0])
                result.best_fit = np.copy(self._best_fit_flat[n])
