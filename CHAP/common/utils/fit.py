#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:22 2021

@author: rv43
"""

import logging

from asteval import Interpreter, get_ast_names
from copy import deepcopy
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from lmfit.models import ConstantModel, LinearModel, QuadraticModel, PolynomialModel,\
        ExponentialModel, StepModel, RectangleModel, ExpressionModel, GaussianModel,\
        LorentzianModel
import numpy as np
from os import cpu_count, getpid, listdir, mkdir, path
from re import compile, sub
from shutil import rmtree
try:
    from sympy import diff, simplify
except:
    pass
try:
    from joblib import Parallel, delayed
    have_joblib = True
except:
    have_joblib = False
try:
    import xarray as xr
    have_xarray = True
except:
    have_xarray = False

try:
    from .general import illegal_value, is_int, is_dict_series, is_index, index_nearest, \
            almost_equal, quick_plot #, eval_expr
except:
    try:
        from sys import path as syspath
        syspath.append(f'/nfs/chess/user/rv43/msnctools/msnctools')
        from general import illegal_value, is_int, is_dict_series, is_index, index_nearest, \
                almost_equal, quick_plot #, eval_expr
    except:
        from general import illegal_value, is_int, is_dict_series, is_index, index_nearest, \
                almost_equal, quick_plot #, eval_expr

from sys import float_info
float_min = float_info.min
float_max = float_info.max

# sigma = fwhm_factor*fwhm
fwhm_factor = {
    'gaussian': f'fwhm/(2*sqrt(2*log(2)))',
    'lorentzian': f'0.5*fwhm',
    'splitlorentzian': f'0.5*fwhm', # sigma = sigma_r
    'voight': f'0.2776*fwhm', # sigma = gamma
    'pseudovoight': f'0.5*fwhm'} # fraction = 0.5

# amplitude = height_factor*height*fwhm
height_factor = {
    'gaussian': f'height*fwhm*0.5*sqrt(pi/log(2))',
    'lorentzian': f'height*fwhm*0.5*pi',
    'splitlorentzian': f'height*fwhm*0.5*pi', # sigma = sigma_r
    'voight': f'3.334*height*fwhm', # sigma = gamma
    'pseudovoight': f'1.268*height*fwhm'} # fraction = 0.5

class Fit:
    """Wrapper class for lmfit
    """
    def __init__(self, y, x=None, models=None, normalize=True, **kwargs):
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
            try_linear_fit = kwargs.pop('try_linear_fit')
            if not isinstance(try_linear_fit, bool):
                illegal_value(try_linear_fit, 'try_linear_fit', 'Fit.fit', raise_error=True)
            self._try_linear_fit = try_linear_fit
        if y is not None:
            if isinstance(y, (tuple, list, np.ndarray)):
                self._x = np.asarray(x)
            elif have_xarray and isinstance(y, xr.DataArray):
                if x is not None:
                    logging.warning('Ignoring superfluous input x ({x}) in Fit.__init__')
                if y.ndim != 1:
                    illegal_value(y.ndim, 'DataArray dimensions', 'Fit:__init__', raise_error=True)
                self._x = np.asarray(y[y.dims[0]])
            else:
                illegal_value(y, 'y', 'Fit:__init__', raise_error=True)
            self._y = y
            if self._x.ndim != 1:
                raise ValueError(f'Invalid dimension for input x ({self._x.ndim})')
            if self._x.size != self._y.size:
                raise ValueError(f'Inconsistent x and y dimensions ({self._x.size} vs '+
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
                    raise ValueError(f'Inconsistent x and mask dimensions ({self._x.size} vs '+
                            f'{self._mask.size})')
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
        return(cls(y, x=x, models=models, normalize=normalize, **kwargs))

    @property
    def best_errors(self):
        if self._result is None:
            return(None)
        return({name:self._result.params[name].stderr for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'})

    @property
    def best_fit(self):
        if self._result is None:
            return(None)
        return(self._result.best_fit)

    @property
    def best_parameters(self):
        if self._result is None:
            return(None)
        parameters = {}
        for name in sorted(self._result.params):
            if name != 'tmp_normalization_offset_c':
                par = self._result.params[name]
                parameters[name] = {'value': par.value, 'error': par.stderr,
                        'init_value': par.init_value, 'min': par.min, 'max': par.max,
                        'vary': par.vary, 'expr': par.expr}
        return(parameters)

    @property
    def best_results(self):
        """Convert the input data array to a data set and add the fit results.
        """
        if self._result is None:
            return(None)
        if isinstance(self._y, xr.DataArray):
            best_results = self._y.to_dataset()
            dims = self._y.dims
            fit_name = f'{self._y.name}_fit'
        else:
            coords = {'x': (['x'], self._x)}
            dims = ('x')
            best_results = xr.Dataset(coords=coords)
            best_results['y'] = (dims, self._y)
            fit_name = 'y_fit'
        best_results[fit_name] = (dims, self.best_fit)
        if self._mask is not None:
            best_results['mask'] = self._mask
        best_results.coords['par_names'] = ('peak', [name for name in self.best_values.keys()])
        best_results['best_values'] = (['par_names'], [v for v in self.best_values.values()])
        best_results['best_errors'] = (['par_names'], [v for v in self.best_errors.values()])
        best_results.attrs['components'] = self.components
        return(best_results)

    @property
    def best_values(self):
        if self._result is None:
            return(None)
        return({name:self._result.params[name].value for name in sorted(self._result.params)
                if name != 'tmp_normalization_offset_c'})

    @property
    def chisqr(self):
        if self._result is None:
            return(None)
        return(self._result.chisqr)

    @property
    def components(self):
        components = {}
        if self._result is None:
            logging.warning('Unable to collect components in Fit.components')
            return(components)
        for component in self._result.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            parameters = {}
            for name in component.param_names:
                par = self._parameters[name]
                parameters[name] = {'free': par.vary, 'value': self._result.params[name].value}
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
                if len(prefix):
                    if prefix[-1] == '_':
                        prefix = prefix[:-1]
                    name = f'{prefix} ({component._name})'
                else:
                    name = f'{component._name}'
            if expr is None:
                components[name] = {'parameters': parameters}
            else:
                components[name] = {'expr': expr, 'parameters': parameters}
        return(components)

    @property
    def covar(self):
        if self._result is None:
            return(None)
        return(self._result.covar)

    @property
    def init_parameters(self):
        if self._result is None or self._result.init_params is None:
            return(None)
        parameters = {}
        for name in sorted(self._result.init_params):
            if name != 'tmp_normalization_offset_c':
                par = self._result.init_params[name]
                parameters[name] = {'value': par.value, 'min': par.min, 'max': par.max,
                        'vary': par.vary, 'expr': par.expr}
        return(parameters)

    @property
    def init_values(self):
        if self._result is None or self._result.init_params is None:
            return(None)
        return({name:self._result.init_params[name].value for name in
                sorted(self._result.init_params) if name != 'tmp_normalization_offset_c'})

    @property
    def normalization_offset(self):
        if self._result is None:
            return(None)
        if self._norm is None:
            return(0.0)
        else:
            if self._result.init_params is not None:
                normalization_offset = self._result.init_params['tmp_normalization_offset_c']
            else:
                normalization_offset = self._result.params['tmp_normalization_offset_c']
            return(normalization_offset)

    @property
    def num_func_eval(self):
        if self._result is None:
            return(None)
        return(self._result.nfev)

    @property
    def parameters(self):
        return({name:{'min': par.min, 'max': par.max, 'vary': par.vary, 'expr': par.expr}
                for name, par in self._parameters.items() if name != 'tmp_normalization_offset_c'})

    @property
    def redchi(self):
        if self._result is None:
            return(None)
        return(self._result.redchi)

    @property
    def residual(self):
        if self._result is None:
            return(None)
        return(self._result.residual)

    @property
    def success(self):
        if self._result is None:
            return(None)
        if not self._result.success:
#            print(f'ier = {self._result.ier}')
#            print(f'lmdif_message = {self._result.lmdif_message}')
#            print(f'message = {self._result.message}')
#            print(f'nfev = {self._result.nfev}')
#            print(f'redchi = {self._result.redchi}')
#            print(f'success = {self._result.success}')
            if self._result.ier == 0 or self._result.ier == 5:
                logging.warning(f'ier = {self._result.ier}: {self._result.message}')
            else:
                logging.warning(f'ier = {self._result.ier}: {self._result.message}')
                return(True)
#            self.print_fit_report()
#            self.plot()
        return(self._result.success)

    @property
    def var_names(self):
        """Intended to be used with covar
        """
        if self._result is None:
            return(None)
        return(getattr(self._result, 'var_names', None))

    @property
    def x(self):
        return(self._x)

    @property
    def y(self):
        return(self._y)

    def print_fit_report(self, result=None, show_correl=False):
        if result is None:
            result = self._result
        if result is not None:
            print(result.fit_report(show_correl=show_correl))

    def add_parameter(self, **parameter):
        if not isinstance(parameter, dict):
            raise ValueError(f'Invalid parameter ({parameter})')
        if parameter.get('expr') is not None:
            raise KeyError(f'Illegal "expr" key in parameter {parameter}')
        name = parameter['name']
        if not isinstance(name, str):
            raise ValueError(f'Illegal "name" value ({name}) in parameter {parameter}')
        if parameter.get('norm') is None:
            self._parameter_norms[name] = False
        else:
            norm = parameter.pop('norm')
            if self._norm is None:
                logging.warning(f'Ignoring norm in parameter {name} in '+
                            f'Fit.add_parameter (normalization is turned off)')
                self._parameter_norms[name] = False
            else:
                if not isinstance(norm, bool):
                    raise ValueError(f'Illegal "norm" value ({norm}) in parameter {parameter}')
                self._parameter_norms[name] = norm
        vary = parameter.get('vary')
        if vary is not None:
            if not isinstance(vary, bool):
                raise ValueError(f'Illegal "vary" value ({vary}) in parameter {parameter}')
            if not vary:
                if 'min' in parameter:
                    logging.warning(f'Ignoring min in parameter {name} in '+
                            f'Fit.add_parameter (vary = {vary})')
                    parameter.pop('min')
                if 'max' in parameter:
                    logging.warning(f'Ignoring max in parameter {name} in '+
                            f'Fit.add_parameter (vary = {vary})')
                    parameter.pop('max')
        if self._norm is not None and name not in self._parameter_norms:
            raise ValueError(f'Missing parameter normalization type for paremeter {name}')
        self._parameters.add(**parameter)

    def add_model(self, model, prefix=None, parameters=None, parameter_norms=None, **kwargs):
        # Create the new model
#        print(f'at start add_model:\nself._parameters:\n{self._parameters}')
#        print(f'at start add_model: kwargs = {kwargs}')
#        print(f'parameters = {parameters}')
#        print(f'parameter_norms = {parameter_norms}')
#        if len(self._parameters.keys()):
#            print('\nAt start adding model:')
#            self._parameters.pretty_print()
#            print(f'parameter_norms:\n{self._parameter_norms}')
        if prefix is not None and not isinstance(prefix, str):
            logging.warning('Ignoring illegal prefix: {model} {type(model)}')
            prefix = None
        if prefix is None:
            pprefix = ''
        else:
            pprefix = prefix
        if parameters is not None:
            if isinstance(parameters, dict):
                parameters = (parameters, )
            elif not is_dict_series(parameters):
                illegal_value(parameters, 'parameters', 'Fit.add_model', raise_error=True)
            parameters = deepcopy(parameters)
        if parameter_norms is not None:
            if isinstance(parameter_norms, dict):
                parameter_norms = (parameter_norms, )
            if not is_dict_series(parameter_norms):
                illegal_value(parameter_norms, 'parameter_norms', 'Fit.add_model', raise_error=True)
        new_parameter_norms = {}
        if callable(model):
            # Linear fit not yet implemented for callable models
            self._try_linear_fit = False
            if parameter_norms is None:
                if parameters is None:
                    raise ValueError('Either "parameters" or "parameter_norms" is required in '+
                            f'{model}')
                for par in parameters:
                    name = par['name']
                    if not isinstance(name, str):
                        raise ValueError(f'Illegal "name" value ({name}) in input parameters')
                    if par.get('norm') is not None:
                        norm = par.pop('norm')
                        if not isinstance(norm, bool):
                            raise ValueError(f'Illegal "norm" value ({norm}) in input parameters')
                        new_parameter_norms[f'{pprefix}{name}'] = norm
            else:
                for par in parameter_norms:
                    name = par['name']
                    if not isinstance(name, str):
                        raise ValueError(f'Illegal "name" value ({name}) in input parameters')
                    norm = par.get('norm')
                    if norm is None or not isinstance(norm, bool):
                        raise ValueError(f'Illegal "norm" value ({norm}) in input parameters')
                    new_parameter_norms[f'{pprefix}{name}'] = norm
            if parameters is not None:
                for par in parameters:
                    if par.get('expr') is not None:
                        raise KeyError(f'Illegal "expr" key ({par.get("expr")}) in parameter '+
                                f'{name} for a callable model {model}')
                    name = par['name']
                    if not isinstance(name, str):
                        raise ValueError(f'Illegal "name" value ({name}) in input parameters')
# RV FIX callable model will need partial deriv functions for any linear pars to get the linearized matrix, so for now skip linear solution option
            newmodel = Model(model, prefix=prefix)
        elif isinstance(model, str):
            if model == 'constant':      # Par: c
                newmodel = ConstantModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}c'] = True
                self._linear_parameters.append(f'{pprefix}c')
            elif model == 'linear':      # Par: slope, intercept
                newmodel = LinearModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}slope'] = True
                new_parameter_norms[f'{pprefix}intercept'] = True
                self._linear_parameters.append(f'{pprefix}slope')
                self._linear_parameters.append(f'{pprefix}intercept')
            elif model == 'quadratic':   # Par: a, b, c
                newmodel = QuadraticModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}a'] = True
                new_parameter_norms[f'{pprefix}b'] = True
                new_parameter_norms[f'{pprefix}c'] = True
                self._linear_parameters.append(f'{pprefix}a')
                self._linear_parameters.append(f'{pprefix}b')
                self._linear_parameters.append(f'{pprefix}c')
            elif model == 'gaussian':    # Par: amplitude, center, sigma (fwhm, height)
                newmodel = GaussianModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center'] = False
                new_parameter_norms[f'{pprefix}sigma'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center')
                self._nonlinear_parameters.append(f'{pprefix}sigma')
                # parameter norms for height and fwhm are needed to get correct errors
                new_parameter_norms[f'{pprefix}height'] = True
                new_parameter_norms[f'{pprefix}fwhm'] = False
            elif model == 'lorentzian':    # Par: amplitude, center, sigma (fwhm, height)
                newmodel = LorentzianModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center'] = False
                new_parameter_norms[f'{pprefix}sigma'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center')
                self._nonlinear_parameters.append(f'{pprefix}sigma')
                # parameter norms for height and fwhm are needed to get correct errors
                new_parameter_norms[f'{pprefix}height'] = True
                new_parameter_norms[f'{pprefix}fwhm'] = False
            elif model == 'exponential': # Par: amplitude, decay
                newmodel = ExponentialModel(prefix=prefix)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}decay'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}decay')
            elif model == 'step':        # Par: amplitude, center, sigma
                form = kwargs.get('form')
                if form is not None:
                    kwargs.pop('form')
                if form is None or form not in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
                    raise ValueError(f'Invalid parameter form for build-in step model ({form})')
                newmodel = StepModel(prefix=prefix, form=form)
                new_parameter_norms[f'{pprefix}amplitude'] = True
                new_parameter_norms[f'{pprefix}center'] = False
                new_parameter_norms[f'{pprefix}sigma'] = False
                self._linear_parameters.append(f'{pprefix}amplitude')
                self._nonlinear_parameters.append(f'{pprefix}center')
                self._nonlinear_parameters.append(f'{pprefix}sigma')
            elif model == 'rectangle':   # Par: amplitude, center1, center2, sigma1, sigma2
                form = kwargs.get('form')
                if form is not None:
                    kwargs.pop('form')
                if form is None or form not in ('linear', 'atan', 'arctan', 'erf', 'logistic'):
                    raise ValueError('Invalid parameter form for build-in rectangle model '+
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
            elif model == 'expression':  # Par: by expression
                expr = kwargs['expr']
                if not isinstance(expr, str):
                    raise ValueError(f'Illegal "expr" value ({expr}) in {model}')
                kwargs.pop('expr')
                if parameter_norms is not None:
                        logging.warning('Ignoring parameter_norms (normalization determined from '+
                                'linearity)}')
                if parameters is not None:
                    for par in parameters:
                        if par.get('expr') is not None:
                            raise KeyError(f'Illegal "expr" key ({par.get("expr")}) in parameter '+
                                    f'({par}) for an expression model')
                        if par.get('norm') is not None:
                            logging.warning(f'Ignoring "norm" key in parameter ({par}) '+
                                '(normalization determined from linearity)}')
                            par.pop('norm')
                        name = par['name']
                        if not isinstance(name, str):
                            raise ValueError(f'Illegal "name" value ({name}) in input parameters')
                ast = Interpreter()
                expr_parameters = [name for name in get_ast_names(ast.parse(expr))
                        if name != 'x' and name not in self._parameters
                        and name not in ast.symtable]
#                print(f'\nexpr_parameters: {expr_parameters}')
#                print(f'expr = {expr}')
                if prefix is None:
                    newmodel = ExpressionModel(expr=expr)
                else:
                    for name in expr_parameters:
                        expr = sub(rf'\b{name}\b', f'{prefix}{name}', expr)
                    expr_parameters = [f'{prefix}{name}' for name in expr_parameters]
#                    print(f'\nexpr_parameters: {expr_parameters}')
#                    print(f'expr = {expr}')
                    newmodel = ExpressionModel(expr=expr, name=name)
#                print(f'\nnewmodel = {newmodel.__dict__}')
#                print(f'params_names = {newmodel._param_names}')
#                print(f'params_names = {newmodel.param_names}')
                # Remove already existing names
                for name in newmodel.param_names.copy():
                    if name not in expr_parameters:
                        newmodel._func_allargs.remove(name)
                        newmodel._param_names.remove(name)
#                print(f'params_names = {newmodel._param_names}')
#                print(f'params_names = {newmodel.param_names}')
            else:
                raise ValueError(f'Unknown build-in fit model ({model})')
        else:
            illegal_value(model, 'model', 'Fit.add_model', raise_error=True)

        # Add the new model to the current one
#        print('\nBefore adding model:')
#        print(f'\nnewmodel = {newmodel.__dict__}')
#        if len(self._parameters):
#            self._parameters.pretty_print()
        if self._model is None:
            self._model = newmodel
        else:
            self._model += newmodel
        new_parameters = newmodel.make_params()
        self._parameters += new_parameters
#        print('\nAfter adding model:')
#        print(f'\nnewmodel = {newmodel.__dict__}')
#        print(f'\nnew_parameters = {new_parameters}')
#        self._parameters.pretty_print()

        # Check linearity of expression model paremeters
        if isinstance(newmodel, ExpressionModel):
            for name in newmodel.param_names:
                if not diff(newmodel.expr, name, name):
                    if name not in self._linear_parameters:
                        self._linear_parameters.append(name)
                        new_parameter_norms[name] = True
#                        print(f'\nADDING {name} TO LINEAR')
                else:
                    if name not in self._nonlinear_parameters:
                        self._nonlinear_parameters.append(name)
                        new_parameter_norms[name] = False
#                        print(f'\nADDING {name} TO NONLINEAR')
#        print(f'new_parameter_norms:\n{new_parameter_norms}')

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
                    if not np.isinf(_min) and abs(_min) != float_min:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != float_min:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
#        print('\nAfter norm defaults:')
#        self._parameters.pretty_print()
#        print(f'parameters:\n{parameters}')
#        print(f'all_parameters:\n{list(self.parameters)}')
#        print(f'new_parameter_norms:\n{new_parameter_norms}')
#        print(f'parameter_norms:\n{self._parameter_norms}')

        # Initialize the model parameters from parameters
        if prefix is None:
            prefix = ""
        if parameters is not None:
            for parameter in parameters:
                name = parameter['name']
                if not isinstance(name, str):
                    raise ValueError(f'Illegal "name" value ({name}) in input parameters')
                if name not in new_parameters:
                    name = prefix+name
                    parameter['name'] = name
                if name not in new_parameters:
                    logging.warning(f'Ignoring superfluous parameter info for {name}')
                    continue
                if name in self._parameters:
                    parameter.pop('name')
                    if 'norm' in parameter:
                        if not isinstance(parameter['norm'], bool):
                            illegal_value(parameter['norm'], 'norm', 'Fit.add_model',
                                    raise_error=True)
                        new_parameter_norms[name] = parameter['norm']
                        parameter.pop('norm')
                    if parameter.get('expr') is not None:
                        if 'value' in parameter:
                            logging.warning(f'Ignoring value in parameter {name} '+
                                    f'(set by expression: {parameter["expr"]})')
                            parameter.pop('value')
                        if 'vary' in parameter:
                            logging.warning(f'Ignoring vary in parameter {name} '+
                                    f'(set by expression: {parameter["expr"]})')
                            parameter.pop('vary')
                        if 'min' in parameter:
                            logging.warning(f'Ignoring min in parameter {name} '+
                                    f'(set by expression: {parameter["expr"]})')
                            parameter.pop('min')
                        if 'max' in parameter:
                            logging.warning(f'Ignoring max in parameter {name} '+
                                    f'(set by expression: {parameter["expr"]})')
                            parameter.pop('max')
                    if 'vary' in parameter:
                        if not isinstance(parameter['vary'], bool):
                            illegal_value(parameter['vary'], 'vary', 'Fit.add_model',
                                    raise_error=True)
                        if not parameter['vary']:
                            if 'min' in parameter:
                                logging.warning(f'Ignoring min in parameter {name} in '+
                                        f'Fit.add_model (vary = {parameter["vary"]})')
                                parameter.pop('min')
                            if 'max' in parameter:
                                logging.warning(f'Ignoring max in parameter {name} in '+
                                        f'Fit.add_model (vary = {parameter["vary"]})')
                                parameter.pop('max')
                    self._parameters[name].set(**parameter)
                    parameter['name'] = name
                else:
                    illegal_value(parameter, 'parameter name', 'Fit.model', raise_error=True)
        self._parameter_norms = {**self._parameter_norms, **new_parameter_norms}
#        print('\nAfter parameter init:')
#        self._parameters.pretty_print()
#        print(f'parameters:\n{parameters}')
#        print(f'new_parameter_norms:\n{new_parameter_norms}')
#        print(f'parameter_norms:\n{self._parameter_norms}')
#        print(f'kwargs:\n{kwargs}')

        # Initialize the model parameters from kwargs
        for name, value in {**kwargs}.items():
            full_name = f'{pprefix}{name}'
            if full_name in new_parameter_norms and isinstance(value, (int, float)):
                kwargs.pop(name)
                if self._parameters[full_name].expr is None:
                    self._parameters[full_name].set(value=value)
                else:
                    logging.warning(f'Ignoring parameter {name} in Fit.fit (set by expression: '+
                            f'{self._parameters[full_name].expr})')
#        print('\nAfter kwargs init:')
#        self._parameters.pretty_print()
#        print(f'parameter_norms:\n{self._parameter_norms}')
#        print(f'kwargs:\n{kwargs}')

        # Check parameter norms (also need it for expressions to renormalize the errors)
        if self._norm is not None and (callable(model) or model == 'expression'):
            missing_norm = False
            for name in new_parameters.valuesdict():
                if name not in self._parameter_norms:
                    print(f'new_parameters:\n{new_parameters.valuesdict()}')
                    print(f'self._parameter_norms:\n{self._parameter_norms}')
                    logging.error(f'Missing parameter normalization type for {name} in {model}')
                    missing_norm = True
            if missing_norm:
                raise ValueError

#        print(f'at end add_model:\nself._parameters:\n{list(self.parameters)}')
#        print(f'at end add_model: kwargs = {kwargs}')
#        print(f'\nat end add_model: newmodel:\n{newmodel.__dict__}\n')
        return(kwargs)

    def fit(self, interactive=False, guess=False, **kwargs):
        # Check inputs
        if self._model is None:
            logging.error('Undefined fit model')
            return
        if not isinstance(interactive, bool):
            illegal_value(interactive, 'interactive', 'Fit.fit', raise_error=True)
        if not isinstance(guess, bool):
            illegal_value(guess, 'guess', 'Fit.fit', raise_error=True)
        if 'try_linear_fit' in kwargs:
            try_linear_fit = kwargs.pop('try_linear_fit')
            if not isinstance(try_linear_fit, bool):
                illegal_value(try_linear_fit, 'try_linear_fit', 'Fit.fit', raise_error=True)
            if not self._try_linear_fit:
                logging.warning('Ignore superfluous keyword argument "try_linear_fit" (not '+
                        'yet supported for callable models)')
            else:
                self._try_linear_fit = try_linear_fit
#        if self._result is None:
#            if 'parameters' in kwargs:
#                raise ValueError('Invalid parameter parameters ({kwargs["parameters"]})')
#        else:
        if self._result is not None:
            if guess:
                logging.warning('Ignoring input parameter guess in Fit.fit during refitting')
                guess = False

        # Check for circular expressions
        # FIX TODO
#        for name1, par1 in self._parameters.items():
#            if par1.expr is not None:

        # Apply mask if supplied:
        if 'mask' in kwargs:
            self._mask = kwargs.pop('mask')
        if self._mask is not None:
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(f'Inconsistent x and mask dimensions ({self._x.size} vs '+
                        f'{self._mask.size})')

        # Estimate initial parameters with build-in lmfit guess method (only for a single model)
#        print(f'\nat start fit: kwargs = {kwargs}')
#RV        print('\nAt start of fit:')
#RV        self._parameters.pretty_print()
#        print(f'parameter_norms:\n{self._parameter_norms}')
        if guess:
            if self._mask is None:
                self._parameters = self._model.guess(self._y, x=self._x)
            else:
                self._parameters = self._model.guess(np.asarray(self._y)[~self._mask],
                        x=self._x[~self._mask])
#            print('\nAfter guess:')
#            self._parameters.pretty_print()

        # Add constant offset for a normalized model
        if self._result is None and self._norm is not None and self._norm[0]:
            self.add_model('constant', prefix='tmp_normalization_offset_', parameters={'name': 'c',
                    'value': -self._norm[0], 'vary': False, 'norm': True})
                    #'value': -self._norm[0]/self._norm[1], 'vary': False, 'norm': False})

        # Adjust existing parameters for refit:
        if 'parameters' in kwargs:
            parameters = kwargs.pop('parameters')
            if isinstance(parameters, dict):
                parameters = (parameters, )
            elif not is_dict_series(parameters):
                illegal_value(parameters, 'parameters', 'Fit.fit', raise_error=True)
            for par in parameters:
                name = par['name']
                if name not in self._parameters:
                    raise ValueError(f'Unable to match {name} parameter {par} to an existing one')
                if self._parameters[name].expr is not None:
                    raise ValueError(f'Unable to modify {name} parameter {par} (currently an '+
                            'expression)')
                if par.get('expr') is not None:
                    raise KeyError(f'Illegal "expr" key in {name} parameter {par}')
                self._parameters[name].set(vary=par.get('vary'))
                self._parameters[name].set(min=par.get('min'))
                self._parameters[name].set(max=par.get('max'))
                self._parameters[name].set(value=par.get('value'))
#RV            print('\nAfter adjust:')
#RV            self._parameters.pretty_print()

        # Apply parameter updates through keyword arguments
#        print(f'kwargs = {kwargs}')
#        print(f'parameter_norms = {self._parameter_norms}')
        for name in set(self._parameters) & set(kwargs):
            value = kwargs.pop(name)
            if self._parameters[name].expr is None:
                self._parameters[name].set(value=value)
            else:
                logging.warning(f'Ignoring parameter {name} in Fit.fit (set by expression: '+
                        f'{self._parameters[name].expr})')

        # Check for uninitialized parameters
        for name, par in self._parameters.items():
            if par.expr is None:
                value = par.value
                if value is None or np.isinf(value) or np.isnan(value):
                    if interactive:
                        value = input_num(f'Enter an initial value for {name}', default=1.0)
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
#        print(f'\n\n--------> linear_model = {linear_model}\n')
        if kwargs.get('check_only_linearity') is not None:
            return(linear_model)

        # Normalize the data and initial parameters
#RV        print('\nBefore normalization:')
#RV        self._parameters.pretty_print()
#        print(f'parameter_norms:\n{self._parameter_norms}')
        self._normalize()
#        print(f'norm = {self._norm}') 
#RV        print('\nAfter normalization:')
#RV        self._parameters.pretty_print()
#        self.print_fit_report()
#        print(f'parameter_norms:\n{self._parameter_norms}')

        if linear_model:
            # Perform a linear fit by direct matrix solution with numpy
            try:
                if self._mask is None:
                    self._fit_linear_model(self._x, self._y_norm)
                else:
                    self._fit_linear_model(self._x[~self._mask],
                            np.asarray(self._y_norm)[~self._mask])
            except:
                linear_model = False
        if not linear_model:
            # Perform a non-linear fit with lmfit
            # Prevent initial values from sitting at boundaries
            self._parameter_bounds = {name:{'min': par.min, 'max': par.max} for name, par in
                    self._parameters.items() if par.vary}
            for par in self._parameters.values():
                if par.vary:
                    par.set(value=self._reset_par_at_boundary(par, par.value))
#            print('\nAfter checking boundaries:')
#            self._parameters.pretty_print()

            # Perform the fit
#            fit_kws = None
#            if 'Dfun' in kwargs:
#                fit_kws = {'Dfun': kwargs.pop('Dfun')}
#            self._result = self._model.fit(self._y_norm, self._parameters, x=self._x,
#                    fit_kws=fit_kws, **kwargs)
            if self._mask is None:
                self._result = self._model.fit(self._y_norm, self._parameters, x=self._x, **kwargs)
            else:
                self._result = self._model.fit(np.asarray(self._y_norm)[~self._mask],
                        self._parameters, x=self._x[~self._mask], **kwargs)
#RV        print('\nAfter fit:')
#        print(f'\nself._result ({self._result}):\n\t{self._result.__dict__}')
#RV        self._parameters.pretty_print()
#        self.print_fit_report()

        # Set internal parameter values to fit results upon success
        if self.success:
            for name, par in self._parameters.items():
                if par.expr is None and par.vary:
                    par.set(value=self._result.params[name].value)
#            print('\nAfter update parameter values:')
#            self._parameters.pretty_print()

        # Renormalize the data and results
        self._renormalize()
#RV        print('\nAfter renormalization:')
#RV        self._parameters.pretty_print()
#        self.print_fit_report()

    def plot(self, y=None, y_title=None, result=None, skip_init=False, plot_comp_legends=False,
            plot_residual=False, plot_masked_data=True, **kwargs):
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
                illegal_value(y, 'y', 'Fit.plot')
            if len(y) != len(self._x):
                logging.warning('Ignoring parameter y in Fit.plot (wrong dimension)')
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
            plots += [(self._x[~mask], result.residual, 'k-')]
            legend += ['residual']
        plots += [(self._x[~mask], result.best_fit, 'k-')]
        legend += ['best fit']
        if not skip_init and hasattr(result, 'init_fit'):
            plots += [(self._x[~mask], result.init_fit, 'g-')]
            legend += ['init']
        components = result.eval_components(x=self._x[~mask])
        num_components = len(components)
        if 'tmp_normalization_offset_' in components:
            num_components -= 1
        if num_components > 1:
            eval_index = 0
            for modelname, y in components.items():
                if modelname == 'tmp_normalization_offset_':
                    continue
                if modelname == '_eval':
                    modelname = f'eval{eval_index}'
                if len(modelname) > 20:
                    modelname = f'{modelname[0:16]} ...'
                if isinstance(y, (int, float)):
                    y *= np.ones(self._x[~mask].size)
                plots += [(self._x[~mask], y, '--')]
                if plot_comp_legends:
                    if modelname[-1] == '_':
                        legend.append(modelname[:-1])
                    else:
                        legend.append(modelname)
        title = kwargs.get('title')
        if title is not None:
            kwargs.pop('title')
        quick_plot(tuple(plots), legend=legend, title=title, block=True, **kwargs)

    @staticmethod
    def guess_init_peak(x, y, *args, center_guess=None, use_max_for_center=True):
        """ Return a guess for the initial height, center and fwhm for a peak
        """
#        print(f'\n\nargs = {args}')
#        print(f'center_guess = {center_guess}')
#        quick_plot(x, y, vlines=center_guess, block=True)
        center_guesses = None
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            logging.error(f'Invalid x and y lengths ({len(x)}, {len(y)}), skip initial guess')
            return(None, None, None)
        if isinstance(center_guess, (int, float)):
            if len(args):
                logging.warning('Ignoring additional arguments for single center_guess value')
            center_guesses = [center_guess]
        elif isinstance(center_guess, (tuple, list, np.ndarray)):
            if len(center_guess) == 1:
                logging.warning('Ignoring additional arguments for single center_guess value')
                if not isinstance(center_guess[0], (int, float)):
                    raise ValueError(f'Invalid parameter center_guess ({type(center_guess[0])})')
                center_guess = center_guess[0]
            else:
                if len(args) != 1:
                    raise ValueError(f'Invalid number of arguments ({len(args)})')
                n = args[0]
                if not is_index(n, 0, len(center_guess)):
                    raise ValueError('Invalid argument')
                center_guesses = center_guess
                center_guess = center_guesses[n]
        elif center_guess is not None:
            raise ValueError(f'Invalid center_guess type ({type(center_guess)})')
#        print(f'x = {x}')
#        print(f'y = {y}')
#        print(f'center_guess = {center_guess}')

        # Sort the inputs
        index = np.argsort(x)
        x = x[index]
        y = y[index]
        miny = y.min()
#        print(f'miny = {miny}')
#        print(f'x_range = {x[0]} {x[-1]} {len(x)}')
#        print(f'y_range = {y[0]} {y[-1]} {len(y)}')
#        quick_plot(x, y, vlines=center_guess, block=True)

#        xx = x
#        yy = y
        # Set range for current peak
#        print(f'n = {n}')
#        print(f'center_guesses = {center_guesses}')
        if center_guesses is not None:
            if len(center_guesses) > 1:
                index = np.argsort(center_guesses)
                n = list(index).index(n)
#                print(f'n = {n}')
#                print(f'index = {index}')
                center_guesses = np.asarray(center_guesses)[index]
#                print(f'center_guesses = {center_guesses}')
            if n == 0:
               low = 0
               upp = index_nearest(x, (center_guesses[0]+center_guesses[1])/2)
            elif n == len(center_guesses)-1:
               low = index_nearest(x, (center_guesses[n-1]+center_guesses[n])/2)
               upp = len(x)
            else:
               low = index_nearest(x, (center_guesses[n-1]+center_guesses[n])/2)
               upp = index_nearest(x, (center_guesses[n]+center_guesses[n+1])/2)
#            print(f'low = {low}')
#            print(f'upp = {upp}')
            x = x[low:upp]
            y = y[low:upp]
#            quick_plot(x, y, vlines=(x[0], center_guess, x[-1]), block=True)

        # Estimate FHHM
        maxy = y.max()
#        print(f'x_range = {x[0]} {x[-1]} {len(x)}')
#        print(f'y_range = {y[0]} {y[-1]} {len(y)} {miny} {maxy}')
#        print(f'center_guess = {center_guess}')
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
#        print(f'center_index = {center_index}')
#        print(f'center = {center}')
#        print(f'height = {height}')
        half_height = miny+0.5*height
#        print(f'half_height = {half_height}')
        fwhm_index1 = 0
        for i in range(center_index, fwhm_index1, -1):
            if y[i] < half_height:
                fwhm_index1 = i
                break
#        print(f'fwhm_index1 = {fwhm_index1} {x[fwhm_index1]}')
        fwhm_index2 = len(x)-1
        for i in range(center_index, fwhm_index2):
            if y[i] < half_height:
                fwhm_index2 = i
                break
#        print(f'fwhm_index2 = {fwhm_index2} {x[fwhm_index2]}')
#        quick_plot((x,y,'o'), vlines=(x[fwhm_index1], center, x[fwhm_index2]), block=True)
        if fwhm_index1 == 0 and fwhm_index2 < len(x)-1:
            fwhm = 2*(x[fwhm_index2]-center)
        elif fwhm_index1 > 0 and fwhm_index2 == len(x)-1:
            fwhm = 2*(center-x[fwhm_index1])
        else:
            fwhm = x[fwhm_index2]-x[fwhm_index1]
#        print(f'fwhm_index1 = {fwhm_index1} {x[fwhm_index1]}')
#        print(f'fwhm_index2 = {fwhm_index2} {x[fwhm_index2]}')
#        print(f'fwhm = {fwhm}')

        # Return height, center and FWHM
#        quick_plot((x,y,'o'), (xx,yy), vlines=(x[fwhm_index1], center, x[fwhm_index2]), block=True)
        return(height, center, fwhm)

    def _check_linearity_model(self):
        """Identify the linearity of all model parameters and check if the model is linear or not
        """
        if not self._try_linear_fit:
            logging.info('Skip linearity check (not yet supported for callable models)')
            return(False)
        free_parameters = [name for name, par in self._parameters.items() if par.vary]
        for component in self._model.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            if isinstance(component, ExpressionModel):
                for name in free_parameters:
                    if diff(component.expr, name, name):
#                        print(f'\t\t{component.expr} is non-linear in {name}')
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
#                                    print(f'\t\t{component} is non-linear in {nname} (through {name} = "{expr}")')
                                    self._nonlinear_parameters.append(nname)
                                    if nname in self._linear_parameters:
                                        self._linear_parameters.remove(nname)
                            else:
                                assert(name in self._linear_parameters)
#                                print(f'\n\nexpr ({type(expr)}) = {expr}\nnname ({type(nname)}) = {nname}\n\n')
                                if diff(expr, nname, nname):
#                                    print(f'\t\t{component} is non-linear in {nname} (through {name} = "{expr}")')
                                    self._nonlinear_parameters.append(nname)
                                    if nname in self._linear_parameters:
                                        self._linear_parameters.remove(nname)
#        print(f'\nfree parameters:\n\t{free_parameters}')
#        print(f'linear parameters:\n\t{self._linear_parameters}')
#        print(f'nonlinear parameters:\n\t{self._nonlinear_parameters}\n')
        if any(True for name in self._nonlinear_parameters if self._parameters[name].vary):
            return(False)
        return(True)

    def _fit_linear_model(self, x, y):
        """Perform a linear fit by direct matrix solution with numpy
        """
        # Construct the matrix and the free parameter vector
#        print(f'\nparameters:')
#        self._parameters.pretty_print()
#        print(f'\nparameter_norms:\n\t{self._parameter_norms}')
#        print(f'\nlinear_parameters:\n\t{self._linear_parameters}')
#        print(f'nonlinear_parameters:\n\t{self._nonlinear_parameters}')
        free_parameters = [name for name, par in self._parameters.items() if par.vary]
#        print(f'free parameters:\n\t{free_parameters}\n')
        expr_parameters = {name:par.expr for name, par in self._parameters.items()
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
#        print(f'expr parameters:\n{expr_parameters}')
#        print(f'model parameters:\n\t{model_parameters}\n')
        norm = 1.0
        if self._normalized:
            norm = self._norm[1]
#        print(f'\n\nself._normalized = {self._normalized}\nnorm = {norm}\nself._norm = {self._norm}\n')
        # Add expression parameters to asteval
        ast = Interpreter()
#        print(f'Adding to asteval sym table:')
        for name, expr in expr_parameters.items():
#            print(f'\tadding {name} {expr}')
            ast.symtable[name] = expr
        # Add constant parameters to asteval
        # (renormalize to use correctly in evaluation of expression models)
        for name, par in self._parameters.items():
            if par.expr is None and not par.vary:
                if self._parameter_norms[name]:
#                    print(f'\tadding {name} {par.value*norm}')
                    ast.symtable[name] = par.value*norm
                else:
#                    print(f'\tadding {name} {par.value}')
                    ast.symtable[name] = par.value
        A = np.zeros((len(x), len(free_parameters)), dtype='float64')
        y_const = np.zeros(len(x), dtype='float64')
        have_expression_model = False
        for component in self._model.components:
            if isinstance(component, ConstantModel):
                name = component.param_names[0]
#                print(f'\nConstant model: {name} {self._parameters[name]}\n')
                if name in free_parameters:
#                    print(f'\t\t{name} is a free constant set matrix column {free_parameters.index(name)} to 1.0')
                    A[:,free_parameters.index(name)] = 1.0
                else:
                    if self._parameter_norms[name]:
                        delta_y_const = self._parameters[name]*np.ones(len(x))
                    else:
                        delta_y_const = (self._parameters[name]*norm)*np.ones(len(x))
                    y_const += delta_y_const
#                    print(f'\ndelta_y_const ({type(delta_y_const)}):\n{delta_y_const}\n')
            elif isinstance(component, ExpressionModel):
                have_expression_model = True
                const_expr = component.expr
#                print(f'\nExpression model:\nconst_expr: {const_expr}\n')
                for name in free_parameters:
                    dexpr_dname = diff(component.expr, name)
                    if dexpr_dname:
                        const_expr = f'{const_expr}-({str(dexpr_dname)})*{name}'
#                        print(f'\tconst_expr: {const_expr}')
                        if not self._parameter_norms[name]:
                            dexpr_dname = f'({dexpr_dname})/{norm}'
#                        print(f'\t{component.expr} is linear in {name}\n\t\tadd "{str(dexpr_dname)}" to matrix as column {free_parameters.index(name)}')
                        fx = [(lambda _: ast.eval(str(dexpr_dname)))(ast(f'x={v}')) for v in x]
#                        print(f'\tfx:\n{fx}')
                        if len(ast.error):
                            raise ValueError(f'Unable to evaluate {dexpr_dname}')
                        A[:,free_parameters.index(name)] += fx
#                        if self._parameter_norms[name]:
#                            print(f'\t\t{component.expr} is linear in {name} add "{str(dexpr_dname)}" to matrix as column {free_parameters.index(name)}')
#                            A[:,free_parameters.index(name)] += fx
#                        else:
#                            print(f'\t\t{component.expr} is linear in {name} add "({str(dexpr_dname)})/{norm}" to matrix as column {free_parameters.index(name)}')
#                            A[:,free_parameters.index(name)] += np.asarray(fx)/norm
                # FIX: find another solution if expr not supported by simplify
                const_expr = str(simplify(f'({const_expr})/{norm}'))
#                print(f'\nconst_expr: {const_expr}')
                delta_y_const = [(lambda _: ast.eval(const_expr))(ast(f'x = {v}')) for v in x]
                y_const += delta_y_const
#                print(f'\ndelta_y_const ({type(delta_y_const)}):\n{delta_y_const}\n')
                if len(ast.error):
                    raise ValueError(f'Unable to evaluate {const_expr}')
            else:
                free_model_parameters = [name for name in component.param_names
                        if name in free_parameters or name in expr_parameters]
#                print(f'\nBuild-in model ({component}):\nfree_model_parameters: {free_model_parameters}\n')
                if not len(free_model_parameters):
                    y_const += component.eval(params=self._parameters, x=x)
                elif isinstance(component, LinearModel):
                    if f'{component.prefix}slope' in free_model_parameters:
                        A[:,free_parameters.index(f'{component.prefix}slope')] = x
                    else:
                        y_const += self._parameters[f'{component.prefix}slope'].value*x
                    if f'{component.prefix}intercept' in free_model_parameters:
                        A[:,free_parameters.index(f'{component.prefix}intercept')] = 1.0
                    else:
                        y_const += self._parameters[f'{component.prefix}intercept'].value* \
                                np.ones(len(x))
                elif isinstance(component, QuadraticModel):
                    if f'{component.prefix}a' in free_model_parameters:
                        A[:,free_parameters.index(f'{component.prefix}a')] = x**2
                    else:
                        y_const += self._parameters[f'{component.prefix}a'].value*x**2
                    if f'{component.prefix}b' in free_model_parameters:
                        A[:,free_parameters.index(f'{component.prefix}b')] = x
                    else:
                        y_const += self._parameters[f'{component.prefix}b'].value*x
                    if f'{component.prefix}c' in free_model_parameters:
                        A[:,free_parameters.index(f'{component.prefix}c')] = 1.0
                    else:
                        y_const += self._parameters[f'{component.prefix}c'].value*np.ones(len(x))
                else:
                    # At this point each build-in model must be strictly proportional to each linear
                    #   model parameter. Without this assumption, the model equation is needed
                    #   For the current build-in lmfit models, this can only ever be the amplitude
                    assert(len(free_model_parameters) == 1)
                    name = f'{component.prefix}amplitude'
                    assert(free_model_parameters[0] == name)
                    assert(self._parameter_norms[name])
                    expr = self._parameters[name].expr
                    if expr is None:
#                        print(f'\t{component} is linear in {name} add to matrix as column {free_parameters.index(name)}')
                        parameters = deepcopy(self._parameters)
                        parameters[name].set(value=1.0)
                        index = free_parameters.index(name)
                        A[:,free_parameters.index(name)] += component.eval(params=parameters, x=x)
                    else:
                        const_expr = expr
#                        print(f'\tconst_expr: {const_expr}')
                        parameters = deepcopy(self._parameters)
                        parameters[name].set(value=1.0)
                        dcomp_dname = component.eval(params=parameters, x=x)
#                        print(f'\tdcomp_dname ({type(dcomp_dname)}):\n{dcomp_dname}')
                        for nname in free_parameters:
                            dexpr_dnname =  diff(expr, nname)
                            if dexpr_dnname:
                                assert(self._parameter_norms[name])
#                                print(f'\t\td({expr})/d{nname} = {dexpr_dnname}')
#                                print(f'\t\t{component} is linear in {nname} (through {name} = "{expr}", add to matrix as column {free_parameters.index(nname)})')
                                fx = np.asarray(dexpr_dnname*dcomp_dname, dtype='float64')
#                                print(f'\t\tfx ({type(fx)}): {fx}')
#                                print(f'free_parameters.index({nname}): {free_parameters.index(nname)}')
                                if self._parameter_norms[nname]:
                                    A[:,free_parameters.index(nname)] += fx
                                else:
                                    A[:,free_parameters.index(nname)] += fx/norm
                                const_expr = f'{const_expr}-({dexpr_dnname})*{nname}'
#                                print(f'\t\tconst_expr: {const_expr}')
                        const_expr = str(simplify(f'({const_expr})/{norm}'))
#                        print(f'\tconst_expr: {const_expr}')
                        fx = [(lambda _: ast.eval(const_expr))(ast(f'x = {v}')) for v in x]
#                        print(f'\tfx: {fx}')
                        delta_y_const = np.multiply(fx, dcomp_dname)
                        y_const += delta_y_const
#                        print(f'\ndelta_y_const ({type(delta_y_const)}):\n{delta_y_const}\n')
#            print(A)
#            print(y_const)
        solution, residual, rank, s = np.linalg.lstsq(A, y-y_const, rcond=None)
#        print(f'\nsolution ({type(solution)} {solution.shape}):\n\t{solution}')
#        print(f'\nresidual ({type(residual)} {residual.shape}):\n\t{residual}')
#        print(f'\nrank ({type(rank)} {rank.shape}):\n\t{rank}')
#        print(f'\ns ({type(s)} {s.shape}):\n\t{s}\n')

        # Assemble result (compensate for normalization in expression models)
        for name, value in zip(free_parameters, solution):
            self._parameters[name].set(value=value)
        if self._normalized and (have_expression_model or len(expr_parameters)):
            for name, norm in self._parameter_norms.items():
                par = self._parameters[name]
                if par.expr is None and norm:
                    self._parameters[name].set(value=par.value*self._norm[1])
#        self._parameters.pretty_print()
#        print(f'\nself._parameter_norms:\n\t{self._parameter_norms}')
        self._result = ModelResult(self._model, deepcopy(self._parameters))
        self._result.best_fit = self._model.eval(params=self._parameters, x=x)
        if self._normalized and (have_expression_model or len(expr_parameters)):
            if 'tmp_normalization_offset_c' in self._parameters:
                offset = self._parameters['tmp_normalization_offset_c']
            else:
                offset = 0.0
            self._result.best_fit = (self._result.best_fit-offset-self._norm[0])/self._norm[1]
            if self._normalized:
                for name, norm in self._parameter_norms.items():
                    par = self._parameters[name]
                    if par.expr is None and norm:
                        value = par.value/self._norm[1]
                        self._parameters[name].set(value=value)
                        self._result.params[name].set(value=value)
#        self._parameters.pretty_print()
        self._result.residual = self._result.best_fit-y
        self._result.components = self._model.components
        self._result.init_params = None
#        quick_plot((x, y, '.'), (x, y_const, 'g'), (x, self._result.best_fit, 'k'), (x, self._result.residual, 'r'), block=True)

    def _normalize(self):
        """Normalize the data and initial parameters
        """
        if self._normalized:
            return
        if self._norm is None:
            if self._y is not None and self._y_norm is None:
                self._y_norm = np.asarray(self._y)
        else:
            if self._y is not None and self._y_norm is None:
                self._y_norm = (np.asarray(self._y)-self._norm[0])/self._norm[1]
            self._y_range = 1.0
            for name, norm in self._parameter_norms.items():
                par = self._parameters[name]
                if par.expr is None and norm:
                    value = par.value/self._norm[1]
                    _min = par.min
                    _max = par.max
                    if not np.isinf(_min) and abs(_min) != float_min:
                        _min /= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != float_min:
                        _max /= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
            self._normalized = True

    def _renormalize(self):
        """Renormalize the data and results
        """
        if self._norm is None or not self._normalized:
            return
        self._normalized = False
        for name, norm in self._parameter_norms.items():
            par = self._parameters[name]
            if par.expr is None and norm:
                value = par.value*self._norm[1]
                _min = par.min
                _max = par.max
                if not np.isinf(_min) and abs(_min) != float_min:
                    _min *= self._norm[1]
                if not np.isinf(_max) and abs(_max) != float_min:
                    _max *= self._norm[1]
                par.set(value=value, min=_min, max=_max)
        if self._result is None:
            return
        self._result.best_fit = self._result.best_fit*self._norm[1]+self._norm[0]
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
                    if not np.isinf(_min) and abs(_min) != float_min:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != float_min:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
        if hasattr(self._result, 'init_fit'):
            self._result.init_fit = self._result.init_fit*self._norm[1]+self._norm[0]
        if hasattr(self._result, 'init_values'):
            init_values = {}
            for name, value in self._result.init_values.items():
                if name not in self._parameter_norms or self._parameters[name].expr is not None:
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
                    if not np.isinf(_min) and abs(_min) != float_min:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != float_min:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
                par.init_value = par.value
        # Don't renormalize chisqr, it has no useful meaning in physical units
        #self._result.chisqr *= self._norm[1]*self._norm[1]
        if self._result.covar is not None:
            for i, name in enumerate(self._result.var_names):
                if self._parameter_norms.get(name, False):
                    for j in range(len(self._result.var_names)):
                        if self._result.covar[i,j] is not None:
                            self._result.covar[i,j] *= self._norm[1]
                        if self._result.covar[j,i] is not None:
                            self._result.covar[j,i] *= self._norm[1]
        # Don't renormalize redchi, it has no useful meaning in physical units
        #self._result.redchi *= self._norm[1]*self._norm[1]
        if self._result.residual is not None:
            self._result.residual *= self._norm[1]

    def _reset_par_at_boundary(self, par, value):
        assert(par.vary)
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
                    return(upp)
        else:
            if np.isinf(_max):
                if self._parameter_norms.get(name, False):
                    low = _min+0.1*self._y_range
                elif _min == 0.0:
                    low = _min+0.1
                else:
                    low = _min+0.1*abs(_min)
                if value <= low:
                    return(low)
            else:
                low = 0.9*_min+0.1*_max
                upp = 0.1*_min+0.9*_max
                if value <= low:
                    return(low)
                elif value >= upp:
                    return(upp)
        return(value)


class FitMultipeak(Fit):
    """Fit data with multiple peaks
    """
    def __init__(self, y, x=None, normalize=True):
        super().__init__(y, x=x, normalize=normalize)
        self._fwhm_max = None
        self._sigma_max = None

    @classmethod
    def fit_multipeak(cls, y, centers, x=None, normalize=True, peak_models='gaussian',
            center_exprs=None, fit_type=None, background_order=None, background_exp=False,
            fwhm_max=None, plot_components=False):
        """Make sure that centers and fwhm_max are in the correct units and consistent with expr
           for a uniform fit (fit_type == 'uniform')
        """
        fit = cls(y, x=x, normalize=normalize)
        success = fit.fit(centers, fit_type=fit_type, peak_models=peak_models, fwhm_max=fwhm_max,
                center_exprs=center_exprs, background_order=background_order,
                background_exp=background_exp, plot_components=plot_components)
        if success:
            return(fit.best_fit, fit.residual, fit.best_values, fit.best_errors, fit.redchi, \
                    fit.success)
        else:
            return(np.array([]), np.array([]), {}, {}, float_max, False)

    def fit(self, centers, fit_type=None, peak_models=None, center_exprs=None, fwhm_max=None,
                background_order=None, background_exp=False, plot_components=False,
                param_constraint=False):
        self._fwhm_max = fwhm_max
        # Create the multipeak model
        self._create_model(centers, fit_type, peak_models, center_exprs, background_order,
                background_exp, param_constraint)

        # RV: Obsolete Normalize the data and results
#        print('\nBefore fit before normalization in FitMultipeak:')
#        self._parameters.pretty_print()
#        self._normalize()
#        print('\nBefore fit after normalization in FitMultipeak:')
#        self._parameters.pretty_print()

        # Perform the fit
        try:
            if param_constraint:
                super().fit(fit_kws={'xtol': 1.e-5, 'ftol': 1.e-5, 'gtol': 1.e-5})
            else:
                super().fit()
        except:
            return(False)

        # Check for valid fit parameter results
        fit_failure = self._check_validity()
        success = True
        if fit_failure:
            if param_constraint:
                logging.warning('  -> Should not happen with param_constraint set, fail the fit')
                success = False
            else:
                logging.info('  -> Retry fitting with constraints')
                self.fit(centers, fit_type, peak_models, center_exprs, fwhm_max=fwhm_max,
                        background_order=background_order, background_exp=background_exp,
                        plot_components=plot_components, param_constraint=True)
        else:
            # RV: Obsolete Renormalize the data and results
#            print('\nAfter fit before renormalization in FitMultipeak:')
#            self._parameters.pretty_print()
#            self.print_fit_report()
#            self._renormalize()
#            print('\nAfter fit after renormalization in FitMultipeak:')
#            self._parameters.pretty_print()
#            self.print_fit_report()

            # Print report and plot components if requested
            if plot_components:
                self.print_fit_report()
                self.plot()

        return(success)

    def _create_model(self, centers, fit_type=None, peak_models=None, center_exprs=None,
                background_order=None, background_exp=False, param_constraint=False):
        """Create the multipeak model
        """
        if isinstance(centers, (int, float)):
            centers = [centers]
        num_peaks = len(centers)
        if peak_models is None:
            peak_models = num_peaks*['gaussian']
        elif isinstance(peak_models, str):
            peak_models = num_peaks*[peak_models]
        if len(peak_models) != num_peaks:
            raise ValueError(f'Inconsistent number of peaks in peak_models ({len(peak_models)} vs '+
                    f'{num_peaks})')
        if num_peaks == 1:
            if fit_type is not None:
                logging.debug('Ignoring fit_type input for fitting one peak')
            fit_type = None
            if center_exprs is not None:
                logging.debug('Ignoring center_exprs input for fitting one peak')
                center_exprs = None
        else:
            if fit_type == 'uniform':
                if center_exprs is None:
                    center_exprs = [f'scale_factor*{cen}' for cen in centers]
                if len(center_exprs) != num_peaks:
                    raise ValueError(f'Inconsistent number of peaks in center_exprs '+
                            f'({len(center_exprs)} vs {num_peaks})')
            elif fit_type == 'unconstrained' or fit_type is None:
                if center_exprs is not None:
                    logging.warning('Ignoring center_exprs input for unconstrained fit')
                    center_exprs = None
            else:
                raise ValueError(f'Invalid fit_type in fit_multigaussian {fit_type}')
        self._sigma_max = None
        if param_constraint:
            min_value = float_min
            if self._fwhm_max is not None:
                self._sigma_max = np.zeros(num_peaks)
        else:
            min_value = None

        # Reset the fit
        self._model = None
        self._parameters = Parameters()
        self._result = None

        # Add background model
        if background_order is not None:
            if background_order == 0:
                self.add_model('constant', prefix='background', parameters=
                        {'name': 'c', 'value': float_min, 'min': min_value})
            elif background_order == 1:
                self.add_model('linear', prefix='background', slope=0.0, intercept=0.0)
            elif background_order == 2:
                self.add_model('quadratic', prefix='background', a=0.0, b=0.0, c=0.0)
            else:
                raise ValueError(f'Invalid parameter background_order ({background_order})')
        if background_exp:
            self.add_model('exponential', prefix='background', parameters=(
                        {'name': 'amplitude', 'value': float_min, 'min': min_value},
                        {'name': 'decay', 'value': float_min, 'min': min_value}))

        # Add peaks and guess initial fit parameters
        ast = Interpreter()
        if num_peaks == 1:
            height_init, cen_init, fwhm_init = self.guess_init_peak(self._x, self._y)
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
            self.add_model(peak_models[0], parameters=(
                    {'name': 'amplitude', 'value': amp_init, 'min': min_value},
                    {'name': 'center', 'value': cen_init, 'min': min_value},
                    {'name': 'sigma', 'value': sig_init, 'min': min_value, 'max': sig_max}))
        else:
            if fit_type == 'uniform':
                self.add_parameter(name='scale_factor', value=1.0)
            for i in range(num_peaks):
                height_init, cen_init, fwhm_init = self.guess_init_peak(self._x, self._y, i,
                        center_guess=centers)
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
                    self.add_model(peak_models[i], prefix=f'peak{i+1}_', parameters=(
                            {'name': 'amplitude', 'value': amp_init, 'min': min_value},
                            {'name': 'center', 'expr': center_exprs[i]},
                            {'name': 'sigma', 'value': sig_init, 'min': min_value,
                            'max': sig_max}))
                else:
                    self.add_model('gaussian', prefix=f'peak{i+1}_', parameters=(
                            {'name': 'amplitude', 'value': amp_init, 'min': min_value},
                            {'name': 'center', 'value': cen_init, 'min': min_value},
                            {'name': 'sigma', 'value': sig_init, 'min': min_value,
                            'max': sig_max}))

    def _check_validity(self):
        """Check for valid fit parameter results
        """
        fit_failure = False
        index = compile(r'\d+')
        for name, par in self.best_parameters.items():
            if 'background' in name:
#                if ((name == 'backgroundc' and par['value'] <= 0.0) or
#                        (name.endswith('amplitude') and par['value'] <= 0.0) or
                 if ((name.endswith('amplitude') and par['value'] <= 0.0) or
                        (name.endswith('decay') and par['value'] <= 0.0)):
                    logging.info(f'Invalid fit result for {name} ({par["value"]})')
                    fit_failure = True
            elif (((name.endswith('amplitude') or name.endswith('height')) and
                    par['value'] <= 0.0) or
                    ((name.endswith('sigma') or name.endswith('fwhm')) and par['value'] <= 0.0) or
                    (name.endswith('center') and par['value'] <= 0.0) or
                    (name == 'scale_factor' and par['value'] <= 0.0)):
                logging.info(f'Invalid fit result for {name} ({par["value"]})')
                fit_failure = True
            if name.endswith('sigma') and self._sigma_max is not None:
                if name == 'sigma':
                    sigma_max = self._sigma_max[0]
                else:
                    sigma_max = self._sigma_max[int(index.search(name).group())-1]
                if par['value'] > sigma_max:
                    logging.info(f'Invalid fit result for {name} ({par["value"]})')
                    fit_failure = True
                elif par['value'] == sigma_max:
                    logging.warning(f'Edge result on for {name} ({par["value"]})')
            if name.endswith('fwhm') and self._fwhm_max is not None:
                if par['value'] > self._fwhm_max:
                    logging.info(f'Invalid fit result for {name} ({par["value"]})')
                    fit_failure = True
                elif par['value'] == self._fwhm_max:
                    logging.warning(f'Edge result on for {name} ({par["value"]})')
        return(fit_failure)


class FitMap(Fit):
    """Fit a map of data
    """
    def __init__(self, ymap, x=None, models=None, normalize=True, transpose=None, **kwargs):
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

        # At this point the fastest index should always be the signal dimension so that the slowest 
        #    ndim-1 dimensions are the map dimensions
        if isinstance(ymap, (tuple, list, np.ndarray)):
            self._x = np.asarray(x)
        elif have_xarray and isinstance(ymap, xr.DataArray):
            if x is not None:
                logging.warning('Ignoring superfluous input x ({x}) in Fit.__init__')
            self._x = np.asarray(ymap[ymap.dims[-1]])
        else:
            illegal_value(ymap, 'ymap', 'FitMap:__init__', raise_error=True)
        self._ymap = ymap

        # Verify the input parameters
        if self._x.ndim != 1:
            raise ValueError(f'Invalid dimension for input x {self._x.ndim}')
        if self._ymap.ndim < 2:
            raise ValueError('Invalid number of dimension of the input dataset '+
                    f'{self._ymap.ndim}')
        if self._x.size != self._ymap.shape[-1]:
            raise ValueError(f'Inconsistent x and y dimensions ({self._x.size} vs '+
                    f'{self._ymap.shape[-1]})')
        if not isinstance(normalize, bool):
            logging.warning(f'Invalid value for normalize ({normalize}) in Fit.__init__: '+
                    'setting normalize to True')
            normalize = True
        if isinstance(transpose, bool) and not transpose:
            transpose = None
        if transpose is not None and self._ymap.ndim < 3:
            logging.warning(f'Transpose meaningless for {self._ymap.ndim-1}D data maps: ignoring '+
                    'transpose')
        if transpose is not None:
            if self._ymap.ndim == 3 and isinstance(transpose, bool) and transpose:
                self._transpose = (1, 0)
            elif not isinstance(transpose, (tuple, list)):
                logging.warning(f'Invalid data type for transpose ({transpose}, '+
                        f'{type(transpose)}) in Fit.__init__: setting transpose to False')
            elif len(transpose) != self._ymap.ndim-1:
                logging.warning(f'Invalid dimension for transpose ({transpose}, must be equal to '+
                        f'{self._ymap.ndim-1}) in Fit.__init__: setting transpose to False')
            elif any(i not in transpose for i in range(len(transpose))):
                logging.warning(f'Invalid index in transpose ({transpose}) '+
                        f'in Fit.__init__: setting transpose to False')
            elif not all(i==transpose[i] for i in range(self._ymap.ndim-1)):
                self._transpose = transpose
            if self._transpose is not None:
                self._inv_transpose = tuple(self._transpose.index(i)
                        for i in range(len(self._transpose)))

        # Flatten the map (transpose if requested)
        # Store the flattened map in self._ymap_norm, whether normalized or not
        if self._transpose is not None:
            self._ymap_norm = np.transpose(np.asarray(self._ymap), list(self._transpose)+
                    [len(self._transpose)])
        else:
            self._ymap_norm = np.asarray(self._ymap)
        self._map_dim = int(self._ymap_norm.size/self._x.size)
        self._map_shape = self._ymap_norm.shape[:-1]
        self._ymap_norm = np.reshape(self._ymap_norm, (self._map_dim, self._x.size))

        # Check if a mask is provided
        if 'mask' in kwargs:
            self._mask = kwargs.pop('mask')
        if self._mask is None:
            ymap_min = float(self._ymap_norm.min())
            ymap_max = float(self._ymap_norm.max())
        else:
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(f'Inconsistent mask dimension ({self._x.size} vs '+
                        f'{self._mask.size})')
            ymap_masked = np.asarray(self._ymap_norm)[:,~self._mask]
            ymap_min = float(ymap_masked.min())
            ymap_max = float(ymap_masked.max())

        # Normalize the data
        self._y_range = ymap_max-ymap_min
        if normalize and self._y_range > 0.0:
            self._norm = (ymap_min, self._y_range)
            self._ymap_norm = (self._ymap_norm-self._norm[0])/self._norm[1]
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
        return(cls(ymap, x=x, models=models, normalize=normalize, **kwargs))

    @property
    def best_errors(self):
        return(self._best_errors)

    @property
    def best_fit(self):
        return(self._best_fit)

    @property
    def best_results(self):
        """Convert the input data array to a data set and add the fit results.
        """
        if self.best_values is None or self.best_errors is None or self.best_fit is None:
            return(None)
        if not have_xarray:
            logging.warning('Unable to load xarray module')
            return(None)
        best_values = self.best_values
        best_errors = self.best_errors
        if isinstance(self._ymap, xr.DataArray):
            best_results = self._ymap.to_dataset()
            dims = self._ymap.dims
            fit_name = f'{self._ymap.name}_fit'
        else:
            coords = {f'dim{n}_index':([f'dim{n}_index'], range(self._ymap.shape[n]))
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
            best_results[f'{self._best_parameters[n]}_values'] = (dims[:-1], best_values[n])
            best_results[f'{self._best_parameters[n]}_errors'] = (dims[:-1], best_errors[n])
        best_results.attrs['components'] = self.components
        return(best_results)

    @property
    def best_values(self):
        return(self._best_values)

    @property
    def chisqr(self):
        logging.warning('property chisqr not defined for fit.FitMap')
        return(None)

    @property
    def components(self):
        components = {}
        if self._result is None:
            logging.warning('Unable to collect components in FitMap.components')
            return(components)
        for component in self._result.components:
            if 'tmp_normalization_offset_c' in component.param_names:
                continue
            parameters = {}
            for name in component.param_names:
                if self._parameters[name].vary:
                    parameters[name] = {'free': True}
                elif self._parameters[name].expr is not None:
                    parameters[name] = {'free': False, 'expr': self._parameters[name].expr}
                else:
                    parameters[name] = {'free': False, 'value': self.init_parameters[name]['value']}
            expr = None
            if isinstance(component, ExpressionModel):
                name = component._name
                if name[-1] == '_':
                    name = name[:-1]
                expr = component.expr
            else:
                prefix = component.prefix
                if len(prefix):
                    if prefix[-1] == '_':
                        prefix = prefix[:-1]
                    name = f'{prefix} ({component._name})'
                else:
                    name = f'{component._name}'
            if expr is None:
                components[name] = {'parameters': parameters}
            else:
                components[name] = {'expr': expr, 'parameters': parameters}
        return(components)

    @property
    def covar(self):
        logging.warning('property covar not defined for fit.FitMap')
        return(None)

    @property
    def max_nfev(self):
        return(self._max_nfev)

    @property
    def num_func_eval(self):
        logging.warning('property num_func_eval not defined for fit.FitMap')
        return(None)

    @property
    def out_of_bounds(self):
        return(self._out_of_bounds)

    @property
    def redchi(self):
        return(self._redchi)

    @property
    def residual(self):
        if self.best_fit is None:
            return(None)
        if self._mask is None:
            return(np.asarray(self._ymap)-self.best_fit)
        else:
            ymap_flat = np.reshape(np.asarray(self._ymap), (self._map_dim, self._x.size))
            ymap_flat_masked = ymap_flat[:,~self._mask]
            ymap_masked = np.reshape(ymap_flat_masked,
                    list(self._map_shape)+[ymap_flat_masked.shape[-1]])
            return(ymap_masked-self.best_fit)

    @property
    def success(self):
        return(self._success)

    @property
    def var_names(self):
        logging.warning('property var_names not defined for fit.FitMap')
        return(None)

    @property
    def y(self):
        logging.warning('property y not defined for fit.FitMap')
        return(None)

    @property
    def ymap(self):
        return(self._ymap)

    def best_parameters(self, dims=None):
        if dims is None:
            return(self._best_parameters)
        if not isinstance(dims, (list, tuple)) or len(dims) != len(self._map_shape):
            illegal_value(dims, 'dims', 'FitMap.best_parameters', raise_error=True)
        if self.best_values is None or self.best_errors is None:
            logging.warning(f'Unable to obtain best parameter values for dims = {dims} in '+
                    'FitMap.best_parameters')
            return({})
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
                parameters_dict[name] = {'value': par.value, 'error': par.stderr,
                        'init_value': self.init_parameters[name]['value'], 'min': par.min,
                        'max': par.max, 'vary': par.vary, 'expr': par.expr}
        return(parameters_dict)

    def freemem(self):
        if self._memfolder is None:
            return
        try:
            rmtree(self._memfolder)
            self._memfolder = None
        except:
            logging.warning('Could not clean-up automatically.')

    def plot(self, dims, y_title=None, plot_residual=False, plot_comp_legends=False,
            plot_masked_data=True):
        if not isinstance(dims, (list, tuple)) or len(dims) != len(self._map_shape):
            illegal_value(dims, 'dims', 'FitMap.plot', raise_error=True)
        if self._result is None or self.best_fit is None or self.best_values is None:
            logging.warning(f'Unable to plot fit for dims = {dims} in FitMap.plot')
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
            plots += [(self._x[mask], np.asarray(self._ymap)[(*dims,mask)], 'bx')]
            legend += ['masked data']
        plots += [(self._x[~mask], self.best_fit[dims], 'k-')]
        legend += ['best fit']
        if plot_residual:
            plots += [(self._x[~mask], self.residual[dims], 'k--')]
            legend += ['residual']
        # Create current parameters
        parameters = deepcopy(self._parameters)
        for name in self._best_parameters:
            if self._parameters[name].vary:
                parameters[name].set(value=
                        self.best_values[self._best_parameters.index(name)][dims])
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
                if len(prefix):
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
        quick_plot(tuple(plots), legend=legend, title=str(dims), block=True)

    def fit(self, **kwargs):
#        t0 = time()
        # Check input parameters
        if self._model is None:
            logging.error('Undefined fit model')
        if 'num_proc' in kwargs:
            num_proc = kwargs.pop('num_proc')
            if not is_int(num_proc, ge=1):
                illegal_value(num_proc, 'num_proc', 'FitMap.fit', raise_error=True)
        else:
            num_proc = cpu_count()
        if num_proc > 1 and not have_joblib:
            logging.warning(f'Missing joblib in the conda environment, running FitMap serially')
            num_proc = 1
        if num_proc > cpu_count():
            logging.warning(f'The requested number of processors ({num_proc}) exceeds the maximum '+
                    f'number of processors, num_proc reduced to ({cpu_count()})')
            num_proc = cpu_count()
        if 'try_no_bounds' in kwargs:
            self._try_no_bounds = kwargs.pop('try_no_bounds')
            if not isinstance(self._try_no_bounds, bool):
                illegal_value(self._try_no_bounds, 'try_no_bounds', 'FitMap.fit', raise_error=True)
        if 'redchi_cutoff' in kwargs:
            self._redchi_cutoff = kwargs.pop('redchi_cutoff')
            if not is_num(self._redchi_cutoff, gt=0):
                illegal_value(self._redchi_cutoff, 'redchi_cutoff', 'FitMap.fit', raise_error=True)
        if 'print_report' in kwargs:
            self._print_report = kwargs.pop('print_report')
            if not isinstance(self._print_report, bool):
                illegal_value(self._print_report, 'print_report', 'FitMap.fit', raise_error=True)
        if 'plot' in kwargs:
            self._plot = kwargs.pop('plot')
            if not isinstance(self._plot, bool):
                illegal_value(self._plot, 'plot', 'FitMap.fit', raise_error=True)
        if 'skip_init' in kwargs:
            self._skip_init = kwargs.pop('skip_init')
            if not isinstance(self._skip_init, bool):
                illegal_value(self._skip_init, 'skip_init', 'FitMap.fit', raise_error=True)

        # Apply mask if supplied:
        if 'mask' in kwargs:
            self._mask = kwargs.pop('mask')
        if self._mask is not None:
            self._mask = np.asarray(self._mask).astype(bool)
            if self._x.size != self._mask.size:
                raise ValueError(f'Inconsistent x and mask dimensions ({self._x.size} vs '+
                        f'{self._mask.size})')

        # Add constant offset for a normalized single component model
        if self._result is None and self._norm is not None and self._norm[0]:
            self.add_model('constant', prefix='tmp_normalization_offset_', parameters={'name': 'c',
                    'value': -self._norm[0], 'vary': False, 'norm': True})
                    #'value': -self._norm[0]/self._norm[1], 'vary': False, 'norm': False})

        # Adjust existing parameters for refit:
        if 'parameters' in kwargs:
#            print('\nIn FitMap before adjusting existing parameters for refit:')
#            self._parameters.pretty_print()
#            if self._result is None:
#                raise ValueError('Invalid parameter parameters ({parameters})')
#            if self._best_values is None:
#                raise ValueError('Valid self._best_values required for refitting in FitMap.fit')
            parameters = kwargs.pop('parameters')
#            print(f'\nparameters:\n{parameters}')
            if isinstance(parameters, dict):
                parameters = (parameters, )
            elif not is_dict_series(parameters):
                illegal_value(parameters, 'parameters', 'Fit.fit', raise_error=True)
            for par in parameters:
                name = par['name']
                if name not in self._parameters:
                    raise ValueError(f'Unable to match {name} parameter {par} to an existing one')
                if self._parameters[name].expr is not None:
                    raise ValueError(f'Unable to modify {name} parameter {par} (currently an '+
                            'expression)')
                value =  par.get('value')
                vary =  par.get('vary')
                if par.get('expr') is not None:
                    raise KeyError(f'Illegal "expr" key in {name} parameter {par}')
                self._parameters[name].set(value=value, vary=vary, min=par.get('min'),
                        max=par.get('max'))
                # Overwrite existing best values for fixed parameters when a value is specified
#                print(f'best values befored resetting:\n{self._best_values}')
                if isinstance(value, (int, float)) and vary is False:
                    for i, nname in enumerate(self._best_parameters):
                        if nname == name:
                            self._best_values[i] = value
#                print(f'best values after resetting (value={value}, vary={vary}):\n{self._best_values}')
#RV            print('\nIn FitMap after adjusting existing parameters for refit:')
#RV            self._parameters.pretty_print()

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

        # Create the best parameter list, consisting of all varying parameters plus the expression
        #   parameters in order to collect their errors
        if self._result is None:
            # Initial fit
            assert(self._best_parameters is None)
            self._best_parameters = [name for name, par in self._parameters.items()
                    if par.vary or par.expr is not None]
            num_new_parameters = 0
        else:
            # Refit
            assert(len(self._best_parameters))
            self._new_parameters = [name for name, par in self._parameters.items()
                if name != 'tmp_normalization_offset_c' and name not in self._best_parameters and
                    (par.vary or par.expr is not None)]
            num_new_parameters = len(self._new_parameters)
        num_best_parameters = len(self._best_parameters)

        # Flatten and normalize the best values of the previous fit, remove the remaining results
        #   of the previous fit
        if self._result is not None:
#            print('\nBefore flatten and normalize:')
#            print(f'self._best_values:\n{self._best_values}')
            self._out_of_bounds = None
            self._max_nfev = None
            self._redchi = None
            self._success = None
            self._best_fit = None
            self._best_errors = None
            assert(self._best_values is not None)
            assert(self._best_values.shape[0] == num_best_parameters)
            assert(self._best_values.shape[1:] == self._map_shape)
            if self._transpose is not None:
                self._best_values = np.transpose(self._best_values,
                        [0]+[i+1 for i in self._transpose])
            self._best_values = [np.reshape(self._best_values[i], self._map_dim)
                for i in range(num_best_parameters)]
            if self._norm is not None:
                for i, name in enumerate(self._best_parameters):
                    if self._parameter_norms.get(name, False):
                        self._best_values[i] /= self._norm[1]
#RV            print('\nAfter flatten and normalize:')
#RV            print(f'self._best_values:\n{self._best_values}')

        # Normalize the initial parameters (and best values for a refit)
#        print('\nIn FitMap before normalize:')
#        self._parameters.pretty_print()
#        print(f'\nparameter_norms:\n{self._parameter_norms}\n')
        self._normalize()
#        print('\nIn FitMap after normalize:')
#        self._parameters.pretty_print()
#        print(f'\nparameter_norms:\n{self._parameter_norms}\n')

        # Prevent initial values from sitting at boundaries
        self._parameter_bounds = {name:{'min': par.min, 'max': par.max}
                for name, par in self._parameters.items() if par.vary}
        for name, par in self._parameters.items():
            if par.vary:
                par.set(value=self._reset_par_at_boundary(par, par.value))
#        print('\nAfter checking boundaries:')
#        self._parameters.pretty_print()

        # Set parameter bounds to unbound (only use bounds when fit fails)
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
            self._best_fit_flat = np.zeros((self._map_dim, x_size),
                    dtype=self._ymap_norm.dtype)
            self._best_errors_flat = [np.zeros(self._map_dim, dtype=np.float64)
                    for _ in range(num_best_parameters+num_new_parameters)]
            if self._result is None:
                self._best_values_flat = [np.zeros(self._map_dim, dtype=np.float64)
                        for _ in range(num_best_parameters)]
            else:
                self._best_values_flat = self._best_values
                self._best_values_flat += [np.zeros(self._map_dim, dtype=np.float64)
                        for _ in range(num_new_parameters)]
        else:
            self._memfolder = './joblib_memmap'
            try:
                mkdir(self._memfolder)
            except FileExistsError:
                pass
            filename_memmap = path.join(self._memfolder, 'out_of_bounds_memmap')
            self._out_of_bounds_flat = np.memmap(filename_memmap, dtype=bool,
                    shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(self._memfolder, 'max_nfev_memmap')
            self._max_nfev_flat = np.memmap(filename_memmap, dtype=bool,
                    shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(self._memfolder, 'redchi_memmap')
            self._redchi_flat = np.memmap(filename_memmap, dtype=np.float64,
                    shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(self._memfolder, 'success_memmap')
            self._success_flat = np.memmap(filename_memmap, dtype=bool,
                    shape=(self._map_dim), mode='w+')
            filename_memmap = path.join(self._memfolder, 'best_fit_memmap')
            self._best_fit_flat = np.memmap(filename_memmap, dtype=self._ymap_norm.dtype,
                    shape=(self._map_dim, x_size), mode='w+')
            self._best_errors_flat = []
            for i in range(num_best_parameters+num_new_parameters):
                filename_memmap = path.join(self._memfolder, f'best_errors_memmap_{i}')
                self._best_errors_flat.append(np.memmap(filename_memmap, dtype=np.float64,
                        shape=self._map_dim, mode='w+'))
            self._best_values_flat = []
            for i in range(num_best_parameters):
                filename_memmap = path.join(self._memfolder, f'best_values_memmap_{i}')
                self._best_values_flat.append(np.memmap(filename_memmap, dtype=np.float64,
                        shape=self._map_dim, mode='w+'))
                if self._result is not None:
                    self._best_values_flat[i][:] = self._best_values[i][:]
            for i in range(num_new_parameters):
                filename_memmap = path.join(self._memfolder,
                        f'best_values_memmap_{i+num_best_parameters}')
                self._best_values_flat.append(np.memmap(filename_memmap, dtype=np.float64,
                        shape=self._map_dim, mode='w+'))

        # Update the best parameter list
        if num_new_parameters:
            self._best_parameters += self._new_parameters

        # Perform the first fit to get model component info and initial parameters
        current_best_values = {}
#        print(f'0 before:\n{current_best_values}')
#        t1 = time()
        self._result = self._fit(0, current_best_values, return_result=True, **kwargs)
#        t2 = time()
#        print(f'0 after:\n{current_best_values}')
#        print('\nAfter the first fit:')
#        self._parameters.pretty_print()
#        print(self._result.fit_report(show_correl=False))

        # Remove all irrelevant content from self._result
        for attr in ('_abort', 'aborted', 'aic', 'best_fit', 'best_values', 'bic', 'calc_covar',
                'call_kws', 'chisqr', 'ci_out', 'col_deriv', 'covar', 'data', 'errorbars',
                'flatchain', 'ier', 'init_vals', 'init_fit', 'iter_cb', 'jacfcn', 'kws',
                'last_internal_values', 'lmdif_message', 'message', 'method', 'nan_policy',
                'ndata', 'nfev', 'nfree', 'params', 'redchi', 'reduce_fcn', 'residual', 'result',
                'scale_covar', 'show_candidates', 'calc_covar', 'success', 'userargs', 'userfcn',
                'userkws', 'values', 'var_names', 'weights', 'user_options'):
            try:
                delattr(self._result, attr)
            except AttributeError:
#                logging.warning(f'Unknown attribute {attr} in fit.FtMap._cleanup_result')
                pass

#        t3 = time()
        if num_proc == 1:
            # Perform the remaining fits serially
            for n in range(1, self._map_dim):
#                print(f'{n} before:\n{current_best_values}')
                self._fit(n, current_best_values, **kwargs)
#                print(f'{n} after:\n{current_best_values}')
        else:
            # Perform the remaining fits in parallel
            num_fit = self._map_dim-1
#            print(f'num_fit = {num_fit}')
            if num_proc > num_fit:
                logging.warning(f'The requested number of processors ({num_proc}) exceeds the '+
                        f'number of fits, num_proc reduced to ({num_fit})')
                num_proc = num_fit
                num_fit_per_proc = 1
            else:
                num_fit_per_proc = round((num_fit)/num_proc)
                if num_proc*num_fit_per_proc < num_fit:
                    num_fit_per_proc +=1
#            print(f'num_fit_per_proc = {num_fit_per_proc}')
            num_fit_batch = min(num_fit_per_proc, 40)
#            print(f'num_fit_batch = {num_fit_batch}')
            with Parallel(n_jobs=num_proc) as parallel:
                parallel(delayed(self._fit_parallel)(current_best_values, num_fit_batch,
                        n_start, **kwargs) for n_start in range(1, self._map_dim, num_fit_batch))
#        t4 = time()

        # Renormalize the initial parameters for external use
        if self._norm is not None and self._normalized:
            init_values = {}
            for name, value in self._result.init_values.items():
                if name not in self._parameter_norms or self._parameters[name].expr is not None:
                    init_values[name] = value
                elif self._parameter_norms[name]:
                    init_values[name] = value*self._norm[1]
            self._result.init_values = init_values
            for name, par in self._result.init_params.items():
                if par.expr is None and self._parameter_norms.get(name, False):
                    _min = par.min
                    _max = par.max
                    value = par.value*self._norm[1]
                    if not np.isinf(_min) and abs(_min) != float_min:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != float_min:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
                par.init_value = par.value

        # Remap the best results
#        t5 = time()
        self._out_of_bounds = np.copy(np.reshape(self._out_of_bounds_flat, self._map_shape))
        self._max_nfev = np.copy(np.reshape(self._max_nfev_flat, self._map_shape))
        self._redchi = np.copy(np.reshape(self._redchi_flat, self._map_shape))
        self._success = np.copy(np.reshape(self._success_flat, self._map_shape))
        self._best_fit = np.copy(np.reshape(self._best_fit_flat,
               list(self._map_shape)+[x_size]))
        self._best_values = np.asarray([np.reshape(par, list(self._map_shape))
                for par in self._best_values_flat])
        self._best_errors = np.asarray([np.reshape(par, list(self._map_shape))
                for par in self._best_errors_flat])
        if self._inv_transpose is not None:
            self._out_of_bounds = np.transpose(self._out_of_bounds, self._inv_transpose)
            self._max_nfev = np.transpose(self._max_nfev, self._inv_transpose)
            self._redchi = np.transpose(self._redchi, self._inv_transpose)
            self._success = np.transpose(self._success, self._inv_transpose)
            self._best_fit = np.transpose(self._best_fit,
                    list(self._inv_transpose)+[len(self._inv_transpose)])
            self._best_values = np.transpose(self._best_values,
                    [0]+[i+1 for i in self._inv_transpose])
            self._best_errors = np.transpose(self._best_errors,
                    [0]+[i+1 for i in self._inv_transpose])
        del self._out_of_bounds_flat
        del self._max_nfev_flat
        del self._redchi_flat
        del self._success_flat
        del self._best_fit_flat
        del self._best_values_flat
        del self._best_errors_flat
#        t6 = time()

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
                    if not np.isinf(_min) and abs(_min) != float_min:
                        _min *= self._norm[1]
                    if not np.isinf(_max) and abs(_max) != float_min:
                        _max *= self._norm[1]
                    par.set(value=value, min=_min, max=_max)
#        t7 = time()
#        print(f'total run time in fit: {t7-t0:.2f} seconds')
#        print(f'run time first fit: {t2-t1:.2f} seconds')
#        print(f'run time remaining fits: {t4-t3:.2f} seconds')
#        print(f'run time remapping results: {t6-t5:.2f} seconds')

#        print('\n\nAt end fit:')
#        self._parameters.pretty_print()
#        print(f'self._best_values:\n{self._best_values}\n\n')

        # Free the shared memory
        self.freemem()

    def _fit_parallel(self, current_best_values, num, n_start, **kwargs):
        num = min(num, self._map_dim-n_start)
        for n in range(num):
#            print(f'{n_start+n} before:\n{current_best_values}')
            self._fit(n_start+n, current_best_values, **kwargs)
#            print(f'{n_start+n} after:\n{current_best_values}')

    def _fit(self, n, current_best_values, return_result=False, **kwargs):
#RV        print(f'\n\nstart FitMap._fit {n}\n')
#RV        print(f'current_best_values = {current_best_values}')
#RV        print(f'self._best_parameters = {self._best_parameters}')
#RV        print(f'self._new_parameters = {self._new_parameters}\n\n')
#        self._parameters.pretty_print()
        # Set parameters to current best values, but prevent them from sitting at boundaries
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
                        par.set(value=self._reset_par_at_boundary(par, current_best_values[name]))
                elif par.expr is None:
                    par.set(value=self._best_values[i][n])
#RV        print(f'\nbefore fit {n}')
#RV        self._parameters.pretty_print()
        if self._mask is None:
            result = self._model.fit(self._ymap_norm[n], self._parameters, x=self._x, **kwargs)
        else:
            result = self._model.fit(self._ymap_norm[n][~self._mask], self._parameters,
                    x=self._x[~self._mask], **kwargs)
#        print(f'\nafter fit {n}')
#        self._parameters.pretty_print()
#        print(result.fit_report(show_correl=False))
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
            # Set parameters to current best values, but prevent them from sitting at boundaries
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
#            print('\nbefore fit')
#            self._parameters.pretty_print()
#            print(result.fit_report(show_correl=False))
            if self._mask is None:
                result = self._model.fit(self._ymap_norm[n], self._parameters, x=self._x, **kwargs)
            else:
                result = self._model.fit(self._ymap_norm[n][~self._mask], self._parameters,
                    x=self._x[~self._mask], **kwargs)
#            print(f'\nafter fit {n}')
#            self._parameters.pretty_print()
#            print(result.fit_report(show_correl=False))
            out_of_bounds = False
            for name, par in self._parameter_bounds.items():
                value = result.params[name].value
                if not np.isinf(par['min']) and value < par['min']:
                    out_of_bounds = True
                    break
                if not np.isinf(par['max']) and value > par['max']:
                    out_of_bounds = True
                    break
#            print(f'{n} redchi < redchi_cutoff = {result.redchi < self._redchi_cutoff} success = {result.success} out_of_bounds = {out_of_bounds}')
            # Reset parameters back to unbound
            for name in self._parameter_bounds.keys():
                self._parameters[name].set(min=-np.inf, max=np.inf)
        assert(not out_of_bounds)
        if result.redchi >= self._redchi_cutoff:
            result.success = False
        if result.nfev == result.max_nfev:
#            print(f'Maximum number of function evaluations reached for n = {n}')
#            logging.warning(f'Maximum number of function evaluations reached for n = {n}')
            if result.redchi < self._redchi_cutoff:
                result.success = True
            self._max_nfev_flat[n] = True
        if result.success:
            assert(all(True for par in current_best_values if par in result.params.values()))
            for par in result.params.values():
                if par.vary:
                    current_best_values[par.name] = par.value
        else:
            logging.warning(f'Fit for n = {n} failed: {result.lmdif_message}')
        # Renormalize the data and results
        self._renormalize(n, result)
        if self._print_report:
            print(result.fit_report(show_correl=False))
        if self._plot:
            dims = np.unravel_index(n, self._map_shape)
            if self._inv_transpose is not None:
                dims= tuple(dims[self._inv_transpose[i]] for i in range(len(dims)))
            super().plot(result=result, y=np.asarray(self._ymap[dims]), plot_comp_legends=True,
                skip_init=self._skip_init, title=str(dims))
#RV        print(f'\n\nend FitMap._fit {n}\n')
#RV        print(f'current_best_values = {current_best_values}')
#        self._parameters.pretty_print()
#        print(result.fit_report(show_correl=False))
#RV        print(f'\nself._best_values_flat:\n{self._best_values_flat}\n\n')
        if return_result:
            return(result)

    def _renormalize(self, n, result):
        self._redchi_flat[n] = np.float64(result.redchi)
        self._success_flat[n] = result.success
        if self._norm is None or not self._normalized:
            self._best_fit_flat[n] = result.best_fit
            for i, name in enumerate(self._best_parameters):
                self._best_values_flat[i][n] = np.float64(result.params[name].value)
                self._best_errors_flat[i][n] = np.float64(result.params[name].stderr)
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
                            if not np.isinf(par.min) and abs(par.min) != float_min:
                                par.min *= self._norm[1]
                            if not np.isinf(par.max) and abs(par.max) != float_min:
                                par.max *= self._norm[1]
            self._best_fit_flat[n] = result.best_fit*self._norm[1]+self._norm[0]
            for i, name in enumerate(self._best_parameters):
                self._best_values_flat[i][n] = np.float64(result.params[name].value)
                self._best_errors_flat[i][n] = np.float64(result.params[name].stderr)
            if self._plot:
                if not self._skip_init:
                    result.init_fit = result.init_fit*self._norm[1]+self._norm[0]
                result.best_fit = np.copy(self._best_fit_flat[n])
