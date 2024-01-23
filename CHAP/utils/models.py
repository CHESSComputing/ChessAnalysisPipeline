"""Utils Pydantic model classes."""

# Third party imports
from numpy import inf
from pydantic import (
    BaseModel,
    StrictBool,
    conint,
    conlist,
    confloat,
    constr,
    validator,
)
from typing import (
    Literal,
    Optional,
    Union,
)

# Local modules
from CHAP.utils.general import not_zero, tiny

def constant(x, c=1.0):
    """Return a linear function.

    constant(x, c) = c

    """
    return c*np.ones((x.size))


def linear(x, slope=1.0, intercept=1.0):
    """Return a linear function.

    linear(x, slope, intercept) = slope * x + intercept

    """
    return slope * x + intercept


def quadratic(x, a=1.0, b=1.0, c=1.0):
    """Return a parabolic function.

    parabolic(x, a, b, c) = a * x**2 + b * x + c

    """
    return (a*x + b) * x + c


def exponential(x, amplitude=1.0, decay=1.0):
    """Return an exponential function.

    exponential(x, amplitude, decay) = amplitude * exp(-x/decay)

    """
    return amplitude * exp(-x/not_zero(decay))


def gaussian(x, amplitude=1.0, center=1.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))


def lorentzian(x, amplitude=1.0, center=1.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.

    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

    """
    return ((amplitude/(1 + ((1.0*x-center)/max(tiny, sigma))**2))
            / max(tiny, (pi*sigma)))


models = {
    'constant': constant,
    'linear': linear,
    'quadratic': quadratic,
    'exponential': exponential,
    'gaussian': gaussian,
    'lorentzian': lorentzian,
}


def validate_parameters(parameters, values):
    """Validate the parameters

    :param parameters: Fit model parameters.
    :type parameters: list[FitParameter]
    :return: List of fit model parameters.
    :rtype: list[FitParameter]
    """
    # System imports
    import inspect
    from copy import deepcopy

    model = values.get('model', None)
    if model is None:
        return
    sig = {
        name:par
        for name, par in inspect.signature(models[model]).parameters.items()}
    sig.pop('x')

    # Check input model parameter validity and set default values
    for par in parameters:
        if par.name not in sig:
            raise ValueError('Invalid parameter {par.name} in {model} model')
        if par.value is None and sig[par.name].default != sig[par.name].empty:
            par.value = sig[par.name].default
        sig.pop(par.name)
    
    # Add unspecified model parameters
    for name, par in sig.items():
        if par.default == par.empty:
            parameters.append(FitParameter(name=name))
        else:
            parameters.append(FitParameter(name=name, value=par.default))

    return parameters


class FitParameter(BaseModel):
    """
    Class representing a specific fit parameter for the fit processor.

    """
    name: constr(strip_whitespace=True, min_length=1)
    value: Optional[confloat(allow_inf_nan=False)]
    min: confloat() = -inf
    max: confloat() = inf
    vary: StrictBool = True
    expr: Optional[constr(strip_whitespace=True, min_length=1)]


class Constant(BaseModel):
    model: Literal['constant']
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Linear(BaseModel):
    model: Literal['linear']
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Quadratic(BaseModel):
    model: Literal['quadratic']
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Exponential(BaseModel):
    model: Literal['exponential']
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Gaussian(BaseModel):
    model: Literal['gaussian']
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Lorentzian(BaseModel):
    model: Literal['lorentzian']
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class FitConfig(BaseModel):
    """
    Class representing the configuration for the fit processor.

    """
    parameters: conlist(item_type=FitParameter) = []
    models: conlist(item_type=Union[
        Constant, Linear, Quadratic, Exponential, Gaussian, Lorentzian],
        min_items=1)
    num_proc: conint(gt=0) = 1
    plot: StrictBool = False
    print_report:  StrictBool = False
