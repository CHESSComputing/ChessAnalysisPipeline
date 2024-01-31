"""Utils Pydantic model classes."""

# Third party imports
import numpy as np
from pydantic import (
    BaseModel,
    PrivateAttr,
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

tiny = np.finfo(np.float64).resolution
s2pi = np.sqrt(2*np.pi)

#def constant(x, c=0.5):
def constant(x, c=0.0):
    """Return a linear function.

    constant(x, c) = c

    """
    return c*np.ones((x.size))


#def linear(x, slope=0.9, intercept=0.1):
def linear(x, slope=1.0, intercept=0.0):
    """Return a linear function.

    linear(x, slope, intercept) = slope * x + intercept

    """
    return slope * x + intercept


#def quadratic(x, a=0.5, b=0.4, c=0.1):
def quadratic(x, a=0.0, b=0.0, c=0.0):
    """Return a parabolic function.

    parabolic(x, a, b, c) = a * x**2 + b * x + c

    """
    return (a*x + b) * x + c


#def exponential(x, amplitude=1.0, decay=0.3):
def exponential(x, amplitude=1.0, decay=1.0):
    """Return an exponential function.

    exponential(x, amplitude, decay) = amplitude * exp(-x/decay)

    """
    return amplitude * np.exp(-x/not_zero(decay))


#def gaussian(x, amplitude=0.25, center=0.5, sigma=0.1):
def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * np.exp(-(x-center)**2 / max(tiny, (2*sigma**2))))


#def lorentzian(x, amplitude=0.3, center=0.5, sigma=0.1):
def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.

    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

    """
    return ((amplitude/(1 + ((x-center)/max(tiny, sigma))**2))
            / max(tiny, (pi*sigma)))


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
    if model is None or model == 'expression':
        return parameters
    sig = {
        name:par
        for name, par in inspect.signature(models[model]).parameters.items()}
    sig.pop('x')

    # Check input model parameter validity
    for par in parameters:
        if par.name not in sig:
            raise ValueError('Invalid parameter {par.name} in {model} model')

    # Set model parameters
    output_parameters = []
    for sig_name, sig_par in sig.items():
        for par in parameters:
            if sig_name == par.name:
                break
        else:
            par = FitParameter(name=sig_name)
        if sig_par.default != sig_par.empty:
            par._default = sig_par.default
        output_parameters.append(par)

    return output_parameters


class FitParameter(BaseModel):
    """
    Class representing a specific fit parameter for the fit processor.

    """
    name: constr(strip_whitespace=True, min_length=1)
    value: Optional[confloat(allow_inf_nan=False)]
    min: confloat() = -np.inf
    max: confloat() = np.inf
    vary: StrictBool = True
    expr: Optional[constr(strip_whitespace=True, min_length=1)]
    _default: float = PrivateAttr()
    _init_value: float = PrivateAttr()
    _prefix: str = PrivateAttr()
    _stderr: float = PrivateAttr()

    @property
    def default(self):
        if hasattr(self, '_default'):
            return self._default
        else:
            return None

    @property
    def init_value(self):
        if hasattr(self, '_init_value'):
            return self._init_value
        else:
            return None

    @property
    def prefix(self):
        if hasattr(self, '_prefix'):
            return self._prefix
        else:
            return None

    @property
    def stderr(self):
        if hasattr(self, '_stderr'):
            return self._stderr
        else:
            return None

    def set(self, value=None, min=None, max=None, vary=None, expr=None):
        if min is not None:
            if not isinstance(min, (int, float)):
                raise ValueError(f'Invalid parameter min ({min})')
            self.min = min
        if max is not None:
            if not isinstance(max, (int, float)):
                raise ValueError(f'Invalid parameter max ({max})')
            self.max = max
        if vary is not None:
            if not isinstance(vary, bool):
                raise ValueError(f'Invalid parameter vary ({vary})')
            self.vary = vary
        if value is not None:
            if not isinstance(value, (int, float)):
                raise ValueError(f'Invalid parameter value ({value})')
            self.value = value
            if self.value > self.max:
                self.value = self.max
            elif self.value < self.min:
                self.value = self.min
            self.expr = None
        if expr is not None:
            if not isinstance(expr, str):
                raise ValueError(f'Invalid parameter expr ({expr})')
            if expr == '':
                expr = None
            self.expr = expr
            if expr is not None:
                self.vary = False

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


class Expression(BaseModel):
    model: Literal['expression']
    expr: constr(strip_whitespace=True, min_length=1)
    parameters: conlist(item_type=FitParameter) = []

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


models = {
    'constant': constant,
    'linear': linear,
    'quadratic': quadratic,
    'exponential': exponential,
    'gaussian': gaussian,
    'lorentzian': lorentzian,
}

model_classes = (
    Constant,
    Linear,
    Quadratic,
    Exponential,
    Gaussian,
    Lorentzian,
)


class FitConfig(BaseModel):
    """
    Class representing the configuration for the fit processor.

    """
    code: Literal['lmfit', 'scipy'] = 'scipy'
    parameters: conlist(item_type=FitParameter) = []
    models: conlist(item_type=Union[
        Constant, Linear, Quadratic, Exponential, Gaussian, Lorentzian,
        Expression], min_items=1)
    num_proc: conint(gt=0) = 1
    plot: StrictBool = False
    print_report:  StrictBool = False
