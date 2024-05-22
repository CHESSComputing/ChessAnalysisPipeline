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


def rectangle(
        x, amplitude=1.0, center1=0.0, sigma1=1.0, center2=1.0,
        sigma2=1.0, form='linear'):
    """Return a rectangle function.

    Starts at 0.0, rises to `amplitude` (at `center1` with width `sigma1`),
    then drops to 0.0 (at `center2` with width `sigma2`) with `form`:
    - `'linear'` (default) = ramp_up + ramp_down
    - `'atan'`, `'arctan`' = amplitude*(atan(arg1) + atan(arg2))/pi
    - `'erf'`              = amplitude*(erf(arg1) + erf(arg2))/2.
    - `'logisitic'`        = amplitude*[1 - 1/(1 + exp(arg1)) - 1/(1+exp(arg2))]

    where ``arg1 = (x - center1)/sigma1`` and
    ``arg2 = -(x - center2)/sigma2``.

    """
    arg1 = (x - center1)/max(tiny, sigma1)
    arg2 = (center2 - x)/max(tiny, sigma2)

    if form == 'erf':
        # Third party modules
        from scipy.special import erf

        rect = 0.5*(erf(arg1) + erf(arg2))
    elif form == 'logistic':
        rect = 1. - 1./(1. + np.exp(arg1)) - 1./(1. + np.exp(arg2))
    elif form in ('atan', 'arctan'):
        rect = (np.arctan(arg1) + np.arctan(arg2))/pi
    elif form == 'linear':
        rect = 0.5*(np.minimum(1, np.maximum(-1, arg1))
                   + np.minimum(1, np.maximum(-1, arg2)))
    else:
        raise ValueError(f'Invalid parameter form ({form})')

    return amplitude*rect


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
        if model == 'rectangle' and sig_name == 'form':
            continue
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
    min: Optional[confloat()] = -np.inf
    max: Optional[confloat()] = np.inf
    vary: StrictBool = True
    expr: Optional[constr(strip_whitespace=True, min_length=1)]
    _default: float = PrivateAttr()
    _init_value: float = PrivateAttr()
    _prefix: str = PrivateAttr()
    _stderr: float = PrivateAttr()

    @validator('min', always=True)
    def validate_min(cls, value):
        """Validate the specified min.

        :param value: Field value to validate (`min`).
        :type value: Union[float, None]
        :return: Lower bound of fit parameter.
        :rtype: float
        """
        if value is None:
            return -np.inf
        return value

    @validator('max', always=True)
    def validate_max(cls, value):
        """Validate the specified max.

        :param value: Field value to validate (`max`).
        :type value: Union[float, None]
        :return: Upper bound of fit parameter.
        :rtype: float
        """
        if value is None:
            return np.inf
        return value

    @property
    def default(self):
        """Return the _default attribute."""
        if hasattr(self, '_default'):
            return self._default
        else:
            return None

    @property
    def init_value(self):
        """Return the _init_value attribute."""
        if hasattr(self, '_init_value'):
            return self._init_value
        else:
            return None

    @property
    def prefix(self):
        """Return the _prefix attribute."""
        if hasattr(self, '_prefix'):
            return self._prefix
        else:
            return None

    @property
    def stderr(self):
        """Return the _stderr attribute."""
        if hasattr(self, '_stderr'):
            return self._stderr
        else:
            return None

    def set(self, value=None, min=None, max=None, vary=None, expr=None):
        """
        Set or update FitParameter attributes.

        :param value: Parameter value.
        :type value: float, optional
        :param min: Lower Parameter value bound. To remove the lower
            bound you must set min to `numpy.inf`.
        :type min: bool, optional
        :param max: Upper Parameter value bound. To remove the lower
            bound you must set max to `numpy.inf`.
        :type max: bool, optional
        :param vary: Whether the Parameter is varied during a fit.
        :type vary: bool, optional
        :param expr: Mathematical expression used to constrain the
            value during the fit. To remove a constraint you must
            supply an empty string.
        :type expr: str, optional
        """
        if expr is not None:
            if not isinstance(expr, str):
                raise ValueError(f'Invalid parameter expr ({expr})')
            if expr == '':
                expr = None
            self.expr = expr
            if expr is not None:
                self.value = None
                self.min = -np.inf
                self.max = np.inf
                self.vary = False
                return
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

class Constant(BaseModel):
    """
    Class representing a Constant model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['constant']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['constant']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Linear(BaseModel):
    """
    Class representing a Linear model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['linear']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['linear']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Quadratic(BaseModel):
    """
    Class representing a Quadratic model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['quadratic']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['quadratic']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Exponential(BaseModel):
    """
    Class representing an Exponential model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['exponential']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['exponential']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Gaussian(BaseModel):
    """
    Class representing a Gaussian model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['gaussian']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['gaussian']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Lorentzian(BaseModel):
    """
    Class representing a Lorentzian model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['lorentzian']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['lorentzian']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Rectangle(BaseModel):
    """
    Class representing a Rectangle model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['rectangle']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['rectangle']
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Expression(BaseModel):
    """
    Class representing an Expression model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :type model: Literal['expression']
    :ivar expr: Mathematical expression to represent the model
        component.
    :type expr: str
    :ivar parameters: Function parameters, defaults to those auto
        generated from the model expression (excluding the
        independent variable), defaults to `[]`.
    :type parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :type prefix: str, optional
    """
    model: Literal['expression']
    expr: constr(strip_whitespace=True, min_length=1)
    parameters: conlist(item_type=FitParameter) = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = validator(
        'parameters', always=True, allow_reuse=True)(validate_parameters)


class Multipeak(BaseModel):
    model: Literal['multipeak']
    centers: conlist(item_type=confloat(allow_inf_nan=False), min_items=1)
    fit_type: Optional[Literal['uniform', 'unconstrained']] = 'unconstrained'
    centers_range: Optional[confloat(allow_inf_nan=False)]
    fwhm_min: Optional[confloat(allow_inf_nan=False)]
    fwhm_max: Optional[confloat(allow_inf_nan=False)]
    peak_models: Literal['gaussian', 'lorentzian'] = 'gaussian'


models = {
    'constant': constant,
    'linear': linear,
    'quadratic': quadratic,
    'exponential': exponential,
    'gaussian': gaussian,
    'lorentzian': lorentzian,
    'rectangle': rectangle,
}

model_classes = (
    Constant,
    Linear,
    Quadratic,
    Exponential,
    Gaussian,
    Lorentzian,
    Rectangle,
)


class FitConfig(BaseModel):
    """
    Class representing the configuration for the fit processor.

    :ivar code: Specifies is lmfit is used to perform the fit or if
        the scipy fit method is called directly, default is `'lmfit'`.
    :type code: Literal['lmfit', 'scipy'], optional
    :ivar parameters: Fit model parameters in addition to those
        implicitly defined through the build-in model functions,
        defaults to `[]`'
    :type parameters: list[FitParameter], optional
    :ivar models: The component(s) of the (composite) fit model.
    :type models: Union[Constant, Linear, Quadratic, Exponential,
        Gaussian, Lorentzian, Rectangle, Expression, Multipeak]
    :ivar rel_height_cutoff: Relative peak height cutoff for
        peak fitting (any peak with a height smaller than
        `rel_height_cutoff` times the maximum height of all peaks 
        gets removed from the fit model), defaults to `None`.
    :type rel_height_cutoff: float, optional
    :ivar num_proc: The number of processors used in fitting a map
        of data, defaults to `1`.
    :type num_proc: int, optional
    :ivar plot: Weather a plot of the fit result is generated,
        defaults to `False`.
    :type plot: bool, optional.
    :ivar print_report:  Weather to generate a fit result printout,
        defaults to `False`.
    :type print_report: bool, optional.
    """
    code: Literal['lmfit', 'scipy'] = 'scipy'
    parameters: conlist(item_type=FitParameter) = []
    models: conlist(item_type=Union[
        Constant, Linear, Quadratic, Exponential, Gaussian, Lorentzian,
        Rectangle, Expression, Multipeak], min_items=1)
    method: Literal[
        'leastsq', 'trf', 'dogbox', 'lm', 'least_squares'] = 'leastsq'
    rel_height_cutoff: Optional[confloat(gt=0, lt=1.0, allow_inf_nan=False)]
    num_proc: conint(gt=0) = 1
    plot: StrictBool = False
    print_report:  StrictBool = False

    @validator('method', always=True)
    def validate_method(cls, value, values):
        """Validate the specified method.

        :param value: Field value to validate (`method`).
        :type value: str
        :param values: Dictionary of validated class field values.
        :type values: dict
        :return: Fit method.
        :rtype: str
        """
        code = values['code']
        if code == 'lmfit':
            if value not in ('leastsq', 'least_squares'):
                value = 'leastsq'
        elif value == 'least_squares':
            value = 'leastsq'

        return value
