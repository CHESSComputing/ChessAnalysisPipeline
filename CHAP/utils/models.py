"""Utils `Pydantic <https://github.com/pydantic/pydantic>`__ model
classes.
"""

# System modules
from typing import (
    Literal,
    Optional,
    Union,
)

# Third party imports
from pydantic import (
    Field,
    PrivateAttr,
    StrictBool,
    conint,
    conlist,
    confloat,
    constr,
    field_validator,
)
from typing_extensions import Annotated
import numpy as np

# Local modules
from CHAP.models import CHAPBaseModel
from CHAP.utils.general import not_zero, tiny

# pylint: disable=no-member
tiny = np.finfo(np.float64).resolution
# pylint: enable=no-member
s2pi = np.sqrt(2*np.pi)
s2ln2 = np.sqrt(2*np.log(2))

#def constant(x, c=0.5):
def constant(x, c=0.0):
    r"""Return a linear function.

    :param c: Constant, defaults to `0`.
    :type c: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; c) = c

    """
    return c*np.ones((x.size))


#def linear(x, slope=0.9, intercept=0.1):
def linear(x, slope=1.0, intercept=0.0):
    r"""Return a linear function.

    :param slope: Slope, defaults to `0`.
    :type slope: float, optional
    :param intercept: Intercept, defaults to `0`.
    :type intercept: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; m, b) = m x + b

    with `slope` for :math:`m` and `intercept` for :math:`b`.

    """
    return slope * x + intercept


#def quadratic(x, a=0.5, b=0.4, c=0.1):
def quadratic(x, a=0.0, b=0.0, c=0.0):
    r"""Return a parabolic function.

    :param a: Quadratic polynomial coefficient, defaults to an
        initial value of `0`.
    :type a: float, optional
    :param b: Linear polynomial coefficient, defaults to an
        initial value of `0`.
    :type b: float, optional
    :param c: Constant polynomial coefficient, defaults to an
        initial value of `0`.
    :type c: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; a, b, c) = a x^2 + b x + c

    """
    return (a*x + b) * x + c


#def exponential(x, amplitude=1.0, decay=0.3):
def exponential(x, amplitude=1.0, decay=1.0):
    r"""Return an
    `exponential function <https://en.wikipedia.org/wiki/Exponential_decay>`__.

    :param amplitude: Amplitude, defaults to `1`.
    :type amplitude: float, optional
    :param decay: Exponential decay, defaults to `1`.
    :type decay: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; A, \tau) = A exp(-x/\tau)

    with `amplitude` for :math:`A` and `decay` for :math:`\tau`.

    """
    return amplitude * np.exp(-x/not_zero(decay))


#def gaussian(x, amplitude=0.25, center=0.5, sigma=0.1):
def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    r"""Return a 1-dimensional
    `Gaussian function <https://en.wikipedia.org/wiki/Normal_distribution>`__.

    :param amplitude: amplitude, defaults to `1`.
    :type amplitude: float, optional
    :param center: Center, defaults to `0`.
    :type center: float, optional
    :param sigma: Standard deviation, defaults to `1`.
    :type sigma: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; A, \mu, \sigma) = frac{A}{\sigma\sqrt{2\pi}}
            e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

    where the parameter `amplitude` corresponds to :math:`A`, `center`
    to :math:`\mu`, and `sigma` to :math:`\sigma`. The full width at
    half maximum is :math:`2\sigma\sqrt{2\ln{2}}`, approximately
    :math:`2.3548\sigma`.where the parameter `amplitude` corresponds to
    :math:`A`, `center` to :math:`\mu`, and `sigma` to :math:`\sigma`.
    The full width at half maximum is :math:`2\sigma\sqrt{2\ln{2}}`,
    and the peak height is :math:`A/(\sigma\sqrt{2\pi})`

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * np.exp(-(x-center)**2 / max(tiny, (2*sigma**2))))


#def lorentzian(x, amplitude=0.3, center=0.5, sigma=0.1):
def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    r"""Return a 1-dimensional
    `Lorentzian function <https://en.wikipedia.org/wiki/Cauchy_distribution>`__.

    :param amplitude: amplitude, defaults to `1`.
    :type amplitude: float, optional
    :param center: Center, defaults to `0`.
    :type center: float, optional
    :param sigma: Standard deviation, defaults to `1`.
    :type sigma: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\pi} \big[
            \frac{\sigma}{(x - \mu)^2 + \sigma^2} \big]

    where the parameter `amplitude` corresponds to :math:`A`, `center`
    to :math:`\mu`, and `sigma` to :math:`\sigma`. The full width at
    half maximum is :math:`2\sigma`, and the peak height is
    :math:`A/(\sigma\pi)`.

    """
    return ((amplitude/(1 + ((x-center)/max(tiny, sigma))**2))
            / max(tiny, (np.pi*sigma)))


def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    r"""Return a 1-dimensional
    `pseudo-Voigt distribution <https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation>`__.

    This is an approximation of the Voigt function, a weighted sum
    of a Gaussian and Lorentzian distribution, with the parameter
    `fraction` setting the relative weight of the Gaussian and
    Lorentzian components.

    :param amplitude: amplitude, defaults to `1`.
    :type amplitude: float, optional
    :param center: Center, defaults to `0`.
    :type center: float, optional
    :param sigma: Standard deviation, defaults to `1`.
    :type sigma: float, optional
    :param fraction: Relative weight of the Gaussian and Lorentzian
        components, defaults to `0.5`.
    :type fraction: float, optional
    :returns: The function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; A, \mu, \sigma) = frac{(1-\alpha)A}{\sigma_g\sqrt{2\pi}}
            e^{[{-{(x-\mu)^2}/{{2\sigma_g}^2}}]} +
            \frac{\alpha A}{\pi} \big[
            \frac{\sigma}{(x - \mu)^2 + \sigma^2} \big]

    where the parameter `amplitude` corresponds to :math:`A`, `center`
    to :math:`\mu`, and `sigma` to :math:`\sigma`. Here
    :math:`\sigma_g = {\sigma}/{\sqrt{2\ln{2}}}` so that the full
    width at half maximum is :math:`2\sigma` and the peak height is
    approximately :math:`A/(2.536\sigma)`.
    """
    return ((1-fraction) * gaussian(x, amplitude, center, sigma/s2ln2) +
        fraction * lorentzian(x, amplitude, center, sigma))


def rectangle(
        x, amplitude=1.0, center1=0.0, sigma1=1.0, center2=1.0,
        sigma2=1.0, form='linear'):
    r"""
    Return a rectangle function.

    Starts at 0.0, rises to ``amplitude`` (at ``center1`` with width
    ``sigma1``), then drops to 0.0 (at ``center2`` with width
    ``sigma2``)

    :param x: Input values where the function is evaluated.
    :type x: float or numpy.ndarray
    :param amplitude: Maximum height of the rectangle, defaults to 1.0.
    :type amplitude: float, optional
    :param center1: Location of the rising edge, defaults to 0.0.
    :type center1: float, optional
    :param sigma1: Width or smoothness of the rising edge,
        defaults to 1.0.
    :type sigma1: float, optional
    :param center2: Location of the falling edge, defaults to 1.0.
    :type center2: float, optional
    :param sigma2: Width or smoothness of the falling edge,
        defaults to 1.0.
    :type sigma2: float, optional
    :param form: Shape type of the transition edges:

        - ``'linear'``: Simple ramp-up and ramp-down.
        - ``'atan'`` or ``'arctan'``: Inverse tangent transitions.
        - ``'erf'``: Error function (Gaussian-like) transitions.
        - ``'logistic'``: Sigmoidal (logistic function) transitions.
    :type form: str, optional

    :returns: The evaluated rectangle function values.
    :rtype: float or numpy.ndarray

    .. note::
        The output is calculated based on the selected ``form``:

        - **atan**: $\frac{A}{\pi} [ \arctan(arg_1) + \arctan(arg_2) ]$
        - **erf**:
          $\frac{1}{2} A [ \text{erf}(arg_1) + \text{erf}(arg_2) ]$
        - **logistic**:
          $A [ \frac{1}{1 + \exp(-arg_1)} +
          \frac{1}{1 + \exp(-arg_2)} - 1 ]$

        The function is constructed using normalized arguments for the
        rising and falling edges:
        $arg_1 = \frac{x - center_1}{\sigma_1}$
        and
        $arg_2 = \frac{center_2 - x}{\sigma_2}$
    """
    arg1 = (x - center1)/max(tiny, sigma1)
    arg2 = (center2 - x)/max(tiny, sigma2)

    if form == 'erf':
        # Third party modules
        # pylint: disable=no-name-in-module
        from scipy.special import erf

        rect = 0.5*(erf(arg1) + erf(arg2))
    elif form == 'logistic':
        rect = 1. - 1./(1. + np.exp(arg1)) - 1./(1. + np.exp(arg2))
    elif form in ('atan', 'arctan'):
        rect = (np.arctan(arg1) + np.arctan(arg2))/np.pi
    elif form == 'linear':
        rect = 0.5*(np.minimum(1, np.maximum(-1, arg1))
                   + np.minimum(1, np.maximum(-1, arg2)))
    else:
        raise ValueError(f'Invalid parameter form ({form})')

    return amplitude*rect


def validate_parameters(parameters, info):
    """Validate the parameters.

    :param parameters: Fit model parameters.
    :type parameters: list[FitParameter]
    :param info: Model parameter validation information.
    :type info: pydantic.ValidationInfo
    :return: List of fit model parameters.
    :rtype: list[FitParameter]
    """
    # System imports
    import inspect

    if 'model' in info.data:
        model = info.data['model']
    else:
        model = None
    if model is None or model == 'expression':
        return parameters
    sig = dict(inspect.signature(models[model]['name']).parameters.items())
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
        if model == 'pvoigt' and sig_name == 'fraction':
            par.min = 0.0
            par.max = 1.0
        output_parameters.append(par)

    return output_parameters


class FitParameter(CHAPBaseModel):
    """Class representing a specific fit parameter for the fit
    processor.

    :ivar name: Parameter name.
    :vartype name: str
    :ivar value: Parameter value.
    :vartype value: float, optional
    :ivar min: Lower Parameter value bound, defaults to `-numpy.inf`.
    :vartype min: bool, optional
    :ivar max: Upper Parameter value bound. defaults to `numpy.inf`.
    :vartype max: bool, optional
    :ivar vary: Whether the Parameter is varied during a fit, defaults
        to `True`.
    :vartype vary: bool, optional
    :ivar expr: Mathematical expression used to constrain the
        value during the fit. To remove a constraint you must
        supply an empty string.
    :vartype expr: str, optional
    """

    name: constr(strip_whitespace=True, min_length=1)
    value: Optional[confloat(allow_inf_nan=False)] = None
    min: Optional[confloat()] = -np.inf
    max: Optional[confloat()] = np.inf
    vary: StrictBool = True
    expr: Optional[constr(strip_whitespace=True, min_length=1)] = None

    _default: float = PrivateAttr()
    _init_value: float = PrivateAttr()
    _prefix: str = PrivateAttr()
    _stderr: float = PrivateAttr()

    @field_validator('min')
    @classmethod
    def validate_min(cls, value):
        """Validate the specified min.

        :param value: Field value to validate (`min`).
        :type value: float or None
        :return: Lower bound of fit parameter.
        :rtype: float
        """
        if value is None:
            return -np.inf
        return value

    @field_validator('max')
    @classmethod
    def validate_max(cls, value):
        """Validate the specified max.

        :param value: Field value to validate (`max`).
        :type value: float or None
        :return: Upper bound of fit parameter.
        :rtype: float
        """
        if value is None:
            return np.inf
        return value

    @property
    def default(self):
        """Return the default parameter value.

        :type: float or None
        """
        if hasattr(self, '_default'):
            return self._default
        return None

    @property
    def init_value(self):
        """Return the initial parameter value.

        :type: float or None
        """
        if hasattr(self, '_init_value'):
            return self._init_value
        return None

    @property
    def prefix(self):
        """Return the parametr prefix.

        :type: str or None
        """
        if hasattr(self, '_prefix'):
            return self._prefix
        return None

    @property
    def stderr(self):
        """Return the parameter's uncertainty value.

        :type: float or None
        """
        if hasattr(self, '_stderr'):
            return self._stderr
        return None

    def set(self, value=None, min=None, max=None, vary=None, expr=None):
        """Set or update FitParameter attributes.

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

class Constant(CHAPBaseModel):
    """Class representing a Constant model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['constant']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['constant']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Linear(CHAPBaseModel):
    """Class representing a Linear model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['linear']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['linear']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Quadratic(CHAPBaseModel):
    """Class representing a Quadratic model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['quadratic']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['quadratic']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Exponential(CHAPBaseModel):
    """Class representing an Exponential model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['exponential']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['exponential']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Gaussian(CHAPBaseModel):
    """Class representing a Gaussian model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['gaussian']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['gaussian']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Lorentzian(CHAPBaseModel):
    """Class representing a Lorentzian model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['lorentzian']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['lorentzian']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class PseudoVoigt(CHAPBaseModel):
    """Class representing a PseudoVoigt model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['pvoigt']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['pvoigt']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Rectangle(CHAPBaseModel):
    """Class representing a Rectangle model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['rectangle']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['rectangle']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Expression(CHAPBaseModel):
    """Class representing an Expression model component.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['expression']
    :ivar expr: Mathematical expression to represent the model
        component.
    :vartype expr: str
    :ivar parameters: Function parameters, defaults to those auto
        generated from the model expression (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: The model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    model: Literal['expression']
    expr: constr(strip_whitespace=True, min_length=1)
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _validate_parameters_parameters = field_validator(
        'parameters')(validate_parameters)


class Multipeak(CHAPBaseModel):
    """Class representing a multipeak model.

    :ivar model: The model component base name (a prefix will be added
        if multiple identical model components are added).
    :vartype model: Literal['expression']
    :ivar centers: Peak centers.
    :vartype center: list[float]
    :ivar centers_range: Range of peak centers around their centers.
    :vartype centers_range: float, optional
    :ivar fit_type: Type of fit, defaults to `'unconstrained'`.
    :vartype fit_type: Literal['uniform', 'unconstrained'], optional.
    :ivar fwhm_min: Lower limit of the fwhm of the peaks.
    :vartype fwhm_min: float, optional
    :ivar fwhm_max: Upper limit of the fwhm of the peaks.
    :vartype fwhm_max: float, optional
    :ivar peak_models: Type of peaks, defaults to `'gaussian'`.
    :vartype peak_models: Literal['gaussian', 'lorentzian', 'pvoigt'],
        optional.
    """

    model: Literal['multipeak']
    centers: conlist(item_type=confloat(allow_inf_nan=False), min_length=1)
    centers_range: Optional[confloat(allow_inf_nan=False)] = None
    fit_type: Optional[Literal['uniform', 'unconstrained']] = 'unconstrained'
    fwhm_min: Optional[confloat(allow_inf_nan=False)] = None
    fwhm_max: Optional[confloat(allow_inf_nan=False)] = None
    peak_models: Literal['gaussian', 'lorentzian', 'pvoigt'] = 'gaussian'


models = {
    'constant': {'name': constant, 'class': Constant},
    'linear': {'name': linear, 'class': Linear},
    'quadratic': {'name': quadratic, 'class': Quadratic},
    'exponential': {'name': exponential, 'class': Exponential},
    'gaussian': {'name': gaussian, 'class': Gaussian},
    'lorentzian': {'name': lorentzian, 'class': Lorentzian},
    'pvoigt': {'name': pvoigt, 'class': PseudoVoigt},
    'rectangle': {'name': rectangle, 'class': Rectangle},
}

model_classes = (
    Constant,
    Linear,
    Quadratic,
    Exponential,
    Gaussian,
    Lorentzian,
    PseudoVoigt,
    Rectangle,
)


class FitConfig(CHAPBaseModel):
    """Class representing the configuration for the fit processor.

    :ivar code: Specifies is lmfit is used to perform the fit or if
        the scipy fit method is called directly, default to `'lmfit'`.
    :vartype code: Literal['lmfit', 'scipy'], optional
    :ivar parameters: Fit model parameters in addition to those
        implicitly defined through the build-in model functions,
        defaults to `[]`'
    :vartype parameters:
        list[:class:`~CHAP.utils.models.FitParameter`], optional
    :ivar models: The component(s) of the (composite) fit model.
    :vartype models:
        list[:attr:`~CHAP.utils.models.FitConfig.models`]
    :ivar rel_height_cutoff: Relative peak height cutoff for
        peak fitting (any peak with a height smaller than
        `rel_height_cutoff` times the maximum height of all peaks 
        gets removed from the fit model).
    :vartype rel_height_cutoff: float, optional
    :ivar num_proc: The number of processors used in fitting a map
        of data, defaults to `1`.
    :vartype num_proc: int, optional
    :ivar plot: Whether a plot of the fit result is generated,
        defaults to `False`.
    :vartype plot: bool, optional.
    :ivar print_report:  Whether to generate a fit result printout,
        defaults to `False`.
    :vartype print_report: bool, optional.
    :ivar memfolder: Folder name for the temporary memory map if
        multiple processors are used, defaults to `'joblib_memmap'`.
    :vartype memfolder: str, optional
    """

    code: Literal['lmfit', 'scipy'] = 'scipy'
    parameters: conlist(item_type=FitParameter) = []
    models: conlist(item_type=Union[
        Constant, Linear, Quadratic, Exponential, Gaussian, Lorentzian,
        PseudoVoigt, Rectangle, Expression, Multipeak], min_length=1)
    method: Literal[
        'leastsq', 'trf', 'dogbox', 'lm', 'least_squares'] = 'leastsq'
    rel_height_cutoff: Optional[
        confloat(gt=0, lt=1.0, allow_inf_nan=False)] = None
    num_proc: conint(gt=0) = 1
    plot: StrictBool = False
    print_report:  StrictBool = False
    memfolder: str = 'joblib_memmap'

    @field_validator('method')
    @classmethod
    def validate_method(cls, method, info):
        """Validate the specified method.

        :param method: The value of `method` to validate.
        :type method: str
        :param info: Model parameter validation information.
        :type info: pydantic.ValidationInfo
        :return: Fit method.
        :rtype: str
        """
        code = info.data['code']
        if code == 'lmfit':
            if method not in ('leastsq', 'least_squares'):
                method = 'leastsq'
        elif method == 'least_squares':
            method = 'leastsq'

        return method
