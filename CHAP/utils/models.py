"""Utils `Pydantic <https://github.com/pydantic/pydantic>`__ model
classes.
"""

# System modules
from typing import (
    ClassVar,
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
    model_validator,
)
from typing_extensions import Annotated
import numpy as np

# Local modules
from CHAP.models import CHAPBaseModel
from CHAP.utils.general import not_zero, tiny

s2pi = np.sqrt(2*np.pi)
s2ln2 = np.sqrt(2*np.log(2))

#def constant(x, c=0.5):
def constant(x, c=0.0):
    r"""Return a linear function.

    :param c: Constant, defaults to `0`.
    :type c: float, optional
    :returns: Function evaluations.
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
    :returns: Function evaluations.
    :rtype: numpy.ndarray

    .. math::

        f(x; m, b) = m x + b

    with `slope` for :math:`m` and `intercept` for :math:`b`.

    """
    return slope * x + intercept


#def parabolic(x, a=0.5, b=0.4, c=0.1):
def parabolic(x, a=0.0, b=0.0, c=0.0):
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
    :returns: Function evaluations.
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
    :returns: Function evaluations.
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
    :returns: Function evaluations.
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
    :returns: Function evaluations.
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
    :returns: Function evaluations.
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

    :returns: Evaluated rectangle function values.
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


class ConstantModel(CHAPBaseModel):
    """Class representing a Constant model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['constant']
    :ivar parameters: Function parameters, defaults to those auto
        generated from the function signature (excluding the
        independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: Model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['c']
    MODEL_PARAMETERS: ClassVar[list[str]] = []
    MODEL_IDENTIFIERS: ClassVar[list[str]] = []
    model_type: Literal['constant']
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _func: PrivateAttr
    _func_args: PrivateAttr

    @model_validator(mode='after')
    def validate_model_after(self):
        """Validate the model configuration and initialize the
        appropriate parameters from the model function signature.

        :return: Validated and initialized configuration.
        :rtype: Model
        """
        # System imports
        from inspect import signature

        if self.model_type == 'expression':
            return self
        self._func = globals()[self.model_type]
        sig = dict(signature(self._func).parameters)
        sig.pop('x')
        self._func_args = [
            arg for arg in sig.keys() if arg not in self.MODEL_IDENTIFIERS]

        # Check input model parameter validity
        par_names = []
        for par in self.parameters:
            if par.name not in sig:
                raise ValueError(
                    'Invalid parameter {par.name} in {self.model_type} model '
                    f'valid function arguments: {list(sig.keys())}')
            par_names.append(par.name)

        # Set model parameters
        for sig_name, sig_par in sig.items():
            if sig_name in self.MODEL_IDENTIFIERS or sig_name in par_names:
#            if ((self.model_type == 'rectangle' and sig_name == 'form')
#                    or sig_name in par_names):
                continue
            par = FitParameter(name=sig_name)
            if sig_par.default != sig_par.empty:
                par._default = sig_par.default
            self.parameters.append(par)
            par_names.append(par.name)

        # Perform any additional validation of model parameters
        if hasattr(self, '_validate_parameters'):
            self._validate_parameters()
        return self

    @property
    def func(self):
        """Return the model function

        :type: function
        """
        if hasattr(self, '_func'):
            return self._func
        return None

    @property
    def func_args(self):
        """Return the model function arguments

        :type: list[str]
        """
        if hasattr(self, '_func_args'):
            return self._func_args
        return None


class LinearModel(ConstantModel):
    """Class representing a Linear model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['linear']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['slope', 'intercept']
    model_type: Literal['linear']


class QuadraticModel(ConstantModel):
    """Class representing a Quadratic model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['parabolic']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['a', 'b', 'c']
    model_type: Literal['parabolic']


class ExponentialModel(ConstantModel):
    """Class representing an Exponential model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['exponential']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    model_type: Literal['exponential']


class GaussianModel(ConstantModel):
    """Class representing a Gaussian model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['gaussian']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    model_type: Literal['gaussian']

    def _validate_parameters(self):
        """Validate the model parameters."""
        for par in self.parameters:
            if par.name == 'sigma':
                par.min = 0.0


class LorentzianModel(ConstantModel):
    """Class representing a Lorentzian model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['lorentzian']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    model_type: Literal['lorentzian']

    def _validate_parameters(self):
        """Validate the model parameters."""
        for par in self.parameters:
            if par.name == 'sigma':
                par.min = 0.0


class PseudoVoigtModel(ConstantModel):
    """Class representing a PseudoVoigt model component.

    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['pvoigt']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    MODEL_PARAMETERS: ClassVar[list[str]] = ['fraction']
    model_type: Literal['pvoigt']

    def _validate_parameters(self):
        """Validate the model parameters."""
        for par in self.parameters:
            if par.name == 'fraction':
                par.min = 0.0
                par.max = 1.0
            elif par.name == 'sigma':
                par.min = 0.0


class RectangleModel(ConstantModel):
    """Class representing a Rectangle model component.

    :ivar form: Shape type of the transition edges, defaults to
        `'linear'`.
    :vartype form: Literal[
        'linear', 'atan', 'arctan', 'erf', 'logistic'], optional
    :ivar model_type: Model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['rectangle']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    MODEL_IDENTIFIERS: ClassVar[list[str]] = ['form']
    form: Literal['linear', 'atan', 'arctan', 'erf', 'logistic'] = 'linear'
    model_type: Literal['rectangle']

    def _validate_parameters(self):
        """Validate the model parameters."""
        for par in self.parameters:
            if par.name == 'form':
                assert form in ('linear', 'atan', 'arctan', 'erf', 'logistic')
            elif par.name == 'sigma1':
                par.min = 0.0
            elif par.name == 'sigma2':
                par.min = 0.0

class ExpressionModel(ConstantModel):
    """Class representing an Expression model component.

    :ivar model_type: The model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['expression']
    :ivar expr: Mathematical expression to represent the model
        component.
    :vartype expr: str
    """

    model_type: Literal['expression']
    expr: constr(strip_whitespace=True, min_length=1)


# Available models for components of the fitting function
#MODEL_CLASSES = [
#    ConstantModel,
#    LinearModel,
#    QuadraticModel,
#    ExponentialModel,
#    GaussianModel,
#    LorentzianModel,
#    PseudoVoigtModel,
#    RectangleModel,
#    ExpressionModel,
#]

# Reusable Discriminator Union for supported fit model components.
Model = Annotated[
# FIX for Python 3.11+    Union[*MODEL_CLASSES],
    Union[
        ConstantModel,
        LinearModel,
        QuadraticModel,
        ExponentialModel,
        GaussianModel,
        LorentzianModel,
        PseudoVoigtModel,
        RectangleModel,
        ExpressionModel,
    ],
    Field(discriminator='model_type')
]

# Peak-like models: with amplitude, center and sigma as their
# function arguments
PEAK_LIKE_MODELS = {
    'gaussian': GaussianModel,
    'lorentzian': LorentzianModel,
    'pvoigt': PseudoVoigtModel,
}

#MODEL_TYPE_TO_CLASS = {#v.model_type:v for v in MODEL_CLASSES}
#    'constant': constant,
#    'linear': linear,
#    'parabolic': parabolic,
#    'exponential': exponential,
#    'gaussian': gaussian,
#    'lorentzian': lorentzian,
#    'pvoigt': pvoigt,
#    'rectangle': rectangle,
#}


class MultipeakModel(CHAPBaseModel):
    """Class representing a multipeak model.

    :ivar model_type: The model component base name (a prefix will be
        added if multiple identical model components are added).
    :vartype model_type: Literal['expression']
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

    model_type: Literal['multipeak']
    centers: conlist(item_type=confloat(allow_inf_nan=False), min_length=1)
    centers_range: Optional[confloat(allow_inf_nan=False)] = None
    fit_type: Optional[Literal['uniform', 'unconstrained']] = 'unconstrained'
    fwhm_min: Optional[confloat(allow_inf_nan=False)] = None
    fwhm_max: Optional[confloat(allow_inf_nan=False)] = None
    peak_models: Literal['gaussian', 'lorentzian', 'pvoigt'] = 'gaussian'


class FitConfig(CHAPBaseModel):
    """Class representing the configuration for the fit processor.

    :ivar code: Specifies is lmfit is used to perform the fit or if
        the scipy fit method is called directly, default to `'lmfit'`.
    :vartype code: Literal['lmfit', 'scipy'], optional
    :ivar max_nfev: Maximum number of function evaluations in the
        the strain analysis peak fitting routine.
    :vartype max_nfev: int, optional
    :ivar memfolder: Folder name for the temporary memory map if
        multiple processors are used, defaults to `'joblib_memmap'`.
    :vartype memfolder: str, optional
    :ivar method: SciPy non-linear fit method, defaults to
        `"leastsq"`.
    :vartype method: Literal[
        'leastsq', 'trf', 'dogbox', 'lm', 'least_squares']
    :ivar models: The component(s) of the (composite) fit model.
    :vartype models: list[Model, MultipeakModel]
    :ivar num_proc: The number of processors used in fitting a map
        of data, defaults to `1`.
    :vartype num_proc: int, optional
    :ivar parameters: Fit model parameters in addition to those
        implicitly defined through the build-in model functions,
        defaults to `[]`'
    :vartype parameters:
        list[:class:`~CHAP.utils.models.FitParameter`], optional
    :ivar plot: Whether a plot of the fit result is generated,
        defaults to `False`.
    :vartype plot: bool, optional.
    :ivar print_report:  Whether to generate a fit result printout,
        defaults to `False`.
    :vartype print_report: bool, optional.
    :ivar rel_height_cutoff: Relative peak height cutoff for
        peak fitting (any peak with a height smaller than
        `rel_height_cutoff` times the maximum height of all peaks 
        gets removed from the fit model).
    :vartype rel_height_cutoff: float, optional
    """

    code: Literal['lmfit', 'scipy'] = 'scipy'
    max_nfev: Optional[conint(gt=0)] = None
    memfolder: str = 'joblib_memmap'
    method: Literal[
        'leastsq', 'trf', 'dogbox', 'lm', 'least_squares'] = 'leastsq'
    models: conlist(item_type=Union[Model, MultipeakModel], min_length=1)
    num_proc: conint(gt=0) = 1
    parameters: conlist(item_type=FitParameter) = []
    plot: StrictBool = False
    print_report:  StrictBool = False
    rel_height_cutoff: Optional[
        confloat(gt=0, lt=1.0, allow_inf_nan=False)] = None

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
