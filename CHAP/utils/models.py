"""Utils `Pydantic <https://github.com/pydantic/pydantic>`__ model
classes.
"""

# System modules
import os
from typing import (
    ClassVar,
    Literal,
    Optional,
    Type,
    Union,
)

# Third party imports
import lmfit.models as lmfit_models
from pydantic import (
    BaseModel,
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

    :ivar name: Parameter name (always includes the component prefix
        if belonging to a model component)
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
    :ivar description: Free-text description of the parameter, defaults
        to `"unspecified"`.
    :vartype description: str, optional
    :ivar units: Units of the parameter, defaults to `"unspecified"`.
    :vartype units: str, optional
    """

    name: constr(strip_whitespace=True, min_length=1)
    value: Optional[confloat(allow_inf_nan=False)] = None
    min: Optional[confloat()] = -np.inf
    max: Optional[confloat()] = np.inf
    vary: StrictBool = True
    expr: Optional[constr(strip_whitespace=True, min_length=1)] = None

    description: Optional[str] = 'unspecified'
    units: Optional[str] = 'unspecified'

    _default: float = PrivateAttr()
    _init_value: float = PrivateAttr()
#    _prefix: str = PrivateAttr()
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

    @init_value.setter
    def init_value(self, value):
        self._init_value = value

# FIX the parameter prefix may be needed to implement expression models
# in scipy, this is not yet implemented
#    @property
#    def prefix(self):
#        """Return the parameter prefix.
#
#        :type: str
#        """
#        if hasattr(self, '_prefix'):
#            return self._prefix
#        return ''

#    @property
#    def long_name(self):
#        """Return the fully-qualified parameter name, combining the
#        model prefix (if any) with the parameter name.
#
#        When no prefix is set, returns :attr:`name` unchanged.
#
#        :type: str
#        """
#        prefix = self.prefix
#        if not prefix:
#            return self.name
#        return f'{prefix}_{self.name}'

    @property
    def stderr(self):
        """Return the parameter's uncertainty value.

        :type: float or None
        """
        if hasattr(self, '_stderr'):
            return self._stderr
        return None

    @stderr.setter
    def stderr(self, value):
        self._stderr = value

    def set(self, value=None, min=None, max=None, vary=None, expr=None,
            is_init_value=True):
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
        :param is_init_value: Whether to set the intial value when
            setting the parameter value, default to `True`.
        :type is_init_value: bool, optional
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
            if is_init_value:
                self._init_value = value

    def zarr_tree(self, dataset_shape, dataset_chunks, nxlinks=None):
        """Return a nested dict representing the Zarr group tree for
        this fit parameter's output container.

        The group contains one dataset per parameter attribute
        (``value``, ``error``, ``initial``, ``min``, ``max``,
        ``vary``, ``expression``), each shaped to hold one scalar per
        point in the scan map.

        :param dataset_shape: Shape of the measurement (scan) dimensions
            of the output dataset, excluding the signal dimensions.
        :type dataset_shape: tuple[int, ...]
        :param dataset_chunks: Chunk shape along the scan dimensions, or
            ``'auto'``.
        :type dataset_chunks: list[int] or str
        :param nxlinks: NeXus path(s) to link into the ``data`` group.
            When the zarr tree is written to a ``.zarr`` file and
            converted to ``.nxs`` with
            :class:`~CHAP.common.processor.ZarrToNexusProcessor`, each
            path produces an ``NXlink`` whose name is
            ``os.path.basename(path)``.  Accepts a single path string or
            a list of path strings.  All links must be explicit; none are
            auto-generated.
        :type nxlinks: str or list[str], optional
        :returns: Nested dict representing the zarr group tree for this
            parameter.
        :rtype: dict
        """
        data_attrs = {
            'NX_class': 'NXdata',
            'description': self.description,
        }
        if isinstance(nxlinks, str):
            nxlinks = [nxlinks]
        if nxlinks:
            data_attrs['__nxlinks__'] = {
                os.path.basename(p): p for p in nxlinks
            }
        if nxlinks:
            data_attrs['__nxlinks__'] = {
                os.path.basename(p): p for p in nxlinks
            }
        return {
            'attributes': data_attrs,
            'children': {
                'value': {
                    'attributes': {
                        'NX_class': 'NXfield',
                        'units': self.units,
                    },
                    'dtype': 'float64',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
                'error': {
                    'attributes': {
                        'NX_class': 'NXfield',
                        'units': self.units,
                    },
                    'dtype': 'float64',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
                'initial': {
                    'attributes': {
                        'NX_class': 'NXfield',
                        'units': self.units,
                    },
                    'dtype': 'float64',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
                'min': {
                    'attributes': {
                        'NX_class': 'NXfield',
                        'units': self.units,
                    },
                    'dtype': 'float64',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
                'max': {
                    'attributes': {
                        'NX_class': 'NXfield',
                        'units': self.units,
                    },
                    'dtype': 'float64',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
                'vary': {
                    'attributes': {
                        'NX_class': 'NXfield',
                    },
                    'dtype': 'bool',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
                'expression': {
                    'attributes': {
                        'NX_class': 'NXfield',
                    },
                    'dtype': 'str',
                    'shape': dataset_shape,
                    'chunks': dataset_chunks,
                },
            },
        }


class FitModel(CHAPBaseModel):
    """Abstract base class representing a generic model component.

    :ivar model_type: Model component type.
    :vartype model: Literal['constant', 'linear', 'parabolic',
        'exponential', 'gaussian', 'lorentzian', 'pvoigt',
        'rectangle', 'expression']
    :ivar parameters: Function parameters (not including the `prefix`),
        defaults to those auto generated from the function signature
        (excluding the independent variable).
    :vartype parameters: list[FitParameter], optional
    :ivar prefix: Model prefix, defaults to `''`.
    :vartype prefix: str, optional
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = []
    MODEL_PARAMETERS: ClassVar[list[str]] = []
    MODEL_IDENTIFIERS: ClassVar[list[str]] = []
    model_type: str
    parameters: Annotated[
        conlist(item_type=FitParameter),
        Field(validate_default=True)] = []
    prefix: Optional[str] = ''

    _func: PrivateAttr
    _func_args: PrivateAttr

    @model_validator(mode='after')
    def validate_fitmodel_after(self):
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

#    def linear_parameters(self, prefix=''):
#        """Return the linear parameters.
#
#        :param prefix: Model prefix.
#        :type prefix: str, optional
#        :returns: The list of linear function parameters.
#        :rtype: list[str]
#        """
#        return [f'{prefix}{v}' for v in self.LINEAR_PARAMETERS]

    @property
    def long_name(self):
        """Return the fully-qualified model name, combining the model
        prefix (if any) with :attr:`model_type`.

        :type: str
        """
        return f'{self.prefix}{self.model_type}'

    def eval(self, x, **kwargs):
        parameters = {}
        for func_arg in self.func_args:
            value = kwargs.pop(func_arg, None)
            if value is None:
                for p in self.parameters:
                    if func_arg == p.name and p.value is not None:
                        parameters[func_arg] = p.value
                        break
                else:
                    raise ValueError(
                        f'Missing or invalid parameter {func_arg}')
            else:
                parameters[func_arg] = value
        return self.func(x, **parameters, **kwargs)

    def zarr_tree(self, dataset_shape, dataset_chunks,
                  signal_shape, nxlinks=None):
        """Return a nested dict representing the Zarr group tree for
        this model component's output container.

        The group contains a ``parameters`` sub-group (one entry per
        parameter, structured by
        :meth:`~CHAP.utils.models.FitParameter.zarr_tree`) and a
        ``data`` sub-group with a ``best_fit`` dataset shaped to hold
        the component's fitted curve for every point in the scan map.

        :param dataset_shape: Shape of the measurement (scan) dimensions
            of the output dataset, excluding the signal dimensions.
        :type dataset_shape: tuple[int, ...]
        :param dataset_chunks: Chunk shape along the scan dimensions, or
            ``'auto'``.
        :type dataset_chunks: list[int] or str
        :param signal_shape: Shape of one frame of the 1-D signal being
            fit (the signal dimension).
        :type signal_shape: tuple[int, ...]
        :param nxlinks: NeXus path(s) to link into the ``data`` group.
            When the zarr tree is written to a ``.zarr`` file and
            converted to ``.nxs`` with
            :class:`~CHAP.common.processor.ZarrToNexusProcessor`, each
            path produces an ``NXlink`` whose name is
            ``os.path.basename(path)``.  Accepts a single path string or
            a list of path strings.  All links must be explicit; none are
            auto-generated.
        :type nxlinks: str or list[str], optional
        :returns: Nested dict representing the zarr group tree for this
            model component.
        :rtype: dict
        """
        if isinstance(nxlinks, str):
            nxlinks = [nxlinks]
        data_attrs = {}
        if nxlinks:
            data_attrs['__nxlinks__'] = {
                os.path.basename(p): p for p in nxlinks
            }
        return {
            'attributes': {
                'NX_class': 'NXcollection',
            },
            'children': {
                'parameters': {
                    'attributes': {
                        'model': self.model_type,
                        'NX_class': 'NXparameters',
                    },
                    'children': {
                        param.long_name: param.zarr_tree(
                            dataset_shape, dataset_chunks, nxlinks=nxlinks
                        ) for param in self.parameters
                    }
                },
                'data': {
                    'attributes': data_attrs,
                    'children': {
                        'best_fit': {
                            'shape': (*dataset_shape, *signal_shape),
                            'dtype': 'float64',
                        }
                    }
                }
            }
        }


class ConstantModel(FitModel):
    """Class representing a Constant model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['constant']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['c']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.ConstantModel
    model_type: Literal['constant']


class LinearModel(FitModel):
    """Class representing a Linear model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['linear']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['slope', 'intercept']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.LinearModel
    model_type: Literal['linear']


class QuadraticModel(FitModel):
    """Class representing a Quadratic model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['parabolic']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['a', 'b', 'c']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.QuadraticModel
    model_type: Literal['parabolic']


class ExponentialModel(FitModel):
    """Class representing an Exponential model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['exponential']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.ExponentialModel
    model_type: Literal['exponential']


class GaussianModel(FitModel):
    """Class representing a Gaussian model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['gaussian']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.GaussianModel
    model_type: Literal['gaussian']

    def _validate_parameters(self):
        """Validate the model parameters."""
        for par in self.parameters:
            if par.name == 'sigma':
                par.min = 0.0


class LorentzianModel(FitModel):
    """Class representing a Lorentzian model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['lorentzian']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.LorentzianModel
    model_type: Literal['lorentzian']

    def _validate_parameters(self):
        """Validate the model parameters."""
        for par in self.parameters:
            if par.name == 'sigma':
                par.min = 0.0


class PseudoVoigtModel(FitModel):
    """Class representing a PseudoVoigt model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['pvoigt']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.PseudoVoigtModel
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


class RectangleModel(FitModel):
    """Class representing a Rectangle model component.

    :ivar form: Shape type of the transition edges, defaults to
        `'linear'`.
    :vartype form: Literal[
        'linear', 'atan', 'arctan', 'erf', 'logistic'], optional
    :ivar model_type: Model component type.
    :vartype model_type: Literal['rectangle']
    """

    LINEAR_PARAMETERS: ClassVar[list[str]] = ['amplitude']
    LMFITMODEL: ClassVar[Type[BaseModel]] = lmfit_models.RectangleModel
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


class ExpressionModel(FitModel):
    """Class representing an Expression model component.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['expression']
    :ivar expr: Mathematical expression to represent the model
        component.
    :vartype expr: str
    """

    model_type: Literal['expression']
    expr: constr(strip_whitespace=True, min_length=1)

    _expr_parameters: PrivateAttr

    @model_validator(mode='after')
    def validate_expressionmodel_after(self):
        """Parse :attr:`expr` for free variable names and append a
        :class:`~CHAP.utils.models.FitParameter` for each one not
        already present in :attr:`parameters` and not a built-in
        ``asteval`` symbol or the independent variable ``x``.

        :return: Validated and updated model instance.
        :rtype: ExpressionModel
        """
        from asteval import (
            Interpreter,
            get_ast_names,
        )
        ast = Interpreter()
        current_params = [param.name for param in self.parameters]
        self._expr_parameters = [
            name for name in get_ast_names(ast.parse(self.expr))
            if (name != 'x' and name not in current_params
                and name not in ast.symtable)]
        for name in self._expr_parameters:
            self.parameters.append(FitParameter(name=name))
        return self

    @property
    def expr_parameters(self):
        """Return the parameter expr_parameters.

        :type: str or None
        """
        if hasattr(self, '_expr_parameters'):
            return self._expr_parameters
        return []

    def lmfit_model(self, prefix=None, parameters=None):
        """Return the corresponding lmfit model.

        :param prefix: Model prefix.
        :type prefix: str, optional
        :param parameters: Current model parameters.
        :type parameters: lmfit.Parameters
        :returns: Corresponding lmfit model.
        :rtype: lmfit.models.ExpressionModel
        """
        # System modules
        from re import sub

        # Third party modules
        from asteval import (
            Interpreter,
            get_ast_names,
        )
        from sympy import diff

        if parameters is None:
            parameters = []
        for par in self.parameters:
            if par.expr is not None:
                raise KeyError(
                    f'Invalid "expr" key ({par.expr}) in '
                    f'parameter ({par}) for an expression model')
        ast = Interpreter()
        expr = self.expr
        self._expr_parameters = [
                name for name in get_ast_names(ast.parse(expr))
                if (name != 'x' and name not in parameters
                    and name not in ast.symtable)]
        if prefix is not None:
            for name in self.expr_parameters:
                expr = sub(rf'\b{name}\b', f'{prefix}{name}', expr)
            self._expr_parameters = [
                f'{prefix}{name}' for name in self.expr_parameters]

        return lmfit_models.ExpressionModel(expr=expr, name=self.model_type)


# Available models for components of the fitting function
MODEL_CLASSES = [
    ConstantModel,
    LinearModel,
    QuadraticModel,
    ExponentialModel,
    GaussianModel,
    LorentzianModel,
    PseudoVoigtModel,
    RectangleModel,
    ExpressionModel,
]

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


class MultipeakModel(CHAPBaseModel):
    """Class representing a multipeak model.

    :ivar model_type: Model component type.
    :vartype model_type: Literal['expression']
    :ivar centers: Peak centers.
    :vartype center: list[float]
    :ivar centers_range: Range of peak centers around their centers,
        defaults to `0.0` in which case it is ignored.
        The actual values used are the larger of the ones determined
        from `centers_range` and `centers_range_fraction`.
    :vartype centers_range: float, optional
    :ivar centers_range_fraction: Range of peak centers around their
        centers as a fraction of their position, defaults to `0.05`.
        The actual values used are the larger of the ones determined
        from `centers_range` and `centers_range_fraction`.
    :vartype centers_range_fraction: float, optional
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
    centers_range: Optional[confloat(ge=0, allow_inf_nan=False)] = 0.0
    centers_range_fraction: Optional[
        confloat(ge=0, allow_inf_nan=False)] = 0.05
    fit_type: Optional[Literal['uniform', 'unconstrained']] = 'unconstrained'
    fwhm_min: Optional[confloat(allow_inf_nan=False)] = None
    fwhm_max: Optional[confloat(allow_inf_nan=False)] = None
    peak_models: Literal['gaussian', 'lorentzian', 'pvoigt'] = 'gaussian'


class FitConfig(CHAPBaseModel):
    """Class representing the configuration for the fit processor.

    :ivar abs_height_cutoff: Absolute peak height cutoff for
        peak fitting (any peak with a height smaller than
        `abs_height_cutoff` gets removed from the fit model).
    :vartype abs_height_cutoff: int, optional
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

    abs_height_cutoff: Optional[conint(gt=0)] = None
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
        confloat(gt=0, lt=1, allow_inf_nan=False)] = None

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

    def zarr_tree(self, dataset_shape, dataset_chunks,
                  signal_shape, nxlinks=None):
        """Return a nested dict representing the Zarr group tree for
        this fit's output container.

        The tree contains a ``data`` group with global fit statistics
        (``best_fit``, ``num_func_eval``, ``redchi``, ``residual``,
        ``success``) and a ``components`` group with one sub-tree per
        model component, structured by
        :meth:`~CHAP.utils.models.FitModel.zarr_tree`.  The paths in
        this tree correspond to those emitted by
        :class:`~CHAP.utils.fit.UpdateValuesProcessor`.

        :param dataset_shape: Shape of the measurement (scan) dimensions
            of the output dataset, excluding the signal dimensions.
        :type dataset_shape: tuple[int, ...]
        :param dataset_chunks: Chunk shape along the scan dimensions, or
            ``'auto'``.
        :type dataset_chunks: list[int] or str
        :param signal_shape: Shape of one frame of the 1-D signal being
            fit (the signal dimension).
        :type signal_shape: tuple[int, ...]
        :param nxlinks: NeXus path(s) to link into the ``data`` group.
            When the zarr tree is written to a ``.zarr`` file and
            converted to ``.nxs`` with
            :class:`~CHAP.common.processor.ZarrToNexusProcessor`, each
            path produces an ``NXlink`` whose name is
            ``os.path.basename(path)``.  Accepts a single path string or
            a list of path strings.  All links must be explicit; none are
            auto-generated.
        :type nxlinks: str or list[str], optional
        :returns: Nested dict representing the zarr group tree for this
            fit.
        :rtype: dict
        """
        if isinstance(nxlinks, str):
            nxlinks = [nxlinks]
        data_attrs = {}
        if nxlinks:
            data_attrs['__nxlinks__'] = {
                os.path.basename(p): p for p in nxlinks
            }
        return {
            'attributes': {
                'description': '''Container for results from
                CHAP.utils.fit.FitProcessor'''
            },
            'children': {
                'data': {
                    'attributes': {
                        'NX_class': 'NXdata',
                        **data_attrs,
                    },
                    'children': {
                        'best_fit': {
                            'shape': (*dataset_shape, *signal_shape),
                            'dtype': 'float64',
                        },
                        'num_func_eval': {
                            'shape': dataset_shape,
                            'dtype': 'uint64',
                        },
                        'redchi': {
                            'shape': dataset_shape,
                            'dtype': 'float64',
                        },
                        'residual': {
                            'shape': (*dataset_shape, *signal_shape),
                            'dtype': 'float64',
                        },
                        'success': {
                            'shape': dataset_shape,
                            'dtype': 'bool'
                        },
                    }
                },
                'components': {
                    'attributes': {
                        'NX_class': 'NXcollection',
                    },
                    'children': {
                        model.long_name: model.zarr_tree(
                            dataset_shape, dataset_chunks,
                            signal_shape, nxlinks=nxlinks
                        ) for model in self.models
                    }
                }
            }
        }
