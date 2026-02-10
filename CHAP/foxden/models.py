"""FOXDEN Pydantic model classes."""

# System modules
from typing import (
#    Literal,
    Optional,
)

# Third party modules
from pydantic import (
    conint,
    constr,
#    field_validator,
)

# Local modules
from CHAP import CHAPBaseModel


class FoxdenRequestConfig(CHAPBaseModel):
    """FOXDEN HTTP request base configuration class.

    :param did: FOXDEN dataset identifier (did).
    :type did: string, optional
    :param limit: Maximum number of returned records,
        defaults to `10`.
    :type limit: int, optional
    :param query: FOXDEN query.
    :type query: string, optional
    :param verbose: Verbose output flag, defaults to `False`.
    :type verbose: bool, optional
    """
#    :param method: HTTP request method (not case sensitive),
#        defaults to `'POST'`.
#    :type method: Literal['DELETE', 'GET', 'POST', 'PUT'], optional
#    :param scope: FOXDEN scope (not case sensitive).
#    :type scope: Literal['read', 'write'], optional
#    :param idx: Ask Valentin, currently it's ignored
#    :type idx: int, optional
    # Mimics golib.services.data.ServiceQuery
    did: Optional[constr(
        strict=True, strip_whitespace=True, to_lower=True)] = None
    limit: Optional[conint(gt=0)] = 10
    query: Optional[constr(
        strict=True, strip_whitespace=True, to_lower=True)] = None
#    method: Optional[Literal['DELETE', 'GET', 'POST', 'PUT']] = 'POST'
#    scope: Optional[Literal['read', 'write']] = None
#    sortkeys: Optional[
#        conlist[item_type=constr(strict=True, strip_whitespace=True)]] = None
#    sortorder: Optional[int] = None
#    spec: Optional[map[string]any] ?
#    sql: Optional[constr(strict=True, strip_whitespace=True)] = None
#    idx: Optional[conint(ge=0)] = 0
    url: constr(strict=True, strip_whitespace=True)
    verbose: Optional[bool] = None

#    @field_validator('method', mode='before')
#    @classmethod
#    def validate_method(cls, method):
#        """Capitalize method."""
#        return method.upper()

#    @field_validator('scope', mode='before')
#    @classmethod
#    def validate_scope(cls, scope):
#        """Enforce lowercase for scope."""
#        if isinstance(scope, str):
#            return scope.lower()
#        return scope
