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
    :param idx: Index of the first record in the list of records to
        be retured, defaults to `0`.
    :type idx: int, optional
    :param limit: Maximum number of returned records,
        defaults to `10`.
    :type limit: int, optional
    :param query: FOXDEN query.
    :type query: string, optional
    :param url: URL of service.
    :type url: str
    :param verbose: Verbose output flag, defaults to `False`.
    :type verbose: bool, optional
    """
#    :param method: HTTP request method (not case sensitive),
#        defaults to `'POST'`.
#    :type method: Literal['DELETE', 'GET', 'POST', 'PUT'], optional
#    :param scope: FOXDEN scope (not case sensitive).
#    :type scope: Literal['read', 'write'], optional
    # Mimics golib.services.data.ServiceQuery
    did: Optional[constr(
        strict=True, strip_whitespace=True, to_lower=True)] = None
    idx: Optional[conint(ge=0)] = 0
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
    url: Optional[constr(strict=True, strip_whitespace=True)] = None
    verbose: Optional[bool] = 'False'

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

    def create_http_request_payload(self, reader):
        # Third party modules
        from json import dumps

        request = {
            'client': f'CHAP-{reader.name}',
            'service_query': {'query': '{}'}}
        if self.did is None:
            if self.query is not None:
                request['service_query'].update({'query': self.query})
        else:
            if reader.name == 'FoxdenMetadataReader':
                if self.idx is not None:
                    reader.logger.warning(
                        f'Ignoring parameter "idx" ({self.idx}), '
                        'when "did" is specified')
                if self.limit is not None:
                    reader.logger.warning(
                        f'Ignoring parameter "limit" ({self.limit}), '
                        'when "did" is specified')
            else:
                request['service_query'].update({
                    'idx': self.idx, 'limit': self.limit})
            if self.query is not None:
                reader.logger.warning(
                    f'Ignoring parameter "query" ({self.query}), '
                    'when "did" is specified')
            request['service_query'].update({'query': f'did:{self.did}'})
        return dumps(request)
