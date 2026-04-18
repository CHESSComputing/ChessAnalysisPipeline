"""`Pydantic <https://github.com/pydantic/pydantic>`__ model
configuration classes unique to the the 
`FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__ integration.
"""

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
    """`FOXDEN <https://github.com/CHESSComputing/FOXDEN>`__
    HTTP request base configuration class.

    :ivar did: FOXDEN dataset identifier (DID).
    :vartype did: string, optional
    :ivar idx: Index of the first record in the list of records to
        be retured, defaults to `0`.
    :vartype idx: int, optional
    :ivar limit: Maximum number of returned records,
        defaults to `10`.
    :vartype limit: int, optional
    :ivar query: FOXDEN query.
    :vartype query: string, optional
    :ivar url: URL of service.
    :vartype url: str
    :ivar verbose: Verbose output flag, defaults to `False`.
    :vartype verbose: bool, optional
    """
#    :ivar method: HTTP request method (not case sensitive),
#        defaults to `'POST'`.
#    :vartype method: Literal['DELETE', 'GET', 'POST', 'PUT'], optional
#    :ivar scope: FOXDEN scope (not case sensitive).
#    :vartype scope: Literal['read', 'write'], optional
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
        """Create the payload for a HTTP request.

        :param reader: Any of the FOXDEN readers in
            :mod:`~CHAP.foxden.reader`.
        :type reader: FoxdenDataDiscoveryReader or
            FoxdenMetadataReader or FoxdenProvenanceReader
        :return: JSON string of the HTTP request.
        :rtype: str
        """
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
