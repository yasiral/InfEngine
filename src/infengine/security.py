"""Smallest credible pilot security layer for the InfEngine control plane.

Not enterprise IAM. A deterministic role model over bearer tokens:

  VIEWER    inspect, capabilities, explain, read history/reports
  OPERATOR  VIEWER + set, apply, restore, evaluate
  APPROVER  review / approve / reject

Tokens are supplied out-of-band (environment variable ``INFENGINE_AUTH_TOKENS``
or a JSON file ``INFENGINE_AUTH_FILE``), never on the command line:

    {"<token>": {"actor": "yasir", "role": "OPERATOR"}, ...}

When no tokens are configured, authentication is disabled (local development).
The service warns prominently when auth is disabled on a non-loopback bind; the
unified CLI refuses to send mutations to a remote service without a token unless
``--insecure`` is passed.
"""

import hmac
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class AuthError(RuntimeError):
    """Authentication or authorization failed."""


class Role(str, Enum):
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    APPROVER = "APPROVER"


class Action(str, Enum):
    READ = "READ"
    MUTATE = "MUTATE"
    APPROVE = "APPROVE"


# Which roles may perform which action class. Deterministic, no hierarchy magic.
_ALLOWED: Dict[Action, Tuple[Role, ...]] = {
    Action.READ: (Role.VIEWER, Role.OPERATOR, Role.APPROVER),
    Action.MUTATE: (Role.OPERATOR, Role.APPROVER),
    Action.APPROVE: (Role.APPROVER,),
}

ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class Identity:
    actor: str
    role: Role
    authenticated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"actor": self.actor, "role": self.role.value,
                "authenticated": self.authenticated}


class AuthConfig:
    """Bearer-token identities loaded from env or a JSON file.

    ``enabled`` is False only when no tokens are configured. ``load`` raises
    ``AuthError`` on a malformed config (fail closed) rather than silently
    disabling auth.
    """

    def __init__(self, tokens: Optional[Dict[str, Dict[str, Any]]] = None):
        self.tokens = dict(tokens or {})
        # Pre-index by role for constant-time-ish lookup.
        self._by_token: Dict[str, Identity] = {}
        for token, spec in self.tokens.items():
            if not isinstance(spec, dict):
                raise AuthError("auth token entry must be an object")
            role = str(spec.get("role", "")).upper()
            if role not in {r.value for r in Role}:
                raise AuthError(f"unknown role {role!r} for a token entry")
            actor = spec.get("actor")
            if not actor:
                raise AuthError("auth token entry requires an 'actor'")
            self._by_token[token] = Identity(str(actor), Role(role), True)

    @property
    def enabled(self) -> bool:
        return bool(self._by_token)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "AuthConfig":
        env = os.environ if env is None else env
        raw = env.get("INFENGINE_AUTH_TOKENS")
        path = env.get("INFENGINE_AUTH_FILE")
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AuthError(f"INFENGINE_AUTH_TOKENS is not valid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise AuthError("INFENGINE_AUTH_TOKENS must be a JSON object")
            return cls(data)
        if path:
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise AuthError(f"INFENGINE_AUTH_FILE unreadable: {exc}") from exc
            if not isinstance(data, dict):
                raise AuthError("INFENGINE_AUTH_FILE must contain a JSON object")
            return cls(data)
        return cls({})

    # ------------------------------------------------------------------ verify
    def authenticate(self, headers: Dict[str, str]) -> Identity:
        """Return the identity for the request, or raise ``AuthError``.

        Header lookup is case-insensitive (Starlette normalizes to lowercase).
        """
        if not self.enabled:
            actor = headers.get("x-infengine-actor") or ANONYMOUS
            return Identity(actor, Role.OPERATOR, False)
        supplied = headers.get("authorization") or ""
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:]
        supplied = supplied or headers.get("x-infengine-token") or ""
        for token, identity in self._by_token.items():
            if supplied and hmac.compare_digest(supplied, token):
                return identity
        raise AuthError("missing or invalid bearer token")

    def require(self, identity: Identity, action: Action) -> None:
        if identity.role not in _ALLOWED[action]:
            raise AuthError(
                f"role {identity.role.value} is not permitted to {action.value}"
            )


def loopback_only(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")
