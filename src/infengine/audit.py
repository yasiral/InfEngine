"""Durable, structured change/audit history (M13 Phase 8).

A single SQLite file (stdlib only) is the lightweight durable store. Every
mutation attempt — applied, failed, conflicted or rejected — is recorded as one
row with deterministic, machine-readable evidence. No observability platform.

The store is append-only from the control plane's perspective; ``get``/``list``
read it back. ``pre_state`` is the verified observed state captured immediately
before the mutation, which is what makes history-based ``infengine restore``
possible without an operator remembering old values.
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

_COLUMNS = (
    "change_id", "created_at", "actor", "role", "action", "engine", "model",
    "identity", "revision_before", "revision_after", "requested", "pre_state",
    "observed", "result", "verification", "rollback", "proposal_id",
    "approved", "dry_run", "detail",
)


@dataclass
class ChangeRecord:
    """One audit row (all fields JSON-serializable)."""

    change_id: str
    created_at: str
    actor: str
    role: str
    action: str
    engine: Optional[str] = None
    model: Optional[str] = None
    identity: Optional[Dict[str, Any]] = None
    revision_before: Optional[int] = None
    revision_after: Optional[int] = None
    requested: Optional[Dict[str, Any]] = None
    pre_state: Optional[Dict[str, Any]] = None
    observed: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    rollback: Optional[Dict[str, Any]] = None
    proposal_id: Optional[str] = None
    approved: Optional[bool] = None
    dry_run: bool = False
    detail: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def make_change_id(*parts: Any) -> str:
    return sha256("|".join(_canonical(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditStore:
    """Append-only SQLite change log.

    ``path=None`` keeps an in-memory database (tests); a path makes history
    durable across process boundaries.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        target = path if path else ":memory:"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS changes ("
                "change_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                "actor TEXT NOT NULL, role TEXT NOT NULL, action TEXT NOT NULL, "
                "engine TEXT, model TEXT, identity TEXT, revision_before INTEGER, "
                "revision_after INTEGER, requested TEXT, pre_state TEXT, "
                "observed TEXT, result TEXT, verification TEXT, rollback TEXT, "
                "proposal_id TEXT, approved INTEGER, dry_run INTEGER, detail TEXT)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def record(self, event: Dict[str, Any], *, now: Optional[str] = None) -> ChangeRecord:
        """Persist one change event and return the stored record."""
        created = now or event.get("created_at") or _iso_now()
        change_id = event.get("change_id") or make_change_id(
            created, event.get("actor"), event.get("action"),
            event.get("requested"), event.get("revision_before"),
        )
        record = ChangeRecord(
            change_id=change_id,
            created_at=created,
            actor=event.get("actor") or "anonymous",
            role=event.get("role") or "OPERATOR",
            action=event.get("action") or "apply",
            engine=event.get("engine"),
            model=event.get("model"),
            identity=event.get("identity"),
            revision_before=event.get("revision_before"),
            revision_after=event.get("revision_after"),
            requested=event.get("requested"),
            pre_state=event.get("pre_state"),
            observed=event.get("observed"),
            result=event.get("result"),
            verification=event.get("verification"),
            rollback=event.get("rollback"),
            proposal_id=event.get("proposal_id"),
            approved=event.get("approved"),
            dry_run=bool(event.get("dry_run", False)),
            detail=event.get("detail"),
        )
        row = record.to_dict()
        values = [
            json.dumps(row[c]) if isinstance(row[c], (dict, list)) else
            (int(row[c]) if isinstance(row[c], bool) else row[c])
            for c in _COLUMNS
        ]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO changes ({','.join(_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in _COLUMNS)})",
                values,
            )
            self._conn.commit()
        return record

    def get(self, change_id: str) -> Optional[ChangeRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM changes WHERE change_id = ?", (change_id,)
            )
            row = cur.fetchone()
        return self._row(row) if row else None

    def latest(self) -> Optional[ChangeRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM changes ORDER BY created_at DESC, rowid DESC LIMIT 1"
            )
            row = cur.fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 50, offset: int = 0) -> List[ChangeRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM changes ORDER BY created_at DESC, rowid DESC "
                "LIMIT ? OFFSET ?", (int(limit), int(offset)),
            )
            rows = cur.fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> ChangeRecord:
        data: Dict[str, Any] = dict(row)
        for key in ("identity", "requested", "pre_state", "observed",
                    "verification", "rollback", "detail"):
            raw = data.get(key)
            if raw:
                try:
                    data[key] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    data[key] = None
        if data.get("approved") is not None:
            data["approved"] = bool(data["approved"])
        data["dry_run"] = bool(data.get("dry_run"))
        return ChangeRecord(**{k: data.get(k) for k in ChangeRecord.__dataclass_fields__})

    def close(self) -> None:
        with self._lock:
            self._conn.close()
