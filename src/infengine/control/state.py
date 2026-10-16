import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class StateStoreError(RuntimeError):
    """Persisted desired state is unreadable/corrupt."""


class StateStore:
    """Persist desired inference state as JSON with a monotonic revision.

    ``path=None`` keeps the desired state in memory only; a path makes it
    durable so a restarted worker can re-apply its last known intent.

    M6B-1: the store maintains ``desired_revision`` — a monotonically
    increasing version of the persisted DesiredState. It is stored as SIBLING
    METADATA (``{path}.revision``), NOT as a DesiredState key, so the
    desired-state JSON file remains exactly the desired key/value dict (the
    DesiredState key model is unchanged). ``desired_revision`` increments on
    every successful ``save()`` (each atomic intent acceptance); reads never
    bump it. A valid persisted store is loaded eagerly at construction so
    ``revision``/``desired`` are correct immediately after restart.

    Single-writer assumption: one process writes the file. Writes are atomic
    (temp file + fsync + ``os.replace``), so a crash never leaves a truncated
    target. Corrupt persisted state raises :class:`StateStoreError` rather than
    being silently replaced with empty intent.
    """

    REVISION_SUFFIX = ".revision"

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self._desired: dict[str, Any] = {}
        self._revision: int = 0
        # Eagerly load a valid persisted store so ``desired``/``revision`` are
        # available immediately after construction (restart persistence). A
        # corrupt file is deferred to the explicit ``load()`` (raises
        # StateStoreError there), matching the pre-existing behavior.
        if self.path is not None and self.path.exists():
            try:
                self.load()
            except StateStoreError:
                pass

    @property
    def revision_path(self) -> Optional[Path]:
        return None if self.path is None else Path(str(self.path) + self.REVISION_SUFFIX)

    def _load_revision(self) -> int:
        rpath = self.revision_path
        if rpath is None or not rpath.exists():
            return 0
        try:
            with open(rpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise StateStoreError(
                f"desired-state revision file {rpath} is corrupt or unreadable: {exc}"
            ) from exc
        rev = data.get("desired_revision", 0) if isinstance(data, dict) else 0
        if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0:
            raise StateStoreError(
                f"desired-state revision file {rpath} has an invalid revision {rev!r}"
            )
        return rev

    def load(self) -> dict[str, Any]:
        if self.path is not None and self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._desired = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                raise StateStoreError(
                    f"desired-state file {self.path} is corrupt or unreadable: {exc}"
                ) from exc
            self._revision = self._load_revision()
        return dict(self._desired)

    def _write_json(self, path: Path, data) -> None:
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def save(self, desired: dict[str, Any]) -> None:
        """Persist a newly accepted DesiredState version and increment the
        revision.

        ``desired_revision`` is reserved metadata; it can never be a
        DesiredState key, so it is defensively excluded from the persisted
        payload. The revision metadata is written FIRST (monotonic — it never
        lags the desired file), then the desired-state file. In-memory state
        commits only after BOTH writes succeed, so a persistence failure leaves
        memory consistent with disk (revision may be one ahead of the desired
        file after a crash between the two writes; that is benign and
        self-healing because If-Match only ever becomes stricter).
        """
        normalized = {k: v for k, v in desired.items() if k != "desired_revision"}
        next_rev = self._revision + 1
        if self.path is not None:
            self._write_json(self.revision_path, {"desired_revision": next_rev})
            self._write_json(self.path, normalized)
        self._desired = normalized
        self._revision = next_rev

    @property
    def desired(self) -> dict[str, Any]:
        return dict(self._desired)

    @property
    def revision(self) -> int:
        return self._revision