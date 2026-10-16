"""Minimal Prometheus text exposition parser (M5B).

Parses the subset of the Prometheus text exposition format emitted by the
engines' native telemetry surfaces:

  * vLLM v0.28.0 ``GET /metrics``  (``vllm:``-namespaced families)
  * SGLang v0.5.18 ``GET /metrics`` (``sglang:``-namespaced families)
  * SGLang v0.5.18 ``GET /v1/loads?format=prometheus`` (``sglang_*`` gauges)

No external dependency (the package deliberately does not depend on
``prometheus_client``). Supports counters, gauges, histograms (``_bucket`` /
``_sum`` / ``_count``), label escaping, ``+Inf``/``-Inf``/``NaN`` values, and
optional per-sample timestamps. ``# HELP`` / ``# TYPE`` lines are honored;
unknown families are retained so callers can decide availability.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

_INF = math.inf


class ExpositionParseError(ValueError):
    """The exposition text is malformed."""


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")


def _parse_number(raw: str) -> float:
    token = raw.strip()
    if token == "+Inf":
        return _INF
    if token == "-Inf":
        return -_INF
    if token == "NaN":
        return math.nan
    try:
        return float(token)
    except ValueError as exc:
        raise ExpositionParseError(f"invalid number {raw!r}") from exc


@dataclass(frozen=True)
class Sample:
    """One exposition sample (value + labels + optional timestamp)."""

    value: float
    labels: Tuple[Tuple[str, str], ...] = ()
    timestamp: Optional[float] = None

    def label(self, name: str) -> Optional[str]:
        for key, value in self.labels:
            if key == name:
                return value
        return None


@dataclass
class Family:
    """A metric family: name, type, and its samples."""

    name: str
    type: str
    samples: List[Sample] = field(default_factory=list)


@dataclass(frozen=True)
class HistogramSeries:
    """A single labeled histogram series (cumulative buckets, sum, count)."""

    labels: Tuple[Tuple[str, str], ...]
    buckets: Tuple[Tuple[float, float], ...]  # sorted by le, ascending; +Inf last
    sum: float
    count: float


class Exposition:
    """Parsed exposition: family name -> :class:`Family`."""

    def __init__(self, families: Dict[str, Family]):
        self.families = families

    def family(self, name: str) -> Optional[Family]:
        return self.families.get(name)

    def samples(self, name: str) -> List[Sample]:
        fam = self.families.get(name)
        return list(fam.samples) if fam else []

    def _matches(self, sample: Sample, match: Optional[Dict[str, str]]) -> bool:
        if not match:
            return True
        for key, value in match.items():
            if sample.label(key) != value:
                return False
        return True

    def find(self, name: str, match: Optional[Dict[str, str]] = None) -> Optional[Sample]:
        for sample in self.samples(name):
            if self._matches(sample, match):
                return sample
        return None

    def values(self, name: str, match: Optional[Dict[str, str]] = None) -> List[float]:
        return [s.value for s in self.samples(name) if self._matches(s, match)]

    def histogram_series(self, name: str) -> List[HistogramSeries]:
        """Group ``<name>_bucket`` / ``_sum`` / ``_count`` into series.

        Returns one :class:`HistogramSeries` per distinct label set (minus the
        ``le`` label). Buckets are sorted ascending by boundary with ``+Inf``
        last. An empty list means the family (or its histogram samples) is
        absent.
        """
        groups: Dict[Tuple[Tuple[str, str], ...], Dict[str, object]] = {}
        prefix = name + "_bucket"
        for fam_name, fam in self.families.items():
            if not fam_name.startswith(name + "_"):
                continue
            suffix = fam_name[len(name) + 1:]
            if suffix not in ("bucket", "sum", "count"):
                continue
            for sample in fam.samples:
                if suffix == "bucket":
                    le = sample.label("le")
                    if le is None:
                        continue
                    labels = tuple((k, v) for k, v in sample.labels if k != "le")
                    group = groups.setdefault(labels, {"buckets": []})
                    group["buckets"].append((_parse_number(le), sample.value))  # type: ignore[attr-defined]
                elif suffix == "sum":
                    labels = tuple(sample.labels)
                    groups.setdefault(labels, {"buckets": []})["sum"] = sample.value  # type: ignore[index]
                else:  # count
                    labels = tuple(sample.labels)
                    groups.setdefault(labels, {"buckets": []})["count"] = sample.value  # type: ignore[index]
        out: List[HistogramSeries] = []
        for labels, data in groups.items():
            buckets = data["buckets"]  # type: ignore[index]
            if not buckets:
                continue
            buckets.sort(key=lambda item: item[0])
            out.append(
                HistogramSeries(
                    labels=labels,
                    buckets=tuple(buckets),
                    sum=float(data.get("sum", 0.0)),  # type: ignore[arg-type]
                    count=float(data.get("count", 0.0)),  # type: ignore[arg-type]
                )
            )
        return out


def _split_sample(line: str) -> Tuple[str, List[Tuple[str, str]], str, Optional[str]]:
    """Parse ``name{labels} value [ts]`` -> (name, labels, value, ts)."""
    rest = line
    if rest.startswith("{"):
        raise ExpositionParseError("sample line missing metric name")
    brace = rest.find("{")
    if brace == -1:
        name = rest.split()[0]
        label_text = ""
        remainder = rest[len(name):].strip()
    else:
        name = rest[:brace]
        close = rest.find("}", brace)
        if close == -1:
            raise ExpositionParseError("unterminated label set")
        label_text = rest[brace + 1:close]
        remainder = rest[close + 1:].strip()
    labels: List[Tuple[str, str]] = []
    if label_text.strip():
        for part in label_text.split(","):
            part = part.strip()
            if not part:
                continue
            key, _, value = part.partition("=")
            if not value.startswith('"') or not value.endswith('"'):
                raise ExpositionParseError(f"invalid label {part!r}")
            labels.append((key.strip(), _unescape(value[1:-1])))
    tokens = remainder.split()
    if not tokens:
        raise ExpositionParseError(f"sample {name!r} missing value")
    value = tokens[0]
    timestamp = tokens[1] if len(tokens) > 1 else None
    return name, labels, value, timestamp


def parse_exposition(text: str) -> Exposition:
    """Parse Prometheus text exposition into an :class:`Exposition`."""
    families: Dict[str, Family] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip() or line.startswith("#"):
            if line.startswith("# TYPE "):
                _, _, rest = line.partition("TYPE ")
                parts = rest.split()
                if len(parts) == 2:
                    name, mtype = parts
                    families.setdefault(name, Family(name=name, type=mtype))
            continue
        name, labels, value, timestamp = _split_sample(line)
        fam = families.setdefault(name, Family(name=name, type="untyped"))
        ts = _parse_number(timestamp) if timestamp is not None else None
        fam.samples.append(Sample(value=_parse_number(value), labels=tuple(labels), timestamp=ts))
    return Exposition(families)


def histogram_quantile(
    q: float, buckets: Iterable[Tuple[float, float]]
) -> Optional[float]:
    """Prometheus ``histogram_quantile`` over cumulative bucket (le, count).

    ``buckets`` must be sorted ascending by ``le`` with the ``+Inf`` bucket
    last. Returns ``None`` when the distribution is empty. Linear interpolation
    inside the bucket containing the target observation (upper bound used when
    the bucket is the ``+Inf`` overflow).
    """
    ordered = []
    for le, count in buckets:
        try:
            upper = float(le)
        except (TypeError, ValueError):
            continue
        ordered.append((upper, count))
    ordered.sort(key=lambda item: item[0])
    if not ordered:
        return None
    total = ordered[-1][1]
    if not math.isfinite(total) or total <= 0:
        return None
    target = q * total
    for index, (upper, count_hi) in enumerate(ordered):
        if count_hi >= target:
            if index == 0:
                return upper
            lower, count_lo = ordered[index - 1]
            if upper == _INF:
                return lower
            if count_hi <= count_lo:
                return lower
            fraction = (target - count_lo) / (count_hi - count_lo)
            return lower + (upper - lower) * fraction
    return None