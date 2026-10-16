"""M9C: self-contained local HTML report from a load-sweep result.json.

The report is a pure presentation of the same ``result.json`` (no separate
collection or analysis pipeline). Charts are drawn with inline vanilla JS on
``<canvas>`` elements; there is no CDN, dashboard server, database or hosted
service. Unavailable measurements are shown explicitly (never plotted as zero).
"""

import html
import json
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------- panel model
def _levels(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(result.get("levels", []), key=lambda lvl: float(lvl["offered_load"]))


def _agg(level: Dict[str, Any], metric: str) -> Optional[float]:
    value = (level.get("aggregate", {}).get(metric) or {}).get("median")
    return float(value) if isinstance(value, (int, float)) else None


def _counter_delta(level: Dict[str, Any], key: str) -> Optional[float]:
    vals = []
    for run in level.get("runs", []):
        entry = (run.get("counter_deltas") or {}).get(key) or {}
        v = entry.get("value")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) if vals else None


def _evidence(level: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    for run in level.get("runs", []):
        if not run.get("ok"):
            continue
        ev = run.get("execution_evidence") or {}
        for section in ("request_stage", "engine_level"):
            entry = (ev.get(section) or {}).get(key)
            if entry is not None:
                return entry
    return None


def _ev_q(level: Dict[str, Any], key: str, quantile: str = "p95") -> Optional[float]:
    entry = _evidence(level, key)
    if not entry:
        return None
    value = (entry.get("quantiles") or {}).get(quantile)
    return float(value) if isinstance(value, (int, float)) else None


def _ev_val(level: Dict[str, Any], key: str) -> Optional[float]:
    entry = _evidence(level, key)
    if not entry:
        return None
    value = entry.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _totals(level: Dict[str, Any], key: str) -> int:
    return int(sum(int(r.get("counts", {}).get(key, 0)) for r in level.get("runs", [])))


def _series(
    name: str,
    unit: str,
    points: List[Tuple[float, Optional[float]]],
    samples: Optional[List[Optional[int]]] = None,
    low_sample: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "unit": unit,
        "points": [[float(x), (None if y is None else float(y))] for x, y in points],
        "samples": list(samples) if samples is not None else None,
        "low_sample": list(low_sample) if low_sample is not None else None,
    }


def _thresholds(result: Dict[str, Any]) -> Tuple[int, int]:
    measurement = (result.get("sweep") or {}).get("measurement") or {}
    return int(measurement.get("min_polls") or 1), int(measurement.get("min_requests") or 1)


def _lat_series(
    result: Dict[str, Any], levels: List[Dict[str, Any]], name: str, metric: str
) -> Dict[str, Any]:
    """Normalized latency series: points + per-point sample counts and a
    low-sample flag. A fresh value is NOT automatically adequately sampled."""
    min_polls, min_req = _thresholds(result)
    points, samples, low = [], [], []
    for lvl in levels:
        entry = lvl.get("aggregate", {}).get(metric) or {}
        value = entry.get("median")
        points.append((float(lvl["offered_load"]), value))
        samples.append(entry.get("sample_count"))
        polls = entry.get("polls_usable")
        ok = (
            value is not None
            and entry.get("dominant_quality") not in ("unavailable", "disabled", "no_samples")
            and (polls is None or polls >= min_polls)
            and (entry.get("sample_count") is None or entry.get("sample_count") >= min_req)
        )
        low.append(not ok)
    return _series(name, "s", points, samples, low)


def _ev_series(
    result: Dict[str, Any], levels: List[Dict[str, Any]], name: str, key: str
) -> Dict[str, Any]:
    """Execution-stage series from the M9B evidence (measured quantiles only)."""
    _, min_req = _thresholds(result)
    points, samples, low = [], [], []
    for lvl in levels:
        measurement = _evidence(lvl, key)
        value = (measurement.get("quantiles") or {}).get("p95") if measurement else None
        count = measurement.get("sample_count") if measurement else None
        classification = measurement.get("classification") if measurement else None
        points.append((float(lvl["offered_load"]),
                       float(value) if isinstance(value, (int, float)) else None))
        samples.append(count)
        low.append(not (
            classification == "MEASURED" and value is not None
            and (count or 0) >= min_req
        ))
    return _series(name, "s", points, samples, low)


def _panel(
    panel_id: str, title: str, x_label: str, y_label: str, series: List[Dict[str, Any]],
    panel_type: str = "line", notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    available: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, str]] = []
    for s in series:
        if any(pt[1] is not None for pt in s["points"]):
            available.append(s)
        else:
            unavailable.append({"name": s["name"],
                                "reason": "no measured value at any level"})
    return {
        "id": panel_id, "title": title, "x_label": x_label, "y_label": y_label,
        "type": panel_type, "series": available, "unavailable": unavailable,
        "notes": list(notes or []),
    }


def build_panels(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pure, deterministic panel/point derivation from the result document."""
    levels = _levels(result)
    x = [float(lvl["offered_load"]) for lvl in levels]
    engine = (result.get("engine") or {}).get("name", "engine")

    def pt(metric: str) -> List[Tuple[float, Optional[float]]]:
        return [(xi, _agg(lvl, metric)) for xi, lvl in zip(x, levels)]

    def counter(key: str) -> List[Tuple[float, Optional[float]]]:
        return [(xi, _counter_delta(lvl, key)) for xi, lvl in zip(x, levels)]

    panels: List[Dict[str, Any]] = []

    # A. throughput
    panels.append(_panel(
        "throughput_req", "A1. Completed request throughput vs offered load",
        "offered load", "requests/s", [_series("completed req/s", "req/s", pt("throughput_req_s"))],
    ))
    panels.append(_panel(
        "output_throughput",
        "A2. Experiment-period output-token throughput vs offered load",
        "offered load", "tokens/s",
        [_series("experiment-period tokens/s", "tokens/s", pt("output_tokens_per_s"))],
        notes=["Derived from the generation-token counter delta over the measurement "
               "window (warmup and phase-end drain excluded). This is NOT the "
               "engine-native windowed gauge; unavailable counter evidence yields a gap, "
               "never a fabricated positive value."],
    ))
    panels.append(_panel(
        "throughput_tokens",
        f"A3. Engine-native windowed token-throughput gauge ({engine}-specific)",
        "offered load", "tokens/s",
        [_series(f"native windowed gauge ({engine})", "tokens/s", pt("throughput"))],
        notes=["Engine-native metric with its own window: SGLang gen_throughput is a "
               "windowed gauge (decode_log_interval) that can read 0 while the cumulative "
               "generation-token counter still advances; vLLM's value is itself derived "
               "from the generation-token counter. Not normalized across engines."],
    ))

    # B. latency
    ttft = _lat_series(result, levels, "TTFT p95", "ttft_p95_s")
    b_notes = [
        "Windowed quantiles. TTFT/queue/e2e are normalized across engines.",
        "Hollow markers = plotted point with inadequate samples (below the configured "
        "min_polls/min_requests): a fresh value is not automatically adequately sampled.",
    ]
    missing = [
        (str(lvl["offered_load"]),
         (lvl.get("aggregate", {}).get("ttft_p95_s") or {}).get("dominant_quality"))
        for lvl, point in zip(levels, ttft["points"]) if point[1] is None
    ]
    if missing:
        b_notes.append(
            "TTFT unavailable (genuine gap, never zero or interpolated) at offered "
            "load(s): " + ", ".join(f"{x} (quality={q})" for x, q in missing)
        )
    panels.append(_panel(
        "latency_tail", "B. TTFT and queue latency vs offered load",
        "offered load", "seconds",
        [ttft,
         _lat_series(result, levels, "queue time p95", "queue_time_p95_s"),
         _lat_series(result, levels, "e2e p95", "e2e_p95_s")],
        notes=b_notes,
    ))

    # C. measured execution-stage (engine-specific; decode only where measured)
    panels.append(_panel(
        "execution_stage", "C. Measured prefill / decode latency vs offered load (p95)",
        "offered load", "seconds",
        [_ev_series(result, levels, "prefill (measured)", "prefill_time"),
         _ev_series(result, levels, "decode (measured)", "decode_time"),
         _ev_series(result, levels, "time per output token (measured)", "time_per_output_token"),
         _ev_series(result, levels, "inter-token latency (engine-specific)", "inter_token_latency")],
        notes=[
            "Hollow markers = inadequate sample count for that point.",
            "vLLM request_prefill/decode/inference and time-per-output-token are measured; "
            "vLLM inter-token latency is per engine step.",
            "SGLang prefill is the measured per_stage 'prefill_forward'; SGLang does NOT "
            "emit a request-scoped decode duration — it is shown as unavailable, never "
            "synthesized. SGLang inter-token latency is token-weighted (not comparable to "
            "vLLM's).",
        ],
    ))

    # D. scheduler / concurrency
    panels.append(_panel(
        "concurrency", "D1. Running / waiting requests vs offered load",
        "offered load", "requests",
        [_series("running (median)", "count", pt("requests_running")),
         _series("waiting (median)", "count", pt("requests_waiting"))],
    ))
    if engine == "sglang":
        sched = _series("decode_sum_seq_lens", "tokens",
                        [(xi, _ev_val(lvl, "decode_sum_seq_lens")) for xi, lvl in zip(x, levels)])
        sched_name = "scheduler batch evidence"
    else:
        sched = _series("iteration_tokens_p95", "tokens/step",
                        [(xi, _ev_val(lvl, "iteration_tokens_p95")) for xi, lvl in zip(x, levels)])
        sched_name = "scheduler batch evidence"
    panels.append(_panel(
        "scheduler", f"D2. Scheduler activity vs offered load ({engine}-specific)",
        "offered load", sched["unit"],
        [dict(sched, name=sched_name)],
        notes=["Engine-specific scheduler/token evidence; not normalized across engines."],
    ))

    # E. KV / cache / preemptions
    panels.append(_panel(
        "kv", "E1. KV utilization vs offered load",
        "offered load", "ratio 0..1", [_series("KV usage (median)", "ratio", pt("kv_usage"))],
    ))
    panels.append(_panel(
        "pressure", "E2. Preemption / retraction evidence vs offered load (within-level delta)",
        "offered load", "events",
        [_series("preemptions (vLLM)", "events", counter("num_preemptions")),
         _series("retractions (SGLang)", "events", counter("num_retracted_requests_total"))],
        notes=["Cumulative counters delta'd within a level (reset/instance-safe: null, not a "
               "fabricated delta). Preemption (vLLM) and retraction (SGLang) are different "
               "mechanisms and are never equated."],
    ))

    # F. errors / timeouts / incomplete. Genuine zeros are DATA and are plotted
    # as zero (only a genuinely missing count becomes unavailable).
    outcome_series = [
        _series(name, "count", [(xi, float(_totals(lvl, name))) for xi, lvl in zip(x, levels)])
        for name in ("timeout", "error", "incomplete")
    ]
    panels.append(_panel(
        "outcomes", "F. Errors, timeouts and incomplete requests vs offered load",
        "offered load", "requests (total over runs)", outcome_series,
        panel_type="bar",
        notes=["Zeros are measured (no failures), not missing. Phase-end drain "
               "completions are excluded from measurement throughput."],
    ))
    return panels


# ------------------------------------------------------- native histograms
def _bucket_label(index: int, lo: Optional[float], hi: Optional[float]) -> str:
    if hi is None:
        return f"≥ {lo:g} s" if lo is not None else "=Inf"
    if index == 0:
        return f"< {hi:g} s"
    return f"{lo:g}–{hi:g} s"


def _bucket_ranges(le: List[Any], counts: List[int]) -> List[Dict[str, Any]]:
    """Convert windowed CUMULATIVE bucket counts to non-overlapping buckets.

    ``le`` bounds and ``counts`` are cumulative within the window
    (observations <= le). The ``+Inf`` bucket is the overflow; its lower bound is
    the last finite bound and it has no upper bound. Bucket widths are arbitrary
    (native buckets are unequal); the label states the actual range.
    """
    out: List[Dict[str, Any]] = []
    cum_prev = 0
    for index, bound in enumerate(le):
        cumulative = int(counts[index]) if index < len(counts) else cum_prev
        frequency = max(0, cumulative - cum_prev)
        is_inf = isinstance(bound, str) or bound is None
        hi = None if is_inf else float(bound)
        prev = le[index - 1] if index > 0 else None
        lo = (None if (prev is None or isinstance(prev, str)) else float(prev))
        if index == 0:
            lo = 0.0
        out.append({
            "lo": lo,
            "hi": hi,
            "label": _bucket_label(index, lo, hi),
            "count": frequency,
            "cumulative": cumulative,
        })
        cum_prev = cumulative
    return out


def _histogram_entry(
    level: Dict[str, Any], key: str, label: str, run_label: str, is_aggregate: bool,
    available: bool, reason: Optional[str], source: Any, unit: str,
    boundary_note: Any, total: Optional[int], window_s: Any,
    le: Optional[List[Any]], counts: Optional[List[int]], min_requests: int,
) -> Dict[str, Any]:
    entry = {
        "level_index": level.get("index", 0),
        "offered_load": float(level["offered_load"]),
        "metric": key,
        "metric_label": label,
        "run_label": run_label,
        "is_aggregate": is_aggregate,
        "available": available,
        "reason": reason,
        "source": source,
        "unit": unit,
        "boundary_note": boundary_note,
        "total": total,
        "window_s": window_s,
        "low_sample": bool(total is not None and total < min_requests),
        "buckets": [],
    }
    if available and le is not None and counts is not None:
        bucket_list = _bucket_ranges(le, counts)
        total_obs = total or 0
        for bucket in bucket_list:
            bucket["pct"] = round(100.0 * bucket["count"] / total_obs, 2) if total_obs else 0.0
        entry["buckets"] = bucket_list
    return entry


def build_histograms(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pure, deterministic selector entries for the native histogram section.

    Each entry is one (load level, metric, run | aggregate) with non-overlapping
    bucket counts. An unavailable selection carries an explicit reason. Old M9C
    results without histogram data yield ``[]``.
    """
    levels = _levels(result)
    min_requests = int((result.get("sweep", {}).get("measurement") or {}).get("min_requests") or 1)
    keys: List[str] = []
    for lvl in levels:
        for run in lvl.get("runs", []):
            for key in ((run.get("histograms") or {}).get("families") or {}):
                if key not in keys:
                    keys.append(key)
    entries: List[Dict[str, Any]] = []
    for lvl in levels:
        runs = lvl.get("runs", [])
        for key in keys:
            available_runs: List[tuple] = []
            label = key
            for index, run in enumerate(runs):
                family = ((run.get("histograms") or {}).get("families") or {}).get(key)
                if family is None:
                    continue
                label = family.get("label") or key
                if family.get("available"):
                    entries.append(_histogram_entry(
                        lvl, key, label, f"run {index + 1}", False, True, None,
                        family.get("source"), family.get("unit") or "seconds",
                        family.get("boundary_note"), int(family.get("total") or 0),
                        family.get("window_s"), family.get("le"), family.get("counts"),
                        min_requests,
                    ))
                    available_runs.append((index, family))
                else:
                    entries.append(_histogram_entry(
                        lvl, key, label, f"run {index + 1}", False, False,
                        family.get("reason") or "unavailable",
                        family.get("source"), family.get("unit") or "seconds",
                        family.get("boundary_note"), None, None, None, None, min_requests,
                    ))
            if len(runs) < 2:
                continue
            run_label = "aggregate"
            if available_runs and len(available_runs) == len(runs):
                le_sets = [tuple(fam.get("le") or ()) for _, fam in available_runs]
                if all(le == le_sets[0] for le in le_sets):
                    counts = [0] * len(le_sets[0])
                    for _, fam in available_runs:
                        for i, c in enumerate(fam.get("counts") or ()):
                            counts[i] += int(c)
                    total = sum(int(fam.get("total") or 0) for _, fam in available_runs)
                    first = available_runs[0][1]
                    entries.append(_histogram_entry(
                        lvl, key, label, run_label, True, True, None,
                        first.get("source"), first.get("unit") or "seconds",
                        first.get("boundary_note"), total, first.get("window_s"),
                        list(le_sets[0]), counts, min_requests,
                    ))
                else:
                    entries.append(_histogram_entry(
                        lvl, key, label, run_label, True, False,
                        "runs have different bucket bounds; aggregation is invalid",
                        None, "seconds", None, None, None, None, None, min_requests,
                    ))
            else:
                entries.append(_histogram_entry(
                    lvl, key, label, run_label, True, False,
                    "not all runs have an available histogram",
                    None, "seconds", None, None, None, None, None, min_requests,
                ))
    return entries


# ------------------------------------------------------------------- HTML
_CSS = """
:root{--bg:#0f1115;--fg:#e6e6e6;--muted:#9aa4b2;--grid:#262b33;--card:#171a21;
--accent:#4f9cff;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--grid)}
h1{margin:0 0 4px;font-size:18px}h2{font-size:14px;margin:22px 0 8px}
.wrap{padding:16px 24px 48px;max-width:1200px}
.meta{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:10px 12px 6px}
.card h3{margin:0 0 6px;font-size:12px;font-weight:600}
canvas{width:100%;height:240px;display:block}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 0;font-size:11px;color:var(--muted)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
.notes,.unavail{font-size:11px;color:var(--muted);margin:6px 0 2px}
.unavail{color:#e0b050}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:600}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px}
.badge.ok{background:#173d24;color:#7fe0a0}.badge.inc{background:#3d2c12;color:#e0b050}
#tip{position:fixed;pointer-events:none;background:#000c;border:1px solid var(--grid);
border-radius:6px;padding:6px 8px;font-size:11px;display:none;z-index:9;white-space:pre}
.disclaimer{color:var(--muted);font-size:11px;margin-top:6px}
.controls{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}
.controls label{color:var(--muted);font-size:11px}
select{background:#0f1115;color:var(--fg);border:1px solid var(--grid);
border-radius:5px;padding:3px 6px;font-size:12px;margin-left:5px}
"""

_JS = r"""
const DATA = JSON.parse(document.getElementById('sweep-data').textContent);
const COLORS = ['#4f9cff','#7fe0a0','#e0b050','#ff7ab6','#9b8cff','#5ad0d0','#ff9d5c'];
const tip = document.getElementById('tip');

function niceStep(range){ if(range<=0) return 1; const p=Math.pow(10,Math.floor(Math.log10(range)));
  const n=range/p; return (n<=2?1:n<=5?2:5)*p; }

function drawPanel(panel, canvas){
  const dpr = window.devicePixelRatio||1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  canvas.width = cssW*dpr; canvas.height = cssH*dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,cssW,cssH);
  const padL=48,padR=12,padT=10,padB=26;
  const allPts = panel.series.flatMap(s=>s.points).filter(p=>p[1]!==null);
  if(allPts.length===0){ ctx.fillStyle='#9aa4b2'; ctx.font='12px sans-serif';
    ctx.fillText('no data', padL, cssH/2); return; }
  const xs=allPts.map(p=>p[0]), ys=allPts.map(p=>p[1]);
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  if(x0===x1){x0-=1;x1+=1;}
  // Always anchor the baseline at 0 for non-negative data; never push y0 below 0
  // (that made all-zero series render as tall bars). Flat series get a range.
  y0=Math.min(0,y0);
  if(y1===y0){ y1 = (y0===0) ? 1 : y0 + Math.abs(y0)*0.1; }
  const yStep=niceStep(y1-y0); y1=Math.ceil(y1/yStep)*yStep;
  const px=v=>padL+(v-x0)/(x1-x0)*(cssW-padL-padR);
  const py=v=>padT+(cssH-padT-padB)*(1-(v-y0)/(y1-y0));
  ctx.strokeStyle='#262b33'; ctx.fillStyle='#9aa4b2'; ctx.font='10px sans-serif';
  for(let v=y0;v<=y1+1e-9;v+=yStep){ const yy=py(v);
    ctx.beginPath();ctx.moveTo(padL,yy);ctx.lineTo(cssW-padR,yy);ctx.stroke();
    ctx.fillText(v.toFixed(yStep<1?2:0), 6, yy+3); }
  const xTicks = panel.series.length? panel.series[0].points.map(p=>p[0]):[];
  xTicks.forEach(v=>{ const xx=px(v);
    ctx.beginPath();ctx.moveTo(xx,padT);ctx.lineTo(xx,cssH-padB);ctx.stroke();
    ctx.fillText(String(v), xx-6, cssH-8); });
  panel.series.forEach((s,si)=>{ const c=COLORS[si%COLORS.length]; ctx.strokeStyle=c;
    ctx.fillStyle=c; ctx.lineWidth=2;
    if(panel.type==='bar'){
      const bw=Math.max(4,(cssW-padL-padR)/Math.max(1,xTicks.length*panel.series.length)-2);
      s.points.forEach((p,pi)=>{ if(p[1]===null) return;
        const xx=px(p[0]) - bw*panel.series.length/2 + si*(bw+2);
        ctx.fillRect(xx, py(p[1]), bw, py(y0)-py(p[1])); });
    } else {
      let started=false; ctx.beginPath();
      s.points.forEach(p=>{ if(p[1]===null){started=false;return;}
        const xx=px(p[0]), yy=py(p[1]);
        if(!started){ctx.moveTo(xx,yy);started=true;} else ctx.lineTo(xx,yy); });
      ctx.stroke();
      s.points.forEach((p,pi)=>{ if(p[1]===null) return;
        const low = s.low_sample && s.low_sample[pi];
        ctx.beginPath();ctx.arc(px(p[0]),py(p[1]), low?4.5:3, 0, 6.283);
        if(low){ ctx.save(); ctx.fillStyle='#0f1115'; ctx.fill();
          ctx.lineWidth=2; ctx.strokeStyle=c; ctx.stroke(); ctx.restore(); }
        else { ctx.fill(); } });
    }
  });
  canvas._geom={px,py,panel,x0,x1,y0,y1};
}

function renderLegend(panel, el){
  el.innerHTML = panel.series.map((s,i)=>
    `<span><i style="background:${COLORS[i%COLORS.length]}"></i>${s.name} (${s.unit})</span>`
  ).join('');
}

function nearest(panel, geom, mx, my){
  let best=null;
  panel.series.forEach((s,si)=>s.points.forEach((p,pi)=>{
    if(p[1]===null) return;
    const dx=geom.px(p[0])-mx, dy=geom.py(p[1])-my, d=dx*dx+dy*dy;
    if(!best||d<best.d) best={d,s,si,p,pi};}));
  return best;
}

function render(){
  document.querySelectorAll('canvas[data-panel]').forEach(canvas=>{
    const panel = DATA.panels.find(p=>p.id===canvas.dataset.panel);
    drawPanel(panel, canvas);
    canvas.onmousemove = e=>{
      const g=canvas._geom; if(!g) return;
      const r=canvas.getBoundingClientRect();
      const b=nearest(g.panel,g,e.clientX-r.left,e.clientY-r.top);
      if(!b||b.d>28*28){tip.style.display='none';return;}
      tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
      const sc=(b.s.samples||[])[b.pi];
      const low=(b.s.low_sample||[])[b.pi];
      tip.textContent=`${b.s.name} (${b.s.unit})\noffered=${b.p[0]}  value=${b.p[1].toFixed(4)}`
        + ((sc===null||sc===undefined)?'' : `\nsamples=${sc}`)
        + (low? '\nLOW SAMPLE (inadequate evidence)' : '');
    };
    canvas.onmouseleave=()=>{tip.style.display='none';};
  });
}
function clearCanvas(cv){
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth, h=cv.clientHeight;
  cv.width=w*dpr; cv.height=h*dpr;
  const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
  cv._hgeo=null;
}

function drawHistogram(canvas, e){
  const dpr=window.devicePixelRatio||1, W=canvas.clientWidth, H=canvas.clientHeight;
  canvas.width=W*dpr; canvas.height=H*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr); ctx.clearRect(0,0,W,H);
  const padL=44,padR=12,padT=12,padB=56;
  const buckets=e.buckets||[]; if(!buckets.length){ canvas._hgeo=null; return; }
  const maxY=Math.max(1, ...buckets.map(b=>b.count));
  const yStep=niceStep(maxY); const yMax=Math.ceil(maxY/yStep)*yStep;
  const py=v=>padT+(H-padT-padB)*(1-(v/(yMax||1)));
  ctx.strokeStyle='#262b33'; ctx.fillStyle='#9aa4b2'; ctx.font='10px sans-serif';
  for(let v=0; v<=yMax+1e-9; v+=yStep){ const yy=py(v);
    ctx.beginPath();ctx.moveTo(padL,yy);ctx.lineTo(W-padR,yy);ctx.stroke();
    ctx.fillText(String(v),6,yy+3); }
  const slot=(W-padL-padR)/buckets.length; const geo=[];
  const labelStep=Math.max(1, Math.ceil(buckets.length/12));
  const maxCount=Math.max(...buckets.map(b=>b.count));
  buckets.forEach((b,i)=>{
    const x=padL+i*slot+slot*0.12, bw=slot*0.76;
    ctx.fillStyle=COLORS[0]; ctx.fillRect(x, py(b.count), bw, py(0)-py(b.count));
    geo.push({x, x2:x+bw, b});
    if(i%labelStep!==0 && b.count!==maxCount) return;  // thin crowded axis labels
    ctx.save(); ctx.translate(x+bw/2, H-padB+6); ctx.rotate(-Math.PI/6);
    ctx.fillStyle='#9aa4b2'; ctx.textAlign='right'; ctx.font='9px sans-serif';
    ctx.fillText(b.label, 0, 0); ctx.restore();
  });
  canvas._hgeo=geo;
}

function histHover(ev){
  const cv=ev.target, geo=cv._hgeo; if(!geo){tip.style.display='none';return;}
  const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left;
  const hit=geo.find(g=>mx>=g.x&&mx<=g.x2);
  if(!hit){tip.style.display='none';return;}
  tip.style.display='block'; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY+12)+'px';
  tip.textContent=`bucket ${hit.b.label}\ncount=${hit.b.count}  (${hit.b.pct}%)\ncumulative=${hit.b.cumulative}`;
}

function renderHistogram(){
  const cv=document.getElementById('hist-canvas'); if(!cv) return;
  const lvl=document.getElementById('hist-level').value;
  const met=document.getElementById('hist-metric').value;
  const run=document.getElementById('hist-run').value;
  const info=document.getElementById('hist-info'), warn=document.getElementById('hist-warn');
  const list=DATA.histograms||[];
  const e=list.find(x=>parseFloat(x.offered_load)===parseFloat(lvl)
    && x.metric_label===met && x.run_label===run);
  cv.onmousemove=histHover; cv.onmouseleave=()=>{tip.style.display='none';};
  if(!e){ clearCanvas(cv); info.textContent='';
    warn.textContent='Native histogram distribution unavailable for this result.'; return; }
  if(!e.available){ clearCanvas(cv); info.textContent='';
    warn.textContent='Unavailable: '+(e.reason||'unknown reason')+' (no distribution fabricated).'; return; }
  warn.textContent = e.low_sample ? ('Low sample count (total='+e.total+') — shown for reference only, not statistically reliable.') : '';
  info.textContent='total observations='+e.total+'  ·  window='+e.window_s+'s  ·  unit='+e.unit
    +'  ·  source='+(e.source||'')+(e.boundary_note?('  ·  note='+e.boundary_note):'');
  drawHistogram(cv, e);
}

function bindHistogramSelectors(){
  ['hist-level','hist-metric','hist-run'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.onchange=renderHistogram; });
}

function initHistogramSelection(){
  const list=DATA.histograms||[]; if(!list.length) return;
  const avail=list.find(e=>e.available); if(!avail) return;
  const levelSel=document.getElementById('hist-level');
  const metricSel=document.getElementById('hist-metric');
  const runSel=document.getElementById('hist-run');
  for(const o of levelSel.options){
    if(parseFloat(o.value)===parseFloat(avail.offered_load)) levelSel.value=o.value; }
  if(metricSel) metricSel.value=avail.metric_label;
  if(runSel) runSel.value=avail.run_label;
}

window.addEventListener('resize', function(){ render(); renderHistogram(); });
document.addEventListener('DOMContentLoaded', function(){
  render(); bindHistogramSelectors(); initHistogramSelection(); renderHistogram(); });
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _summary_html(result: Dict[str, Any]) -> str:
    s = result.get("summary") or {}
    engine = result.get("engine") or {}
    evidence = s.get("evidence") or {}
    experiment = s.get("experiment") or {}
    observations = s.get("observations") or {}
    saturation = s.get("saturation") or {}
    verdict = s.get("verdict", "UNKNOWN")
    badge = "inc" if verdict == "INCONCLUSIVE" else "ok"
    rows = [
        ("engine", f"{engine.get('name')} {(engine.get('version') or {}).get('version') or ''}".strip()),
        ("model", result.get("model") or "n/a"),
        ("load model", (result.get("sweep") or {}).get("load_model")),
        ("offered range", json.dumps(s.get("observed_load_range"))),
        ("experiment", f"levels ok {experiment.get('levels_ok')}/{experiment.get('levels_total')} "
                       f"repeats={experiment.get('repeats')} completed={experiment.get('completed')}"),
        ("evidence quality", evidence.get("quality")),
        ("censored levels", json.dumps(evidence.get("censored_levels"))),
        ("low-sample levels", json.dumps(evidence.get("low_sample_levels"))),
        ("saturation demonstrated", str(saturation.get("demonstrated"))),
        ("timeouts per level", json.dumps((observations.get("timeout_increase") or {}).get("timeouts_per_level"))),
        ("runtime config", json.dumps({
            "verified": (result.get("runtime_config") or {}).get("verified"),
            "changed": (result.get("runtime_config") or {}).get("changed"),
        })),
    ]
    rows_html = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    caveats = s.get("caveats") or []
    caveat_html = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    plateau = observations.get("throughput_plateau")
    plateau_html = "none observed"
    if plateau:
        plateau_html = (f"flattening between offered load "
                        f"{plateau['observed_between_offered_load']} "
                        f"(marginal gain {plateau['marginal_relative_gain']:+.1%}) — "
                        f"{_esc(plateau['note'])}")
    sat_note = saturation.get("note", "")
    return f"""
    <h2>Sweep summary <span class="badge {badge}">{_esc(verdict)}</span></h2>
    <table>{rows_html}<tr><th>observed throughput flattening</th><td>{plateau_html}</td></tr></table>
    <p class="notes">{_esc(sat_note)}</p>
    <p class="notes">Caveats: <ul>{caveat_html or '<li>none</li>'}</ul></p>
    <p class="disclaimer">{_esc(s.get('comparability_disclaimer',''))}</p>
    """


def _boundaries_html(result: Dict[str, Any]) -> str:
    assumptions = result.get("assumptions") or {}
    runtime = result.get("runtime_config") or {}
    boundaries: Dict[str, Any] = {}
    for level in _levels(result):
        for run in level.get("runs", []):
            if run.get("evidence_windows"):
                boundaries = run["evidence_windows"]
                break
        if boundaries:
            break
    items = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>"
        for k, v in {**boundaries, **assumptions}.items()
    )
    return f"""
    <h2>Measurement boundaries &amp; assumptions</h2>
    <table>{items}</table>
    <p class="notes">Runtime configuration: {_esc(runtime.get('note', ''))}
    (verified={_esc(runtime.get('verified'))}, changed={_esc(runtime.get('changed'))}).</p>
    <p class="unavail">Latency charts show p50/p95/p99 windowed quantiles only. Raw native
    histogram bucket distributions are not preserved by the current collection path, so
    no distribution is reconstructed from percentiles.</p>
    """


def _panels_html(panels: List[Dict[str, Any]]) -> str:
    cards = []
    for panel in panels:
        legend_id = f"legend-{panel['id']}"
        notes = "".join(f'<div class="notes">{_esc(n)}</div>' for n in panel.get("notes", []))
        unavail = ""
        if panel.get("unavailable"):
            items = ", ".join(f"{u['name']} ({u['reason']})" for u in panel["unavailable"])
            unavail = f'<div class="unavail">unavailable: {_esc(items)}</div>'
        cards.append(
            f'<div class="card"><h3>{_esc(panel["title"])}</h3>'
            f'<canvas data-panel="{_esc(panel["id"])}"></canvas>'
            f'<div class="legend" id="{legend_id}"></div>{notes}{unavail}</div>'
        )
    return '<div class="grid">' + "".join(cards) + "</div>"


def _histograms_html(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return """
    <h2>Latency distribution by load level</h2>
    <p class="unavail">Native histogram distribution unavailable for this result.</p>
    """
    preferred = ["TTFT", "Queue latency", "Prefill (measured)", "Decode (measured)",
                 "End-to-end"]
    levels: List[Any] = []
    seen_metrics: List[str] = []
    runs: List[str] = []
    for entry in entries:
        if entry["offered_load"] not in levels:
            levels.append(entry["offered_load"])
        if entry["metric_label"] not in seen_metrics:
            seen_metrics.append(entry["metric_label"])
        if entry["run_label"] not in runs:
            runs.append(entry["run_label"])
    metrics = [m for m in preferred if m in seen_metrics]
    metrics += [m for m in seen_metrics if m not in metrics]

    def options(values: List[Any]) -> str:
        return "".join(
            f'<option value="{_esc(v)}">{_esc(v)}</option>' for v in values
        )

    return f"""
    <h2>Latency distribution by load level</h2>
    <div class="controls">
      <label>Load level <select id="hist-level">{options(levels)}</select></label>
      <label>Latency metric <select id="hist-metric">{options(metrics)}</select></label>
      <label>Run / aggregate <select id="hist-run">{options(runs)}</select></label>
    </div>
    <div class="card"><canvas id="hist-canvas" style="height:300px"></canvas></div>
    <div id="hist-info" class="notes"></div>
    <div id="hist-warn" class="unavail"></div>
    <p class="notes">Bars are non-overlapping bucket counts converted from the
    engine's native cumulative histogram over the measurement window. Bucket
    widths are the engine's own (unequal) and each bar is labelled with its
    actual range. Unavailable selections state the reason; nothing is fabricated
    or substituted with zeros.</p>
    """


def _levels_table_html(result: Dict[str, Any]) -> str:
    """Compact per-level request-outcome / evidence table so an operator can see
    why a throughput curve behaves as it does without reading result.json."""
    levels = _levels(result)
    headers = ("offered", "runs", "started", "completed", "drained", "timeout",
               "error", "incomplete", "wall_s", "req/s", "out tok/s", "TTFT p95",
               "TTFT samples", "latency evidence")
    body = []
    for lvl in levels:
        runs = lvl.get("runs", [])

        def total(key: str) -> int:
            return int(sum(int(r.get("counts", {}).get(key, 0)) for r in runs))

        walls = [r.get("wall_s") for r in runs if isinstance(r.get("wall_s"), (int, float))]
        wall = round(sum(walls) / len(walls), 2) if walls else None
        agg = lvl.get("aggregate", {})
        ttft = agg.get("ttft_p95_s") or {}
        if ttft.get("median") is None:
            lat_evidence = "unavailable (gap)"
        elif ttft.get("adequate"):
            lat_evidence = "adequate"
        else:
            lat_evidence = "low sample"
        cells = [
            lvl["offered_load"], len(runs), total("started"), total("completed"),
            total("drained"), total("timeout"), total("error"), total("incomplete"),
            wall,
            None if (agg.get("throughput_req_s") or {}).get("median") is None
            else round((agg.get("throughput_req_s") or {}).get("median"), 3),
            None if (agg.get("output_tokens_per_s") or {}).get("median") is None
            else round((agg.get("output_tokens_per_s") or {}).get("median"), 2),
            None if ttft.get("median") is None else round(ttft.get("median"), 4),
            ttft.get("sample_count"),
            lat_evidence,
        ]
        body.append("".join(f"<td>{_esc(c)}</td>" for c in cells))
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    rows_html = "".join(f"<tr>{r}</tr>" for r in body)
    return f"""
    <h2>Per-level request outcomes</h2>
    <table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>
    <p class="notes">completed = measurement-period completions (phase-end drain is
    excluded and shown separately). Latency sample counts are the windowed observations
    per level. A fall in request throughput alone does NOT establish GPU saturation or a
    scheduler bottleneck.</p>
    """


def render_report(result: Dict[str, Any]) -> str:
    """Return a self-contained HTML document (no external resources)."""
    panels = build_panels(result)
    histograms = build_histograms(result)
    payload = json.dumps(
        {"result": result, "panels": panels, "histograms": histograms}
    ).replace("</", "<\\/")
    legend_script = "".join(
        f"renderLegend(DATA.panels.find(p=>p.id==={json.dumps(p['id'])}),"
        f"document.getElementById({json.dumps('legend-' + p['id'])}));"
        for p in panels
    )
    generated = _esc(result.get("generated_at", ""))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfEngine load sweep report</title>
<style>{_CSS}</style></head>
<body>
<header><h1>InfEngine continuous-batching load sweep</h1>
<div class="meta">generated {generated} &middot; schema {_esc(result.get('schema',''))} &middot;
self-contained local report (no server, no CDN)</div></header>
<div class="wrap">
{_summary_html(result)}
{_levels_table_html(result)}
{_boundaries_html(result)}
{_panels_html(panels)}
{_histograms_html(histograms)}
</div>
<div id="tip"></div>
<script id="sweep-data" type="application/json">{payload}</script>
<script>{_JS}
{legend_script}
</script>
</body></html>"""


def write_report(result: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_report(result))
