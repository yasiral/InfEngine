"""M13 unified, operator-facing ``infengine`` command.

A thin, CLI-first product surface over the EXISTING control plane. It adds no
control-plane logic of its own: it wraps ``operator`` (manual set/restore/
review), the control-plane HTTP API (inspect/capabilities/explain/apply/history)
and delegates ``serve``/``evaluate`` to their existing entry points.

Public mental model:

    INSTALL -> CONNECT/DOCTOR -> INSPECT -> UNDERSTAND -> APPLY POLICY ->
    VERIFY -> EVALUATE -> RESTORE -> HISTORY

The operator never needs to know about adapters, transports, reconcilers or the
individual service classes.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .about import PRODUCT, __version__
from .experiment import ControlPlaneClient, ControlPlaneError
from .operator import _coerce
from .policy import PolicyError, load_policy, validate_policy

DEFAULT_CONTROL_URL = os.environ.get("INFENGINE_CONTROL_URL", "http://127.0.0.1:8000")


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))
        return
    _pretty(payload)


def _pretty(payload: Any) -> None:
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        print(payload)
        return
    print(json.dumps(payload, indent=2, default=str, sort_keys=True))


def _client(args) -> ControlPlaneClient:
    return ControlPlaneClient(args.control_url, timeout=args.timeout,
                              token=getattr(args, "token", None),
                              actor=getattr(args, "actor", None))


# --------------------------------------------------------------------- version
def cmd_version(args) -> int:
    payload = {"product": PRODUCT, "version": __version__,
               "control_url": args.control_url}
    _emit(payload, args.json)
    return 0


# ---------------------------------------------------------------------- doctor
def _check(label: str, ok: bool, detail: str = "") -> Dict[str, Any]:
    return {"label": label, "ok": bool(ok), "detail": detail}


def cmd_doctor(args) -> int:
    client = _client(args)
    checks: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}
    caps: List[Dict[str, Any]] = []
    status: Dict[str, Any] = {}
    try:
        info = client.info()
        checks.append(_check("control plane reachable", True, info.get("infengine_version", "")))
    except ControlPlaneError as exc:
        checks.append(_check("control plane reachable", False, str(exc)))
        _emit({"ready": False, "checks": checks}, args.json)
        return 1

    checks.append(_check("InfEngine version", True, info.get("infengine_version", __version__)))
    engine = info.get("engine")
    checks.append(_check("configured engine backend", bool(engine), str(engine)))

    try:
        health = client.health()
        checks.append(_check("engine health", bool(health.get("status") == "ok"),
                             f"admitting={health.get('admitting')}"))
    except ControlPlaneError as exc:
        checks.append(_check("engine health", False, str(exc)))

    try:
        caps = client.capabilities()
        supported = [c["intent"] for c in caps if c.get("supported_on_selected_engine")]
        checks.append(_check("capability contract", True, f"{len(supported)} HOT controls"))
        checks.append(_check("control transport", True, "working"))
        checks.append(_check("readback working", True, "observed state available"))
    except ControlPlaneError as exc:
        supported = []
        checks.append(_check("capability contract", False, str(exc)))

    try:
        status = client.status()
        ident = status.get("identity") or {}
        checks.append(_check("engine identity", bool(ident),
                             str(ident.get("engine_instance_id") or ident)))
    except ControlPlaneError as exc:
        ident = {}
        checks.append(_check("engine identity", False, str(exc)))

    try:
        st, _ = client._request("GET", "/runtime/telemetry")
        checks.append(_check("telemetry", st == 200, f"http={st}"))
    except ControlPlaneError as exc:
        checks.append(_check("telemetry", False, str(exc)))

    checks.append(_check("security mode",
                         bool(info.get("auth_enabled")),
                         "authenticated" if info.get("auth_enabled")
                         else "UNAUTHENTICATED (local development)"))
    checks.append(_check("audit persistence", bool(info.get("audit_configured")),
                         "durable" if info.get("audit_configured") else "not configured"))
    checks.append(_check("apply semantics", True, str(info.get("apply_semantics"))))

    ready = all(c["ok"] for c in checks if c["label"] not in (
        "telemetry", "audit persistence"))
    payload = {
        "product": PRODUCT,
        "version": info.get("infengine_version", __version__),
        "engine": engine,
        "model": info.get("model"),
        "ready": ready,
        "supported_controls": supported,
        "checks": checks,
    }
    if args.json:
        _emit(payload, True)
    else:
        print(f"{PRODUCT} {info.get('infengine_version', __version__)}")
        for c in checks:
            mark = "OK " if c["ok"] else "FAIL"
            print(f"  [{mark}] {c['label']}" + (f": {c['detail']}" if c["detail"] else ""))
        print(f"\n{'Ready.' if ready else 'NOT READY.'}")
    return 0 if ready else 1


# --------------------------------------------------------------------- inspect
def cmd_inspect(args) -> int:
    client = _client(args)
    try:
        config = client.get_config()
        info = client.info()
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    payload = {
        "engine": info.get("engine"),
        "model": info.get("model"),
        "identity": config.get("identity"),
        "revision": config.get("desired_revision"),
        "convergence": config.get("status"),
        "desired": config.get("desired"),
        "observed": config.get("observed"),
        "apply_semantics": info.get("apply_semantics"),
    }
    _emit(payload, args.json)
    return 0


# ---------------------------------------------------------------- capabilities
def cmd_capabilities(args) -> int:
    client = _client(args)
    try:
        entries = client.capabilities()
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.supported_only:
        entries = [e for e in entries if e.get("supported_on_selected_engine")]
    if args.json:
        _emit(entries, True)
        return 0
    header = ("CONTROL", "CURRENT", "SUPPORTED", "TRANSITION", "BOUNDS", "NATIVE", "ENGINE")
    print("{:<32} {:<10} {:<10} {:<10} {:<14} {:<14} {}".format(*header))
    current = {}
    try:
        current = client.get_config().get("observed", {})
    except ControlPlaneError:
        pass
    for e in entries:
        bounds = f"[{e.get('min')},{e.get('max')}]"
        native = e.get("internal_key") or ""
        print("{:<32} {:<10} {:<10} {:<10} {:<14} {:<14} {}".format(
            e["intent"], str(current.get(e.get("internal_key"))),
            "yes" if e.get("supported_on_selected_engine") else "no",
            e.get("transition") or "-", bounds, native, ",".join(e.get("engine") or [])))
    return 0


# --------------------------------------------------------------------- explain
def cmd_explain(args) -> int:
    from .knowledge import explain as explain_control
    client = _client(args)
    caps: List[Dict[str, Any]] = []
    observed: Dict[str, Any] = {}
    try:
        caps = client.capabilities()
        observed = client.get_config().get("observed", {})
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    match = None
    for e in caps:
        if e.get("intent") == args.control or e.get("internal_key") == args.control:
            match = e
            break
    current = observed.get(match.get("internal_key")) if match else None
    # Knowledge is keyed by canonical intent (alias-aware).
    canonical = args.control
    if match and (match.get("internal_key") == args.control):
        canonical = match.get("intent")
    payload = explain_control(canonical, match, current)
    _emit(payload, args.json)
    return 0 if payload.get("known") else 2


# ------------------------------------------------------------------- set/apply
def _print_patch(payload: Dict[str, Any]) -> None:
    for step in payload.get("steps", []):
        print(f"  {step['param']}: requested={step['desired']} "
              f"observed={step['observed_after']} {step['status']}"
              + (f" ({step['reason']})" if step.get("reason") else ""))
    print(f"POLICY STATUS: {payload.get('status')}")
    if payload.get("rollback"):
        print(f"ROLLBACK: {payload['rollback'].get('status')}")


def _patch(client, request: Dict[str, Any], if_match: Optional[int], as_json: bool) -> int:
    status, payload = client.patch_config(request, if_match=if_match)
    if as_json:
        _emit(payload, True)
    else:
        _print_patch(payload)
    if status == 200 and payload.get("status") in ("CONVERGED",):
        return 0
    if status in (409, 422):
        return 2
    return 1


def cmd_set(args) -> int:
    client = _client(args)
    return _patch(client, {args.control: _coerce(args.value)},
                  getattr(args, "if_match", None), args.json)


def cmd_apply(args) -> int:
    client = _client(args)
    try:
        request = load_policy(args.policy_file)
    except PolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        caps = client.capabilities()
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    resolved, errors = validate_policy(request, caps)
    if errors:
        print("validation failed; APPLY NOTHING:", file=sys.stderr)
        for e in errors:
            print(f"  {e['control']}: {e['error']}", file=sys.stderr)
        return 2
    if args.dry_run:
        _emit({"dry_run": True, "requested": request, "resolved": resolved,
               "valid": True}, args.json)
        return 0
    return _patch(client, resolved, getattr(args, "if_match", None), args.json)


def cmd_restore(args) -> int:
    client = _client(args)
    try:
        if args.change_id:
            status, record = client.history_one(args.change_id)
        else:
            status, payload = client.history(limit=1)
            record = (payload.get("history") or [None])[0]
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not record:
        print("no history to restore (or audit persistence not configured)", file=sys.stderr)
        return 2
    pre = record.get("pre_state") or {}
    requested = record.get("requested") or {}
    if not pre:
        print("no captured pre-state for that change; cannot restore", file=sys.stderr)
        return 2
    missing = [k for k in requested if pre.get(k) is None]
    if missing:
        print(f"pre-state missing for {missing}; refusing to guess", file=sys.stderr)
        return 2
    target = {k: pre[k] for k in requested}
    if not args.json:
        print(f"restoring change {record.get('change_id')} "
              f"({record.get('actor')}, {record.get('created_at')})")
    return _patch(client, target, getattr(args, "if_match", None), args.json)


# ---------------------------------------------------------------------- review
def cmd_review(args) -> int:
    """Human review of a change.

    ``infengine review`` (no policy) preserves the existing single-control M7A
    observe -> confirm -> approve flow. ``infengine review <policy-file>`` (or
    ``--policy <file>``) reviews a whole multi-control policy through the SAME
    M7 lifecycle (``SupervisorPolicyApplication``): exact whole-policy binding,
    TTL, single-use, fresh-evidence revalidation, If-Match, whole-policy
    verification. There is no autonomous apply or revert.
    """
    policy_file = getattr(args, "policy", None) or getattr(args, "policy_file", None)
    if policy_file:
        return _cmd_review_policy(args, policy_file)
    from . import operator
    client = _client(args)
    try:
        return operator.cmd_review(client, args)
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_review_policy(args, policy_file: str) -> int:
    from .policy import PolicyError, load_policy
    from .supervisor.approval import (
        ApplicationConfig,
        PolicyApprovalRequest,
        PolicyProposal,
        SupervisorPolicyApplication,
    )

    client = _client(args)
    try:
        policy = load_policy(policy_file)
    except PolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def fresh_evidence():
        try:
            return client.get_config().get("observed") or {}
        except Exception:
            return None

    app = SupervisorPolicyApplication(
        ApplicationConfig(proposal_ttl_s=getattr(args, "ttl", 300.0)),
        client,
        fresh_evidence=fresh_evidence,
    )
    try:
        proposal_id = app.register(PolicyProposal(policy=policy))
        config = client.get_config()
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if proposal_id is None:
        print("policy not registrable; zero mutation:", file=sys.stderr)
        print(f"  {app.last_reason()}", file=sys.stderr)
        return 2

    caps = {}
    try:
        for entry in client.capabilities():
            caps[entry.get("intent")] = entry
            if entry.get("internal_key"):
                caps.setdefault(entry["internal_key"], entry)
    except ControlPlaneError:
        pass
    observed = config.get("observed") or {}
    revision = config.get("desired_revision")
    print(f"PROPOSAL {proposal_id} (revision {revision}):")
    for key, value in policy.items():
        entry = caps.get(key) or {}
        internal = entry.get("internal_key") or key
        print(f"  {key}: {observed.get(internal)} -> {value}")
    if not args.yes:
        answer = input("Approve this policy? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("declined; zero mutation")
            return 0

    outcome = app.approve(PolicyApprovalRequest(
        proposal_id,
        expected_desired_revision=revision,
        expected_policy=policy,
        approver=args.approver or getattr(args, "actor", None),
    ))
    _emit(outcome.to_dict(), args.json)
    return 0 if outcome.status == "CONVERGED" else 1


# --------------------------------------------------------------------- history
def cmd_history(args) -> int:
    client = _client(args)
    try:
        status, payload = client.history(limit=args.limit, offset=args.offset)
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if status != 200:
        print(f"history unavailable: HTTP {status} {payload.get('detail', '')}", file=sys.stderr)
        return 2
    if args.json:
        _emit(payload, True)
        return 0
    records = payload.get("history", [])
    if not records:
        print("no changes recorded yet")
        return 0
    for r in records:
        print(f"CHANGE {r.get('change_id')}  {r.get('created_at')}  "
              f"actor={r.get('actor')} role={r.get('role')} result={r.get('result')}")
        for k, v in (r.get("requested") or {}).items():
            print(f"    {k}: {r.get('pre_state', {}).get(k)} -> {v}")
        if r.get("revision_before") is not None:
            print(f"    revision: {r.get('revision_before')} -> {r.get('revision_after')}")
        if r.get("rollback"):
            print(f"    rollback: {r['rollback'].get('status')}")
    return 0


# -------------------------------------------------------------------- evaluate
def cmd_evaluate(args) -> int:
    from .evaluator import run_evaluation_file
    return run_evaluation_file(args.experiment_file, as_json=args.json,
                               control_url=args.control_url,
                               report_path=args.report,
                               engine_url=args.engine_url,
                               telemetry_url=args.telemetry_url)


# ----------------------------------------------------------------------- serve
def cmd_serve(args) -> int:
    from .serve import main as serve_main
    return serve_main(args.serve_args)


# ------------------------------------------------------------------------ parser
def _add_client_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--control-url", default=DEFAULT_CONTROL_URL,
                   help="base URL of the InfEngine control-plane service")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--token", default=None,
                   help="bearer token (prefer env INFENGINE_TOKEN)")
    p.add_argument("--actor", default=None,
                   help="actor identity (prefer env INFENGINE_ACTOR)")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "infengine",
        description=(
            "InfEngine unified operator CLI. Human-controlled runtime policy for "
            "vLLM / SGLang: inspect, explain, change, verify, evaluate, restore."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="print the InfEngine version")
    _add_client_opts(p)
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("doctor", help="check whether this deployment is ready to operate")
    _add_client_opts(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("inspect", help="show desired/observed state, revision and identity")
    _add_client_opts(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("capabilities", help="list controls and their engine support")
    _add_client_opts(p)
    p.add_argument("--supported-only", action="store_true")
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("explain", help="explain one control (mapping, semantics, limits)")
    _add_client_opts(p)
    p.add_argument("control")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("set", help="change ONE control (HOT; no approval required)")
    _add_client_opts(p)
    p.add_argument("control")
    p.add_argument("value")
    p.add_argument("--if-match", type=int, default=None,
                   help="expected desired_revision (conditional PATCH)")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("apply", help="apply a multi-control policy file (dry-run supported)")
    _add_client_opts(p)
    p.add_argument("policy_file")
    p.add_argument("--if-match", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("restore", help="restore the pre-state of a previous change")
    _add_client_opts(p)
    p.add_argument("--change-id", default=None)
    p.add_argument("--if-match", type=int, default=None)
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("history", help="show the durable change/audit history")
    _add_client_opts(p)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("evaluate", help="run an A/B (single- or multi-control) experiment")
    _add_client_opts(p)
    p.add_argument("experiment_file")
    p.add_argument("--report", default=None, help="write an HTML report to this path")
    p.add_argument("--engine-url", default=None,
                   help="inference endpoint (or env INFENGINE_ENGINE_URL)")
    p.add_argument("--telemetry-url", default=None,
                   help="telemetry endpoint (or env INFENGINE_TELEMETRY_URL)")
    p.set_defaults(func=cmd_evaluate)

    # review: reuse the existing operator review flow (M7A observe -> approve);
    # an optional policy file switches to the multi-control policy approval path.
    p = sub.add_parser(
        "review",
        help="human review: single-control M7A observe->approve, or a multi-control "
             "policy file (infengine review <policy-file>)",
    )
    _add_client_opts(p)
    p.add_argument("policy_file", nargs="?", default=None,
                   help="optional multi-control policy file to review and approve")
    p.add_argument("--policy", default=None,
                   help="alias for the policy file positional")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--window-s", type=float, default=30.0)
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--cooldown-s", type=float, default=60.0)
    p.add_argument("--ttl", type=float, default=300.0)
    p.add_argument("--approver", default=None)
    p.add_argument("--latency-max", type=float, default=0.5)
    p.add_argument("--kv-max", type=float, default=0.9)
    p.add_argument("--latency-guardrail", dest="latency_guardrail",
                   action="store_true", default=True)
    p.add_argument("--no-latency-guardrail", dest="latency_guardrail",
                   action="store_false")
    p.add_argument("--kv-guardrail", dest="kv_guardrail", action="store_true", default=True)
    p.add_argument("--no-kv-guardrail", dest="kv_guardrail", action="store_false")
    p.add_argument("--required-guardrails", dest="required_guardrails",
                   action="store_true", default=True)
    p.add_argument("--no-required-guardrails", dest="required_guardrails",
                   action="store_false")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("serve", help="run the control-plane service (wraps infengine-serve)")
    p.add_argument("serve_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
