"""Minimal manual operator CLI for the engine-neutral control plane.

Thin client over the EXISTING service API (``/runtime/*``) and the EXISTING
supervisor/approval objects. It adds no control-plane logic of its own: it
inspects, manually sets/restores, verifies, and runs the single-invocation
review-and-approve flow (M7A observe -> human confirmation -> M7B approve) with
the ``SupervisorApplication`` and its registered proposal kept in the SAME
process. Manual HOT changes (``set``/``restore``) go straight through the
existing conditional-PATCH path and do NOT require an M7 proposal.

Engine selection (builtin / sglang / vllm) lives in ``infengine-serve``; this
CLI only talks to a running control-plane service. The vLLM control token, when
needed, is read from the environment by the service, never passed here.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .experiment import ControlPlaneClient, ControlPlaneError
from .supervisor.approval import (
    ApplicationConfig,
    ApprovalRequest,
    SupervisorApplication,
)
from .supervisor.policy import Guardrail, Guardrails, SupervisorConfig
from .supervisor.supervisor import Supervisor
from .telemetry import RuntimeSnapshot

_ACTIONABLE = ("PROPOSE_UP", "PROPOSE_DOWN", "PROPOSE_CHANGE")


def _dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str, sort_keys=True))


def _coerce(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except (TypeError, ValueError):
            pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


# --------------------------------------------------------------------- manual
def cmd_inspect(control: ControlPlaneClient, args) -> int:
    _dump(control.get_config())
    return 0


def cmd_status(control: ControlPlaneClient, args) -> int:
    _dump(control.status())
    return 0


def cmd_capabilities(control: ControlPlaneClient, args) -> int:
    entries = control.capabilities()
    if args.supported_only:
        entries = [e for e in entries if e.get("supported_on_selected_engine")]
    _dump(entries)
    return 0


def _patch(control: ControlPlaneClient, intent: str, raw_value: str,
           if_match: Optional[int]) -> int:
    value = _coerce(raw_value)
    status, payload = control.patch_config({intent: value}, if_match=if_match)
    _dump(payload)
    if status == 200 and payload.get("status") in ("CONVERGED", "NOOP"):
        return 0
    if status in (409, 422):
        return 2
    return 1


def cmd_set(control: ControlPlaneClient, args) -> int:
    return _patch(control, args.intent, args.value, args.if_match)


def cmd_restore(control: ControlPlaneClient, args) -> int:
    return _patch(control, args.intent, args.value, args.if_match)


# --------------------------------------------------------------------- review
def _collect_telemetry(control: ControlPlaneClient, samples: int,
                       interval: float) -> List[Tuple[float, RuntimeSnapshot]]:
    out: List[Tuple[float, RuntimeSnapshot]] = []
    for _ in range(samples):
        status, payload = control._request("GET", "/runtime/telemetry")
        if status == 200:
            out.append((time.monotonic(), RuntimeSnapshot.from_dict(payload)))
        if interval > 0:
            time.sleep(interval)
    return out


def _guardrails(args) -> Guardrails:
    """Build the (configurable) guardrail set from CLI flags.

    Thresholds are configuration, not hidden constants: an operator whose engine
    does not expose a guardrail metric can disable that guardrail explicitly.
    Enabled + ``required_for_scale_up`` guardrails still fail closed when their
    telemetry is unavailable.
    """
    return Guardrails(
        latency=Guardrail(
            name="latency", metric="ttft_p95_s", max_value=args.latency_max,
            enabled=args.latency_guardrail,
            required_for_scale_up=args.required_guardrails,
        ),
        kv_usage=Guardrail(
            name="kv_usage", metric="kv_usage", max_value=args.kv_max,
            enabled=args.kv_guardrail,
            required_for_scale_up=args.required_guardrails,
        ),
    )


def cmd_review(control: ControlPlaneClient, args) -> int:
    config = SupervisorConfig(
        window_s=args.window_s,
        min_samples=args.min_samples,
        cooldown_s=args.cooldown_s,
        guardrails=_guardrails(args),
    )
    record = Supervisor(config, control).observe(
        _collect_telemetry(control, args.samples, args.interval), now=time.monotonic()
    )
    _dump(record.to_dict())

    if record.decision not in _ACTIONABLE:
        print(f"no actionable proposal (decision={record.decision}); nothing to approve")
        return 0

    print(
        f"\nPROPOSAL {record.decision}: {record.current_value} -> "
        f"{record.proposed_value} (intent={record.intent}, "
        f"desired_revision={record.desired_revision})"
    )
    if not args.yes:
        answer = input("Apply this proposal? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("declined; zero mutation")
            return 0

    def _fresh():
        return Supervisor(config, control).observe(
            _collect_telemetry(control, args.samples, args.interval),
            now=time.monotonic(),
        )

    app = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=args.ttl),
        control,
        fresh_evidence=_fresh,
    )
    decision_id = app.register(record)
    if decision_id is None:
        print("proposal was not registered as actionable; zero mutation")
        return 1
    outcome = app.approve(
        ApprovalRequest(
            decision_id,
            expected_desired_revision=record.desired_revision,
            expected_current_value=record.current_value,
            expected_proposed_value=record.proposed_value,
            approver=args.approver,
        )
    )
    _dump(outcome.to_dict())
    return 0 if outcome.status == "CONVERGED" else 1


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "infengine-operator",
        description="Manual operator CLI for the InfEngine control plane.",
    )
    parser.add_argument(
        "--control-url", default=os.environ.get("INFENGINE_CONTROL_URL",
                                                 "http://127.0.0.1:8000"),
        help="base URL of the InfEngine control-plane service",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="show desired/observed state + revision")
    sub.add_parser("status", help="show runtime status")

    p_caps = sub.add_parser("capabilities", help="list capabilities")
    p_caps.add_argument("--supported-only", action="store_true",
                        help="only capabilities supported on the selected engine")

    for name, help_text in (
        ("set", "manually apply a HOT intent change (no M7 proposal required)"),
        ("restore", "manually restore a HOT intent value"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("intent")
        p.add_argument("value")
        p.add_argument("--if-match", type=int, default=None,
                       help="expected desired_revision (conditional PATCH)")

    p_rev = sub.add_parser(
        "review",
        help="single-invocation M7A observe -> confirm -> M7B approve",
    )
    p_rev.add_argument("--samples", type=int, default=5)
    p_rev.add_argument("--interval", type=float, default=0.5)
    p_rev.add_argument("--window-s", type=float, default=30.0)
    p_rev.add_argument("--min-samples", type=int, default=3)
    p_rev.add_argument("--cooldown-s", type=float, default=60.0)
    p_rev.add_argument("--ttl", type=float, default=300.0,
                       help="proposal TTL for the approval")
    p_rev.add_argument("--approver", default=None)
    p_rev.add_argument("--latency-max", type=float, default=0.5,
                       help="ttft_p95_s guardrail limit")
    p_rev.add_argument("--kv-max", type=float, default=0.9,
                       help="kv_usage guardrail limit")
    p_rev.add_argument("--latency-guardrail", dest="latency_guardrail",
                       action="store_true", default=True)
    p_rev.add_argument("--no-latency-guardrail", dest="latency_guardrail",
                       action="store_false",
                       help="disable the latency guardrail (engine telemetry lacks it)")
    p_rev.add_argument("--kv-guardrail", dest="kv_guardrail",
                       action="store_true", default=True)
    p_rev.add_argument("--no-kv-guardrail", dest="kv_guardrail",
                       action="store_false")
    p_rev.add_argument("--required-guardrails", dest="required_guardrails",
                       action="store_true", default=True)
    p_rev.add_argument("--no-required-guardrails", dest="required_guardrails",
                       action="store_false",
                       help="enabled guardrails no longer gate scale-up on availability")
    p_rev.add_argument("--yes", action="store_true",
                       help="skip the interactive confirmation (non-interactive)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    control = ControlPlaneClient(args.control_url, timeout=args.timeout)
    handlers = {
        "inspect": cmd_inspect,
        "status": cmd_status,
        "capabilities": cmd_capabilities,
        "set": cmd_set,
        "restore": cmd_restore,
        "review": cmd_review,
    }
    try:
        return handlers[args.command](control, args)
    except ControlPlaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
