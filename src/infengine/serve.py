import sys

try:
    from fastapi import FastAPI, Request as FastAPIRequest
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
except ImportError:
    print("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import torch

from .about import PRODUCT, __version__
from .audit import AuditStore
from .backend import build_model_from_hf, build_tokenizer, resolve_device, resolve_dtype
from .engine import LLMEngine, AdmissionError
from .security import Action, AuthConfig, AuthError
from .control import (
    ControlContract,
    InfengineAdapter,
    Reconciler,
    ReconcileStatus,
    StateStore,
    collect_engine_metrics,
)
from .sglang import HttpSglangRuntimeClient, SglangAdapter
from .vllm import HttpVllmRuntimeClient, VllmAdapter
from .telemetry import (
    SglangTelemetryCollector,
    TelemetryCollector,
    VllmTelemetryCollector,
)

_BACKENDS = ("builtin", "sglang", "vllm")


def create_app(get_engine=None, state_store=None, model="Qwen/Qwen2.5-0.5B-Instruct",
               bits=0, engine_kwargs=None, drain_timeout=30.0,
               runtime=None, runtime_url=None, runtime_client=None,
               vllm_control_url=None, vllm_control_token=None,
               telemetry=None, telemetry_url=None, telemetry_engine=None,
               telemetry_interval=5.0, audit_store=None, auth=None):
    """FastAPI app.

    ``get_engine`` injects a built-in engine (tests); otherwise the default
    provider lazily loads ``model`` from HuggingFace.

    M4B-C: the same service also drives a remote SGLang runtime. ``runtime``
    selects the backend: ``"builtin"`` (default) or ``"sglang"``. The sglang
    backend is selected explicitly (``runtime="sglang"`` or a supplied
    ``runtime_client``) and is never auto-derived from the ``SGLANG_BASE_URL``
    environment variable. ``runtime_url`` / ``runtime_client`` (or, for the
    sglang backend, the ``SGLANG_BASE_URL`` environment variable) point at the
    remote SGLang admin surface.
    """
    # Single-flight reconciliation: exactly one reconcile transaction runs per
    # control-plane instance at a time. The single-worker executor both keeps
    # blocking reconciliation off the event loop and serializes concurrent
    # PATCHes. Normal inference is NOT routed through this executor.
    reconcile_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="infengine-reconcile"
    )

    @asynccontextmanager
    async def lifespan(_app):
        if telemetry is not None:
            telemetry.start()
        try:
            yield
        finally:
            if telemetry is not None:
                telemetry.stop()
            reconcile_executor.shutdown(wait=False)

    # Backend selection (Control Contract V2 targets the selected engine). The
    # sglang backend is selected explicitly via ``runtime="sglang"`` or by
    # supplying a ``runtime_client``; it is NEVER auto-derived from the
    # SGLANG_BASE_URL environment variable, so a globally-set SGLANG_BASE_URL
    # (e.g. present for integration tests) cannot hijack the default builtin
    # path. SGLANG_BASE_URL is only read as the URL for the sglang backend once
    # that backend has been selected.
    backend = runtime or ("sglang" if runtime_client is not None else "builtin")
    if backend not in _BACKENDS:
        # No silent fallback: an unknown backend is a hard error.
        raise ValueError(
            f"unknown runtime backend {backend!r}; expected one of {_BACKENDS}"
        )

    # Out-of-band telemetry (M5B): independent of the control adapters and the
    # Reconciler. A collector may be passed directly, or built from an explicit
    # telemetry_url (+ telemetry_engine). It only GETs the engine's native
    # telemetry endpoints; inference traffic is never proxied through it.
    if telemetry is None and telemetry_url:
        engine = telemetry_engine or ("sglang" if backend == "sglang" else "vllm")
        if engine == "vllm":
            telemetry = VllmTelemetryCollector(telemetry_url,
                                               poll_interval=telemetry_interval)
        elif engine == "sglang":
            telemetry = SglangTelemetryCollector(telemetry_url,
                                                 poll_interval=telemetry_interval)

    app = FastAPI(title="infengine", lifespan=lifespan)
    engine_state = {"model": model, "bits": bits, "engine": None}
    control = {"adapter": None, "reconciler": None, "contract": None, "lock": threading.Lock()}
    kwargs = dict(engine_kwargs or {})

    def default_engine():
        if engine_state["engine"] is None:
            device = resolve_device()
            dtype = resolve_dtype(device)
            m, cfg = build_model_from_hf(engine_state["model"], device=device,
                                         dtype=dtype, bits=engine_state["bits"])
            tokenizer = build_tokenizer(engine_state["model"])
            engine_state["engine"] = LLMEngine(m, tokenizer, device, dtype, **kwargs)
        return engine_state["engine"]

    provider = get_engine if get_engine is not None else default_engine

    # M13 Phase 7/8: authentication + durable audit are independent of the
    # control adapters. ``auth=None`` defers to the environment; ``audit=None``
    # disables history persistence (in-memory single-process only).
    security = auth if auth is not None else AuthConfig.from_env()
    audit = audit_store

    def _guard(req, action):
        """Authenticate + authorize a request. Returns (identity, error_response)."""
        try:
            identity = security.authenticate(dict(req.headers))
            security.require(identity, action)
            return identity, None
        except AuthError as exc:
            code = 401 if "token" in str(exc) else 403
            return None, JSONResponse(status_code=code, content={"detail": str(exc)})

    def _engine_name():
        return backend

    def _record(event):
        if audit is None:
            return None
        try:
            return audit.record(event)
        except Exception:
            return None

    def ensure_control():
        if control["adapter"] is not None:
            return control["adapter"], control["reconciler"], control["contract"]
        with control["lock"]:
            if control["adapter"] is not None:
                return control["adapter"], control["reconciler"], control["contract"]
            if backend == "sglang":
                client = runtime_client or HttpSglangRuntimeClient(
                    runtime_url or os.environ["SGLANG_BASE_URL"]
                )
                adapter = SglangAdapter(client)
                reconciler = Reconciler(adapter, state_store, drain_timeout=drain_timeout)
                contract = ControlContract("sglang")
            elif backend == "vllm":
                client = runtime_client or HttpVllmRuntimeClient(
                    vllm_control_url or os.environ["INFENGINE_VLLM_CONTROL_URL"],
                    token=vllm_control_token,
                )
                adapter = VllmAdapter(client)
                reconciler = Reconciler(adapter, state_store, drain_timeout=drain_timeout)
                contract = ControlContract("vllm")
            else:  # builtin
                eng = provider()
                adapter = InfengineAdapter(eng)
                reconciler = Reconciler(adapter, state_store, drain_timeout=drain_timeout)
                contract = ControlContract("builtin")
            control["adapter"] = adapter
            control["reconciler"] = reconciler
            control["contract"] = contract
        return control["adapter"], control["reconciler"], control["contract"]

    @app.get("/health")
    async def health():
        adapter, _, _ = ensure_control()
        return {"status": "ok", "admitting": adapter.admitting}

    @app.get("/metrics")
    async def metrics():
        if backend == "sglang":
            # The remote SGLang backend is control-plane only; per-engine
            # inference metrics (tokens/throughput/prefill_frac) are not
            # available here. This is a separate milestone from the control
            # contract.
            return JSONResponse(
                status_code=501,
                content={"detail": "engine metrics not available for the sglang backend"},
            )
        eng = provider()
        m = dict(eng.metrics)
        m.update(collect_engine_metrics(eng))
        if eng.metrics["requests"] > 0:
            m["avg_tokens_per_request"] = round(eng.metrics["tokens"] / eng.metrics["requests"], 2)
            tt = eng.metrics["total_time_s"]
            m["throughput_tok_per_s"] = round(eng.metrics["tokens"] / tt, 2) if tt > 0 else 0.0
            m["prefill_frac"] = round(eng.metrics["prefill_time_s"] / tt, 3) if tt > 0 else 0.0
        return m

    @app.get("/runtime/config")
    async def runtime_config():
        adapter, reconciler, _ = ensure_control()
        return {
            "desired": reconciler.desired(),
            "observed": adapter.observed_state(),
            "identity": adapter.identity(),
            "status": reconciler.last.status.value if reconciler.last else None,
            "desired_revision": reconciler.store.revision,
        }

    @app.get("/runtime/info")
    async def runtime_info():
        """Product/readiness metadata (no secrets). Used by ``infengine doctor``."""
        adapter, reconciler, _ = ensure_control()
        return {
            "product": PRODUCT,
            "infengine_version": __version__,
            "engine": backend,
            "model": engine_state["model"],
            "apply_semantics": reconciler.apply_semantics(),
            "desired_revision": reconciler.store.revision,
            "identity": adapter.identity(),
            "telemetry_configured": telemetry is not None,
            "audit_configured": audit is not None,
            "auth_enabled": security.enabled,
        }

    @app.get("/runtime/history")
    async def runtime_history(req: FastAPIRequest, limit: int = 50, offset: int = 0):
        _, err = _guard(req, Action.READ)
        if err is not None:
            return err
        if audit is None:
            return {"history": [], "persisted": False,
                    "detail": "audit persistence not configured"}
        records = audit.list(limit=limit, offset=offset)
        return {"history": [r.to_dict() for r in records], "persisted": True}

    @app.get("/runtime/history/{change_id}")
    async def runtime_history_one(change_id: str, req: FastAPIRequest):
        _, err = _guard(req, Action.READ)
        if err is not None:
            return err
        if audit is None:
            return JSONResponse(status_code=404,
                                content={"detail": "audit persistence not configured"})
        record = audit.get(change_id)
        if record is None:
            return JSONResponse(status_code=404,
                                content={"detail": f"no change {change_id!r}"})
        return record.to_dict()

    @app.patch("/runtime/config")
    async def runtime_config_patch(req: FastAPIRequest):
        identity, err = _guard(req, Action.MUTATE)
        if err is not None:
            return err
        adapter, reconciler, contract = ensure_control()
        body = await req.json()
        revision_before = reconciler.store.revision
        try:
            observed_before = adapter.observed_state()
        except Exception:
            observed_before = {}

        def _audit(requested, pre_state, status, result_payload=None, rollback=None,
                   detail=None):
            keys = list(requested or {})
            observed = {k: (result_payload or {}).get("observed", {}).get(k) for k in keys} \
                if result_payload else None
            _record({
                "actor": identity.actor,
                "role": identity.role.value,
                "action": "apply",
                "engine": backend,
                "model": engine_state["model"],
                "identity": dict(adapter.identity()),
                "revision_before": revision_before,
                "revision_after": reconciler.store.revision,
                "requested": requested,
                "pre_state": pre_state,
                "observed": observed,
                "result": status,
                "verification": (result_payload or {}).get("steps"),
                "rollback": rollback,
                "detail": detail,
            })

        # Control Contract V2 (operator/API boundary): resolve operator intents
        # (e.g. the common alias runtime_concurrency, or V1 internal keys)
        # against the selected engine and validate semantics BEFORE any engine
        # mutation. Invalid / unsupported / cross-engine requests are rejected
        # here with zero mutation and never reach the Reconciler.
        resolved = contract.resolve(adapter, body)
        if not resolved.valid:
            payload = {
                "status": "FAILED",
                "valid": False,
                "intents": [i.to_dict() for i in resolved.intents],
                "desired_revision": reconciler.store.revision,
            }
            _audit(resolved.resolved, None, "VALIDATION_FAILED", detail=payload)
            return JSONResponse(status_code=422, content=payload)
        # M6B-1 conditional-PATCH guard: an optional If-Match header carries the
        # expected desired_revision. The comparison happens ATOMICALLY with the
        # mutation inside the Reconciler's single-flight critical section
        # (never a read-then-reconcile here); a stale value yields HTTP 409 with
        # zero mutation. Requests without If-Match behave exactly as before
        # (backward compatible).
        expected_revision = None
        if_match = req.headers.get("if-match")
        if if_match is not None:
            try:
                expected_revision = int(if_match)
            except (TypeError, ValueError):
                payload = {
                    "status": "FAILED",
                    "valid": False,
                    "reason": f"invalid If-Match desired_revision {if_match!r}",
                    "desired_revision": reconciler.store.revision,
                }
                _audit(resolved.resolved, None, "VALIDATION_FAILED", detail=payload)
                return JSONResponse(status_code=422, content=payload)
        # M13 Phase 5A: bind the mutation to ONE captured expected revision and
        # preserve the exact pre-change observed state for restoration.
        pre_state = {k: observed_before.get(k) for k in resolved.resolved}
        # Reconciliation (incl. DRAIN's wait loop) is synchronous and may block
        # for up to drain_timeout. The dedicated single-worker executor keeps it
        # off the event loop AND enforces single-flight reconciliation: a second
        # PATCH queues behind the first and only begins after it completes.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            reconcile_executor,
            lambda: reconciler.reconcile(resolved.resolved, expected_revision=expected_revision),
        )
        payload = result.to_dict()
        payload["desired"] = reconciler.desired()
        payload["observed"] = adapter.observed_state()
        payload["desired_revision"] = reconciler.store.revision
        if result.status is ReconcileStatus.CONFLICT:
            _audit(resolved.resolved, pre_state, "CONFLICT", payload, detail=payload)
            return JSONResponse(status_code=409, content=payload)
        if result.status is ReconcileStatus.FAILED:
            _audit(resolved.resolved, pre_state, "FAILED", payload,
                   rollback=result.rollback, detail=payload)
            return JSONResponse(status_code=422, content=payload)
        _audit(resolved.resolved, pre_state, result.status.value, payload,
               rollback=result.rollback, detail=payload)
        return payload

    @app.get("/runtime/capabilities")
    async def runtime_capabilities():
        """Control Contract V2 capability discovery: the UNION of operator
        intents (vLLM + SGLang + the common intent), annotated for the selected
        engine. Each entry carries canonical intent, existing internal key,
        engine(s), supported status on the selected engine, exact semantic,
        scope, bounds, transition, operation-vs-policy, and quiescence metadata.
        Unsupported / other-engine capabilities stay discoverable but are not
        executable."""
        adapter, _, contract = ensure_control()
        return contract.discover(adapter)

    @app.get("/runtime/status")
    async def runtime_status():
        adapter, reconciler, _ = ensure_control()
        return {
            "admitting": adapter.admitting,
            "draining": adapter.drain_active,
            "apply_semantics": reconciler.apply_semantics(),
            "capabilities": [
                {
                    "name": n,
                    "transition": c.transition.value,
                    "dtype": c.dtype,
                    "min": c.min_value,
                    "max": c.max_value,
                    "supported": c.supported,
                }
                for n, c in sorted(adapter.capabilities().items())
            ],
            "identity": adapter.identity(),
            "last_reconcile": reconciler.last.to_dict() if reconciler.last else None,
        }

    @app.get("/runtime/telemetry")
    async def runtime_telemetry():
        """Latest out-of-band telemetry snapshot (M5B).

        Independent of the control adapters and the Reconciler: this returns
        whatever the configured collector has most recently scraped from the
        engine's native telemetry surface(s). 404 when telemetry is not
        configured; 503 before the first successful poll.
        """
        if telemetry is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "telemetry not configured"},
            )
        snap = telemetry.latest()
        if snap is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "no telemetry snapshot collected yet"},
            )
        return snap.to_dict()

    @app.post("/v1/completions")
    async def completions(req: FastAPIRequest):
        adapter, _, _ = ensure_control()
        if not adapter.admitting:
            return JSONResponse(status_code=503,
                                content={"detail": "not admitting; drain in progress"})
        if backend in ("sglang", "vllm"):
            # Remote inference serving is out of scope for the control-contract
            # milestone: these backends are control-plane only (the engine's own
            # OpenAI router serves inference).
            return JSONResponse(
                status_code=501,
                content={"detail": f"completions not available for the {backend} backend"},
            )
        eng = provider()
        body = await req.json()
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 64)
        temperature = body.get("temperature", 0.0)
        top_p = body.get("top_p", 1.0)
        stream = body.get("stream", False)
        tokenizer = eng.tokenizer

        tokens = eng.generate(prompt, max_new_tokens=max_tokens,
                              temperature=temperature, top_p=top_p)

        if stream:
            async def sse():
                try:
                    for tok in tokens:
                        text = tokenizer.decode([tok], skip_special_tokens=True)
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': text, 'token': tok}}]})}\n\n"
                except AdmissionError:
                    yield 'data: {"error": "not admitting"}\n\n'
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        else:
            try:
                out = tokenizer.decode(list(tokens))
            except AdmissionError:
                return JSONResponse(status_code=503,
                                    content={"detail": "not admitting; drain in progress"})
            return {"choices": [{"text": out}]}

    return app


app = create_app()


def validate_runtime(backend, *, runtime_url=None, vllm_control_url=None,
                     vllm_control_token=None):
    """Fail fast, with a clear message, when the selected backend's transport is
    unreachable or incompatible. Never silently falls back to ``builtin``.

    Returns ``{"identity": ..., "capabilities": [...]}`` on success; raises
    ``SystemExit`` (non-zero) otherwise. The token is read from the environment
    only (``INFENGINE_VLLM_CONTROL_TOKEN``); it is never taken from CLI args or
    printed.
    """
    if backend == "builtin":
        return None
    try:
        if backend == "sglang":
            client = HttpSglangRuntimeClient(
                runtime_url or os.environ["SGLANG_BASE_URL"]
            )
            adapter = SglangAdapter(client)
        elif backend == "vllm":
            client = HttpVllmRuntimeClient(
                vllm_control_url or os.environ["INFENGINE_VLLM_CONTROL_URL"],
                token=vllm_control_token,
            )
            adapter = VllmAdapter(client)
        else:
            raise SystemExit(
                f"unknown runtime backend {backend!r}; expected one of {_BACKENDS}"
            )
        identity = adapter.identity()
        capabilities = adapter.capabilities()
    except SystemExit:
        raise
    except Exception as exc:  # unreachable / incompatible / missing config
        raise SystemExit(
            f"runtime backend {backend!r} is unreachable or incompatible: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not capabilities:
        raise SystemExit(f"runtime backend {backend!r} reported no capabilities")
    return {"identity": identity, "capabilities": sorted(capabilities)}


def main():
    import argparse
    parser = argparse.ArgumentParser("infengine-serve")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--bits", type=int, default=0, help="quantize weights to INT* (0 = fp)")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--state-file", type=str, default=None,
                        help="persist desired runtime state to this JSON file")
    parser.add_argument("--drain-timeout", type=float, default=30.0,
                        help="max seconds to wait for engine quiescence on DRAIN")
    parser.add_argument("--telemetry-url", type=str, default=None,
                        help="engine base URL for out-of-band telemetry collection")
    parser.add_argument("--telemetry-engine", type=str, default=None,
                        choices=["vllm", "sglang"],
                        help="engine kind for --telemetry-url (default: sglang when runtime=sglang, else vllm)")
    parser.add_argument("--telemetry-interval", type=float, default=5.0,
                        help="seconds between out-of-band telemetry polls")
    parser.add_argument("--runtime", type=str, default="builtin",
                        choices=list(_BACKENDS),
                        help="runtime backend to control (default: builtin, in-process)")
    parser.add_argument("--runtime-url", type=str, default=None,
                        help="SGLang admin/base URL (required for --runtime sglang; "
                             "else env SGLANG_BASE_URL)")
    parser.add_argument("--vllm-control-url", type=str, default=None,
                        help="patched vLLM control-listener base URL (required for "
                             "--runtime vllm; else env INFENGINE_VLLM_CONTROL_URL). "
                             "Token comes from env INFENGINE_VLLM_CONTROL_TOKEN only.")
    parser.add_argument("--audit-file", type=str, default=None,
                        help="SQLite audit/history database path (enables durable history)")
    parser.add_argument("--insecure", action="store_true",
                        help="explicitly allow mutation endpoints without authentication "
                             "(local development only)")
    args = parser.parse_args()

    if args.runtime != "builtin":
        info = validate_runtime(
            args.runtime,
            runtime_url=args.runtime_url,
            vllm_control_url=args.vllm_control_url,
        )
        print(f"infengine-serve: runtime={args.runtime} "
              f"identity={info['identity']} capabilities={info['capabilities']}")

    auth = AuthConfig.from_env()
    if not auth.enabled and not args.insecure and args.host not in (
        "127.0.0.1", "localhost", "::1"
    ):
        print(
            "infengine-serve: REFUSING to expose unauthenticated mutation endpoints on "
            f"host {args.host!r}. Configure INFENGINE_AUTH_TOKENS/INFENGINE_AUTH_FILE "
            "(bearer roles) or pass --insecure to override (local development only).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not auth.enabled:
        print(
            "infengine-serve: WARNING: authentication is disabled; mutation endpoints "
            "are unauthenticated. Do not expose this service beyond loopback.",
            file=sys.stderr,
        )

    state_store = StateStore(Path(args.state_file)) if args.state_file else None
    audit_store = AuditStore(args.audit_file) if args.audit_file else None
    engine_kwargs = {
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "max_batch": args.max_batch,
        "chunk_size": args.chunk_size,
    }
    uvicorn.run(create_app(model=args.model, bits=args.bits, state_store=state_store,
                           engine_kwargs=engine_kwargs, drain_timeout=args.drain_timeout,
                           runtime=args.runtime, runtime_url=args.runtime_url,
                           vllm_control_url=args.vllm_control_url,
                           telemetry_url=args.telemetry_url,
                           telemetry_engine=args.telemetry_engine,
                           telemetry_interval=args.telemetry_interval,
                           audit_store=audit_store, auth=auth),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()