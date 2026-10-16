"""Narrow, opt-in InfEngine control listener for the patched vLLM API server.

Runs the normal vLLM OpenAI server in this process and, alongside it, a SEPARATE
FastAPI app on its own port that forwards a small set of runtime-control
operations to the SAME ``AsyncLLM``/EngineCore instance that serves inference.
It adds no control-plane logic (no capabilities, validation, desired state,
revision, approval or reconciliation) — those stay in InfEngine.

Enabled only when ``INFENGINE_CONTROL_PORT`` is set. All control endpoints
require the shared token from ``INFENGINE_CONTROL_TOKEN`` (env/secret; never
logged). Intended to be published to host loopback only, e.g.:
    docker run ... -p 8000:8000 -p 127.0.0.1:9001:9001 \
        -e INFENGINE_CONTROL_PORT=9001 -e INFENGINE_CONTROL_TOKEN=... \
        infengine-vllm:latest python3 /workspace/contrib/vllm/infengine_control.py \
        --model <m> --port 8000 ...

Scope: HOT runtime-scheduler config read/apply + identity. DRAIN/reset_prefix_cache
and cache stats are intentionally NOT exposed here.
"""

import asyncio
import os
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai import api_server as srv

_TOKEN = os.environ.get("INFENGINE_CONTROL_TOKEN") or ""
_PORT = int(os.environ.get("INFENGINE_CONTROL_PORT") or "0")
_HOST = os.environ.get("INFENGINE_CONTROL_HOST") or "0.0.0.0"
_HEADER = "x-infengine-token"


def _authorized(req: Request) -> bool:
    # Constant-time compare; token comes only from the environment.
    import hmac

    supplied = req.headers.get(_HEADER) or ""
    return bool(_TOKEN) and hmac.compare_digest(supplied, _TOKEN)


def _control_app(engine_client) -> FastAPI:
    app = FastAPI(title="infengine-vllm-control", docs_url=None, redoc_url=None)

    def _guard(req: Request):
        if not _authorized(req):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return None

    @app.get("/infengine/control/config")
    async def get_config(req: Request):
        denied = _guard(req)
        if denied is not None:
            return denied
        return await engine_client.get_runtime_scheduler_config()

    @app.post("/infengine/control/config")
    async def set_config(req: Request):
        denied = _guard(req)
        if denied is not None:
            return denied
        overrides = await req.json()
        if not isinstance(overrides, dict):
            return JSONResponse(status_code=400, content={"detail": "expected an object"})
        try:
            return await engine_client.set_runtime_scheduler_config(overrides)
        except ValueError as exc:  # engine-side allowlist/validation
            return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/infengine/control/identity")
    async def identity(req: Request):
        denied = _guard(req)
        if denied is not None:
            return denied
        import vllm

        return {
            "engine": "vllm",
            "version": getattr(vllm, "__version__", None),
            "api_pid": os.getpid(),
        }

    return app


# ---- start the control listener in the SAME process/loop as the API server ---
_original_build_and_serve = srv.build_and_serve


async def _build_and_serve_with_control(engine_client, listen_address, sock, args, **kwargs):
    if _PORT:
        control = uvicorn.Server(
            uvicorn.Config(_control_app(engine_client), host=_HOST, port=_PORT,
                           log_level="warning")
        )
        # Tied to this process/loop; exits with the owning API server.
        asyncio.get_running_loop().create_task(control.serve())
    return await _original_build_and_serve(
        engine_client, listen_address, sock, args, **kwargs
    )


srv.build_and_serve = _build_and_serve_with_control


def main() -> int:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.api_server import validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if not _TOKEN:
        print("INFENGINE_CONTROL_TOKEN is required when the control listener is on",
              file=sys.stderr)
        return 2
    if not _PORT:
        print("INFENGINE_CONTROL_PORT is required", file=sys.stderr)
        return 2

    srv.cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM + InfEngine control listener")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    import uvloop

    uvloop.run(srv.run_server(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
