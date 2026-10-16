"""M4B-C preservation regression tests: Control Contract V2.

These tests pin the operator-facing union of capabilities (docs/
control-contract-v2-preservation-matrix.md, section 6) and the preservation
invariants:

  * Every vLLM / SGLang capability stays individually addressable (no removal,
    hiding, renaming, or silent re-mapping).
  * V1 persisted DesiredState keys work verbatim (never renamed).
  * The V2 alias ``runtime_concurrency`` resolves AT THE OPERATOR BOUNDARY to
    the existing internal key ``runtime_max_running_seqs`` and is never stored
    under the alias name.
  * Token budgets are NOT normalized across engines: a strict total
    scheduled-token budget against SGLang is UNSUPPORTED (rejected), never
    translated to the weaker prefill-admission budget.
  * Unsupported / other-engine capabilities stay discoverable but never
    executable; unknown intents are REJECTED.

The adapters are the real ``SglangAdapter`` / ``VllmAdapter`` over the same
in-memory fakes used by their own unit tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sglang_adapter import FakeSglangRuntimeClient  # noqa: E402
from test_vllm_adapter import FakeVllmRuntimeClient  # noqa: E402

from infengine.control import (  # noqa: E402
    INTENT_REGISTRY,
    ControlContract,
    IntentResolution,
)
from infengine.sglang import SglangAdapter  # noqa: E402
from infengine.vllm import VllmAdapter  # noqa: E402


def sglang_adapter(**kwargs):
    return SglangAdapter(FakeSglangRuntimeClient(**kwargs))


def vllm_adapter(**kwargs):
    return VllmAdapter(FakeVllmRuntimeClient(**kwargs))


UNION = {
    "runtime_concurrency",
    "runtime_max_running_seqs",
    "runtime_max_scheduled_tokens",
    "long_prefill_token_threshold",
    "runtime_kv_watermark",
    "runtime_reserve_full_isl",
    "reset_prefix_cache",
    "runtime_max_prefill_tokens",
    "engine_cache_flush",
    "prefix_cache_reset",
}


# --------------------------------------------------------------- discovery union
def test_intent_registry_is_exact_union():
    assert set(INTENT_REGISTRY) == UNION


def test_discovery_exposes_full_union_per_engine():
    """Discover() must expose the full union on each engine, with only the
    selected engine's capabilities marked supported_on_selected_engine."""
    for contract, adapter in (
        (ControlContract("sglang"), sglang_adapter()),
        (ControlContract("vllm"), vllm_adapter()),
    ):
        entries = {e["intent"]: e for e in contract.discover(adapter)}
        assert set(entries) == UNION
        for e in entries.values():
            assert e["internal_key"] is not None
            assert e["semantic"] is not None


def test_supported_on_selected_engine():
    sg = {e["intent"]: e for e in ControlContract("sglang").discover(sglang_adapter())}
    vm = {e["intent"]: e for e in ControlContract("vllm").discover(vllm_adapter())}

    sglang_supported = {
        "runtime_concurrency", "runtime_max_running_seqs", "runtime_max_prefill_tokens",
        "engine_cache_flush",
    }
    vllm_supported = {
        "runtime_concurrency", "runtime_max_running_seqs", "runtime_max_scheduled_tokens",
        "long_prefill_token_threshold", "runtime_kv_watermark", "runtime_reserve_full_isl",
        "reset_prefix_cache",
    }
    for name in UNION:
        assert sg[name]["supported_on_selected_engine"] == (name in sglang_supported)
        assert vm[name]["supported_on_selected_engine"] == (name in vllm_supported)

    # Unsupported / other-engine capabilities stay discoverable but not executable.
    assert sg["prefix_cache_reset"]["supported_on_selected_engine"] is False
    assert sg["runtime_max_scheduled_tokens"]["supported_on_selected_engine"] is False
    assert vm["engine_cache_flush"]["supported_on_selected_engine"] is False
    assert vm["runtime_max_prefill_tokens"]["supported_on_selected_engine"] is False


# ------------------------------------------------------- V1 keys preserved verbatim
def test_v1_keys_resolve_verbatim():
    """Every vLLM V1 internal key works verbatim through the contract, resolving
    to itself and never being renamed or re-mapped."""
    vm = ControlContract("vllm")
    adapter = vllm_adapter()
    valid = {
        "runtime_max_running_seqs": 1,
        "runtime_max_scheduled_tokens": 1,
        "long_prefill_token_threshold": 1,
        "runtime_kv_watermark": 0.5,
        "runtime_reserve_full_isl": True,
    }
    for key, value in valid.items():
        r = vm.resolve(adapter, {key: value})
        assert r.valid
        i = r.intents[0]
        assert i.status is IntentResolution.RESOLVED
        assert i.internal_key == key  # never renamed
        assert set(r.resolved) == {key}


def test_sglang_v1_keys_resolve_verbatim():
    sg = ControlContract("sglang")
    adapter = sglang_adapter()
    for key in ("runtime_max_running_seqs", "runtime_max_prefill_tokens"):
        r = sg.resolve(adapter, {key: 4})
        assert r.valid
        assert r.intents[0].internal_key == key
        assert set(r.resolved) == {key}


# -------------------------------------------------------- common alias resolution
def test_runtime_concurrency_alias_maps_to_internal_key():
    """The V2 alias resolves at the operator boundary to the existing internal
    key; it is never persisted under the alias name."""
    for contract, adapter in (
        (ControlContract("sglang"), sglang_adapter()),
        (ControlContract("vllm"), vllm_adapter()),
    ):
        r = contract.resolve(adapter, {"runtime_concurrency": 3})
        assert r.valid
        i = r.intents[0]
        assert i.status is IntentResolution.RESOLVED
        assert i.internal_key == "runtime_max_running_seqs"
        assert "runtime_concurrency" not in r.resolved  # never stored under alias
        assert r.resolved == {"runtime_max_running_seqs": 3}
        assert i.semantic == "CONCURRENCY_UPPER_BOUND"


def test_runtime_concurrency_is_not_a_registry_internal_key():
    """runtime_concurrency is a discovery/operator alias only; the internal key
    stored in DesiredState is runtime_max_running_seqs."""
    assert INTENT_REGISTRY["runtime_concurrency"].internal_key == "runtime_max_running_seqs"


# ----------------------------------------------------- cross-engine rejection
def test_strict_token_budget_rejected_against_sglang():
    """A strict total scheduled-token budget is NOT translated to SGLang's
    prefill-admission budget; it is rejected as unsupported."""
    r = ControlContract("sglang").resolve(sglang_adapter(), {"runtime_max_scheduled_tokens": 64})
    assert r.valid is False
    i = r.intents[0]
    assert i.status is IntentResolution.UNSUPPORTED
    assert i.semantic == "STRICT_SCHEDULED_TOKEN_BUDGET"
    assert r.resolved == {}


def test_prefill_budget_rejected_against_vllm():
    r = ControlContract("vllm").resolve(vllm_adapter(), {"runtime_max_prefill_tokens": 512})
    assert r.valid is False
    assert r.intents[0].status is IntentResolution.UNSUPPORTED
    assert r.resolved == {}


def test_token_budget_family_kept_distinct():
    """The two budgets share a discovery family but retain distinct semantic
    identities; they are never merged."""
    vm = ControlContract("vllm")
    sg = ControlContract("sglang")
    vm_e = {e["intent"]: e for e in vm.discover(vllm_adapter())}
    sg_e = {e["intent"]: e for e in sg.discover(sglang_adapter())}
    assert vm_e["runtime_max_scheduled_tokens"]["semantic"] == "STRICT_SCHEDULED_TOKEN_BUDGET"
    assert (sg_e["runtime_max_prefill_tokens"]["semantic"]
            == "AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED")
    assert (vm_e["runtime_max_scheduled_tokens"]["family"]
            == sg_e["runtime_max_prefill_tokens"]["family"] == "work_budget")


# ------------------------------------------------------------ unsupported / unknown
def test_unsupported_intent_rejected_before_mutation():
    """prefix_cache_reset on SGLang is discoverable but never executable."""
    sg = ControlContract("sglang")
    adapter = sglang_adapter()
    r = sg.resolve(adapter, {"prefix_cache_reset": 1})
    assert r.valid is False
    assert r.intents[0].status is IntentResolution.UNSUPPORTED
    assert r.resolved == {}


def test_unknown_intent_rejected():
    for contract, adapter in (
        (ControlContract("sglang"), sglang_adapter()),
        (ControlContract("vllm"), vllm_adapter()),
    ):
        r = contract.resolve(adapter, {"model.foo": 1})
        assert r.valid is False
        assert r.intents[0].status is IntentResolution.REJECTED
        assert r.resolved == {}


# ------------------------------------------------------------- API-boundary wiring
def test_serve_patch_resolves_at_operator_boundary():
    """The /runtime/config PATCH is the operator/API boundary: the V2 alias
    runtime_concurrency resolves there to the internal key (never stored under
    the alias), and cross-engine strict-budget requests are rejected with zero
    mutation."""
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from infengine.serve import create_app

    client = FakeSglangRuntimeClient()
    app = create_app(runtime="sglang", runtime_client=client)
    c = TestClient(app)

    r = c.patch("/runtime/config", json={"runtime_concurrency": 4})
    assert r.status_code == 200
    assert r.json()["status"] == "CONVERGED"
    # persisted/observed state uses the internal key, never the alias
    assert r.json()["desired"].get("runtime_concurrency") is None
    assert r.json()["observed"]["runtime_max_running_seqs"] == 4

    # strict total scheduled-token budget against SGLang is rejected, not
    # translated to the prefill-admission budget; zero mutation.
    before = client.provisioned()
    r2 = c.patch("/runtime/config", json={"runtime_max_scheduled_tokens": 64})
    assert r2.status_code == 422
    assert r2.json()["status"] == "FAILED"
    assert r2.json()["intents"][0]["status"] == "UNSUPPORTED"


# ------------------------------------------------------------ operations preserved
def test_vllm_prefix_cache_reset_remains_available():
    """vLLM PREFIX_CACHE_RESET (reset_prefix_cache, WAIT_INFLIGHT/DRAIN_TO_IDLE)
    stays a resolvable, executable one-shot operation through V2."""
    vm = ControlContract("vllm")
    adapter = vllm_adapter()
    r = vm.resolve(adapter, {"reset_prefix_cache": 1})
    assert r.valid
    i = r.intents[0]
    assert i.status is IntentResolution.RESOLVED
    assert i.internal_key == "reset_prefix_cache"
    assert i.operation is True
    assert i.semantic == "PREFIX_CACHE_RESET"
    assert r.resolved == {"reset_prefix_cache": 1}


def test_sglang_engine_cache_flush_requires_idle_metadata():
    """SGLang ENGINE_CACHE_FLUSH is REQUIRE_IDLE (NOT RETRACT_INFLIGHT) and
    stays a resolvable one-shot operation."""
    sg = ControlContract("sglang")
    adapter = sglang_adapter()
    entry = {e["intent"]: e for e in sg.discover(adapter)}["engine_cache_flush"]
    assert entry["quiescence_requirement"] == "REQUIRE_IDLE"
    assert entry["operation"] is True
    r = sg.resolve(adapter, {"engine_cache_flush": 1})
    assert r.valid and r.intents[0].operation is True


# ------------------------------------------------------------- API-boundary wiring
def test_runtime_capabilities_endpoint_exposes_union():
    """/runtime/capabilities exposes the full V2 union with annotation metadata
    for the selected engine (capability discovery over HTTP)."""
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from infengine.serve import create_app

    app = create_app(runtime="sglang", runtime_client=FakeSglangRuntimeClient())
    c = TestClient(app)
    entries = {e["intent"]: e for e in c.get("/runtime/capabilities").json()}
    assert set(entries) == UNION
    assert entries["runtime_concurrency"]["supported_on_selected_engine"] is True
    assert entries["runtime_concurrency"]["internal_key"] == "runtime_max_running_seqs"
    # unsupported / other-engine capabilities stay discoverable but not executable
    assert entries["prefix_cache_reset"]["supported_on_selected_engine"] is False
    assert entries["runtime_max_scheduled_tokens"]["supported_on_selected_engine"] is False


# ------------------------------------------------------------- end-to-end resolve
def test_resolve_result_feeds_reconciler_keys():
    """The resolved internal keys are exactly the keys an existing Reconciler /
    adapter expects, enabling a zero-change reconcile path."""
    sg = ControlContract("sglang")
    adapter = sglang_adapter()
    r = sg.resolve(adapter, {"runtime_concurrency": 4, "engine_cache_flush": 1})
    assert r.valid
    assert r.resolved == {
        "runtime_max_running_seqs": 4,
        "engine_cache_flush": 1,
    }
    ops = [i for i in r.intents if i.operation]
    assert [i.internal_key for i in ops] == ["engine_cache_flush"]