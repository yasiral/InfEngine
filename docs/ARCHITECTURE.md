# Control-plane architecture

InfEngine is an external runtime-control and evaluation layer. Its built-in engine supports local development and deterministic tests; vLLM and SGLang continue serving inference through their ordinary endpoints.

## One mutation path

`infengine` is a CLI over the HTTP control plane and existing evaluation/review implementations. Operator intents resolve through `control/contract.py` into engine-specific controls. The reconciler validates a whole plan, reads native pre-state, checks the expected desired revision under its mutation lock, persists accepted intent and applies supported transitions through the adapter.

Desired state and observed state remain distinct. A native getter must confirm the requested value; returning an adapter's cached request is insufficient. Accepted intent can outlive a blocked transition or remain different from compensated physical state. The service records outcomes in the audit store; an application response includes status and per-step verification.

## Engine boundaries

- **vLLM:** an authenticated control listener shares the API-server process/event loop and uses the existing AsyncLLM/EngineCoreClient utility connection. EngineCore may be another process. It changes live scheduler operating fields inside provisioned limits, not model memory allocation, buffers or graph captures.
- **SGLang:** the adapter uses existing internal-state transport through TokenizerManager and scheduler RPC. The small patch adds a live prefill-admission field and exposes native readback. The proven concurrency scope is a local/per-rank scheduler at PP=1/DP=1.
- **Inference:** clients still call the serving engine directly. InfEngine is not an inference proxy, model deployment system or fleet scheduler.

The capability contract preserves exact meanings, types, bounds, transition classes and support on the selected transport. vLLM's total scheduled-token step budget is never translated into SGLang's weaker prefill-input budget with its first-request exception.

## Failure and restoration

vLLM advertises `CONTROL_PLANE_TRANSACTIONAL`, not `ENGINE_ATOMIC`: its native hook performs some provisioned checks in the mutating phase. SGLang advertises `COORDINATED_NON_ATOMIC` and applies fields sequentially. The reconciler attempts compensation for observed HOT changes on failure and verifies recovery where possible. A compensation failure remains visible, and the overall failed application does not become successful because rollback worked.

One-shot cache operations are not persistent desired configuration. The tested vLLM HTTP transport does not expose its in-process DRAIN/reset lifecycle. SGLang flush requires an already-idle engine and reports BLOCKED when busy; it does not retract requests to manufacture idleness.

Audit-based restore uses captured pre-state and a fresh desired revision. Restoring values cannot undo completed tokens, prior latency or cache history. Desired state/revision files, SQLite audit and pending approvals are separate stores/lifecycles; the pilot does not establish distributed crash-atomic recovery or multi-replica ownership.

## Evidence and authority

Native telemetry is collected outside inference traffic and retains timestamps, counter epochs, quality and availability. Cumulative histograms require valid deltas and adjacent-bucket differencing. Window percentile summaries are not pooled per-request quantiles, and unavailable data is never silently converted to zero.

The supervisor observes and proposes but has no autonomous mutation authority. Human approval binds content, revision, expiry and fresh evidence. A direct authenticated operator change is a separate authorized workflow. An explicitly launched evaluator temporarily applies named phase policies, verifies them, measures the defined workload and attempts original-state restoration. It reports trade-offs or insufficient evidence rather than automatically installing a winner.

Authenticated actor identity comes from configured bearer tokens. VIEWER, OPERATOR and APPROVER permissions are distinct. Protect native administrative endpoints separately and terminate TLS outside the process for remote use.
