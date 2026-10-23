# InfEngine Recipes

Operator-oriented tasks. All commands are read-only unless stated otherwise.

## How do I check whether InfEngine is ready?

```bash
infengine doctor
```
Non-zero exit means not ready; each failing check includes an actionable detail.

## How do I see what the engine supports?

```bash
infengine capabilities --supported-only
```
The full discovery list (including controls for the other engine) is available
without `--supported-only`.

## How do I change concurrency?

```bash
infengine set runtime_concurrency 32
infengine inspect
```

## How do I change several controls together?

```bash
infengine apply examples/policy-multi.yaml
```
Preflight validates every field first. If one field is invalid, nothing is
applied. If one field fails to converge, the change is reported as a partial
failure and the captured pre-state is restored.

## How do I safely try a policy without applying it?

```bash
infengine apply examples/policy-multi.yaml --dry-run
```

## How do I safely restore?

```bash
infengine history                 # find the change id
infengine restore --change-id <id>
```
Restore uses the VERIFIED pre-change state recorded in the audit history. If the
pre-state is missing or incomplete, restore refuses rather than guessing.

## How do I evaluate a candidate policy?

```bash
INFENGINE_ENGINE_URL=http://127.0.0.1:8000 \
infengine evaluate examples/evaluation-multi.yaml --report evaluation.html
```
Single-control is the default recommendation for causal interpretability; multi-
control is supported. The report states an observed trade-off, not a winner.

## How do I investigate high TTFT?

```bash
infengine inspect
infengine explain runtime_max_scheduled_tokens
infengine explain long_prefill_token_threshold
infengine evaluate examples/evaluation-single.yaml   # isolate one control
```
Prefer evaluating one control at a time to attribute the effect.

## How do I inspect queue/admission behavior?

```bash
infengine capabilities --supported-only     # concurrency + work-budget controls
infengine evaluate ...                       # queue/throughput metrics in the report
```

## How do I compare two runtime policies?

Use `infengine evaluate` with `baseline`/`candidate` policies. For repeated
runs, `alternate: true` interleaves A/B so drift is visible.

## How do I know the engine actually accepted my change?

`infengine set`/`apply` prints per-control `requested` vs `observed` and a
`POLICY STATUS`. Local `CONVERGED` means the engine's own readback matched
**every** changed control. `FAILED`/partial means it did not; the previous state
is restored where possible.

## How do I review history?

```bash
infengine history
infengine history --limit 50 --json
```

## How does human approval work?

Single control (observe → approve):

```bash
infengine review --approver you
```

Multi-control policy (whole-policy human approval):

```bash
infengine review examples/policy-multi.yaml --approver you
# or: infengine review --policy examples/policy-multi.yaml
```

Both use the same approval lifecycle. For a policy, the proposal is bound to the exact
whole-policy hash and desired revision; approval is single-use, expires (TTL),
is re-validated against fresh evidence immediately before mutation, uses
conditional `If-Match`, and requires whole-policy verification. Modifying any
control/value after registration invalidates the approval. There is no autonomous
apply or revert. Every applied approval appears in `infengine history` with the
authenticated actor.
