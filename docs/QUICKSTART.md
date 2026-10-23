# InfEngine Quickstart

Operate InfEngine in 10–15 minutes. You do **not** need to know the internal
architecture — only the `infengine` CLI.

## 1. Install

```bash
# from the repository root
pip install -e ".[serve,eval]"
infengine version
```

## 2. Configure the engine

Point the control plane at a running engine. The token (if any) comes from the
environment, never the command line.

vLLM (patched v0.28.0 control listener):

```bash
export INFENGINE_VLLM_CONTROL_URL=http://127.0.0.1:9001
export INFENGINE_VLLM_CONTROL_TOKEN=...      # token for the vLLM control hop
```

SGLang (native admin surface):

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000
```

Security (recommended; see DEPLOYMENT.md):

```bash
export INFENGINE_AUTH_TOKENS='{"<operator-token>":{"actor":"you","role":"OPERATOR"}}'
export INFENGINE_TOKEN=<operator-token>
export INFENGINE_ACTOR=you
```

## 3. Start / connect

```bash
infengine serve --runtime vllm \
  --vllm-control-url http://127.0.0.1:9001 \
  --state-file ./desired.json \
  --audit-file ./audit.db \
  --host 127.0.0.1 --port 8010
```

(`infengine serve` wraps `infengine-serve`.)

Set `INFENGINE_CONTROL_URL=http://127.0.0.1:8010` in the operator environment and provide its configured operator bearer token through `INFENGINE_TOKEN`. Engine inference remains on port 8000.

## 4. Doctor

```bash
infengine doctor
```

Answers whether the deployment is ready: engine reachable, control transport
authenticated, readback working, supported controls, telemetry, audit
persistence and security mode. Failures are actionable; secrets are never
printed.

## 5. Inspect

```bash
infengine inspect
```

Shows engine, identity, current revision, desired state, observed state and
convergence status.

## 6. Capabilities

```bash
infengine capabilities --supported-only
```

## 7. Explain

```bash
infengine explain runtime_max_scheduled_tokens
```

Native mapping, semantics, transition class, current/bounds, operational
effects, conditions where it is inert, verification mechanism, limitations and
(clearly labelled historical) evidence.

## 8. Change one control

```bash
infengine set runtime_concurrency 32
```

## 9. Change several controls together

```bash
infengine apply examples/policy-multi.yaml
```

Every field is validated before anything is applied. If one field is invalid,
nothing is applied. If the whole policy does not converge, compensation is attempted for observed HOT changes and verified where possible.
The application remains failed; recovery can itself fail. Neither engine is
claimed engine-atomic.

## 10. Verify

```bash
infengine inspect
```

`CONVERGED` means the engine's own readback matches the requested policy for
**every** changed control.

## 11. Restore

```bash
infengine restore                 # latest change
infengine restore --change-id <id>
```

Restoration uses the verified pre-change state captured in the audit history;
you do not need to remember previous values.

## 12. Run a small evaluation

```bash
INFENGINE_ENGINE_URL=http://127.0.0.1:8000 \
infengine evaluate examples/evaluation-multi.yaml --report evaluation.html
```

The evaluator applies the baseline, measures, applies the candidate, verifies
every control, measures, restores, and writes `evaluation.json`/HTML. The report
states an observed trade-off — a human decides what policy is preferable.

## 13. History

```bash
infengine history
```

Every mutation (and attempted mutation) is recorded with actor, revisions,
requested policy, pre-state, post-state, verification and rollback.

## 14. Human approval (optional)

For a change that requires explicit human approval rather than a direct `set`/
`apply`:

```bash
infengine review examples/policy-multi.yaml --approver you   # multi-control
infengine review --approver you                              # single-control
```

A policy review binds to the exact whole policy and desired revision, expires
(TTL), is single-use, is re-validated against fresh evidence immediately before
mutation, and requires whole-policy verification. Modifying any control/value
invalidates the approval. There is no autonomous apply or revert.

