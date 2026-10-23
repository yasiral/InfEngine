# Architecture and engineering diagrams

These nine Mermaid sources explain the system, native control paths and engineering workflow. They are architectural documentation, not measured performance results. For the simpler introduction, keep using the three diagrams in the main README.

## System architecture

[Mermaid source](diagrams/d01_system_architecture.mmd)

```mermaid
flowchart TB
  O[Operator CLI or service client] --> S[Control-plane service and audit]
  S --> D[DesiredState and StateStore]
  S --> R[Reconciler and transition plan]
  D --> R
  R --> A[Selected engine adapter]
  A --> V[vLLM native control hook]
  A --> G[SGLang internal-state RPC]
  A --> B[Built-in reference engine]
  I[Inference client] --> E[Engine inference endpoint]
  V --> E
  G --> E
  T[Out-of-band native telemetry] --> N[Quality-aware normalized snapshot]
  E --> T
  N --> U[Observe-only supervisor]
  U --> H[Human approval and revalidation]
  H --> S
  X[Explicitly invoked evaluator] --> S
  N --> X
```

## Reconciliation sequence

[Mermaid source](diagrams/d02_reconciliation_sequence.mmd)

```mermaid
sequenceDiagram
  actor O as Operator
  participant S as Service
  participant R as Reconciler
  participant D as StateStore
  participant A as Adapter
  participant E as Engine
  O->>S: Policy plus expected revision
  S->>R: Conditional reconciliation
  R->>D: Check revision under mutation lock
  R->>A: Read native state and validate all controls
  A->>E: Native readback
  alt Invalid policy or revision conflict
    R-->>S: Reject with no accepted mutation
  else Valid plan
    R->>D: Persist accepted desired state and revision
    R->>A: Execute planned transitions
    A->>E: Native mutation
    R->>A: Verify resulting native state
    R-->>S: Converged, blocked or failed
  end
  S->>S: Record audit outcome
  S-->>O: Desired, observed and transition result
```

## vLLM control path

[Mermaid source](diagrams/d03_vllm_control_path.mmd)

```mermaid
flowchart TB
  C[InfEngine vLLM HTTP adapter] --> L[Authenticated control listener]
  subgraph API[vLLM API-server]
    L --> U[AsyncLLM and EngineCoreClient utility call]
    I[Existing inference HTTP endpoint]
  end
  U --> E[Existing EngineCore process]
  I --> E
  E --> S[Scheduler runtime fields and KV admission]
  S --> R[Native config and provisioned-limit readback]
  R --> U
  U --> C
  style API fill:#f3f7f9,stroke:#748895,color:#182c3d
```

## SGLang control path

[Mermaid source](diagrams/d04_sglang_control_path.mmd)

```mermaid
flowchart TB
  A[InfEngine SGLang adapter] --> P[POST set_internal_state with server_args]
  P --> T[TokenizerManager]
  T --> Z[Existing ZMQ transport]
  Z --> S[Scheduler runtime controls]
  S --> R[GET server_info: live internal_states]
  R --> A
  A --> L[Runtime-client load read: get_load]
  N[Normalized telemetry collector] --> V[v1/loads with Prometheus format]
  N --> M[Native metrics endpoint]
  S --> L
  S --> V
  S --> M
```

## Capability semantics

[Mermaid source](diagrams/d05_capability_semantics.mmd)

```mermaid
flowchart TB
  I[Operator intent] --> C[Capability contract]
  C --> V[Type, live bounds and engine support]
  V --> H[HOT bounded runtime policy]
  V --> D[DRAIN requires supported quiescence path]
  V --> O[One-shot operation with explicit outcome]
  V --> U[Unsupported: reject before application]
  H --> M[Native mapping and scope]
  M --> X[vLLM total-step scheduling semantics]
  M --> Y[SGLang prefill-admission semantics]
  D --> Q[Busy or unavailable path: BLOCKED]
  O --> N[Do not persist as desired configuration]
```

## Approval lifecycle

[Mermaid source](diagrams/d06_approval_lifecycle.mmd)

```mermaid
flowchart TB
  S[Observe-only supervisor proposal] --> R[Register exact decision and revision]
  P[Human-authored multi-control policy] --> W[Register whole-policy hash and native pre-state]
  R --> H[Human approve or reject]
  W --> H
  H -->|Reject| T[Terminal record; no mutation]
  H -->|Approve| C[TTL, capability, revision and live-state checks]
  C --> F[Fresh evidence required]
  F --> A[Single-control: reproduce eligible proposal]
  F --> B[Whole-policy: confirm captured control pre-state]
  A --> I[Conditional PATCH through reconciler]
  B --> I
  I --> V[Verify all requested native values]
  V --> K[Terminal record and verified-change cooldown]
  C -->|Stale or unavailable| T
  F -->|Unavailable or ineligible| T
```

## Multi control failure recovery

[Mermaid source](diagrams/d07_multi_control_failure_recovery.mmd)

```mermaid
flowchart TB
  P[Validate whole policy and capture pre-state] --> A[Accept desired revision]
  A --> E[Execute supported transitions]
  E --> V{Native verification}
  V -->|All requested values match| C[CONVERGED]
  V -->|HOT failure or mismatch| F[FAILED application]
  F --> R[Compensate observed HOT changes]
  R --> B{Readback of compensation}
  B -->|Matches pre-state| K[Recovery verified; application still FAILED]
  B -->|Unavailable or mismatched| U[Recovery unverified; operator intervention]
  E -->|DRAIN cannot proceed| D[BLOCKED; accepted intent not converged]
  A -. durable intent can remain .-> S[Desired and physical state may differ]
  K --> S
  U --> S
```

## Evaluation telemetry lifecycle

[Mermaid source](diagrams/d08_evaluation_telemetry_lifecycle.mmd)

```mermaid
flowchart TB
  V[vLLM native metrics] --> C[Out-of-band collectors]
  S[SGLang metrics and v1/loads] --> C
  C --> Q[Values, units, source and quality]
  Q --> E[Contiguous counter epoch and rolling windows]
  E --> N[Normalized snapshot]
  N --> O[Read-only load sweep and supervisor evidence]
  N --> X[Explicit policy evaluator measurements]
  X --> A[Apply verified baseline or candidate]
  A --> W[Warmup and bounded measurement]
  W --> R[Report outcomes and availability]
  R --> F[Finally attempt original desired-state restore]
  F --> H[Human interpretation; no automatic winner]
```

## Operator workflow

[Mermaid source](diagrams/d10_operator_workflow.mmd)

```mermaid
flowchart TB
  D[Doctor, inspect and capabilities] --> C[Choose authorized workflow]
  C --> M[Direct manual policy]
  C --> H[Human-reviewed proposal]
  C --> E[Explicit temporary evaluation]
  M --> P[Conditional apply through reconciler]
  H --> R[Register, approve and freshly revalidate]
  R --> P
  E --> B[Verify baseline and candidate phases]
  B --> F[Attempt original-state restore]
  P --> V[Native verification and durable history]
  F --> V
  V --> A[Inspect outcome and interpret evidence]
  A --> S[Optional separate audit-based restore]
```
