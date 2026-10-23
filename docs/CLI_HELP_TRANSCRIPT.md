### infengine --help
usage: infengine [-h]
                 {version,doctor,inspect,capabilities,explain,set,apply,restore,history,evaluate,review,serve}
                 ...

InfEngine unified operator CLI. Human-controlled runtime policy for vLLM /
SGLang: inspect, explain, change, verify, evaluate, restore.

positional arguments:
  {version,doctor,inspect,capabilities,explain,set,apply,restore,history,evaluate,review,serve}
    version             print the InfEngine version
    doctor              check whether this deployment is ready to operate
    inspect             show desired/observed state, revision and identity
    capabilities        list controls and their engine support
    explain             explain one control (mapping, semantics, limits)
    set                 change ONE control (HOT; no approval required)
    apply               apply a multi-control policy file (dry-run supported)
    restore             restore the pre-state of a previous change
    history             show the durable change/audit history
    evaluate            run an A/B (single- or multi-control) experiment
    review              human review: single-control observe->approve, or
                        a multi-control policy file (infengine review <policy-
                        file>)
    serve               run the control-plane service (wraps infengine-serve)

options:
  -h, --help            show this help message and exit

### infengine doctor --help
usage: infengine doctor [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                        [--token TOKEN] [--actor ACTOR] [--json]

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output

### infengine inspect --help
usage: infengine inspect [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                         [--token TOKEN] [--actor ACTOR] [--json]

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output

### infengine capabilities --help
usage: infengine capabilities [-h] [--control-url CONTROL_URL]
                              [--timeout TIMEOUT] [--token TOKEN]
                              [--actor ACTOR] [--json] [--supported-only]

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --supported-only

### infengine explain --help
usage: infengine explain [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                         [--token TOKEN] [--actor ACTOR] [--json]
                         control

positional arguments:
  control

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output

### infengine set --help
usage: infengine set [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                     [--token TOKEN] [--actor ACTOR] [--json]
                     [--if-match IF_MATCH]
                     control value

positional arguments:
  control
  value

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --if-match IF_MATCH   expected desired_revision (conditional PATCH)

### infengine apply --help
usage: infengine apply [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                       [--token TOKEN] [--actor ACTOR] [--json]
                       [--if-match IF_MATCH] [--dry-run]
                       policy_file

positional arguments:
  policy_file

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --if-match IF_MATCH
  --dry-run

### infengine restore --help
usage: infengine restore [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                         [--token TOKEN] [--actor ACTOR] [--json]
                         [--change-id CHANGE_ID] [--if-match IF_MATCH]

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --change-id CHANGE_ID
  --if-match IF_MATCH

### infengine evaluate --help
usage: infengine evaluate [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                          [--token TOKEN] [--actor ACTOR] [--json]
                          [--report REPORT] [--engine-url ENGINE_URL]
                          [--telemetry-url TELEMETRY_URL]
                          experiment_file

positional arguments:
  experiment_file

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --report REPORT       write an HTML report to this path
  --engine-url ENGINE_URL
                        inference endpoint (or env INFENGINE_ENGINE_URL)
  --telemetry-url TELEMETRY_URL
                        telemetry endpoint (or env INFENGINE_TELEMETRY_URL)

### infengine history --help
usage: infengine history [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                         [--token TOKEN] [--actor ACTOR] [--json]
                         [--limit LIMIT] [--offset OFFSET]

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --limit LIMIT
  --offset OFFSET

### infengine review --help
usage: infengine review [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                        [--token TOKEN] [--actor ACTOR] [--json]
                        [--policy POLICY] [--samples SAMPLES]
                        [--interval INTERVAL] [--window-s WINDOW_S]
                        [--min-samples MIN_SAMPLES] [--cooldown-s COOLDOWN_S]
                        [--ttl TTL] [--approver APPROVER]
                        [--latency-max LATENCY_MAX] [--kv-max KV_MAX]
                        [--latency-guardrail] [--no-latency-guardrail]
                        [--kv-guardrail] [--no-kv-guardrail]
                        [--required-guardrails] [--no-required-guardrails]
                        [--yes]
                        [policy_file]

positional arguments:
  policy_file           optional multi-control policy file to review and
                        approve

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output
  --policy POLICY       alias for the policy file positional
  --samples SAMPLES
  --interval INTERVAL
  --window-s WINDOW_S
  --min-samples MIN_SAMPLES
  --cooldown-s COOLDOWN_S
  --ttl TTL
  --approver APPROVER
  --latency-max LATENCY_MAX
  --kv-max KV_MAX
  --latency-guardrail
  --no-latency-guardrail
  --kv-guardrail
  --no-kv-guardrail
  --required-guardrails
  --no-required-guardrails
  --yes

### infengine serve --help
usage: infengine serve [-h] ...

positional arguments:
  serve_args

options:
  -h, --help  show this help message and exit

### infengine version --help
usage: infengine version [-h] [--control-url CONTROL_URL] [--timeout TIMEOUT]
                         [--token TOKEN] [--actor ACTOR] [--json]

options:
  -h, --help            show this help message and exit
  --control-url CONTROL_URL
                        base URL of the InfEngine control-plane service
  --timeout TIMEOUT
  --token TOKEN         bearer token (prefer env INFENGINE_TOKEN)
  --actor ACTOR         actor identity (prefer env INFENGINE_ACTOR)
  --json                machine-readable JSON output

