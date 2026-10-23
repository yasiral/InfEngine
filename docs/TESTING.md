# Testing and validation

InfEngine has separate checks for local software behavior, native integration and historical performance measurements.

## Local tests

With Python 3.11, install the development dependencies and run:

```bash
python -m pip install -e ".[dev,serve,eval]"
python -m pytest -q -m "not integration"
```

The recorded Windows release run passed **619 tests**, with **nine live GPU integration tests excluded** and two dependency deprecation warnings. Local fixtures and the reference engine test software behavior; they do not establish live GPU performance. The [GitHub workflow](../.github/workflows/evidence.yml) runs the CPU suite and evidence checks. Refer to [Actions](https://github.com/yasiral/InfEngine/actions/workflows/evidence.yml) for current results.

## Integration and packaging

The recorded source checks applied the vLLM hook to version 0.28.0, verified repeat application, and checked the SGLang patch against version 0.5.18. The wheel built and installed with all six entry points and the required license/notice files. Source and packaging checks do not replace a container build or live engine acceptance.

Documented native acceptance covers the seven HOT controls on the pinned patched engines. The A100 performance study covers selected vLLM controls. See [compatibility](COMPATIBILITY.md) and [runtime control evidence](evidence/support/runtime-control-verification.md) for scope and limits.

## Offline measurement verification

```bash
python -B docs/evidence/scripts/verify_and_analyze.py
python -B docs/evidence/scripts/make_figures.py
```

The evidence checks cover 48 A/B arm records, 15 initial pilot points, the pressure pilot, 12 factorial cell runs, native cache-counter fractions, paired-ratio reconciliation, histogram accounting and live readback/restoration. All eight figures can be regenerated without a GPU or model download. See [reproduction instructions](evidence/REPRODUCE.md).

Checksums cover the evidence files and link compact captures to the fuller sanitized archive. Recorded measurements remain historical observations: the original measured InfEngine commit is unknown, short telemetry windows have limited resolution, and the results do not establish a universal tuning policy.

## License

InfEngine is licensed under Apache-2.0. Upstream license texts and attributions are included in [third-party notices](../THIRD_PARTY_NOTICES.md) and [NOTICE](../NOTICE).
