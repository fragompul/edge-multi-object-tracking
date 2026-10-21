# Contributing

```bash
pip install -e ".[dev,dashboard,agent,docs]"
pre-commit install            # optional: ruff + hygiene hooks
make check                    # ruff, mypy, pytest, web parity tests -- what CI runs
```

- **Tests first for algorithms.** Trackers, metrics and analytics are numeric code: add a unit test with a hand-checkable case.
- **Ports need parity fixtures.** If you change tracker, analytics or YOLO post-processing, regenerate
  `python scripts/make_ts_fixtures.py` and keep `cd web && npm test` green.
- **Generated artefacts are generated.** Do not edit `docs/results.md`, `docs/edge.md` or `docs/figures/*`; change the
  script and re-run it.
- **Benchmarks:** state the machine, and run `make edge` on an otherwise idle system.
- Style: ruff (line length 100), type hints, comments that explain *why*.
