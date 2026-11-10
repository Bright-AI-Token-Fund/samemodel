# Contributing

Thanks for looking. This is a small tool with a narrow job, and it should stay easy
to read in one sitting.

## Development setup

```bash
git clone https://github.com/Bright-AI-Token-Fund/samemodel
cd samemodel
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff
pytest -q
ruff check src tests
```

The tests start the built-in mock endpoint on a free local port. They need no
model, no GPU and no network access, and take about half a minute.

## What makes a change easy to merge

- **No runtime dependencies.** The package uses the standard library only, so it
  installs anywhere. Test-only dependencies are fine.
- **A test that fails without the change.** For anything that touches the numbers,
  the mock endpoint can inject a known perturbation; assert against that.
- **Engine quirks go in the client, not in the maths.** If an engine returns
  log-probabilities in an unusual shape, normalise it in `client.py` and add a test
  with a recorded response.

## Measurements

Results depend on the engine, backend, quantisation and hardware. If you measured a
setup that is not in `benchmarks/`, open a "Measurement report" issue with the exact
versions and commands. Numbers without a way to reproduce them will not be added.

## Reporting a problem in an inference engine

samemodel measures; it does not decide whether a difference is a bug. If a
measurement looks like an engine defect, report it to that engine with the samemodel
output and the server command attached.
