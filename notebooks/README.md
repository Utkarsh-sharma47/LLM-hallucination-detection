# Notebooks

No analysis notebooks are checked in. Model and dataset files stay under `artifacts/` or the Hugging Face cache, which Git does not track.

Add notebooks here when an exploration is worth keeping. Before committing one:

- Strip large outputs and widget state.
- Keep downloaded data, activations, and checkpoints in `artifacts/`, which Git ignores.
- State in the first cell that the notebook is exploratory and point to the config and git commit it used.
- Move any result that should be compared across runs into `artifacts/experiments/<run_id>/`. The record schema is in [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md).

There is no `notebooks` extra in `pyproject.toml`. Install Jupyter in the same virtualenv if you need it. Do not add it to the default install.
