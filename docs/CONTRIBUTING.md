# Contributing

Setup and the commands to run before a pull request are in [DEVELOPMENT.md](DEVELOPMENT.md) and [TESTING.md](TESTING.md). The license is unselected. Do not add a `LICENSE` file or copyright headers until the maintainer chooses one.

## Branches

`main` is the integration branch. Work happens on a short-lived branch and lands through a pull request:

```text
feat/terminal-demo
feat/probe-training
docs/project-documentation
ci/github-actions
fix/probe-metadata
test/model-validation
```

Do not commit feature work directly on `main`. Do not force-push `main` or any published branch. Fetch `main` before merging. One branch should be one reviewable change. Do not split a change so that an intermediate merge cannot import the package. The published history starts at `9b8bcc4` (`Initial commit`, README only). Do not rewrite that commit. There is no older tag or release.

## Commits

Use a conventional subject: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, or `chore:`, then an imperative line that says why the change exists. Keep each commit to one coherent change. Inspect the staged diff before committing. Do not commit `.env`, Hugging Face tokens, `artifacts/` contents, weight files, dataset snapshots, caches, or `*.egg-info`. Those paths are gitignored; `git add -f` on them is a mistake.

## Pull requests

Open a pull request against `main`. GitHub uses [the PR template](../.github/PULL_REQUEST_TEMPLATE.md). Fill in the commands you actually ran and their results. A docs-only change still runs the test commands in [TESTING.md](TESTING.md) when it touches config keys, metric names, or CLI flags the tests lock.

Recommended branch protection, once an administrator turns it on:

- Require the `checks` job from [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
- Require one review. [CODEOWNERS](../.github/CODEOWNERS) names the repository owner as the default reviewer. Feature ownership inside the roadmap is otherwise unassigned.
- Do not allow force-pushes to `main`.

The optional model workflow is not a required check. It downloads weights and is manual.

## Review

Review for the claim the diff makes, not only for style.

- A new number in a doc or README must come from a command in the pull request. Leave the cell blank or say "not run" otherwise.
- Do not relabel token probability, entropy, or an uncalibrated score as hallucination risk.
- Do not treat a reference-answer wording mismatch as a hallucination.
- Do not pool HaluEval and TruthfulQA labels.
- Do not describe `halueval-generation-001` as a validated detector. Its test AUROC does not beat TF-IDF, and calibration was rejected. Do not quote the pair AUROC of 0.996 as that experiment. The index is [EXPERIMENTS.md](EXPERIMENTS.md).
- Keep downloads behind `allow_download=True` or an explicit CLI flag. Tests and smoke stay offline.
- Probe metadata keys listed in [ARCHITECTURE.md](ARCHITECTURE.md) are a compatibility contract.

Ruff is the formatter and linter (`pyproject.toml`). Line length E501 is ignored until [the roadmap task](ROADMAP.md) removes that ignore. Do not add a second formatter.

## Releases

Stay on semantic versioning. `0.1.0` is the package version in `pyproject.toml` and `glassbox.__version__`. They must match. The release rule says to cut `0.2.0` when a saved probe and its held-out record exist, or when the public CLI flags change. `train-generation-probe` is a new public subcommand, and an uncalibrated probe already exists under gitignored `artifacts/`. Tagging `0.2.0` is still a maintainer action. Do not tag it from a training or documentation pull request, and do not publish a release that contains weights or datasets. A release changelog does not exist yet; add one with the first tagged release. The research index is [EXPERIMENTS.md](EXPERIMENTS.md), which is not that changelog.
