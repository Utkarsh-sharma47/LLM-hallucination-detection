# Datasets

Moved to [RESEARCH.md](RESEARCH.md). That page has the HaluEval pin, the filter counts, the curated demo file, and the TruthfulQA reference rules.

`glassbox.data.loading.load_examples` reads a local pinned file and does not download. HaluEval rows use the pair-protocol labels from `glassbox.data.halueval`. TruthfulQA rows are reference questions with `UNVALIDATED` labels. A missing snapshot stays offline: `DownloadDisabledError` when downloads are disabled, and `NotImplementedError` when a download is requested.
