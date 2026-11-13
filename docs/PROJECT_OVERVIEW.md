# Project overview

Moved. The current architecture and the limits of the probe are in [ARCHITECTURE.md](ARCHITECTURE.md) and [RESEARCH.md](RESEARCH.md).

The three constraints in `src/glassbox/caveats.py` are unchanged: labels need validation, a wording mismatch is not a hallucination, and probe predictability is not causality. A Task 7 prefix-level early-warning training/evaluation path is implemented and writes per-prefix evidence/features plus held-out lead-time metrics; its current result is exploratory and has a 100% false-warning rate on faithful answers. It does not compute lead time for ordinary live CLI answers without stored protocol labels. Activation intervention is not implemented.
