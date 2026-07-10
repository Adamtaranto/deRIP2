<!--
Thanks for contributing to deRIP2! See .github/CONTRIBUTING.md for setup and
style guidelines. Delete any section that does not apply to your change.
-->

## What

<!-- What does this PR change? One or two sentences, in plain language. -->

## Why

<!--
The reasoning a reviewer cannot get from the diff: the problem this solves, the
approach you chose and what you rejected. For changes to how RIP is detected,
counted or corrected, state the biological justification — a reviewer needs to
know a `TA` was called a RIP product for the right reason.
-->

Fixes #<!-- issue number, or delete this line -->

## How it was verified

<!--
Say what you actually ran and what you observed, not just that CI is green.
For plotting or output changes, attach a before/after image or paste the output.
-->

```bash
pytest tests/
pre-commit run --all-files
```

## Checklist

- [ ] Tests cover the new behaviour, and they fail without the change
- [ ] `pytest tests/` passes
- [ ] `pre-commit run --all-files` passes (ruff, numpydoc, formatting)
- [ ] Public functions and classes have NumPy-style docstrings
- [ ] Docs updated (`docs/`, `README.md`) if behaviour or the CLI changed
- [ ] `mkdocs build` succeeds if docs changed
- [ ] Benchmarks run if hot paths changed: `pytest tests/benchmarks --codspeed`

## Breaking changes

<!--
Renamed or removed arguments, changed defaults, changed output formats. Note
what downstream callers must do. Write "None" if there are none.
-->

None
