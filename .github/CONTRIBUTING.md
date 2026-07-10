# Contributing to deRIP2

Thank you for your interest in contributing to deRIP2! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Contributing Code](#contributing-code)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Questions and Contact](#questions-and-contact)

## Code of Conduct

This project is committed to providing a welcoming and inclusive experience for everyone. We expect all participants to be respectful and constructive in all project spaces — issues, pull requests and discussions. Unacceptable behaviour can be reported to the maintainer.

## How to Contribute

### Reporting Bugs

We use GitHub issues to track bugs. Report a bug by [opening a new issue](https://github.com/Adamtaranto/derip2/issues/new). Be sure to include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Any relevant logs, error messages, or screenshots
- The version of deRIP2 you're using
- Your operating system and Python version

### Suggesting Features

We welcome feature suggestions! To suggest a feature:

1. Check the [issues page](https://github.com/Adamtaranto/derip2/issues) to see if the feature has already been suggested
2. If not, [open a new issue](https://github.com/Adamtaranto/derip2/issues/new) with the label "enhancement"
3. Clearly describe the feature and its potential benefits
4. If possible, outline how it might be implemented

### Contributing Code

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Set up your environment and install the pre-commit hooks (see below)
4. Make your changes, with tests
5. Run `pytest tests/` and `pre-commit run --all-files`
6. Submit a pull request

For anything larger than a bugfix, open an issue first so the approach can be
discussed before you invest the time.

## Development Setup

To set up your development environment:

1. Clone your fork of the repository

   ```bash
   git clone https://github.com/[YOUR_USER_NAME]/derip2.git
   cd derip2
   ```


2. Create and activate a virtual environment

```bash
conda env create -f environment.yml
conda activate derip2
```

3. Install development dependencies

```bash
pip install -e ".[dev,test,docs]"
```

4. Install pre-commit hooks to enforce style

```bash
pre-commit install
```

## Style Guidelines

We follow PEP 8 and use NumPy-style docstrings. Formatting and linting are both
handled by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`
and run automatically by pre-commit. You should not need to run a formatter by
hand, but you can:

```bash
ruff format .   # format
ruff check .    # lint, with --fix applied automatically by pre-commit
```

Key points:

- Maximum line length of 88 characters
- Single quotes for strings (`ruff format` enforces this)
- Use type hints
- Use descriptive variable names

### Docstrings

Every public module, class and function needs a NumPy-style docstring, and this
is **enforced by a pre-commit hook** (`numpydoc-validation`) that will block your
commit. A docstring needs a summary line starting with an infinitive verb
("Compute…", not "Computes…"), plus `Parameters`, `Returns` and `Raises`
sections where they apply. Functions returning nothing still need a `Returns`
section with a description:

```python
def mask_columns(alignment, columns):
    """
    Overwrite the given columns with IUPAC ambiguity codes.

    Parameters
    ----------
    alignment : Bio.Align.MultipleSeqAlignment
        Alignment to modify in place.
    columns : list of int
        Ascending column indices to mask.

    Returns
    -------
    None
        Nothing is returned; the alignment is modified in place.
    """
```

Short local closures can be exempted by adding them to the `exclude` list under
`[tool.numpydoc_validation]` in `pyproject.toml`. Do this sparingly, and only
for functions that are genuinely local to their enclosing scope.

## Testing

deRIP2 is tested against Python 3.9 through 3.14. Before submitting a pull
request:

1. For new functionality, add tests. A good test fails without your change.

2. Ensure test coverage remains high.

3. Ensure all tests and hooks pass:

   ```bash
   pytest tests/
   pre-commit run --all-files
   ```

4. If you changed a hot path (alignment scanning, column classification), check
   the benchmarks:

   ```bash
   pytest tests/benchmarks --codspeed
   ```

5. If you changed the docs, check the site builds:

   ```bash
   mkdocs build
   mkdocs serve
   ```

### A note on correctness

deRIP2 makes biological calls: it decides that a given `TA` dinucleotide is a
RIP product, and rewrites sequence accordingly. A change to how columns are
classified, counted or corrected is a change to the scientific output, not just
to the code. Such changes need tests that pin the *reasoning*, not only the
result — see `tests/test_strand_bias.py`, which fuzzes the vectorised classifier
against a transcription of the original per-column scan.

## Pull Request Process

1. Update documentation (`docs/`, `README.md`) if behaviour or the CLI changed
2. Ensure all tests pass and code quality checks succeed
3. Fill out the pull request template, especially the **Why** section — explain
   the reasoning a reviewer cannot recover from the diff
4. Address any feedback from maintainers

## Questions and Contact

If you have questions about contributing, please [open an issue](https://github.com/Adamtaranto/deRIP2/issues) or contact the maintainer directly.

Thank you for contributing to deRIP2!
