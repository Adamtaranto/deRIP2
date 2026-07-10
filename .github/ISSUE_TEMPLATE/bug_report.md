---
name: Bug report
about: Report incorrect behaviour, a crash, or a wrong result
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
A clear and concise description of what goes wrong.

**To Reproduce**

1. The command or code you ran:

```bash
derip2 -i alignment.fa -d out -p sample
```

2. What happened — the full error message and traceback, or the incorrect output:

```code
YOUR ERROR MESSAGE OR OUTPUT HERE
```

**Expected behavior**
What you expected to happen instead.

**Input data**
A bug in deRIP2 usually depends on the alignment. Please attach a **minimal**
alignment that reproduces it — often just a few sequences and a few columns. If
the data cannot be shared, describe its shape instead:

- Number of sequences and alignment length:
- Proportion of gaps, roughly:
- Any non-ACGT characters (IUPAC codes, `N`, lowercase)?

**Parameters used**
The non-default settings matter. If you used any of `--max-snp-noise`,
`--min-rip-like`, `--reaminate`, `--max-gaps`, `--fill-index`, `--fill-max-gc`,
or the strand-bias options, list them here.

**Is the result wrong, or does it crash?**
If deRIP2 produced a result you believe is biologically incorrect, say which
column or position is wrong and what you expected the corrected base to be. This
is the most useful thing you can tell us.

**Screenshots**
If the bug is in a figure or in the alignment visualisation, attach the image.

**Environment**

- deRIP2 version: [output of `derip2 --version`]
- Python version: [output of `python --version`]
- Installed via: [pip / conda / from source]
- OS: [e.g. macOS 15.1, Ubuntu 24.04, Windows 11]

**Additional context**
Anything else that might be relevant.
