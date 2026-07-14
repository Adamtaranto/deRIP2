"""
Shared fixtures for the deRIP2 test suite.

Centralises the alignment/tree/state data-file paths, the IQ-TREE availability
gate, and the offline ``TreeReconstruction`` builder that several modules test
against so they no longer have to be re-declared per file.
"""

import os
import shutil

import numpy as np
import pytest

HERE = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def data_dir():
    """Absolute path to the committed test-data directory."""
    return os.path.join(HERE, 'data')


@pytest.fixture(scope='session')
def mintest_path(data_dir):
    """Path to the 6-sequence, 35-column reference alignment."""
    return os.path.join(data_dir, 'mintest.fa')


@pytest.fixture(scope='session')
def treefile_path(data_dir):
    """Path to the committed IQ-TREE ``.treefile`` for mintest."""
    return os.path.join(data_dir, 'mintest.treefile')


@pytest.fixture(scope='session')
def statefile_path(data_dir):
    """Path to the committed IQ-TREE ``.state`` for mintest."""
    return os.path.join(data_dir, 'mintest.state')


@pytest.fixture(scope='session')
def have_iqtree():
    """True when an IQ-TREE binary is on PATH (gates live-run tests)."""
    return any(shutil.which(name) for name in ('iqtree3', 'iqtree2', 'iqtree'))


@pytest.fixture
def make_reconstruction():
    """
    Return a factory that builds a ``TreeReconstruction`` from strings/edges.

    The factory takes ``(node_seqs, edges, root_name, probs=None)`` where
    ``node_seqs`` maps node names to sequence strings and ``edges`` is a list of
    directed ``(parent, child)`` name pairs. No IQ-TREE binary is required, so it
    is usable for pure-Python traversal/guard tests.
    """
    from derip2.spectra.tree_asr import TreeReconstruction

    def _make(node_seqs, edges, root_name, *, probs=None):
        node_seq = {
            name: np.frombuffer(s.encode('ascii'), dtype='S1').copy()
            for name, s in node_seqs.items()
        }
        n_cols = len(next(iter(node_seqs.values())))
        node_prob = {
            name: (
                np.full(n_cols, probs[name], dtype=float)
                if probs and name in probs
                else np.ones(n_cols, dtype=float)
            )
            for name in node_seqs
        }
        tips = [name for name in node_seqs if not any(name == p for p, _ in edges)]
        return TreeReconstruction(
            edges=edges,
            node_seq=node_seq,
            node_prob=node_prob,
            root_name=root_name,
            tip_names=tips,
            n_cols=n_cols,
        )

    return _make
