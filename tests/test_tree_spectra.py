"""
Tests for the phylogenetic mutation-spectrum path.

The correctness-critical behaviour — parent -> child polarity, per-branch
homoplasy counting, probability filtering — is pinned with hand-built
``TreeReconstruction`` objects that need no external binary. Rooting and ``.state``
parsing are checked against committed IQ-TREE fixtures
(``tests/data/mintest.treefile`` / ``mintest.state``). A live IQ-TREE run is
exercised only when the binary is on ``PATH``.
"""

import logging
import os
import shutil

import numpy as np
import pytest

from derip2.spectra.call_mutations import compute_spectra_from_tree
from derip2.spectra.channels import SBS96_INDEX, SBS192_INDEX
from derip2.spectra.tree_asr import (
    TreeReconstruction,
    build_reconstruction,
    find_iqtree,
    orientation_flip_fraction,
    parse_state,
)

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')
TREEFILE = os.path.join(HERE, 'data', 'mintest.treefile')
STATEFILE = os.path.join(HERE, 'data', 'mintest.state')

_HAVE_IQTREE = any(shutil.which(name) for name in ('iqtree3', 'iqtree2', 'iqtree'))


def _seq(s):
    """Return a sequence string as an 'S1' byte array."""
    return np.frombuffer(s.encode('ascii'), dtype='S1').copy()


def _reconstruction(node_seqs, edges, root_name, *, probs=None):
    """Build a TreeReconstruction from sequence strings and directed edges."""
    node_seq = {name: _seq(s) for name, s in node_seqs.items()}
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


# --------------------------------------------------------------------------
# Branch traversal: polarity, homoplasy, filtering (no binary needed)
# --------------------------------------------------------------------------
def test_single_branch_event():
    """One C>T on a branch is one event in the A[C>T]G channel."""
    rec = _reconstruction(
        {'root': 'ACGTA', 'child': 'ATGTA'}, [('root', 'child')], 'root'
    )
    res = compute_spectra_from_tree(rec)
    assert res.method == 'phylogenetic'
    assert res.sbs96.sum() == 1
    assert res.sbs96[SBS96_INDEX['A[C>T]G'], 0] == 1
    assert res.event_parent_names == ['root']
    assert res.event_child_names == ['child']


def test_direction_follows_polarity():
    """Reversing the edge reverses the substitution direction (C>T vs T>C)."""
    forward = _reconstruction({'p': 'ACGTA', 'c': 'ATGTA'}, [('p', 'c')], 'p')
    reverse = _reconstruction({'p': 'ACGTA', 'c': 'ATGTA'}, [('c', 'p')], 'c')
    fwd = compute_spectra_from_tree(forward)
    rev = compute_spectra_from_tree(reverse)
    # Forward parent C -> child T is C>T; the reversed edge sees T -> C.
    assert fwd.sbs192[SBS192_INDEX['A[C>T]G'], 0] == 1
    assert rev.sbs192[SBS192_INDEX['A[T>C]G'], 0] == 1


def test_homoplasy_counts_independent_branches():
    """The same C>T planted on two independent branches is two events."""
    # root -> a and root -> b, each acquiring the col-1 C>T independently.
    rec = _reconstruction(
        {'root': 'ACGTA', 'a': 'ATGTA', 'b': 'ATGTA'},
        [('root', 'a'), ('root', 'b')],
        'root',
    )
    res = compute_spectra_from_tree(rec)
    assert res.sbs96.sum() == 2
    table = res.homoplasy_table(min_hits=2)
    assert table == [{'col': 1, 'ref': 'C', 'alt': 'T', 'n_independent': 2}]


def test_inherited_mutation_counted_once():
    """A mutation on one branch, then inherited, is a single event."""
    # root=C, mid acquired T (one event), tip inherits T (no further event).
    rec = _reconstruction(
        {'root': 'ACGTA', 'mid': 'ATGTA', 'tip': 'ATGTA'},
        [('root', 'mid'), ('mid', 'tip')],
        'root',
    )
    res = compute_spectra_from_tree(rec)
    assert res.sbs96.sum() == 1  # only the root->mid branch carries the change


def test_min_prob_filters_low_confidence():
    """A low posterior on a branch drops the event under min_prob."""
    rec = _reconstruction(
        {'root': 'ACGTA', 'child': 'ATGTA'},
        [('root', 'child')],
        'root',
        probs={'root': 0.6},  # child prob defaults to 1.0 -> combined 0.6
    )
    assert compute_spectra_from_tree(rec, min_prob=0.0).sbs96.sum() == 1
    assert compute_spectra_from_tree(rec, min_prob=0.8).sbs96.sum() == 0


def test_indel_on_branch_is_skipped():
    """A gap opposite a base on a branch is not a substitution."""
    rec = _reconstruction(
        {'root': 'ACGT', 'child': 'A-GT'}, [('root', 'child')], 'root'
    )
    res = compute_spectra_from_tree(rec)
    assert res.sbs96.sum() == 0
    assert res.n_indel_or_ambiguous == 1


def test_clade_partition_samples():
    """A samples_by_child map splits events into per-clade sample columns."""
    # 'a' has a col-1 C>T, 'b' has a col-2 G>T; both are internal columns with
    # resolvable context.
    rec = _reconstruction(
        {'root': 'ACGTA', 'a': 'ATGTA', 'b': 'ACTTA'},
        [('root', 'a'), ('root', 'b')],
        'root',
    )
    res = compute_spectra_from_tree(
        rec, samples_by_child={'a': 'cladeA', 'b': 'cladeB'}
    )
    assert set(res.sample_names) == {'cladeA', 'cladeB'}
    assert res.sbs96.sum() == 2
    assert res.sbs96[:, res.sample_names.index('cladeA')].sum() == 1


# --------------------------------------------------------------------------
# .state parsing and rooting against committed IQ-TREE fixtures
# --------------------------------------------------------------------------
def test_parse_state_fixture():
    """The committed .state parses to four internal nodes over 35 sites."""
    node_seq, node_prob, n_sites = parse_state(STATEFILE)
    assert n_sites == 35
    assert set(node_seq) == {'Node1', 'Node2', 'Node3', 'Node4'}
    assert node_seq['Node4'].shape == (35,)
    # Probabilities lie in [0, 1].
    assert float(node_prob['Node4'].min()) >= 0.0
    assert float(node_prob['Node4'].max()) <= 1.0


def test_build_reconstruction_all_nodes_sequenced():
    """Every node in the rooted reconstruction carries a sequence."""
    from derip2.aln_ops import loadAlign

    aln = loadAlign(MINTEST)
    rec = build_reconstruction(TREEFILE, STATEFILE, aln, rooting='none')
    assert len(rec.edges) == 9  # 6 tips + 4 internal -> 10 nodes, 9 edges
    for parent, child in rec.edges:
        assert parent in rec.node_seq and child in rec.node_seq


def test_rooting_flips_polarity():
    """Outgroup and midpoint rooting orient at least one edge differently."""
    from derip2.aln_ops import loadAlign

    aln = loadAlign(MINTEST)
    og = build_reconstruction(
        TREEFILE, STATEFILE, aln, rooting='outgroup', outgroup='Seq1'
    )
    mid = build_reconstruction(TREEFILE, STATEFILE, aln, rooting='midpoint')
    assert og.root_name == 'Seq1'
    assert orientation_flip_fraction(og.edges, mid.edges) > 0.0


def test_outgroup_rooting_requires_outgroup():
    """Outgroup rooting without a named outgroup is an error."""
    from derip2.aln_ops import loadAlign

    aln = loadAlign(MINTEST)
    with pytest.raises(ValueError):
        build_reconstruction(TREEFILE, STATEFILE, aln, rooting='outgroup')


def test_find_iqtree_missing_raises():
    """An explicit bogus binary name raises a clear error."""
    with pytest.raises(FileNotFoundError):
        find_iqtree('definitely-not-iqtree-xyz')


# --------------------------------------------------------------------------
# Live IQ-TREE integration (only when the binary is installed)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_IQTREE, reason='IQ-TREE not on PATH')
def test_reconstruct_end_to_end(tmp_path):
    """A full IQ-TREE run over mintest yields a usable reconstruction."""
    from derip2.aln_ops import loadAlign
    from derip2.spectra.tree_asr import reconstruct

    aln = loadAlign(MINTEST)
    rec = reconstruct(
        aln, str(tmp_path / 'run'), model='JC', threads='1', rooting='midpoint'
    )
    assert rec.n_cols == aln.get_alignment_length()
    assert len(rec.tip_names) == len(aln)
    res = compute_spectra_from_tree(rec)
    assert res.sbs96.sum() == res.sbs192.sum()
    assert 'iqtree_version' in rec.manifest
