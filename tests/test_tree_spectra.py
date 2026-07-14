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


def test_assign_groups_homogeneous_and_mixed():
    """Branches inside one group get that label; spanning branches are 'mixed'."""
    from derip2.spectra.tree_asr import assign_groups

    # root -> g (a mixed internal node) and root -> c3 (group B).
    # g -> c1 (group A), g -> c2 (group A). The g subtree is all A; the root's
    # other side is B; branches above are mixed.
    rec = _reconstruction(
        {
            'root': 'ACGTA',
            'g': 'ACGTA',
            'c1': 'ATGTA',
            'c2': 'ATGTA',
            'c3': 'ACGTT',
        },
        [('root', 'g'), ('g', 'c1'), ('g', 'c2'), ('root', 'c3')],
        'root',
    )
    group_by_tip = {'c1': 'A', 'c2': 'A', 'c3': 'B'}
    assigned = assign_groups(rec, group_by_tip)
    assert assigned['c1'] == 'A'
    assert assigned['c2'] == 'A'
    assert assigned['g'] == 'A'  # whole g-subtree is group A
    assert assigned['c3'] == 'B'


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


def test_sanitize_name_matches_iqtree():
    """Name sanitisation matches IQ-TREE: only [A-Za-z0-9._-] survive."""
    from derip2.spectra.tree_asr import _sanitize_name

    assert _sanitize_name('UNSE01.1:1-9(-)') == 'UNSE01.1_1-9_-_'
    assert _sanitize_name('scf|a b+c') == 'scf_a_b_c'
    assert _sanitize_name('Seq1') == 'Seq1'  # clean names are unchanged


def test_build_reconstruction_maps_sanitised_leaf_names(tmp_path):
    """Tree leaves with IQ-TREE-sanitised names attach to the right sequences."""
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from derip2.spectra.tree_asr import _sanitize_name

    # Alignment ids carry characters IQ-TREE would rewrite to '_'.
    aln = MultipleSeqAlignment(
        [
            SeqRecord(Seq('ACGT'), id='a:1(+)'),
            SeqRecord(Seq('ATGT'), id='b:2(-)'),
            SeqRecord(Seq('ACGA'), id='c'),
        ]
    )
    # A tree written with the sanitised tip names, plus two internal nodes.
    a_name = _sanitize_name('a:1(+)')
    b_name = _sanitize_name('b:2(-)')
    treefile = tmp_path / 'san.treefile'
    treefile.write_text(f'(({a_name},{b_name})Node2,c)Node1;')
    # A minimal .state for the two internal nodes over four sites.
    statefile = tmp_path / 'san.state'
    lines = ['# comment', 'Node\tSite\tState\tp_A\tp_C\tp_G\tp_T']
    for node in ('Node1', 'Node2'):
        for site, base in enumerate('ACGT', start=1):
            probs = ['0.01', '0.01', '0.01', '0.01']
            probs['ACGT'.index(base)] = '0.97'
            lines.append(f'{node}\t{site}\t{base}\t' + '\t'.join(probs))
    statefile.write_text('\n'.join(lines) + '\n')

    rec = build_reconstruction(str(treefile), str(statefile), aln, rooting='none')
    # The sanitised leaf name resolves to the original sequence's bases.
    assert rec.node_seq[a_name].tobytes() == b'ACGT'
    assert rec.node_seq[b_name].tobytes() == b'ATGT'


# --------------------------------------------------------------------------
# Live IQ-TREE integration (only when the binary is installed)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Clade assignment and pure-Python guard branches (no binary needed)
# --------------------------------------------------------------------------
def test_assign_clades_labels_each_subtree():
    """Every non-root node is labelled by the root-child at the top of its clade."""
    from derip2.spectra.tree_asr import assign_clades

    # root -> a -> a1, root -> b. Clade 'a' covers {a, a1}; clade 'b' covers {b}.
    rec = _reconstruction(
        {'root': 'ACGTA', 'a': 'ACGTA', 'a1': 'ATGTA', 'b': 'ACGTT'},
        [('root', 'a'), ('a', 'a1'), ('root', 'b')],
        'root',
    )
    clades = assign_clades(rec)
    assert clades == {'a': 'a', 'a1': 'a', 'b': 'b'}
    assert 'root' not in clades  # the root belongs to no clade


def test_parse_state_empty_raises(tmp_path):
    """A .state file with a header but no data rows is an error."""
    statefile = tmp_path / 'empty.state'
    statefile.write_text('# comment\nNode\tSite\tState\tp_A\tp_C\tp_G\tp_T\n')
    with pytest.raises(ValueError, match='No ancestral states'):
        parse_state(str(statefile))


def test_parse_state_out_of_range_site_raises(tmp_path):
    """A site index beyond the declared width is rejected."""
    statefile = tmp_path / 'oor.state'
    lines = ['Node\tSite\tState\tp_A\tp_C\tp_G\tp_T']
    # Sites 1 and 3 present but no 2: max is 3 so n_sites=3; indices in range.
    # Force out-of-range by using site 0 (idx -1).
    lines.append('Node1\t0\tA\t0.97\t0.01\t0.01\t0.01')
    lines.append('Node1\t1\tC\t0.01\t0.97\t0.01\t0.01')
    statefile.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='out-of-range'):
        parse_state(str(statefile))


def test_build_reconstruction_width_mismatch_raises(tmp_path):
    """An alignment wider/narrower than the .state sites is rejected."""
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    # .state fixture is 35 sites; give a 4-column alignment.
    aln = MultipleSeqAlignment(
        [SeqRecord(Seq('ACGT'), id='Seq1'), SeqRecord(Seq('ATGT'), id='Seq2')]
    )
    with pytest.raises(ValueError, match='does not match .state sites'):
        build_reconstruction(TREEFILE, STATEFILE, aln, rooting='none')


def test_build_reconstruction_sanitised_collision_raises(tmp_path):
    """Two ids that collapse to the same sanitised name are rejected."""
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    treefile = tmp_path / 't.treefile'
    treefile.write_text('((a_1,b)Node2,c)Node1;')
    statefile = tmp_path / 't.state'
    lines = ['Node\tSite\tState\tp_A\tp_C\tp_G\tp_T']
    for node in ('Node1', 'Node2'):
        for site, base in enumerate('ACGT', start=1):
            probs = ['0.01', '0.01', '0.01', '0.01']
            probs['ACGT'.index(base)] = '0.97'
            lines.append(f'{node}\t{site}\t{base}\t' + '\t'.join(probs))
    statefile.write_text('\n'.join(lines) + '\n')
    # 'a:1' and 'a_1' both sanitise to 'a_1'.
    aln = MultipleSeqAlignment(
        [
            SeqRecord(Seq('ACGT'), id='a:1'),
            SeqRecord(Seq('ATGT'), id='a_1'),
            SeqRecord(Seq('ACGA'), id='b'),
            SeqRecord(Seq('ACGA'), id='c'),
        ]
    )
    with pytest.raises(ValueError, match='collide'):
        build_reconstruction(str(treefile), str(statefile), aln, rooting='none')


def test_build_reconstruction_leaf_without_sequence_raises(tmp_path):
    """A tree leaf absent from the alignment is rejected."""
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    treefile = tmp_path / 't.treefile'
    treefile.write_text('((a,ghost)Node2,c)Node1;')
    statefile = tmp_path / 't.state'
    lines = ['Node\tSite\tState\tp_A\tp_C\tp_G\tp_T']
    for node in ('Node1', 'Node2'):
        for site, base in enumerate('ACGT', start=1):
            probs = ['0.01', '0.01', '0.01', '0.01']
            probs['ACGT'.index(base)] = '0.97'
            lines.append(f'{node}\t{site}\t{base}\t' + '\t'.join(probs))
    statefile.write_text('\n'.join(lines) + '\n')
    aln = MultipleSeqAlignment(
        [
            SeqRecord(Seq('ACGT'), id='a'),
            SeqRecord(Seq('ATGT'), id='b'),  # 'ghost' is in the tree, not here
            SeqRecord(Seq('ACGA'), id='c'),
        ]
    )
    with pytest.raises(ValueError, match='does not match any alignment sequence'):
        build_reconstruction(str(treefile), str(statefile), aln, rooting='none')


def test_choose_root_name_outgroup_not_found():
    """Outgroup rooting names a tip that is not in the tree."""
    from derip2.spectra.tree_asr import _choose_root_name, _load_tree

    tree = _load_tree(TREEFILE)
    with pytest.raises(ValueError, match='not found in the tree'):
        _choose_root_name(tree, 'outgroup', 'no-such-tip')


def test_choose_root_name_outgroup_mrca():
    """A multi-tip outgroup roots at the tips' common ancestor."""
    from derip2.spectra.tree_asr import _choose_root_name, _load_tree

    tree = _load_tree(TREEFILE)
    root_name, method = _choose_root_name(tree, 'outgroup', ['Seq1', 'Seq2'])
    assert method.startswith('outgroup-mrca:')
    assert root_name  # an internal node name


def test_orient_edges_duplicate_name_raises(tmp_path):
    """A tree with a repeated node name is ambiguous and rejected."""
    from derip2.spectra.tree_asr import _load_tree, _orient_edges

    treefile = tmp_path / 'dup.treefile'
    # Two leaves both named 'x'.
    treefile.write_text('((x,x)Node2,c)Node1;')
    tree = _load_tree(str(treefile))
    with pytest.raises(ValueError, match='Duplicate node name'):
        _orient_edges(tree, 'Node1')


def test_iqtree_version_unknown_on_bad_binary():
    """A binary that cannot be executed yields the 'unknown' version banner."""
    from derip2.spectra.tree_asr import iqtree_version

    assert iqtree_version('definitely-not-iqtree-xyz') == 'unknown'


def test_run_iqtree_nonzero_exit_raises(monkeypatch, tmp_path):
    """A non-zero IQ-TREE exit is surfaced as a RuntimeError."""
    import subprocess

    from derip2.spectra import tree_asr

    monkeypatch.setattr(tree_asr, 'find_iqtree', lambda binary=None: 'fake-iqtree')
    monkeypatch.setattr(tree_asr, 'iqtree_version', lambda binary: 'fake 1.0')

    class _Result:
        returncode = 1
        stderr = 'boom'

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _Result())
    with pytest.raises(RuntimeError, match='exited with status 1'):
        tree_asr.run_iqtree(str(tmp_path / 'aln.fa'), str(tmp_path / 'out'))


def test_run_iqtree_missing_output_raises(monkeypatch, tmp_path):
    """A zero exit that leaves no .treefile is surfaced as FileNotFoundError."""
    import subprocess

    from derip2.spectra import tree_asr

    monkeypatch.setattr(tree_asr, 'find_iqtree', lambda binary=None: 'fake-iqtree')
    monkeypatch.setattr(tree_asr, 'iqtree_version', lambda binary: 'fake 1.0')

    class _Result:
        returncode = 0
        stderr = ''

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: _Result())
    with pytest.raises(FileNotFoundError, match='was not produced'):
        tree_asr.run_iqtree(str(tmp_path / 'aln.fa'), str(tmp_path / 'out'))


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


@pytest.mark.skipif(not _HAVE_IQTREE, reason='IQ-TREE not on PATH')
def test_reconstruct_fixed_tree(tmp_path):
    """A user-supplied topology is used while ancestral states are recomputed."""
    from derip2.aln_ops import loadAlign
    from derip2.spectra.tree_asr import reconstruct

    aln = loadAlign(MINTEST)
    # Reconstruct ancestral states on the committed fixture topology.
    rec = reconstruct(
        aln,
        str(tmp_path / 'fixed'),
        model='JC',
        threads='1',
        rooting='none',
        tree=TREEFILE,
    )
    assert rec.manifest['user_tree'] == TREEFILE
    assert len(rec.edges) == 9
    # Every internal node still has a reconstructed sequence on the fixed tree.
    res = compute_spectra_from_tree(rec)
    assert res.method == 'phylogenetic'
