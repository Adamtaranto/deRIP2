"""
Phylogeny and ancestral-state reconstruction for the mutation-spectrum pipeline.

This module wraps IQ-TREE to infer a maximum-likelihood tree and marginal
ancestral sequences, then roots the tree and orients every edge parent -> child
so that substitutions can be polarised. It is the only place in deRIP2 that shells
out to an external binary; all subprocess handling is isolated here.

The workflow is:

1. :func:`run_iqtree` runs ``iqtree --ancestral`` on a written alignment, producing
   ``<prefix>.treefile`` (Newick, named internal nodes) and ``<prefix>.state``
   (per-internal-node marginal posteriors).
2. :func:`parse_state` reads the ``.state`` file into per-node ancestral sequences
   and per-site probabilities.
3. :func:`build_reconstruction` loads the tree with ete4, roots it, orients edges
   away from the root, and assembles a :class:`TreeReconstruction` giving every
   node a sequence (tips from the alignment, internal nodes from the ``.state``).

IQ-TREE must be on ``PATH`` as ``iqtree3``, ``iqtree2`` or ``iqtree``; ete4 must be
installed (the optional ``spectra`` dependency group).
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Candidate IQ-TREE executable names, most recent first.
_IQTREE_BINARIES = ('iqtree3', 'iqtree2', 'iqtree')

# IQ-TREE keeps only these characters in taxon names; everything else becomes '_'.
_NAME_UNSAFE = re.compile(r'[^A-Za-z0-9._-]')


def _sanitize_name(name: str) -> str:
    """
    Apply IQ-TREE's taxon-name sanitisation to a sequence id.

    IQ-TREE replaces every character outside ``[A-Za-z0-9._-]`` with an
    underscore in its tree and state outputs, so a FASTA id such as
    ``UNSE01.1:1-9(-)`` is written as ``UNSE01.1_1-9_-_``. Reproducing that rule
    lets tree leaves be matched back to alignment sequences.

    Parameters
    ----------
    name : str
        The original sequence id.

    Returns
    -------
    str
        The sanitised name as IQ-TREE would write it.
    """
    return _NAME_UNSAFE.sub('_', name)


def find_iqtree(binary: Optional[str] = None) -> str:
    """
    Locate an IQ-TREE executable on ``PATH``.

    Parameters
    ----------
    binary : str, optional
        An explicit executable name or path to use. When ``None`` the known
        IQ-TREE names are tried in order (``iqtree3``, ``iqtree2``, ``iqtree``).

    Returns
    -------
    str
        The resolved path to the executable.

    Raises
    ------
    FileNotFoundError
        If no IQ-TREE executable can be found.
    """
    candidates = [binary] if binary else list(_IQTREE_BINARIES)
    for name in candidates:
        resolved = shutil.which(name)
        if resolved:
            logger.debug('Using IQ-TREE executable: %s', resolved)
            return resolved
    raise FileNotFoundError(
        'IQ-TREE was not found on PATH. Install it (e.g. '
        '`conda install -c bioconda iqtree`) and ensure one of '
        f'{", ".join(_IQTREE_BINARIES)} is callable, or pass an explicit path.'
    )


def iqtree_version(binary: str) -> str:
    """
    Return the version banner of an IQ-TREE executable.

    Parameters
    ----------
    binary : str
        Path to the IQ-TREE executable.

    Returns
    -------
    str
        The first non-empty line of ``iqtree --version``, or ``'unknown'`` if it
        cannot be determined.
    """
    try:
        out = subprocess.run(
            [binary, '--version'], capture_output=True, text=True, timeout=60
        )
        for line in out.stdout.splitlines():
            if line.strip():
                return line.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return 'unknown'


def run_iqtree(
    alignment_path: str,
    prefix: str,
    *,
    model: str = 'MFP',
    threads: str = 'AUTO',
    fixed_tree: Optional[str] = None,
    binary: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Run IQ-TREE with marginal ancestral state reconstruction.

    Parameters
    ----------
    alignment_path : str
        Path to the input alignment (FASTA).
    prefix : str
        Output prefix; IQ-TREE writes ``<prefix>.treefile``, ``<prefix>.state``
        and friends.
    model : str, optional
        Substitution model passed to ``-m`` (default ``'MFP'``, ModelFinder Plus).
    threads : str, optional
        Value for ``-T`` (default ``'AUTO'``).
    fixed_tree : str, optional
        Path to a user tree. When given it is passed via ``-te`` so IQ-TREE
        reconstructs ancestral states on that fixed topology instead of inferring
        a new one.
    binary : str, optional
        Explicit IQ-TREE executable; otherwise auto-detected.
    extra_args : list of str, optional
        Additional command-line arguments appended verbatim.

    Returns
    -------
    dict
        Paths of the key outputs: ``treefile``, ``state``, ``iqtree`` (the report)
        and ``binary`` / ``version`` used.

    Raises
    ------
    FileNotFoundError
        If IQ-TREE is not found, or an expected output file is missing.
    RuntimeError
        If IQ-TREE exits with a non-zero status.
    """
    executable = find_iqtree(binary)
    version = iqtree_version(executable)

    cmd = [
        executable,
        '-s',
        alignment_path,
        '-m',
        model,
        '--ancestral',
        '-T',
        threads,
        '--prefix',
        prefix,
        '-redo',
    ]
    # A fixed topology (``-te``) constrains the tree while the model, branch
    # lengths and ancestral states are still estimated from ``alignment_path``.
    # This is what the RIP-masked-topology workflow needs: infer the topology
    # from the masked alignment, then reconstruct ancestral states for the
    # unmasked sequences on that same topology.
    if fixed_tree is not None:
        cmd += ['-te', fixed_tree]
    if extra_args:
        cmd += list(extra_args)

    logger.info('Running IQ-TREE (%s): %s', version, ' '.join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        logger.error('IQ-TREE stderr:\n%s', completed.stderr)
        raise RuntimeError(
            f'IQ-TREE exited with status {completed.returncode}. '
            'See the log above for details.'
        )

    outputs = {
        'treefile': f'{prefix}.treefile',
        'state': f'{prefix}.state',
        'iqtree': f'{prefix}.iqtree',
        'binary': executable,
        'version': version,
    }
    for key in ('treefile', 'state'):
        if not os.path.exists(outputs[key]):
            raise FileNotFoundError(
                f'Expected IQ-TREE output {outputs[key]} was not produced.'
            )
    return outputs


# Base-code lookup shared with the spectrum assembler.
_CODE_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate((b'A', b'C', b'G', b'T')):
    _CODE_LUT[_b[0]] = _i
    _CODE_LUT[_b.lower()[0]] = _i


def parse_state(
    state_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    """
    Parse an IQ-TREE ``.state`` file into per-node sequences and probabilities.

    Parameters
    ----------
    state_path : str
        Path to the ``.state`` file.

    Returns
    -------
    tuple
        ``(node_seq, node_prob, n_sites)`` where ``node_seq`` maps each internal
        node name to a ``(n_sites,)`` ``'S1'`` array of most-likely bases, and
        ``node_prob`` maps it to a ``(n_sites,)`` float array of the posterior
        probability of that base. ``n_sites`` is the alignment width.

    Raises
    ------
    ValueError
        If the file has no data rows or a node has an unexpected number of sites.
    """
    import pandas as pd

    df = pd.read_csv(state_path, sep='\t', comment='#')
    if df.empty:
        raise ValueError(f'No ancestral states found in {state_path}')

    n_sites = int(df['Site'].max())
    state_char = df['State'].to_numpy(dtype='U1')
    state_bytes = np.array([c.encode('ascii') for c in state_char], dtype='S1')
    state_code = _CODE_LUT[state_bytes.view(np.uint8)]

    pcols = df[['p_A', 'p_C', 'p_G', 'p_T']].to_numpy(dtype=np.float64)
    # Probability of the reconstructed base; fall back to the row max where the
    # state is a gap or ambiguous character (code < 0).
    row = np.arange(len(df))
    safe_code = np.where(state_code >= 0, state_code, 0)
    prob = pcols[row, safe_code]
    prob = np.where(state_code >= 0, prob, pcols.max(axis=1))

    node_seq: Dict[str, np.ndarray] = {}
    node_prob: Dict[str, np.ndarray] = {}
    nodes = df['Node'].to_numpy()
    sites = df['Site'].to_numpy()
    # IQ-TREE writes node-major, site-ascending blocks; group by node preserving
    # order and place each row at its (site - 1) index defensively.
    for node in dict.fromkeys(nodes):  # ordered unique
        mask = nodes == node
        seq = np.full(n_sites, b'-', dtype='S1')
        pr = np.zeros(n_sites, dtype=np.float64)
        idx = sites[mask] - 1
        if idx.min() < 0 or idx.max() >= n_sites:
            raise ValueError(f'Node {node} has out-of-range sites in {state_path}')
        seq[idx] = state_bytes[mask]
        pr[idx] = prob[mask]
        node_seq[str(node)] = seq
        node_prob[str(node)] = pr

    logger.debug(
        'Parsed %d internal nodes x %d sites from %s',
        len(node_seq),
        n_sites,
        state_path,
    )
    return node_seq, node_prob, n_sites


@dataclass
class TreeReconstruction:
    """
    A rooted, ancestrally-reconstructed tree ready for branch traversal.

    Attributes
    ----------
    edges : list of tuple of str
        Directed ``(parent_name, child_name)`` edges, oriented away from the root.
    node_seq : dict of str to numpy.ndarray
        Every node name (tips and internal) mapped to its ``(n_cols,)`` ``'S1'``
        sequence.
    node_prob : dict of str to numpy.ndarray
        Every node name mapped to its ``(n_cols,)`` per-site state probability
        (tips are all 1.0).
    root_name : str
        Name of the node chosen as the root.
    tip_names : list of str
        Names of the leaf nodes (alignment sequences).
    n_cols : int
        Alignment width.
    manifest : dict
        Provenance: rooting method, outgroup, IQ-TREE version, node counts, etc.
    """

    edges: List[Tuple[str, str]]
    node_seq: Dict[str, np.ndarray]
    node_prob: Dict[str, np.ndarray]
    root_name: str
    tip_names: List[str]
    n_cols: int
    manifest: dict = field(default_factory=dict)


def _load_tree(treefile: str):
    """
    Load a Newick tree with ete4, keeping internal node names.

    Parameters
    ----------
    treefile : str
        Path to the ``.treefile``.

    Returns
    -------
    ete4.Tree
        The parsed tree.
    """
    from ete4 import Tree

    with open(treefile) as handle:
        newick = handle.read().strip()
    # parser=1 reads internal-node labels as names (not support values), which is
    # how IQ-TREE writes NodeN labels.
    return Tree(newick, parser=1)


def _choose_root_name(tree, rooting: str, outgroup) -> Tuple[str, str]:
    """
    Choose an existing node to root at, avoiding phantom root creation.

    Rooting *at a node* (rather than mid-branch) keeps every node backed by a
    reconstructed sequence, so no edge is lost to an unsequenced root. Midpoint
    rooting therefore roots at the node just below the midpoint branch.

    Parameters
    ----------
    tree : ete4.Tree
        The loaded tree.
    rooting : {'outgroup', 'midpoint', 'none'}
        Rooting strategy.
    outgroup : str or list of str or None
        Outgroup tip name(s), required when ``rooting == 'outgroup'``.

    Returns
    -------
    tuple of str
        ``(root_name, resolved_method)`` where ``resolved_method`` records what
        was actually used (it may differ from the request on fallback).

    Raises
    ------
    ValueError
        If ``rooting == 'outgroup'`` but no valid outgroup is supplied.
    """
    if rooting == 'outgroup':
        if not outgroup:
            raise ValueError("rooting='outgroup' requires an outgroup name")
        names = [outgroup] if isinstance(outgroup, str) else list(outgroup)
        if len(names) == 1:
            node = next(tree.search_nodes(name=names[0]), None)
            if node is None:
                raise ValueError(f'Outgroup {names[0]!r} not found in the tree')
            return names[0], f'outgroup:{names[0]}'
        mrca = tree.common_ancestor(names)
        return mrca.name, f'outgroup-mrca:{",".join(names)}'

    if rooting == 'midpoint':
        node = tree.get_midpoint_outgroup()
        # get_midpoint_outgroup returns the node below the midpoint branch; root
        # there so the root keeps a reconstructed sequence.
        if node is not None and node.name:
            return node.name, 'midpoint'
        logger.warning('Midpoint node has no name; falling back to arbitrary root')

    # Arbitrary: use the current top of the tree.
    return tree.name, 'arbitrary'


def _orient_edges(tree, root_name: str) -> List[Tuple[str, str]]:
    """
    Orient every edge away from ``root_name`` by breadth-first search.

    Parameters
    ----------
    tree : ete4.Tree
        The loaded tree.
    root_name : str
        Name of the node to treat as the root.

    Returns
    -------
    list of tuple of str
        Directed ``(parent, child)`` edges.

    Raises
    ------
    ValueError
        If the tree contains duplicate node names (which would make the
        undirected adjacency ambiguous).
    """
    # Build an undirected adjacency keyed by node name.
    adjacency: Dict[str, List[str]] = {}
    seen = set()
    for node in tree.traverse():
        name = node.name
        if name in seen:
            raise ValueError(f'Duplicate node name {name!r} in tree')
        seen.add(name)
        neighbours = [c.name for c in node.children]
        if node.up is not None:
            neighbours.append(node.up.name)
        adjacency[name] = neighbours

    edges: List[Tuple[str, str]] = []
    visited = {root_name}
    queue = deque([root_name])
    while queue:
        parent = queue.popleft()
        for child in adjacency[parent]:
            if child not in visited:
                visited.add(child)
                edges.append((parent, child))
                queue.append(child)
    return edges


def build_reconstruction(
    treefile: str,
    state_path: str,
    alignment,
    *,
    rooting: str = 'midpoint',
    outgroup=None,
    manifest_extra: Optional[dict] = None,
) -> TreeReconstruction:
    """
    Assemble a rooted, oriented, ancestrally-reconstructed tree.

    Parameters
    ----------
    treefile : str
        Path to the IQ-TREE ``.treefile``.
    state_path : str
        Path to the IQ-TREE ``.state`` file.
    alignment : Bio.Align.MultipleSeqAlignment
        The alignment IQ-TREE was run on; supplies tip sequences.
    rooting : {'midpoint', 'outgroup', 'none'}, optional
        How to root the tree (default ``'midpoint'``).
    outgroup : str or list of str, optional
        Outgroup tip name(s), required when ``rooting == 'outgroup'``.
    manifest_extra : dict, optional
        Extra key/values to record in the reconstruction manifest.

    Returns
    -------
    TreeReconstruction
        The rooted tree with a sequence for every node.

    Raises
    ------
    ValueError
        If a tree node has no reconstructed or observed sequence, or the
        alignment width does not match the ``.state`` sites.
    """
    from derip2.aln_ops import alignment_to_array

    node_seq, node_prob, n_sites = parse_state(state_path)

    arr = alignment_to_array(alignment)
    if arr.shape[1] != n_sites:
        raise ValueError(
            f'Alignment width {arr.shape[1]} does not match .state sites {n_sites}'
        )

    tree = _load_tree(treefile)
    leaf_names = [leaf.name for leaf in tree.leaves()]

    # IQ-TREE sanitises tip names in its outputs (every character outside
    # [A-Za-z0-9._-] becomes '_'), so a header like 'seq:1(-)' appears as
    # 'seq_1_-_' in the tree. Map each tree leaf back to the alignment row via the
    # same sanitisation so tip sequences attach to the right node.
    sanitized_to_id = {}
    for record in alignment:
        key = _sanitize_name(record.id)
        if key in sanitized_to_id:
            raise ValueError(
                f'Alignment ids {sanitized_to_id[key]!r} and {record.id!r} collide '
                'after IQ-TREE name sanitisation; rename the sequences.'
            )
        sanitized_to_id[key] = record.id
    id_to_row = {record.id: i for i, record in enumerate(alignment)}

    tip_names = []
    for leaf in leaf_names:
        original = sanitized_to_id.get(leaf, leaf)
        if original not in id_to_row:
            raise ValueError(
                f'Tree leaf {leaf!r} does not match any alignment sequence id'
            )
        node_seq[leaf] = arr[id_to_row[original]]
        node_prob[leaf] = np.ones(n_sites, dtype=np.float64)
        tip_names.append(leaf)

    root_name, resolved_method = _choose_root_name(tree, rooting, outgroup)
    edges = _orient_edges(tree, root_name)

    # Every node touched by an edge must have a sequence.
    for parent, child in edges:
        for name in (parent, child):
            if name not in node_seq:
                raise ValueError(
                    f'Tree node {name!r} has no reconstructed or observed sequence'
                )

    manifest = {
        'rooting_requested': rooting,
        'rooting_used': resolved_method,
        'outgroup': outgroup,
        'root': root_name,
        'n_tips': len(tip_names),
        'n_internal_nodes': len(node_seq) - len(tip_names),
        'n_edges': len(edges),
        'n_cols': n_sites,
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    logger.info(
        'Reconstruction rooted at %s (%s): %d tips, %d edges',
        root_name,
        resolved_method,
        len(tip_names),
        len(edges),
    )
    return TreeReconstruction(
        edges=edges,
        node_seq=node_seq,
        node_prob=node_prob,
        root_name=root_name,
        tip_names=tip_names,
        n_cols=n_sites,
        manifest=manifest,
    )


def reconstruct(
    alignment,
    work_prefix: str,
    *,
    model: str = 'MFP',
    threads: str = 'AUTO',
    rooting: str = 'midpoint',
    outgroup=None,
    tree: Optional[str] = None,
    binary: Optional[str] = None,
) -> TreeReconstruction:
    """
    Run IQ-TREE on an alignment and build a rooted reconstruction in one call.

    Parameters
    ----------
    alignment : Bio.Align.MultipleSeqAlignment
        The alignment to analyse.
    work_prefix : str
        Prefix for the written alignment and all IQ-TREE outputs.
    model : str, optional
        Substitution model for ``-m`` (default ``'MFP'``).
    threads : str, optional
        Value for IQ-TREE ``-T`` (default ``'AUTO'``). ``'AUTO'`` benchmarks the
        best thread count, which adds noticeable overhead on tiny alignments;
        pass a fixed integer string (e.g. ``'1'``) to skip it.
    rooting : {'midpoint', 'outgroup', 'none'}, optional
        Rooting strategy (default ``'midpoint'``).
    outgroup : str or list of str, optional
        Outgroup tip name(s) when ``rooting == 'outgroup'``.
    tree : str, optional
        Path to a fixed user tree; passed to IQ-TREE via ``-te`` so ancestral
        states are reconstructed on that topology.
    binary : str, optional
        Explicit IQ-TREE executable.

    Returns
    -------
    TreeReconstruction
        The rooted, oriented, ancestrally-reconstructed tree.
    """
    from Bio import AlignIO

    aln_path = f'{work_prefix}.aln.fasta'
    AlignIO.write(alignment, aln_path, 'fasta')

    outputs = run_iqtree(
        aln_path,
        work_prefix,
        model=model,
        threads=threads,
        fixed_tree=tree,
        binary=binary,
    )
    return build_reconstruction(
        outputs['treefile'],
        outputs['state'],
        alignment,
        rooting=rooting,
        outgroup=outgroup,
        manifest_extra={
            'iqtree_binary': outputs['binary'],
            'iqtree_version': outputs['version'],
            'model': model,
            'user_tree': tree,
        },
    )


def assign_clades(reconstruction: TreeReconstruction) -> Dict[str, str]:
    """
    Assign every non-root node to the clade of its root-child ancestor.

    Each subtree hanging directly off the root is one clade, named by its
    root-child node. This yields the ``samples_by_child`` mapping used to
    partition the spectra by lineage.

    Parameters
    ----------
    reconstruction : TreeReconstruction
        The rooted reconstruction.

    Returns
    -------
    dict of str to str
        Maps each non-root node name to its clade label.
    """
    children_of: Dict[str, List[str]] = {}
    for parent, child in reconstruction.edges:
        children_of.setdefault(parent, []).append(child)

    node_to_clade: Dict[str, str] = {}
    for root_child in children_of.get(reconstruction.root_name, []):
        stack = [root_child]
        while stack:
            node = stack.pop()
            node_to_clade[node] = root_child
            stack.extend(children_of.get(node, []))
    return node_to_clade


def orientation_flip_fraction(
    edges_a: List[Tuple[str, str]], edges_b: List[Tuple[str, str]]
) -> float:
    """
    Fraction of shared edges whose parent -> child direction differs.

    Parameters
    ----------
    edges_a, edges_b : list of tuple of str
        Two directed edge lists over the same undirected tree (e.g. under two
        rooting choices).

    Returns
    -------
    float
        Proportion of undirected edges present in both lists that are oriented in
        opposite directions. ``0.0`` when there are no shared edges.
    """
    dir_a = set(edges_a)
    shared = 0
    flipped = 0
    for parent, child in edges_b:
        if (parent, child) in dir_a or (child, parent) in dir_a:
            shared += 1
            if (child, parent) in dir_a:
                flipped += 1
    return flipped / shared if shared else 0.0
