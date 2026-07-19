"""Draw a tanglegram comparing two phylogenies with ete4 + matplotlib.

Loads two Newick trees, ladderises each, lays them out facing one another, and
draws curved connector lines between equivalent tips. Connectors are coloured by
how far a tip moves between the two trees (its leaf-order rank change), so
entangled crossings -- clades reshaped by RIP homoplasy -- stand out.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from ete4 import Tree


def _load(path):
    with open(path) as fh:
        return Tree(fh.read())


def _ladderise_layout(tree):
    """Return {leaf_name: y} and node (x, y) coords; x = depth, y = leaf order."""
    for node in tree.traverse():
        node.children.sort(key=lambda c: len(list(c.leaves())))

    leaves = list(tree.leaves())
    yof = {leaf.name: i for i, leaf in enumerate(leaves)}
    xy = {}

    def place(node, depth):
        if node.is_leaf:
            y = yof[node.name]
        else:
            ys = [place(c, depth + (c.dist or 0.0)) for c in node.children]
            y = sum(ys) / len(ys)
        xy[node] = (depth, y)
        return y

    place(tree, 0.0)
    return yof, xy


def _draw_scaled(ax, tree, xy, scale, offset, flip):
    """Draw branches with x normalised to a fixed display width."""
    depth = max(x for x, _ in xy.values()) or 1.0

    def dx(x):
        f = (x / depth) * scale
        return offset - f if flip else offset + f

    for node in tree.traverse():
        px = dx(xy[node][0])
        _, y = xy[node]
        for child in node.children:
            cpx = dx(xy[child][0])
            cy = xy[child][1]
            ax.plot([px, px], [y, cy], color='0.35', lw=0.8)
            ax.plot([px, cpx], [cy, cy], color='0.35', lw=0.8)


def draw_tanglegram(
    tree1_path,
    tree2_path,
    outfile=None,
    labelmap=None,
    gap=1.3,
    treew=1.0,
    title=None,
    dpi=150,
):
    t1, t2 = _load(tree1_path), _load(tree2_path)
    y1, xy1 = _ladderise_layout(t1)
    y2, xy2 = _ladderise_layout(t2)
    n = max(len(y1), len(y2))

    # Normalise both trees to the same display width: RIP-masking shortens the
    # masked tree's branch lengths, so raw scale would collapse it.
    left_tip = treew
    right_tip = treew + gap
    right_root = treew + gap + treew

    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.30)), dpi=dpi)
    _draw_scaled(ax, t1, xy1, scale=treew, offset=0.0, flip=False)
    _draw_scaled(ax, t2, xy2, scale=treew, offset=right_root, flip=True)

    rankspan = max(1, n - 1)
    cmap = plt.get_cmap('viridis')
    for name, ya in y1.items():
        if name not in y2:
            continue
        yb = y2[name]
        col = cmap(0.12 + 0.83 * abs(ya - yb) / rankspan)
        xs = np.linspace(left_tip, right_tip, 40)
        t = (xs - left_tip) / (right_tip - left_tip)
        ys = ya + (yb - ya) * (t * t * (3 - 2 * t))  # smoothstep
        ax.plot(xs, ys, color=col, lw=1.2, alpha=0.9)

    mid = (left_tip + right_tip) / 2
    for name, ya in y1.items():
        lab = labelmap.get(name, name) if labelmap else name
        ax.text(
            mid,
            ya,
            lab,
            va='center',
            ha='center',
            fontsize=6,
            color='0.15',
            bbox={
                'boxstyle': 'round,pad=0.15',
                'fc': 'white',
                'ec': 'none',
                'alpha': 0.75,
            },
        )

    ax.set_xlim(-0.05 * right_root, right_root * 1.05)
    ax.set_ylim(-1, n + 0.5)
    ax.axis('off')
    ax.text(
        0.0,
        n,
        'topology from RAW alignment',
        fontsize=9,
        fontweight='bold',
        ha='left',
        va='bottom',
    )
    ax.text(
        right_root,
        n,
        'topology from RIP-MASKED alignment',
        fontsize=9,
        fontweight='bold',
        ha='right',
        va='bottom',
    )
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    return fig
