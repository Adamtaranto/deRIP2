# Mutation spectra (SBS-96 / SBS-192)

RIP is one of several processes that deaminate cytosine in fungal repeats. To tell
them apart you need more than a count of C→T changes — you need the **sequence
context** each change happened in. `derip2-spectra` builds the standard
single-base-substitution spectra used in mutational-signature analysis:

- **SBS-96** — every substitution folded onto the pyrimidine strand, classified by
  its 5′ and 3′ neighbours: 6 substitution types × 16 contexts = 96 channels.
- **SBS-192** — the strand-resolved form (12 types × 16 contexts) that keeps the
  reference base as observed on the coding strand, so strand asymmetries stay
  visible.

RIP (CpA → TpA) shows up as a sharp `C>T` peak concentrated in `NCA` contexts:

![SBS-96 spectrum of the Sahana transposon](../img/spectra_sbs96.png)

The matrices are written in SigProfiler-compliant format, so they drop straight
into `SigProfilerPlotting` / `SigProfilerAssignment` if you want to decompose them
against COSMIC signatures later.

## Quick start

```bash
derip2-spectra -i family.fasta -d out -p family
```

This writes, into `out/`:

| File | Contents |
|---|---|
| `family.SBS96.txt`, `family.SBS192.txt` | SigProfiler-compliant count matrices |
| `family_SBS96.png`, `family_SBS192.png` | spectrum bar plots |
| `family_strand_asymmetry.png` | coding- vs template-strand counts per class |
| `family_homoplasy.png`, `family_homoplasy.tsv` | recurrently-hit sites |
| `family_events.tsv` | one row per called substitution |

## Two methods: baseline and phylogenetic

Direction and recurrence are not free from an alignment — you need an ancestor. The
tool offers two ways to get one.

### Baseline (`--method baseline`, the default)

Every sequence is compared to a **single reference** — deRIP2's reconstructed
ancestral consensus — and each difference is one event, with its context read from
that ancestor. It needs no tree and no external tools.

Its blind spot is **recurrence**. If the same C→T deamination struck independently
on many lineages, comparing every tip to one reference records the derived state on
each tip and cannot tell "one ancestral event inherited by twenty tips" from
"twenty independent events". The baseline therefore *over-counts* homoplasic sites
and reports recurrence only as a *multi-hit-column* proxy.

### Phylogenetic (`--method phylo`)

The rigorous path reconstructs ancestral sequences at every internal node of a tree
(IQ-TREE marginal ASR) and walks every parent→child branch, logging each
substitution as an independent event with its context read from the **parent**
sequence — the state at the moment the mutation occurred.

The difference is large and biological. On the full Sahana family (396 copies):

| | Events (SBS-96) | Interpretation |
|---|---|---|
| Baseline | 326,778 | every tip vs one reference |
| Phylogenetic | ~47,000 | independent branch events |

The baseline's extra ~280,000 "events" are shared/inherited RIP mutations counted
once per descendant. The phylogenetic path assigns each to the single branch it
arose on — and flags the sites that really were hit again and again:

![SBS-96 spectrum, phylogenetic](../img/spectra_sbs96_phylo.png)

!!! warning "Read direction with care on heavily-RIP'd families"
    Notice the `T>C` peak in the phylogenetic spectrum, absent from the baseline.
    It is largely an artefact of **maximum-likelihood ancestral reconstruction**,
    which is biased toward the majority state. When RIP has converted *most* copies
    of a column from C to T, IQ-TREE reconstructs the internal nodes as the
    majority `T`, so the minority copies that *retained* the ancestral C read as
    `T>C` reversals. The **recurrence counting is still correct** (46,952 vs
    326,778 events) — this bias affects the inferred *direction*, not the event
    count.

    deRIP2's consensus-based **baseline is more robust to RIP direction**: its
    ancestor is the deRIP'd (un-RIP'd) sequence, recovered by finding the copies
    that escaped RIP even when they are the minority. Use the two together — the
    baseline for polarity, the phylogenetic path for correct recurrence — and lean
    on the masked-topology workflow below to keep the tree itself honest.

!!! note "IQ-TREE is required for `--method phylo`"
    Install it separately (`conda install -c bioconda iqtree`) and make sure
    `iqtree3`, `iqtree2` or `iqtree` is on your `PATH`. `ete4` (a Python
    dependency) handles the tree.

```bash
# Infer the tree and reconstruct ancestors in one step
derip2-spectra -i family.fasta --method phylo -d out -p family
```

## Supplying your own phylogeny

You will usually get a better tree from a dedicated run than from the built-in
one-shot inference. Pass any Newick tree with `--tree`; IQ-TREE then keeps that
**topology fixed** (`-te`) and re-estimates the model, branch lengths and ancestral
states from your alignment.

```bash
# Build a well-supported tree however you like...
iqtree3 -s family.fasta -m MFP -B 1000 -T AUTO --prefix family_tree

# ...then reconstruct ancestral states on it and call the spectrum
derip2-spectra -i family.fasta --method phylo --tree family_tree.treefile \
    -d out -p family
```

!!! tip "Tip names"
    Tree leaf names must correspond to the FASTA sequence ids. IQ-TREE rewrites
    characters outside `[A-Za-z0-9._-]` to `_` in its output (so `scf:1-9(+)`
    becomes `scf_1-9___`); deRIP2 applies the same rule to match leaves back to
    sequences, so trees produced by IQ-TREE Just Work. If you supply a tree from
    another tool, keep tip names to those safe characters.

## Recommended: infer topology from a RIP-masked alignment

RIP mutations are **convergent** — the same CpA→TpA change happens independently in
many copies. To a tree-builder that convergence looks like shared ancestry, so a
tree built directly from RIP-riddled repeats can be pulled out of shape, grouping
copies by *how much they were RIP'd* rather than by their true history. That
distorted topology would then mis-assign the very substitutions you are trying to
count.

The fix is to **infer the topology from a RIP-masked alignment**, then reconstruct
ancestral states for the **unmasked** sequences on that same topology:

```bash
# 1. Mask RIP-corrected positions (degenerate IUPAC codes); no consensus appended
derip2 -i family.fasta --mask --no-append -d out -p family
#    -> out/family_masked_alignment.fasta

# 2. Infer the topology from the masked alignment (RIP signal removed).
#    -st DNA forces the DNA model: a heavily masked alignment carries many IUPAC
#    ambiguity codes, and IQ-TREE's sequence-type auto-detection can otherwise
#    fail with "Unknown sequence type".
iqtree3 -s out/family_masked_alignment.fasta -m MFP -B 1000 -T AUTO -st DNA \
    --prefix out/family_masked

# 3. Reconstruct ancestral states for the UNMASKED sequences on that fixed
#    topology, and call the spectrum
derip2-spectra -i family.fasta --method phylo --tree out/family_masked.treefile \
    -d out -p family_spectrum
```

Why this works: masking removes the homoplasic RIP columns that mislead tree
search, giving a topology that reflects true descent. Step 3 then *fixes* that
topology (`--tree`) but re-derives branch lengths, the substitution model and the
ancestral sequences from the **full, unmasked** alignment — so the spectrum is
computed from the real substitutions while the tree shape is not an artefact of
RIP. The masked and unmasked alignments share identical tips and columns (masking
only rewrites bases in place), so the topology transfers exactly.

!!! note "Same topology, unmasked ancestors"
    The point of `--tree` here is precisely that the **topology comes from the
    masked tree** while the **ancestral sequences are recomputed for the unmasked
    data**. Do not run IQ-TREE ancestral reconstruction on the masked alignment —
    its ancestors would be masked too, and the spectrum would be blank exactly
    where RIP acted.

The resulting spectrum for the full Sahana family (topology from the masked
alignment, ancestral states from the unmasked sequences) still resolves the RIP
`C>T`/CpA signal cleanly, now on a topology that RIP homoplasy could not distort:

![SBS-96 spectrum, phylogenetic on a RIP-masked topology](../img/spectra_sbs96_maskedtopo.png)

## Reading the outputs

### Homoplasy (recurrence)

The homoplasy report lists sites hit by the same substitution on two or more
independent lineages — the explicit, measured record of recurrent deamination.
Each stem is one (column, derived base); its colour is the pyrimidine-folded
substitution class (see the legend) and its height is the number of independent
hits. Markers are drawn semi-transparent so that stems stacked at the same column
and height remain visible.

The recurrence *unit* differs by method, and the two plots make the point of the
whole feature. Under the **baseline** every hit is one sequence carrying the
derived state, so shared/inherited RIP makes almost every column look "recurrent"
— the multi-hit-column proxy saturates:

![Recurrent sites, baseline](../img/spectra_homoplasy_baseline.png)

Under **`--method phylo`** each hit is an independent *branch* event, so only sites
that truly mutated more than once on the tree remain. The plot is far sparser and
each stem is a real recurrence (here, sites hit on ≥3 independent branches):

![Recurrent sites, phylogenetic](../img/spectra_homoplasy_phylo.png)

### Strand asymmetry

From the SBS-192 matrix, each pyrimidine class is compared to its
reverse-complement partner. For every class two bars are drawn: **coding-strand**
counts (blue) and **template-strand** counts (orange) — the legend colours the two
strands, and the substitution class is read off the x-axis. An **asterisk** marks a
class whose coding-vs-template split departs from an even 50:50 at a binomial
`p < 0.05` (the figure footnote states this). This is where enzyme-, replication-
or transcription-linked strand biases show up.

![Strand asymmetry](../img/spectra_strand_asymmetry.png)

### Per-lineage spectra

`--partition-by clade` (phylo) splits the matrices into one sample column per
subtree hanging off the root, so both the matrix files and the plots carry one
panel per lineage — useful for spotting lineage-specific deamination. Each branch
is attributed to the clade its child subtree belongs to; branches above the split
form the ancestral trunk.

```bash
derip2-spectra -i family.fasta --method phylo --partition-by clade \
    -d out -p family
```

![Per-clade spectra](../img/spectra_clade.png)

`--partition-by row` does the analogous per-sequence split for the baseline.

### Per-group spectra (species or user-defined sets)

When you already know which sequences belong together — species, populations,
sub-families — pass a two-column mapping with `--groups` to get one spectrum per
group. This works for **both** methods.

The mapping file is whitespace- or tab-separated: sequence name, then group label.
A header row and `#` comments are optional:

```text
# sequence            group
UNSE01000019.1:422682-431483(-)   speciesA
UNSE01000019.1:709761-718562(-)   speciesA
UNSE01000006.1:12043-20871(+)     speciesB
Sahana_prime                      reference
```

```bash
# Baseline: one spectrum per group, each sequence compared to the deRIP ancestor
derip2-spectra -i family.fasta --groups groups.tsv -d out -p family

# Phylogenetic: a branch is attributed to a group only when its whole descendant
# clade belongs to that group (spanning branches become 'mixed')
derip2-spectra -i family.fasta --method phylo --groups groups.tsv -d out -p family
```

Names are matched leniently: the label file may use the original FASTA ids even
though IQ-TREE rewrites special characters in tree tip names, so the same file
works for both methods. Sequences absent from the map fall into an `ungrouped`
sample.

![Per-group spectra](../img/spectra_groups.png)

!!! note "Grouping in the example"
    The Sahana copies have no species labels, so this figure bins them by genomic
    scaffold purely to illustrate the mechanic. In practice the labels would be
    your species or population names, and the panels would let you compare, say,
    RIP intensity between a methylation-competent and a methylation-deficient
    lineage.

### Run manifest

The phylo path writes `*_run_manifest.json` recording the IQ-TREE version, the
model, the rooting method and the root node, node/edge counts, and — with
`--root-sensitivity` — the fraction of edges whose direction flips under midpoint
rooting. Directionality depends on the root, so record it.

## Useful options

| Option | Purpose |
|---|---|
| `--sbs {96,192,both}` | which matrices/plots to produce |
| `--rooting {midpoint,outgroup,none}` | how to root (sets substitution direction) |
| `--outgroup NAME[,NAME…]` | outgroup tip(s) for `--rooting outgroup` |
| `--iqtree-model` | IQ-TREE model (default `MFP`) |
| `--threads` | IQ-TREE `-T` (default `AUTO`; pass an integer to skip its benchmark on small alignments) |
| `--min-prob` | drop phylo events below this parent × child ancestral posterior |
| `--partition-by {none,row,clade}` | pool, per-sequence, or per-clade samples |
| `--groups FILE` | report one spectrum per user-defined group (both methods) |
| `--root-sensitivity` | report how much polarity depends on the rooting choice |
| `--ancestor FASTA` | baseline only: call against a supplied ancestor instead of the deRIP consensus |

## Decomposing against COSMIC signatures

Because the matrices are SigProfiler-compliant, you can fit them to reference
signatures. First render the standard SBS-96 figure (`SigProfilerPlotting`), then
fit COSMIC signatures with `SigProfilerAssignment` (both installed separately):

```python
from SigProfilerAssignment import Analyzer as Analyze

Analyze.cosmic_fit(
    samples='out/family.SBS96.txt',
    output='out/cosmic',
    input_type='matrix',
    context_type='96',
    cosmic_version=3.4,
    make_plots=True,
)
```

Fitting the full Sahana spectrum (326,778 events) to COSMIC v3.4 gives:

| COSMIC signature | Assigned mutations | Share | Human-cancer aetiology |
|---|---:|---:|---|
| SBS44 | 135,625 | 41.5% | defective mismatch repair |
| SBS96 | 92,238 | 28.2% | unknown |
| SBS2  | 51,312 | 15.7% | APOBEC cytosine deamination |
| SBS1  | 47,603 | 14.6% | spontaneous 5-methyl-cytosine deamination |

The reconstruction **cosine similarity is only 0.80** (a good fit is usually
> 0.9). That poor fit is itself the result: RIP has no COSMIC signature, so the
fitter approximates it with a blend of the human deamination signatures (SBS1,
the CpG 5mC-deamination clock, and SBS2, APOBEC) plus catch-all signatures.

!!! warning "COSMIC signatures do not apply directly to fungi"
    The COSMIC reference set was derived from **human cancers**, on whole-genome
    trinucleotide opportunities. RIP and fungal methylation-driven deamination are
    not represented, so a decomposition like the one above is at best a loose
    analogy — treat the assigned signatures as "the nearest human look-alikes",
    not as mechanisms operating in your fungus. The low cosine similarity is the
    honest signal of that mismatch. The same machinery would, however, work well
    against a **fungal-specific reference library** (a custom signature matrix in
    the same SBS-96 format), which is the appropriate way to interpret these
    spectra — and building one is a natural next step for the community.

Interpret single-gene or single-family fits cautiously in any case: reference
signatures assume genome-wide trinucleotide opportunities, so a normalisation
caveat applies on top of the species-mismatch one.
