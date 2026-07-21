"""
GFF3 gene annotation and RIP effect prediction over a gapped alignment.

deRIP2 reconstructs an un-RIP'd ancestral sequence for a family of aligned
repeats. When a gene model is supplied for one or more of those sequences, this
module answers a follow-on question: *what did RIP do to the protein?* It

1. parses a GFF3 file into per-sequence genes (:func:`parse_gff3`),
2. maps each gene's ungapped coordinates onto gapped alignment columns
   (:func:`ungapped_to_column_map`), so a feature lines up with the alignment,
   and
3. translates each sequence's CDS and compares it to the reconstructed ancestor,
   reporting premature stops, non-synonymous changes, frameshifts and broken
   splice sites (:func:`predict_gene_effects`, :func:`translate_cds`).

GFF3 coordinates are 1-based and inclusive, in the *ungapped* sequence's own
frame; the alignment is gapped. The coordinate map is the bridge between the two.

No new dependency is introduced: GFF3 is parsed with the standard library and
translation uses Biopython's :meth:`Bio.Seq.Seq.translate`, already a project
dependency. The genetic code defaults to the NCBI standard table (1) and is
configurable for organisms that use an alternate code.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

import numpy as np

logger = logging.getLogger(__name__)

# Feature types we retain from a GFF3. Everything else (region, match, ...) is
# ignored: only the gene hierarchy and its coding parts matter here.
_KEPT_TYPES = frozenset({'gene', 'mRNA', 'transcript', 'CDS', 'exon'})

# Canonical intron boundary dinucleotides (GT-AG rule), read 5'->3' on the
# coding strand. A RIP change that destroys either is flagged.
_DONOR = 'GT'
_ACCEPTOR = 'AG'

# Complement used when reading a minus-strand CDS off the forward alignment.
_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '-': '-'}

# Default colours for the alignment annotation track, keyed by feature type. A
# user-supplied two-column file overrides these (:func:`load_annotation_colors`).
DEFAULT_ANNOTATION_COLORS = {
    'gene': '#4a3aa7',  # violet
    'mRNA': '#2a78d6',  # blue
    'CDS': '#008300',  # green
    'exon': '#5aa469',  # light green
}
_FALLBACK_ANNOTATION_COLOR = '#898781'  # grey for unlisted types


@dataclass(frozen=True)
class Feature:
    """
    One GFF3 feature line, coordinates 1-based inclusive on the forward strand.

    Attributes
    ----------
    seqid : str
        Sequence identifier; matched against alignment record IDs.
    ftype : str
        Feature type (``'gene'``, ``'mRNA'``, ``'CDS'``, ``'exon'``, ...).
    start, end : int
        1-based inclusive bounds in the ungapped sequence's own coordinates.
    strand : str
        ``'+'`` or ``'-'`` (``'.'`` is treated as ``'+'``).
    phase : int or None
        CDS phase (0, 1 or 2) where given, else ``None``.
    attributes : dict of str to str
        Parsed column-9 key/value attributes.
    feature_id : str or None
        The ``ID`` attribute, if present.
    parent : str or None
        The ``Parent`` attribute, if present.
    """

    seqid: str
    ftype: str
    start: int
    end: int
    strand: str
    phase: Optional[int]
    attributes: Dict[str, str]
    feature_id: Optional[str]
    parent: Optional[str]


@dataclass(frozen=True)
class Gene:
    """
    A gene grouped to its coding exons, in transcription order via ``strand``.

    Attributes
    ----------
    gene_id : str
        Identifier for the gene / transcript the CDS features belong to.
    seqid : str
        Sequence identifier, matched against alignment record IDs.
    strand : str
        ``'+'`` or ``'-'``.
    cds : list of Feature
        CDS features, sorted by ``start`` ascending (forward-strand order). The
        transcription order is derived from ``strand`` when the CDS is assembled.
    """

    gene_id: str
    seqid: str
    strand: str
    cds: List[Feature] = field(default_factory=list)


@dataclass(frozen=True)
class EffectRecord:
    """
    One predicted effect of RIP on a sequence's coding sequence.

    Attributes
    ----------
    seq_id : str
        The sequence the effect was found in.
    gene_id : str
        The gene / transcript affected.
    kind : str
        One of ``'missense'``, ``'premature_stop'``, ``'frameshift'``,
        ``'splice_site'`` or ``'synonymous'``.
    aa_pos : int or None
        1-based amino-acid position of the change (``None`` for whole-CDS
        effects such as frameshifts).
    ref_aa, alt_aa : str or None
        Ancestral and observed amino acid (or splice dinucleotide) at the site.
    gapped_col : int or None
        Alignment column of the affected codon's middle base (or the splice
        site), for cross-referencing the figures.
    nt_ref, nt_alt : str or None
        Ancestral and observed nucleotide context, where meaningful.
    """

    seq_id: str
    gene_id: str
    kind: str
    aa_pos: Optional[int] = None
    ref_aa: Optional[str] = None
    alt_aa: Optional[str] = None
    gapped_col: Optional[int] = None
    nt_ref: Optional[str] = None
    nt_alt: Optional[str] = None


def _parse_attributes(column9: str) -> Dict[str, str]:
    """
    Parse a GFF3 column-9 attribute string into a dict, unescaping ``%XX``.

    Parameters
    ----------
    column9 : str
        The raw 9th column, e.g. ``'ID=cds1;Parent=mRNA1'``.

    Returns
    -------
    dict of str to str
        Attribute keys to values. Empty when the column is ``.`` or blank.
    """
    attributes: Dict[str, str] = {}
    if not column9 or column9 == '.':
        return attributes
    for field_str in column9.strip().rstrip(';').split(';'):
        if not field_str or '=' not in field_str:
            continue
        key, _, value = field_str.partition('=')
        attributes[unquote(key.strip())] = unquote(value.strip())
    return attributes


def parse_gff3(path: str) -> Dict[str, List[Gene]]:
    """
    Parse a GFF3 file into genes grouped by sequence identifier.

    CDS features are grouped by their ``Parent`` (falling back to the mRNA/gene
    ``ID`` or a synthesised key) and sorted into forward-strand order. Only the
    gene hierarchy is retained; other feature types are ignored.

    Parameters
    ----------
    path : str
        Path to the GFF3 file.

    Returns
    -------
    dict of str to list of Gene
        Mapping of sequence identifier to its genes, in first-seen order.

    Raises
    ------
    ValueError
        If a coordinate field is not an integer or start > end.

    Notes
    -----
    A gene whose CDS features disagree on strand is dropped with a warning: a
    single transcript cannot be transcribed from both strands.
    """
    # gene_key -> (seqid, strand, [CDS Feature, ...]); preserves first-seen order.
    grouped: Dict[str, Tuple[str, str, List[Feature]]] = {}
    order: List[str] = []

    with open(path, 'r', encoding='utf-8') as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            cols = line.split('\t')
            if len(cols) < 9:
                logger.warning(
                    'Skipping malformed GFF3 line %d (%d columns): %s',
                    line_no,
                    len(cols),
                    line[:60],
                )
                continue
            seqid, _source, ftype, start_s, end_s, _score, strand, phase_s, attr_s = (
                cols[:9]
            )
            if ftype not in _KEPT_TYPES:
                continue
            try:
                start, end = int(start_s), int(end_s)
            except ValueError as exc:
                raise ValueError(
                    f'GFF3 line {line_no}: non-integer coordinate'
                ) from exc
            if start > end:
                raise ValueError(f'GFF3 line {line_no}: start {start} > end {end}')
            strand = strand if strand in ('+', '-') else '+'
            phase = int(phase_s) if phase_s in ('0', '1', '2') else None
            attributes = _parse_attributes(attr_s)
            feature = Feature(
                seqid=seqid,
                ftype=ftype,
                start=start,
                end=end,
                strand=strand,
                phase=phase,
                attributes=attributes,
                feature_id=attributes.get('ID'),
                parent=attributes.get('Parent'),
            )
            if ftype != 'CDS':
                continue  # only CDS rows define the translated sequence

            key = feature.parent or feature.feature_id or f'{seqid}:{ftype}:{line_no}'
            if key not in grouped:
                grouped[key] = (seqid, strand, [])
                order.append(key)
            grouped[key][2].append(feature)

    by_seqid: Dict[str, List[Gene]] = {}
    for key in order:
        seqid, strand, cds_list = grouped[key]
        strands = {c.strand for c in cds_list}
        if len(strands) > 1:
            logger.warning('Gene %r has CDS features on both strands; skipping', key)
            continue
        cds_sorted = sorted(cds_list, key=lambda c: c.start)
        gene = Gene(gene_id=key, seqid=seqid, strand=strand, cds=cds_sorted)
        by_seqid.setdefault(seqid, []).append(gene)

    return by_seqid


def ungapped_to_column_map(row_bytes: np.ndarray) -> np.ndarray:
    """
    Map a sequence's ungapped positions to their gapped alignment columns.

    Parameters
    ----------
    row_bytes : numpy.ndarray
        A single alignment row as an ``'S1'`` byte array (one entry per column),
        e.g. ``ColumnClassification.arr[row_index]``.

    Returns
    -------
    numpy.ndarray
        Int array where element ``u`` is the column index of the ``u``-th
        non-gap base. Its length equals the ungapped sequence length, so a
        1-based GFF coordinate ``pos`` maps to column ``result[pos - 1]``.
    """
    return np.where(row_bytes != b'-')[0]


def warn_unmatched_seqids(genes_by_seqid: Dict[str, List[Gene]], alignment_ids):
    """
    Warn about GFF sequence identifiers that are absent from the alignment.

    Parameters
    ----------
    genes_by_seqid : dict of str to list of Gene
        Parsed genes keyed by sequence identifier.
    alignment_ids : iterable of str
        The alignment record identifiers.

    Returns
    -------
    list of str
        The unmatched sequence identifiers (also logged as a warning).
    """
    known = set(alignment_ids)
    unmatched = [seqid for seqid in genes_by_seqid if seqid not in known]
    if unmatched:
        logger.warning(
            'GFF3 sequence id(s) not found in alignment: %s',
            ', '.join(sorted(unmatched)),
        )
    return unmatched


def load_annotation_colors(path: str) -> Dict[str, str]:
    """
    Load a two-column ``type<TAB>hex`` annotation-colour override file.

    Parameters
    ----------
    path : str
        Path to a whitespace/tab-separated file of ``feature_type colour`` rows.
        Blank lines and ``#`` comments are ignored.

    Returns
    -------
    dict of str to str
        Feature type to colour, merged over :data:`DEFAULT_ANNOTATION_COLORS`.
    """
    colors = dict(DEFAULT_ANNOTATION_COLORS)
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                logger.warning('Ignoring malformed annotation-colour line: %s', line)
                continue
            colors[parts[0]] = parts[1]
    return colors


def build_annotation_spans(
    genes_by_seqid: Dict[str, List[Gene]],
    row_lookup: Dict[str, np.ndarray],
    colors: Optional[Dict[str, str]] = None,
) -> List[Tuple[int, int, str, str, int]]:
    """
    Project gene CDS exons onto alignment columns as stacked track spans.

    Each gene occupies its own track row; its CDS exons are drawn as separate
    coloured spans (gaps split an exon into contiguous column runs). The result
    is ready to pass to
    :func:`derip2.plotting.minialign.drawMiniAlignment` as ``annotation_track``.

    Parameters
    ----------
    genes_by_seqid : dict of str to list of Gene
        Parsed genes keyed by sequence identifier.
    row_lookup : dict of str to numpy.ndarray
        Maps each sequence identifier to its alignment row (``'S1'`` byte array),
        used to convert ungapped coordinates to columns.
    colors : dict of str to str, optional
        Feature-type colour map; defaults to :data:`DEFAULT_ANNOTATION_COLORS`.

    Returns
    -------
    list of tuple
        ``(start_col, end_col, color, label, track_row)`` spans.
    """
    colors = colors or DEFAULT_ANNOTATION_COLORS
    cds_color = colors.get('CDS', _FALLBACK_ANNOTATION_COLOR)
    spans: List[Tuple[int, int, str, str, int]] = []
    track_row = 0

    for seqid, genes in genes_by_seqid.items():
        row = row_lookup.get(seqid)
        if row is None:
            continue  # unmatched seqid, already warned elsewhere
        ungapped_to_col = ungapped_to_column_map(row)
        n_ungapped = ungapped_to_col.shape[0]
        for gene in genes:
            drew = False
            for exon in gene.cds:
                if exon.end > n_ungapped:
                    continue
                cols = ungapped_to_col[exon.start - 1 : exon.end]
                if cols.size:
                    # Label only the first exon so the gene id is not repeated.
                    label = gene.gene_id if not drew else ''
                    spans.append(
                        (int(cols.min()), int(cols.max()), cds_color, label, track_row)
                    )
                    drew = True
            if drew:
                track_row += 1

    return spans


def _cds_columns_transcription_order(
    gene: Gene, ungapped_to_col: np.ndarray
) -> Optional[List[int]]:
    """
    List the alignment columns of a gene's CDS in 5'->3' transcription order.

    Parameters
    ----------
    gene : Gene
        The gene whose CDS columns are wanted.
    ungapped_to_col : numpy.ndarray
        The owning sequence's ungapped-to-column map
        (:func:`ungapped_to_column_map`).

    Returns
    -------
    list of int or None
        Column indices in transcription order (reversed for a minus-strand
        gene). ``None`` if any exon runs past the end of the ungapped sequence,
        which cannot be mapped.
    """
    n_ungapped = ungapped_to_col.shape[0]
    columns: List[int] = []
    for exon in gene.cds:  # already sorted ascending by start
        if exon.end > n_ungapped:
            logger.warning(
                'Gene %r exon %d-%d runs past sequence %r (length %d); skipping gene',
                gene.gene_id,
                exon.start,
                exon.end,
                gene.seqid,
                n_ungapped,
            )
            return None
        columns.extend(int(c) for c in ungapped_to_col[exon.start - 1 : exon.end])

    if gene.strand == '-':
        columns.reverse()
    return columns


def _read_coding_bases(
    row_bytes: np.ndarray, columns: List[int], strand: str
) -> Tuple[str, List[int]]:
    """
    Read a row's coding bases at the given columns, complementing if minus.

    Gap characters are dropped (they are not part of the translated sequence),
    and the columns backing the surviving bases are returned in parallel so a
    codon can be traced back to its alignment column.

    Parameters
    ----------
    row_bytes : numpy.ndarray
        Alignment row as an ``'S1'`` byte array.
    columns : list of int
        Column indices in transcription order.
    strand : str
        ``'+'`` or ``'-'``; a minus-strand base is complemented.

    Returns
    -------
    tuple
        ``(bases, kept_columns)`` — the coding sequence string and the column
        index behind each of its bases.
    """
    bases: List[str] = []
    kept: List[int] = []
    for col in columns:
        base = row_bytes[col].decode('ascii').upper()
        if base == '-':
            continue
        bases.append(_COMPLEMENT.get(base, 'N') if strand == '-' else base)
        kept.append(col)
    return ''.join(bases), kept


def _trim_phase(bases: str, columns: List[int], phase: Optional[int]):
    """
    Drop leading bases so the coding sequence starts in frame.

    Parameters
    ----------
    bases : str
        The coding sequence.
    columns : list of int
        Column behind each base, parallel to ``bases``.
    phase : int or None
        CDS phase of the first exon; ``None`` and ``0`` both mean no trim.

    Returns
    -------
    tuple
        ``(bases, columns)`` trimmed by ``phase`` leading bases.
    """
    if phase:
        return bases[phase:], columns[phase:]
    return bases, columns


def translate_cds(
    gene: Gene,
    row_bytes: np.ndarray,
    ungapped_to_col: np.ndarray,
    genetic_code: int = 1,
) -> str:
    """
    Translate one sequence's CDS for a gene, rendering stops as ``'*'``.

    Parameters
    ----------
    gene : Gene
        The gene to translate.
    row_bytes : numpy.ndarray
        The sequence's alignment row (``'S1'`` byte array).
    ungapped_to_col : numpy.ndarray
        The sequence's ungapped-to-column map.
    genetic_code : int, optional
        NCBI translation table (default: 1, the standard code).

    Returns
    -------
    str
        The amino-acid sequence, ``'*'`` for each stop codon. Empty if the CDS
        could not be mapped.
    """
    from Bio.Seq import Seq

    columns = _cds_columns_transcription_order(gene, ungapped_to_col)
    if columns is None:
        return ''
    bases, kept = _read_coding_bases(row_bytes, columns, gene.strand)
    phase = gene.cds[0].phase if gene.cds else None
    bases, _kept = _trim_phase(bases, kept, phase)
    usable = len(bases) - (len(bases) % 3)
    if usable <= 0:
        return ''
    return str(Seq(bases[:usable]).translate(table=genetic_code))


def _revcomp(bases: str) -> str:
    """
    Reverse-complement a short base string using the module complement table.

    Parameters
    ----------
    bases : str
        A DNA string (upper-case A/C/G/T; other characters map to ``N``).

    Returns
    -------
    str
        The reverse complement.
    """
    return ''.join(_COMPLEMENT.get(b, 'N') for b in reversed(bases))


def _forward_dinucleotide(row: np.ndarray, u0: int, ungapped_to_col: np.ndarray):
    """
    Read the forward-strand dinucleotide at ungapped positions ``u0, u0+1``.

    Parameters
    ----------
    row : numpy.ndarray
        Alignment row (``'S1'`` byte array).
    u0 : int
        1-based ungapped position of the first base of the pair.
    ungapped_to_col : numpy.ndarray
        The sequence's ungapped-to-column map.

    Returns
    -------
    tuple
        ``(bases, first_column)`` — the two forward-strand bases and the column
        of the first base.
    """
    c0 = int(ungapped_to_col[u0 - 1])
    c1 = int(ungapped_to_col[u0])
    b0 = row[c0].decode('ascii').upper()
    b1 = row[c1].decode('ascii').upper()
    return b0 + b1, c0


def _intron_boundary_effects(
    gene: Gene,
    target_row: np.ndarray,
    ref_row: np.ndarray,
    ungapped_to_col: np.ndarray,
    seq_id: str,
) -> List[EffectRecord]:
    """
    Flag introns whose canonical GT-AG boundary is broken in the target.

    For each intron (the gap between two consecutive exons), the coding-strand
    donor (5', canonical ``GT``) and acceptor (3', canonical ``AG``) are read
    for both the observed and ancestral rows. A boundary is reported when the
    ancestor was canonical and the observed base differs — i.e. RIP broke it. On
    the minus strand the donor/acceptor swap genomic ends and are
    reverse-complemented to the coding strand.

    Parameters
    ----------
    gene : Gene
        The multi-exon gene to check (single-exon genes have no introns).
    target_row, ref_row : numpy.ndarray
        The observed and ancestral alignment rows (``'S1'`` byte arrays).
    ungapped_to_col : numpy.ndarray
        The sequence's ungapped-to-column map.
    seq_id : str
        Identifier stamped onto each returned record.

    Returns
    -------
    list of EffectRecord
        One ``'splice_site'`` record per broken boundary.
    """
    effects: List[EffectRecord] = []
    n_ungapped = ungapped_to_col.shape[0]

    for left, right in zip(gene.cds, gene.cds[1:]):
        # Intron spans forward positions [left.end + 1, right.start - 1].
        donor_u = left.end + 1  # forward 5' end of the intron
        acceptor_u = right.start - 2  # first of the forward 3'-end pair
        if donor_u + 1 > n_ungapped or acceptor_u < 1:
            continue

        fwd_donor, donor_col = _forward_dinucleotide(
            target_row, donor_u, ungapped_to_col
        )
        fwd_acceptor, acc_col = _forward_dinucleotide(
            target_row, acceptor_u, ungapped_to_col
        )
        ref_fwd_donor, _ = _forward_dinucleotide(ref_row, donor_u, ungapped_to_col)
        ref_fwd_acceptor, _ = _forward_dinucleotide(
            ref_row, acceptor_u, ungapped_to_col
        )

        if gene.strand == '+':
            sites = (
                (fwd_donor, ref_fwd_donor, _DONOR, donor_col),
                (fwd_acceptor, ref_fwd_acceptor, _ACCEPTOR, acc_col),
            )
        else:
            # Coding strand runs the other way: the forward 3'-end pair is the
            # coding donor, the forward 5'-end pair is the coding acceptor.
            sites = (
                (_revcomp(fwd_acceptor), _revcomp(ref_fwd_acceptor), _DONOR, acc_col),
                (_revcomp(fwd_donor), _revcomp(ref_fwd_donor), _ACCEPTOR, donor_col),
            )

        for observed, ancestral, canonical, col in sites:
            if ancestral == canonical and observed != ancestral:
                effects.append(
                    EffectRecord(
                        seq_id=seq_id,
                        gene_id=gene.gene_id,
                        kind='splice_site',
                        gapped_col=col,
                        nt_ref=ancestral,
                        nt_alt=observed,
                    )
                )
    return effects


def predict_gene_effects(
    gene: Gene,
    target_row: np.ndarray,
    ref_row: np.ndarray,
    ungapped_to_col: np.ndarray,
    *,
    seq_id: str = '',
    genetic_code: int = 1,
    include_synonymous: bool = False,
) -> List[EffectRecord]:
    """
    Predict the coding effects of RIP on one sequence, versus the ancestor.

    The gene's CDS is assembled in transcription order for both the observed
    sequence (``target_row``) and the reconstructed ancestor (``ref_row``),
    translated, and compared codon by codon. Length differences that are not a
    multiple of three are reported as frameshifts; canonical splice boundaries
    broken in the target are reported as splice-site effects.

    Parameters
    ----------
    gene : Gene
        The gene to evaluate.
    target_row, ref_row : numpy.ndarray
        The observed and ancestral alignment rows (``'S1'`` byte arrays). Both
        must be indexed by the same columns, i.e. drawn from the same alignment.
    ungapped_to_col : numpy.ndarray
        The *observed* sequence's ungapped-to-column map, used to place the CDS.
    seq_id : str, optional
        Identifier stamped onto each returned record.
    genetic_code : int, optional
        NCBI translation table (default: 1).
    include_synonymous : bool, optional
        Also emit ``'synonymous'`` records (default: False; usually noise).

    Returns
    -------
    list of EffectRecord
        Effects in codon order, splice-site effects appended.
    """
    from Bio.Seq import Seq

    columns = _cds_columns_transcription_order(gene, ungapped_to_col)
    if columns is None:
        return []

    tgt_bases, tgt_cols = _read_coding_bases(target_row, columns, gene.strand)
    ref_bases, _ref_cols = _read_coding_bases(ref_row, columns, gene.strand)
    phase = gene.cds[0].phase if gene.cds else None
    tgt_bases, tgt_cols = _trim_phase(tgt_bases, tgt_cols, phase)
    ref_bases, _ref_cols = _trim_phase(ref_bases, _ref_cols, phase)

    effects: List[EffectRecord] = []

    # A length change that is not a whole number of codons shifts the frame.
    if (len(tgt_bases) - len(ref_bases)) % 3 != 0 or len(tgt_bases) % 3 != 0:
        effects.append(
            EffectRecord(
                seq_id=seq_id,
                gene_id=gene.gene_id,
                kind='frameshift',
                nt_ref=f'{len(ref_bases)} nt',
                nt_alt=f'{len(tgt_bases)} nt',
            )
        )
    else:
        tgt_aa = str(Seq(tgt_bases).translate(table=genetic_code))
        ref_aa = str(Seq(ref_bases).translate(table=genetic_code))
        for k, (r_aa, a_aa) in enumerate(zip(ref_aa, tgt_aa)):
            middle_col = tgt_cols[3 * k + 1] if 3 * k + 1 < len(tgt_cols) else None
            codon_changed = ref_bases[3 * k : 3 * k + 3] != tgt_bases[3 * k : 3 * k + 3]
            if a_aa == '*' and r_aa != '*':
                effects.append(
                    EffectRecord(
                        seq_id=seq_id,
                        gene_id=gene.gene_id,
                        kind='premature_stop',
                        aa_pos=k + 1,
                        ref_aa=r_aa,
                        alt_aa=a_aa,
                        gapped_col=middle_col,
                    )
                )
                break  # translation ends at the first premature stop
            if r_aa != a_aa:
                effects.append(
                    EffectRecord(
                        seq_id=seq_id,
                        gene_id=gene.gene_id,
                        kind='missense',
                        aa_pos=k + 1,
                        ref_aa=r_aa,
                        alt_aa=a_aa,
                        gapped_col=middle_col,
                    )
                )
            elif include_synonymous and codon_changed:
                effects.append(
                    EffectRecord(
                        seq_id=seq_id,
                        gene_id=gene.gene_id,
                        kind='synonymous',
                        aa_pos=k + 1,
                        ref_aa=r_aa,
                        alt_aa=a_aa,
                        gapped_col=middle_col,
                    )
                )

    effects.extend(
        _intron_boundary_effects(gene, target_row, ref_row, ungapped_to_col, seq_id)
    )

    return effects


def compute_effects_for_alignment(
    derip, genes_by_seqid: Dict[str, List[Gene]], genetic_code: int = 1
) -> Dict[str, List[EffectRecord]]:
    """
    Predict RIP effects for every annotated sequence in a DeRIP alignment.

    Each gene is evaluated against the reconstructed ancestor (deRIP2's gapped
    consensus). Genes whose sequence identifier is not in the alignment are
    skipped.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        A DeRIP object on which ``calculate_rip()`` has run.
    genes_by_seqid : dict of str to list of Gene
        Parsed genes keyed by sequence identifier.
    genetic_code : int, optional
        NCBI translation table (default: 1).

    Returns
    -------
    dict of str to list of EffectRecord
        Effects keyed by sequence identifier (only sequences with a gene and at
        least one effect appear).
    """
    ref_row = np.frombuffer(
        str(derip.gapped_consensus.seq).upper().encode('ascii'), dtype='S1'
    ).copy()
    id_to_row = {rec.id: i for i, rec in enumerate(derip.alignment)}

    effects_by_seq: Dict[str, List[EffectRecord]] = {}
    for seqid, genes in genes_by_seqid.items():
        row_index = id_to_row.get(seqid)
        if row_index is None:
            continue
        target_row = derip.column_classes.arr[row_index]
        ungapped_to_col = ungapped_to_column_map(target_row)
        collected: List[EffectRecord] = []
        for gene in genes:
            collected.extend(
                predict_gene_effects(
                    gene,
                    target_row,
                    ref_row,
                    ungapped_to_col,
                    seq_id=seqid,
                    genetic_code=genetic_code,
                )
            )
        if collected:
            effects_by_seq[seqid] = collected
    return effects_by_seq


def deripd_translations(
    derip, genes_by_seqid: Dict[str, List[Gene]], genetic_code: int = 1
) -> Dict[str, str]:
    """
    Translate each gene's CDS on the reconstructed deRIP'd sequence.

    Columns are taken from the gene's owning sequence, but the bases are read
    from the deRIP'd consensus, so the returned protein is what the restored
    (un-RIP'd) coding sequence encodes.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        A DeRIP object on which ``calculate_rip()`` has run.
    genes_by_seqid : dict of str to list of Gene
        Parsed genes keyed by sequence identifier.
    genetic_code : int, optional
        NCBI translation table (default: 1).

    Returns
    -------
    dict of str to str
        Gene identifier to amino-acid string (``'*'`` for stops).
    """
    ref_row = np.frombuffer(
        str(derip.gapped_consensus.seq).upper().encode('ascii'), dtype='S1'
    ).copy()
    id_to_row = {rec.id: i for i, rec in enumerate(derip.alignment)}

    translations: Dict[str, str] = {}
    for seqid, genes in genes_by_seqid.items():
        row_index = id_to_row.get(seqid)
        if row_index is None:
            continue
        ungapped_to_col = ungapped_to_column_map(derip.column_classes.arr[row_index])
        for gene in genes:
            translations[gene.gene_id] = translate_cds(
                gene, ref_row, ungapped_to_col, genetic_code=genetic_code
            )
    return translations


def write_snp_effects(
    output_file: str,
    effects_by_seq: Dict[str, List[EffectRecord]],
    deripd_aa: Dict[str, str],
) -> str:
    """
    Write a tab-separated summary of RIP coding effects.

    Parameters
    ----------
    output_file : str
        Destination path.
    effects_by_seq : dict of str to list of EffectRecord
        Per-sequence effects (:func:`compute_effects_for_alignment`).
    deripd_aa : dict of str to str
        Per-gene deRIP'd translations (:func:`deripd_translations`).

    Returns
    -------
    str
        The path written.
    """
    columns = (
        'seq_id',
        'gene_id',
        'kind',
        'aa_pos',
        'ref_aa',
        'alt_aa',
        'gapped_col',
        'nt_change',
    )
    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write('\t'.join(columns) + '\n')
        for seqid in sorted(effects_by_seq):
            for effect in effects_by_seq[seqid]:
                nt_change = (
                    f'{effect.nt_ref}>{effect.nt_alt}'
                    if effect.nt_ref is not None or effect.nt_alt is not None
                    else ''
                )
                row = (
                    effect.seq_id,
                    effect.gene_id,
                    effect.kind,
                    '' if effect.aa_pos is None else str(effect.aa_pos),
                    effect.ref_aa or '',
                    effect.alt_aa or '',
                    '' if effect.gapped_col is None else str(effect.gapped_col),
                    nt_change,
                )
                handle.write('\t'.join(row) + '\n')

        # A trailing block records the restored (deRIP'd) protein per gene.
        handle.write('\n# deRIP-restored CDS translations\n')
        handle.write('# gene_id\tprotein\n')
        for gene_id in sorted(deripd_aa):
            handle.write(f'# {gene_id}\t{deripd_aa[gene_id]}\n')

    logger.info('SNP-effect summary written to %s', output_file)
    return output_file
