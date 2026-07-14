"""
Tests for the alignment QC profile used by the phylogenetic spectrum path.
"""

import logging
import os

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from derip2.spectra.qc import profile_alignment, write_column_profile, write_qc_report

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')


def _aln(seqs):
    """Build an alignment from sequence strings."""
    return MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )


def test_gap_fraction_per_column():
    """Gap fractions are computed per column."""
    profile = profile_alignment(_aln(['AC-T', 'A--T', 'ACGT']))
    # Column 1: one gap of three; column 2: two gaps of three.
    assert profile.gap_fraction[1] == 1 / 3
    assert profile.gap_fraction[2] == 2 / 3
    assert profile.n_cols == 4
    assert profile.n_rows == 3


def test_context_unreliable_flag():
    """Columns above the gap threshold are flagged context-unreliable."""
    profile = profile_alignment(_aln(['A-GT', 'A-GT', 'ACGT']), gap_threshold=0.5)
    # Column 1 is 2/3 gaps (> 0.5) -> flagged; others are not.
    assert bool(profile.context_unreliable[1])
    assert not bool(profile.context_unreliable[0])
    assert profile.n_flagged == 1


def test_ambiguous_fraction():
    """Non-ACGT, non-gap symbols count towards the ambiguity fraction."""
    profile = profile_alignment(_aln(['ANGT', 'ACGT']))
    assert profile.ambiguous_fraction[1] == 0.5


def test_write_reports(tmp_path):
    """The QC report and column profile are written to disk."""
    from derip2.aln_ops import loadAlign

    profile = profile_alignment(loadAlign(MINTEST))
    report = write_qc_report(loadAlign(MINTEST), profile, str(tmp_path / 'qc.txt'))
    table = write_column_profile(profile, str(tmp_path / 'cols.tsv'))
    assert os.path.getsize(report) > 0
    with open(table) as handle:
        header = handle.readline()
    assert header.startswith('column\tgap_fraction')
