"""
DeRIP class for detecting and correcting RIP mutations in DNA alignments.

This module provides a class-based interface to the deRIP2 tool for correcting
Repeat-Induced Point (RIP) mutations in fungal DNA alignments.
"""

from derip2._derip.base import DeRIPBase
from derip2._derip.plots import PlotsReportsMixin
from derip2._derip.rip import RIPCorrectionMixin
from derip2._derip.spectra import SpectraMixin
from derip2._derip.stats import StatsMixin

__all__ = ['DeRIP']


class DeRIP(RIPCorrectionMixin, StatsMixin, SpectraMixin, PlotsReportsMixin, DeRIPBase):
    """
    A class to detect and correct RIP (Repeat-Induced Point) mutations in DNA alignments.

    This class encapsulates the functionality to analyze DNA sequence alignments for
    RIP-like mutations, correct them, and generate deRIPed consensus sequences.

    Attributes
    ----------
    alignment : MultipleSeqAlignment
        The loaded DNA sequence alignment.
    masked_alignment : MultipleSeqAlignment
        The alignment with RIP-corrected positions masked with IUPAC codes.
    consensus : SeqRecord
        The deRIPed consensus sequence.
    gapped_consensus : SeqRecord
        The deRIPed consensus sequence with gaps.
    rip_counts : Dict
        Dictionary tracking RIP mutation counts for each sequence.
    corrected_positions : Dict
        Dictionary of corrected positions {col_idx: {row_idx: {observed_base, corrected_base}}}.
    colored_consensus : str
        Consensus sequence with corrected positions highlighted in green.
    colored_alignment : str
        Alignment with corrected positions highlighted in green.
    colored_masked_alignment : str
        Masked alignment with RIP positions highlighted in color.
    markupdict : Dict
        Dictionary of markup codes for masked positions.
    """
