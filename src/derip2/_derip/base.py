"""Shared state, loading and result invalidation for :class:`derip2.derip.DeRIP`."""

import logging
from os import path
from typing import Optional

from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord

import derip2.aln_ops as ao

logger = logging.getLogger(__name__)


class DeRIPBase:
    """
    Shared state, loading and result invalidation for :class:`derip2.derip.DeRIP`.

    Holds the parameters and every derived attribute, loads the alignment, and
    provides the guards the other mixins rely on.

    Parameters
    ----------
    alignment_input : str or Bio.Align.MultipleSeqAlignment
        Path to the alignment file in FASTA format or a pre-loaded MultipleSeqAlignment object.
    max_snp_noise : float, optional
        Maximum proportion of conflicting SNPs permitted before excluding column
        from RIP/deamination assessment (default: 0.5).
    min_rip_like : float, optional
        Minimum proportion of deamination events in RIP context required for
        column to be deRIP'd in final sequence (default: 0.1).
    reaminate : bool, optional
        Whether to correct all deamination events independent of RIP context (default: False).
    fill_index : int, optional
        Index of row to use for filling uncorrected positions (default: None).
    fill_max_gc : bool, optional
        Whether to use sequence with highest GC content for filling if
        no row index is specified (default: False).
    max_gaps : float, optional
        Maximum proportion of gaps in a column before considering it a gap
        in consensus (default: 0.7).
    """

    # Parameters (set once in __init__).
    max_snp_noise: float
    min_rip_like: float
    reaminate: bool
    fill_index: Optional[int]
    fill_max_gc: bool
    max_gaps: float

    # Derived state, populated by calculate_rip() and the analysis mixins.
    alignment: Optional[MultipleSeqAlignment]
    masked_alignment: Optional[MultipleSeqAlignment]
    consensus: Optional[SeqRecord]
    gapped_consensus: Optional[SeqRecord]
    consensus_tracker: Optional[dict]
    rip_counts: Optional[dict]
    corrected_positions: dict
    markupdict: Optional[dict]
    column_classes: Optional[ao.ColumnClassification]
    rsi_result: object
    spectra_result: object
    flank_spectra_result: object
    max_rip_results: dict

    def __init__(
        self,
        alignment_input,
        max_snp_noise: float = 0.5,
        min_rip_like: float = 0.1,
        reaminate: bool = False,
        fill_index: Optional[int] = None,
        fill_max_gc: bool = False,
        max_gaps: float = 0.7,
    ) -> None:
        """
        Initialize DeRIP with an alignment file or MultipleSeqAlignment object and parameters.

        Parameters
        ----------
        alignment_input : str or Bio.Align.MultipleSeqAlignment
            Path to the alignment file in FASTA format or a pre-loaded MultipleSeqAlignment object.
            If a MultipleSeqAlignment is provided, it must contain at least 2 sequences.
        max_snp_noise : float, optional
            Maximum proportion of conflicting SNPs permitted before excluding column
            from RIP/deamination assessment (default: 0.5).
        min_rip_like : float, optional
            Minimum proportion of deamination events in RIP context required for
            column to be deRIP'd in final sequence (default: 0.1).
        reaminate : bool, optional
            Whether to correct all deamination events independent of RIP context (default: False).
        fill_index : int, optional
            Index of row to use for filling uncorrected positions (default: None).
        fill_max_gc : bool, optional
            Whether to use sequence with highest GC content for filling if
            no row index is specified (default: False).
        max_gaps : float, optional
            Maximum proportion of gaps in a column before considering it a gap
            in consensus (default: 0.7).
        """
        # Store parameters
        self.max_snp_noise = max_snp_noise
        self.min_rip_like = min_rip_like
        self.reaminate = reaminate
        self.fill_index = fill_index
        self.fill_max_gc = fill_max_gc
        self.max_gaps = max_gaps

        # Lazily rendered ANSI-coloured views (see the ``colored_*`` properties).
        self._colored = {}

        # Initialize attributes
        self.alignment = None
        self.masked_alignment = None
        self.consensus = None
        self.gapped_consensus = None
        self.consensus_tracker = None
        self.rip_counts = None
        self.corrected_positions = {}
        self.colored_consensus = None
        self.colored_alignment = None
        self.colored_masked_alignment = None
        self.markupdict = None
        self.column_classes = None
        self.rsi_result = None
        self.spectra_result = None
        self.flank_spectra_result = None
        self.max_rip_results = {}

        # Load the alignment
        self._load_alignment(alignment_input)

    def _load_alignment(self, alignment_input):
        """
        Load and validate the alignment from file or MultipleSeqAlignment object.

        Parameters
        ----------
        alignment_input : str or Bio.Align.MultipleSeqAlignment
            Path to the alignment file or a pre-loaded MultipleSeqAlignment object.

        Raises
        ------
        FileNotFoundError
            If the alignment file path does not exist.
        ValueError
            If the alignment contains fewer than two sequences, has duplicate IDs,
            or if the input type is not supported.
        """
        # Check if input is a MultipleSeqAlignment object
        if isinstance(alignment_input, MultipleSeqAlignment):
            # Directly use the provided alignment
            self.alignment = alignment_input
            logger.info(
                f'Using provided MultipleSeqAlignment with {len(self.alignment)} sequences'
            )

            # Validate the alignment has at least 2 sequences
            if len(self.alignment) < 2:
                raise ValueError('Alignment must contain at least 2 sequences')

        # Check if input is a string (file path)
        elif isinstance(alignment_input, str):
            # Check if file exists
            if not path.isfile(alignment_input):
                raise FileNotFoundError(f'Alignment file not found: {alignment_input}')

            # Load alignment using aln_ops function
            try:
                self.alignment = ao.loadAlign(alignment_input, alnFormat='fasta')
                logger.info(
                    f'Loaded alignment from file with {len(self.alignment)} sequences'
                )

                # Validate the alignment has at least 2 sequences
                if len(self.alignment) < 2:
                    raise ValueError('Alignment must contain at least 2 sequences')

            except Exception as e:
                raise ValueError(f'Error loading alignment: {str(e)}') from e
        else:
            # Neither a string nor a MultipleSeqAlignment
            raise ValueError(
                f'alignment_input must be either a file path (str) or a MultipleSeqAlignment object, '
                f'got {type(alignment_input).__name__}'
            )

    def __str__(self) -> str:
        """
        String representation of the DeRIP object.

        Returns a formatted string representing the current state of the object.
        If calculate_rip() has been called, returns the colored alignment and
        colored consensus sequence. Otherwise, returns the basic alignment.

        Returns
        -------
        str
            Formatted string representation of the DeRIP object
        """
        # Check if calculate_rip() has been called by checking if colored_alignment exists
        if self.colored_alignment is not None and self.colored_consensus is not None:
            # Calculate_rip has been called, show colored alignment and consensus
            consensus_id = self.consensus.id if self.consensus else 'deRIPseq'
            rows = len(self.alignment)
            cols = self.alignment.get_alignment_length()
            header = f'DeRIP alignment with {rows} rows and {cols} columns:'
            return f'{header}\n{self.colored_alignment}\n{self.colored_consensus} {consensus_id}'
        elif self.alignment is not None:
            # calculate_rip has not been called yet, show basic alignment
            rows = []
            rows_count = len(self.alignment)
            cols_count = self.alignment.get_alignment_length()
            header = f'DeRIP alignment with {rows_count} rows and {cols_count} columns:'
            rows.append(header)

            for seq_record in self.alignment:
                rows.append(f'{seq_record.seq} {seq_record.id}')
            return '\n'.join(rows)
        else:
            # No alignment loaded
            return 'DeRIP object (no alignment loaded)'

    def _invalidate_results(self) -> None:
        """
        Discard everything derived from the alignment.

        Called whenever the alignment itself is replaced in place, since every
        cached result is keyed to the old row order or row membership.
        :meth:`calculate_rip` must be run again before the results are usable.

        Returns
        -------
        None
            The cached results are cleared in place.
        """
        self.masked_alignment = None
        self.consensus = None
        self.gapped_consensus = None
        self.consensus_tracker = None
        self.rip_counts = None
        self.corrected_positions = {}
        self.colored_consensus = None
        self.colored_alignment = None
        self.colored_masked_alignment = None
        self.markupdict = None
        self.column_classes = None
        self.rsi_result = None
        self.spectra_result = None
        self.flank_spectra_result = None
        self.max_rip_results = {}

    def _require_rip(self, action: str) -> None:
        """
        Raise if :meth:`calculate_rip` has not been run.

        Parameters
        ----------
        action : str
            Description of what the caller was trying to do, used in the message.

        Returns
        -------
        None
            Nothing is returned; the check either passes or raises.

        Raises
        ------
        ValueError
            If the column classification has not been computed.
        """
        if self.column_classes is None:
            raise ValueError(f'Must call calculate_rip before {action}')
