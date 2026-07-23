"""
DeRIP class for detecting and correcting RIP mutations in DNA alignments.

This module provides a class-based interface to the deRIP2 tool for correcting
Repeat-Induced Point (RIP) mutations in fungal DNA alignments.
"""

import logging
from os import path
import sys
import time
from typing import List, Optional, Tuple

from Bio.Align import MultipleSeqAlignment
import numpy as np

import derip2.aln_ops as ao

logger = logging.getLogger(__name__)


class DeRIP:
    """
    A class to detect and correct RIP (Repeat-Induced Point) mutations in DNA alignments.

    This class encapsulates the functionality to analyze DNA sequence alignments for
    RIP-like mutations, correct them, and generate deRIPed consensus sequences.

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
        from Bio.Align import MultipleSeqAlignment

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

    def calculate_rip(self, label: str = 'deRIPseq') -> None:
        """
        Calculate RIP locations and corrections in the alignment.

        This method performs RIP detection and correction, fills in the consensus
        sequence, and populates the class attributes.

        Parameters
        ----------
        label : str, optional
            ID for the generated deRIPed sequence (default: "deRIPseq").

        Returns
        -------
        None
            Updates class attributes with results.
        """
        # Timing helper for performance debugging; emits DEBUG logs only.
        _t0 = time.perf_counter()

        def _lap(_stage, _since):
            _now = time.perf_counter()
            logger.debug(f'calculate_rip: {_stage} took {_now - _since:.3f}s')
            return _now

        # Initialize tracking structures
        # tracker is a dict of tuples, keys are column indices, values are tuples of (col_idx, corrected_base)
        # used to compose the consensus sequence
        tracker = ao.initTracker(self.alignment)
        # rip_counts is a dict of rowItem('idx', 'SeqID', 'revRIPcount', 'RIPcount', 'nonRIPcount', 'GC'), keys are row IDs
        # used to track RIP mutations in each sequence
        rip_counts = ao.initRIPCounter(self.alignment)
        _t = _lap('init trackers', _t0)

        # Pre-fill conserved positions
        tracker = ao.fillConserved(self.alignment, tracker, self.max_gaps)
        _t = _lap('fillConserved', _t)

        # Classify every cell and column by RIP context. The classification is
        # cached because the strand-bias statistics and figures consume the same
        # structure, so they can never disagree with the correction.
        self.column_classes = ao.classify_alignment(
            self.alignment,
            max_snp_noise=self.max_snp_noise,
            min_rip_like=self.min_rip_like,
            reaminate=self.reaminate,
        )
        _t = _lap('classify_columns', _t)

        # Apply the classification to the consensus tracker, counters and mask.
        tracker, rip_counts, masked_alignment, _corrected_positions, markupdict = (
            ao.apply_classification(
                self.alignment, tracker, rip_counts, self.column_classes
            )
        )
        _t = _lap('apply_classification', _t)

        # Store the markupdict for later use in colored alignment
        self.markupdict = markupdict

        # Populate corrected positions dictionary
        # TODO: Avoid double pass of data to calculate this.
        self._build_corrected_positions(self.alignment, masked_alignment)
        _t = _lap('build_corrected_positions', _t)

        # Select reference sequence for filling uncorrected positions
        if self.fill_index is not None:
            # Validate index is within range
            ao.checkrow(self.alignment, idx=self.fill_index)
            ref_id = self.fill_index
        else:
            # Select based on RIP counts or GC content
            ref_id = ao.setRefSeq(
                self.alignment,
                rip_counts,
                getMinRIP=not self.fill_max_gc,  # Use sequence with fewest RIPs if not filling with max GC
                getMaxGC=self.fill_max_gc,
            )
            # Set fill_index to the selected reference sequence ID
            self.fill_index = ref_id

        # Fill remaining positions from selected reference sequence
        tracker = ao.fillRemainder(self.alignment, ref_id, tracker)

        # Create consensus sequence
        consensus = ao.getDERIP(tracker, ID=label, deGAP=True)
        gapped_consensus = ao.getDERIP(tracker, ID=label, deGAP=False)

        # Store results in attributes
        self.masked_alignment = masked_alignment
        self.consensus = consensus
        self.gapped_consensus = gapped_consensus
        self.consensus_tracker = tracker
        self.rip_counts = rip_counts

        # Create colorized consensus
        self._colorize_corrected_positions()

        # Create colorized alignment
        self.colored_alignment = self._create_colored_alignment(self.alignment)

        # Create colorized masked alignment
        self.colored_masked_alignment = self._create_colored_alignment(
            self.masked_alignment
        )
        _lap('fill + colorize', _t)
        logger.debug(f'calculate_rip: total {time.perf_counter() - _t0:.3f}s')

        # Log summary
        logger.info(
            f'RIP correction complete. Reference sequence used for filling: {ref_id}'
        )

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

    def _build_corrected_positions(
        self, original: MultipleSeqAlignment, masked: MultipleSeqAlignment
    ) -> None:
        """
        Build dictionary of corrected positions by comparing original and masked alignments.

        Parameters
        ----------
        original : MultipleSeqAlignment
            The original alignment.
        masked : MultipleSeqAlignment
            The masked alignment with RIP positions marked.

        Returns
        -------
        None
            Updates the corrected_positions attribute.
        """
        self.corrected_positions = {}

        # Decode both alignments to byte arrays and diff them vectorised, rather
        # than indexing every cell of two Biopython Seq objects.
        orig = ao.alignment_to_array(original)
        mask = ao.alignment_to_array(masked)
        diff = orig != mask

        # Map masked IUPAC code -> corrected ancestral base
        corrected_for = {'Y': 'C', 'R': 'G'}

        # Only visit columns that contain at least one masked position
        for col_idx in np.where(diff.any(axis=0))[0].tolist():
            col_dict = {}
            for row_idx in np.where(diff[:, col_idx])[0].tolist():
                masked_base = mask[row_idx, col_idx].decode('ascii')
                corrected_base = corrected_for.get(masked_base)
                if corrected_base:
                    col_dict[row_idx] = {
                        'observed_base': orig[row_idx, col_idx].decode('ascii'),
                        'corrected_base': corrected_base,
                    }

            # Only add column to dict if corrections were made
            if col_dict:
                self.corrected_positions[col_idx] = col_dict

        logger.info(
            f'Identified {len(self.corrected_positions)} columns with RIP corrections'
        )

    def _colorize_corrected_positions(self) -> str:
        """
        Create a colorized version of the gapped consensus sequence.

        Bases at positions that were corrected during RIP analysis
        are highlighted in green.

        Returns
        -------
        str
            Consensus sequence with corrected positions highlighted in green.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.gapped_consensus is None:
            raise ValueError('Must call calculate_rip before colorizing consensus')

        # Get the consensus sequence as a string
        seq_str = str(self.gapped_consensus.seq)

        # Convert to list for easier manipulation
        seq_chars = list(seq_str)

        # Add ANSI color codes for each corrected position
        BOLD_GREEN = '\033[1;32m'  # 1 for bold, 32 for green
        RESET = '\033[0m'

        for pos in self.corrected_positions:
            if 0 <= pos < len(seq_chars):
                # Only colorize if position is in range (safety check)
                seq_chars[pos] = f'{BOLD_GREEN}{seq_chars[pos]}{RESET}'

        # Join back into string
        colored_seq = ''.join(seq_chars)

        # Store as attribute
        self.colored_consensus = colored_seq

        return colored_seq

    def _create_colored_alignment(self, alignment) -> str:
        """
        Create a colorized version of the entire alignment.

        Bases are colored according to their RIP status as defined in the markupdict:
        - RIP products (typically T from C→T mutations) are highlighted in red
        - RIP substrates (unmutated nucleotides in RIP context) are highlighted in blue
        - Non-RIP deaminations are highlighted in yellow (only if reaminate=True)
        - Target bases are bold + colored, while bases in offset range are only colored

        Parameters
        ----------
        alignment : Bio.Align.MultipleSeqAlignment
            The alignment to colorize. Can be either the original alignment or masked alignment.

        Returns
        -------
        str
            Alignment with sequences displayed with colored bases and labels.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if alignment is None or self.markupdict is None:
            raise ValueError(
                'Must call calculate_rip before creating colored alignment'
            )

        # Define ANSI color codes - separate bold+color from just color
        RED_BOLD = '\033[1;31m'  # Bold red for target RIP products
        BLUE_BOLD = '\033[1;34m'  # Bold blue for target RIP substrates
        YELLOW_BOLD = '\033[1;33m'  # Bold yellow for target non-RIP deaminations

        RED = '\033[0;31m'  # Red (not bold) for offset bases
        BLUE = '\033[0;34m'  # Blue (not bold) for offset bases
        YELLOW = '\033[0;33m'  # Orange (not bold) for offset bases

        RESET = '\033[0m'

        # Define color maps for each category
        target_color_map = {
            'rip_product': RED_BOLD,
            'rip_substrate': BLUE_BOLD,
            'non_rip_deamination': YELLOW_BOLD,
        }

        offset_color_map = {
            'rip_product': RED,
            'rip_substrate': BLUE,
            'non_rip_deamination': YELLOW,
        }

        # Create a colored representation of each sequence in the alignment
        lines = []

        # Process each sequence in the alignment
        for row_idx in range(len(alignment)):
            seq = alignment[row_idx].seq
            seq_id = alignment[row_idx].id

            # Create list of characters for this sequence with their default coloring
            colored_chars = list(str(seq))

            # Process each RIP category
            for category, positions in self.markupdict.items():
                # Skip non_rip_deamination highlighting if reaminate is False
                if category == 'non_rip_deamination' and not self.reaminate:
                    continue

                target_color = target_color_map[category]
                offset_color = offset_color_map[category]

                # Process each position in this category
                for pos in positions:
                    if (
                        pos.rowIdx == row_idx
                    ):  # Only apply if this position is in the current row
                        col_idx = pos.colIdx
                        offset = pos.offset

                        # Apply bold+color formatting to the target base
                        if 0 <= col_idx < len(colored_chars):
                            colored_chars[col_idx] = (
                                f'{target_color}{colored_chars[col_idx]}{RESET}'
                            )

                        # Determine range of offset positions to color (but not bold)
                        if offset is not None:
                            if offset > 0:
                                # Color bases to the right (excluding target)
                                start_col = col_idx + 1
                                end_col = min(col_idx + offset, len(seq) - 1)
                            else:  # offset < 0
                                # Color bases to the left (excluding target)
                                start_col = max(
                                    0, col_idx + offset
                                )  # offset is negative
                                end_col = col_idx - 1

                            # Apply color-only formatting to the offset bases
                            for i in range(start_col, end_col + 1):
                                if 0 <= i < len(colored_chars):
                                    colored_chars[i] = (
                                        f'{offset_color}{colored_chars[i]}{RESET}'
                                    )

            # Join the characters and add sequence ID
            colored_seq = ''.join(colored_chars)
            lines.append(f'{colored_seq} {seq_id}')

        # Join all lines with newlines
        colored_alignment = '\n'.join(lines)

        return colored_alignment

    def write_alignment(
        self,
        output_file: str,
        append_consensus: bool = True,
        mask_rip: bool = True,
        consensus_id: str = 'deRIPseq',
        format: str = 'fasta',
    ) -> None:
        """
        Write alignment to file with options to append consensus and mask RIP positions.

        Parameters
        ----------
        output_file : str
            Path to the output alignment file.
        append_consensus : bool, optional
            Whether to append the consensus sequence to the alignment (default: True).
        mask_rip : bool, optional
            Whether to mask RIP positions in the output alignment (default: True).
        consensus_id : str, optional
            ID for the consensus sequence if appended (default: "deRIPseq").
        format : str, optional
            Format for the output alignment file (default: "fasta").

        Returns
        -------
        None
            Writes alignment to file.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.consensus_tracker is None:
            raise ValueError('Must call calculate_rip before writing output')

        # Select alignment based on masking preference
        source_alignment = self.masked_alignment if mask_rip else self.alignment

        # Write the alignment file
        ao.writeAlign(
            self.consensus_tracker,
            source_alignment,
            output_file,
            ID=consensus_id,
            outAlnFormat=format,
            noappend=not append_consensus,
        )

        logger.info(f'Alignment written to {output_file}')

    def write_consensus(self, output_file: str, consensus_id: str = 'deRIPseq') -> None:
        """
        Write the deRIPed consensus sequence to a FASTA file.

        Parameters
        ----------
        output_file : str
            Path to the output FASTA file.
        consensus_id : str, optional
            ID for the consensus sequence (default: "deRIPseq").

        Returns
        -------
        None
            Writes consensus sequence to file.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.consensus_tracker is None:
            raise ValueError('Must call calculate_rip before writing output')

        # Write the sequence to file
        ao.writeDERIP(self.consensus_tracker, output_file, ID=consensus_id)

        logger.info(f'Consensus sequence written to {output_file}')

    def get_consensus_string(self) -> str:
        """
        Get the deRIPed consensus sequence as a string.

        Returns
        -------
        str
            The deRIPed consensus sequence.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.consensus is None:
            raise ValueError('Must call calculate_rip before accessing consensus')

        return str(self.consensus.seq)

    def rip_summary(self) -> None:
        """
        Return a summary of RIP mutations found in each sequence as str.

        Returns
        -------
        str
            Summary of RIP mutations by sequence.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.rip_counts is None:
            raise ValueError('Must call calculate_rip before printing RIP summary')

        return ao.summarizeRIP(self.rip_counts)

    def plot_alignment(
        self,
        output_file: Optional[str] = None,
        dpi: int = 300,
        title: Optional[str] = None,
        width: int = 20,
        height: int = 15,
        palette: str = 'derip2',
        column_ranges: Optional[List[Tuple[int, int, str, str]]] = None,
        show_chars: bool = False,
        draw_boxes: bool = False,
        show_rip: str = 'both',
        highlight_corrected: bool = True,
        flag_corrected: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate a visualization of the alignment with RIP mutations highlighted.

        This method creates a PNG image showing the aligned sequences with color-coded
        highlighting of RIP mutations and corrections. It displays the consensus sequence
        below the alignment with asterisks marking corrected positions.

        Parameters
        ----------
        output_file : str
            Path to save the output image file.
        dpi : int, optional
            Resolution of the output image in dots per inch (default: 300).
        title : str, optional
            Title to display on the image (default: None).
        width : int, optional
            Width of the output image in inches (default: 20).
        height : int, optional
            Height of the output image in inches (default: 15).
        palette : str, optional
            Color palette to use: 'colorblind', 'bright', 'tetrimmer', 'basegrey', or 'derip2' (default: 'basegrey').
        column_ranges : List[Tuple[int, int, str, str]], optional
            List of column ranges to mark, each as (start_col, end_col, color, label) (default: None).
        show_chars : bool, optional
            Whether to display sequence characters inside the colored cells (default: False).
        draw_boxes : bool, optional
            Whether to draw black borders around highlighted bases (default: False).
        show_rip : str, optional
            Which RIP markup categories to include: 'substrate', 'product', or 'both' (default: 'both').
        highlight_corrected : bool, optional
            If True, only corrected positions in the consensus will be colored, all others will be gray (default: True).
        flag_corrected : bool, optional
            If True, corrected positions in the alignment will be marked with asterisks (default: False).
        **kwargs
            Additional keyword arguments to pass to drawMiniAlignment function.

        Returns
        -------
        str
            Path to the output image file.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.

        Notes
        -----
        The visualization uses different colors to distinguish RIP-related mutations:
        - Red: RIP products (typically T from C→T mutations)
        - Blue: RIP substrates (unmutated nucleotides in RIP context)
        - Yellow: Non-RIP deaminations (only if reaminate=True)
        - Target bases are displayed in black text, while surrounding context is in grey text
        """
        # Check if calculate_rip has been run
        if self.markupdict is None or self.consensus is None:
            raise ValueError('Must call calculate_rip before plotting alignment')

        # Import minialign here to avoid circular imports
        from derip2.plotting.minialign import drawMiniAlignment

        # Extract column indices of corrected positions for marking with asterisks
        corrected_pos = (
            list(self.corrected_positions.keys()) if self.corrected_positions else []
        )

        # Call drawMiniAlignment with the alignment object and parameters from this object and user inputs
        _t_plot = time.perf_counter()
        result = drawMiniAlignment(
            alignment=self.alignment,
            outfile=output_file or '',
            dpi=dpi,
            title=title,
            width=width,
            height=height,
            markupdict=self.markupdict,
            palette=palette,
            column_ranges=column_ranges,
            show_chars=show_chars,
            draw_boxes=draw_boxes,
            consensus_seq=str(self.gapped_consensus.seq),
            corrected_positions=corrected_pos,
            reaminate=self.reaminate,
            reference_seq_index=self.fill_index,
            show_rip=show_rip,
            highlight_corrected=highlight_corrected,
            flag_corrected=flag_corrected,
            **kwargs,  # Pass any additional customization options
        )

        logger.debug(
            f'plot_alignment: drawMiniAlignment took {time.perf_counter() - _t_plot:.3f}s'
        )
        if not kwargs.get('return_figure'):
            logger.info(f'Alignment visualization saved to {output_file}')
        return result

    def calculate_rsi(self, ambiguous: str = 'split', substrate_scope: str = 'all'):
        """
        Calculate the RIP Strandedness Imbalance (RSI) for every sequence.

        RSI is ``p_fwd - p_rev``, the difference between the proportion of
        forward-strand substrate (CpA) and reverse-strand substrate (TpG) that
        RIP has converted to TpA. It lies in ``[-1, 1]``: positive means RIP
        acted mainly on the forward strand, negative mainly on the reverse.

        Because a single round of meiotic RIP acts on one strand of a duplex,
        a strongly imbalanced sequence is the signature of one round of RIP,
        while a balanced one has either escaped RIP or been RIP'd repeatedly on
        both strands. ``p_fwd`` and ``p_rev`` separate those two cases.

        Parameters
        ----------
        ambiguous : {'split', 'exclude', 'weight', 'both'}, optional
            How to attribute TpA dinucleotides that could have arisen from RIP
            on either strand (default: ``'split'``, half to each).
        substrate_scope : {'all', 'assessable', 'rip_like_columns'}, optional
            Which unmutated substrate dinucleotides enter the denominators
            (default: ``'all'``).

        Returns
        -------
        derip2.stats.strand_bias.RSIResult
            Per-sequence RSI, its components, ambiguity counts and significance.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first.

        See Also
        --------
        derip2.stats.strand_bias.compute_rsi : The underlying calculation.

        Examples
        --------
        >>> d = DeRIP('alignment.fa')            # doctest: +SKIP
        >>> d.calculate_rip()                    # doctest: +SKIP
        >>> d.calculate_rsi().rsi                # doctest: +SKIP
        array([ 0.9, -0.8,  0.0])
        """
        from derip2.stats import compute_rsi

        self._require_rip('calculating RSI')
        self.rsi_result = compute_rsi(
            self.column_classes,
            ambiguous=ambiguous,
            substrate_scope=substrate_scope,
        )

        # Annotate the records, mirroring calculate_cri_for_all.
        for i, record in enumerate(self.alignment):
            if not hasattr(record, 'annotations'):
                record.annotations = {}
            record.annotations['RSI'] = float(self.rsi_result.rsi[i])
            record.annotations['p_fwd'] = float(self.rsi_result.p_fwd[i])
            record.annotations['p_rev'] = float(self.rsi_result.p_rev[i])

        logger.info(f'Calculated RSI for {len(self.alignment)} sequences')
        return self.rsi_result

    def get_rsi_values(self, **kwargs):
        """
        Return per-sequence RSI values, calculating them if needed.

        Parameters
        ----------
        **kwargs
            Passed to :meth:`calculate_rsi` when RSI has not yet been computed.

        Returns
        -------
        list of dict
            One record per sequence, in alignment order.
        """
        if self.rsi_result is None:
            self.calculate_rsi(**kwargs)
        return self.rsi_result.as_records([r.id for r in self.alignment])

    def sort_by_rsi(self, descending: bool = True, inplace: bool = False):
        """
        Sort the alignment by RIP strandedness imbalance.

        Parameters
        ----------
        descending : bool, optional
            If True (default), sequences with the most forward-strand RIP come
            first and those with the most reverse-strand RIP last.
        inplace : bool, optional
            If True, replace the current alignment and discard all computed
            results, which must then be recalculated (default: False).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            The sorted alignment.

        Notes
        -----
        Sequences whose RSI is undefined (NaN, because one strand carries no
        substrate and no product) sort to the end regardless of direction. They
        carry no evidence, so placing them at either extreme would misrepresent
        them.
        """
        import math

        from Bio.Align import MultipleSeqAlignment

        self.get_rsi_values()

        def _sort_key(record):
            rsi = record.annotations['RSI']
            # NaN sorts last in both directions.
            if math.isnan(rsi):
                return (1, 0.0)
            return (0, -rsi if descending else rsi)

        sorted_alignment = MultipleSeqAlignment(sorted(self.alignment, key=_sort_key))

        if inplace:
            self.alignment = sorted_alignment
            logger.info('Updated alignment in-place with RSI-sorted sequences')
            self._invalidate_results()

        return sorted_alignment

    def summarize_stats(self, ambiguous: str = 'split'):
        """
        Build a per-sequence table of every RIP statistic deRIP2 computes.

        Combines the RIP event counts from the alignment scan, the classical
        composite RIP index (CRI) and its components, GC content, and the
        strandedness imbalance (RSI) with its components and significance.

        Parameters
        ----------
        ambiguous : {'split', 'exclude', 'weight', 'both'}, optional
            Ambiguity policy for RSI (default: ``'split'``). RSI is recomputed
            whenever this differs from the cached result's policy.

        Returns
        -------
        pandas.DataFrame
            One row per sequence, in alignment order.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first.
        """
        import pandas as pd

        self._require_rip('summarizing stats')

        if self.rsi_result is None or self.rsi_result.ambiguous != ambiguous:
            self.calculate_rsi(ambiguous=ambiguous)
        cri_values = self.get_cri_values()
        rsi_records = self.rsi_result.as_records([r.id for r in self.alignment])

        rows = []
        for i, record in enumerate(self.alignment):
            counts = self.rip_counts[i]
            rsi = rsi_records[i]
            rows.append(
                {
                    'index': i,
                    'ID': record.id,
                    'GC': counts.GC,
                    'CRI': cri_values[i]['CRI'],
                    'PI': cri_values[i]['PI'],
                    'SI': cri_values[i]['SI'],
                    'RSI': rsi['RSI'],
                    'p_fwd': rsi['p_fwd'],
                    'p_rev': rsi['p_rev'],
                    'fwd_product': rsi['fwd_product'],
                    'fwd_substrate': rsi['fwd_substrate'],
                    'rev_product': rsi['rev_product'],
                    'rev_substrate': rsi['rev_substrate'],
                    'n_ambiguous': rsi['n_ambiguous'],
                    'RIP_fwd': counts.RIPcount,
                    'RIP_rev': counts.revRIPcount,
                    'non_RIP': counts.nonRIPcount,
                    'pvalue': rsi['pvalue'],
                }
            )

        return pd.DataFrame(rows)

    def stats_summary(self, ambiguous: str = 'split') -> str:
        """
        Format :meth:`summarize_stats` as a table for terminal output.

        Parameters
        ----------
        ambiguous : str, optional
            Ambiguity policy (default: ``'split'``).

        Returns
        -------
        str
            The stats table, ready to print.
        """
        from io import StringIO

        df = self.summarize_stats(ambiguous=ambiguous).copy()
        for col in ('GC', 'CRI', 'PI', 'SI', 'RSI', 'p_fwd', 'p_rev'):
            df[col] = df[col].map('{:.3f}'.format)
        for col in ('fwd_product', 'fwd_substrate', 'rev_product', 'rev_substrate'):
            df[col] = df[col].map('{:.1f}'.format)
        df['pvalue'] = df['pvalue'].map('{:.3g}'.format)

        buffer = StringIO()
        df.to_string(buffer, index=False)
        return buffer.getvalue()

    def write_stats(self, output_file: str, ambiguous: str = 'split') -> str:
        """
        Write the per-sequence statistics table to a TSV file.

        Parameters
        ----------
        output_file : str
            Destination path.
        ambiguous : str, optional
            Ambiguity policy (default: ``'split'``).

        Returns
        -------
        str
            The path written.
        """
        df = self.summarize_stats(ambiguous=ambiguous)
        df.to_csv(output_file, sep='\t', index=False, float_format='%.6g')
        logger.info(f'Statistics table written to {output_file}')
        return output_file

    def plot_strand_bias(
        self,
        output_file: Optional[str] = None,
        mode: str = 'rip',
        **kwargs,
    ):
        """
        Draw a diverging stacked-bar chart of per-column RIP strand bias.

        Bars are drawn above the axis where the deamination is observed on the
        forward strand and below it where it is observed on the reverse strand.

        Parameters
        ----------
        output_file : str, optional
            Path to write the figure to. Use ``.svg`` or ``.pdf`` for
            publication output.
        mode : {'rip', 'non_rip', 'all_deamination'}, optional
            Which deamination events to display (default: ``'rip'``).
        **kwargs
            Additional options forwarded to
            :func:`derip2.plotting.strandbias.plot_strand_bias`, such as
            ``scale``, ``stack``, ``xaxis``, ``color_by``, ``emphasis`` and
            ``column_range``. ``columns`` selects which positions are lettered
            when ``xaxis`` is ``'logo'`` or ``'derip'``; every column is drawn
            as a bar regardless.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first.
        """
        from derip2.plotting.strandbias import plot_strand_bias

        self._require_rip('plotting strand bias')

        if kwargs.get('xaxis') == 'derip' and 'consensus_seq' not in kwargs:
            kwargs['consensus_seq'] = str(self.gapped_consensus.seq)

        return plot_strand_bias(
            self.column_classes, outfile=output_file, mode=mode, **kwargs
        )

    def calculate_spectra(
        self,
        partition_by: str = 'none',
        ancestor=None,
        samples=None,
        context: str = 'trinucleotide',
    ):
        """
        Compute the SBS-96 and SBS-192 trinucleotide mutation spectra.

        Every alignment cell whose base differs from the ancestral reference at
        that column is counted as one substitution event, with its trinucleotide
        context read from the ancestor using nearest non-gap bases. Events are
        folded to the pyrimidine strand for the SBS-96 matrix and kept
        strand-resolved for the SBS-192 matrix.

        This is the tree-free *baseline* spectrum: the ancestor is deRIP2's
        reconstructed consensus (or a user-supplied sequence), so recurrence is
        reported only as a multi-hit-column proxy. Correct independent-event
        counting requires the phylogenetic path.

        Parameters
        ----------
        partition_by : {'none', 'row'}, optional
            How to split the spectra into samples. ``'none'`` (default) pools
            every sequence into one ``AllSequences`` sample; ``'row'`` gives one
            sample column per input sequence. Ignored when ``samples`` is given.
        ancestor : str or Bio.SeqRecord.SeqRecord, optional
            Ancestral reference sequence, one base per alignment column. Defaults
            to deRIP2's gapped consensus (:attr:`gapped_consensus`).
        samples : sequence of str, optional
            An explicit per-row sample label (length equal to the number of
            sequences), e.g. species or group names. Overrides ``partition_by``
            when provided.
        context : {'trinucleotide', 'downstream'}, optional
            Which sequence context to classify substitutions by (default:
            ``'trinucleotide'``). ``'downstream'`` builds the pyrimidine-folded
            downstream-triplet matrix (CHG-aware) with no strand-resolved form.

        Returns
        -------
        derip2.stats.mutation_spectra.SpectraResult
            The assembled spectra, per-event detail and homoplasy proxy.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first, or if
            ``partition_by`` is not recognised.

        See Also
        --------
        derip2.stats.mutation_spectra.compute_spectra : The underlying calculation.
        """
        from derip2.stats import compute_spectra

        self._require_rip('calculating spectra')

        if ancestor is None:
            ancestor_seq = str(self.gapped_consensus.seq)
        elif hasattr(ancestor, 'seq'):
            ancestor_seq = str(ancestor.seq)
        else:
            ancestor_seq = str(ancestor)

        if samples is not None:
            pass  # explicit per-row labels take precedence
        elif partition_by == 'none':
            samples = None
        elif partition_by == 'row':
            samples = [record.id for record in self.alignment]
        else:
            raise ValueError(
                f"partition_by must be 'none' or 'row', got {partition_by!r}"
            )

        self.spectra_result = compute_spectra(
            self.column_classes, ancestor_seq, samples=samples, context=context
        )
        logger.info(
            'Computed mutation spectra: %d events across %d sample(s)',
            self.spectra_result.event_rows.size,
            len(self.spectra_result.sample_names),
        )
        return self.spectra_result

    def write_spectra_matrix(self, output_file: str, kind: str = '96', **kwargs) -> str:
        """
        Write a SigProfiler-compliant SBS matrix, computing spectra if needed.

        Parameters
        ----------
        output_file : str
            Destination path for the tab-separated matrix file.
        kind : {'96', '192'}, optional
            Which spectrum matrix to write (default: ``'96'``).
        **kwargs
            Passed to :meth:`calculate_spectra` when spectra have not yet been
            computed.

        Returns
        -------
        str
            The path written.
        """
        from derip2.spectra import write_sbs_matrix

        if self.spectra_result is None:
            self.calculate_spectra(**kwargs)
        return write_sbs_matrix(self.spectra_result, output_file, kind=kind)

    def plot_spectra(
        self, output_file: Optional[str] = None, kind: str = '96', **kwargs
    ):
        """
        Draw an SBS mutation-spectrum figure, computing spectra if needed.

        Parameters
        ----------
        output_file : str, optional
            Path to write the figure to. Use ``.svg`` or ``.pdf`` for
            publication output.
        kind : {'96', '192', 'downstream', 'strand', 'homoplasy'}, optional
            Which figure to draw: the SBS-96 spectrum (default), the SBS-192
            strand-resolved spectrum, the pyrimidine-folded downstream-triplet
            spectrum, the strand-asymmetry panel, or the homoplasy (recurrence)
            plot.
        **kwargs
            Forwarded to the underlying plotting function (e.g. ``title``,
            ``percentage``, ``min_hits``), except any that
            :meth:`calculate_spectra` consumes when spectra are first computed.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        Raises
        ------
        ValueError
            If ``kind`` is not recognised.
        """
        from derip2.plotting import spectra as spectra_plots

        if self.spectra_result is None:
            self.calculate_spectra()

        plotters = {
            '96': spectra_plots.plot_sbs96,
            '192': spectra_plots.plot_sbs192,
            'downstream': spectra_plots.plot_downstream,
            'strand': spectra_plots.plot_strand_asymmetry,
            'homoplasy': spectra_plots.plot_homoplasy,
        }
        if kind not in plotters:
            raise ValueError(f'kind must be one of {sorted(plotters)}, got {kind!r}')
        return plotters[kind](self.spectra_result, output_file, **kwargs)

    def calculate_flank_spectra(self):
        """
        Compute the flanking-context spectra of RIP-like sites.

        Classifies every RIP-like dinucleotide by the single base one position
        upstream and one downstream (a 4 bp motif). Surviving substrate sites
        (``CpA``/``TpG`` anywhere in each sequence) and RIP product sites
        (``TpA`` in RIP-informative columns) are counted separately, one sample
        column per input sequence, folded onto ``CA``/``TA``-equivalent channels.
        The result is cached on :attr:`flank_spectra_result`.

        Returns
        -------
        derip2.stats.flank_spectra.FlankSpectraResult
            The four ``(16, n_rows)`` count matrices and per-state skipped counts.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first.

        See Also
        --------
        derip2.stats.flank_spectra.compute_flank_spectra : The calculation.
        """
        from derip2.stats.flank_spectra import compute_flank_spectra

        self._require_rip('calculating flank spectra')
        sample_names = [record.id for record in self.alignment]
        self.flank_spectra_result = compute_flank_spectra(
            self.column_classes, sample_names=sample_names
        )
        return self.flank_spectra_result

    def write_flank_spectra_matrix(self, output_file: str) -> str:
        """
        Write the flank-context spectra as a tidy TSV, computing them if needed.

        Parameters
        ----------
        output_file : str
            Destination path for the tab-separated matrix file.

        Returns
        -------
        str
            The path written.
        """
        from derip2.stats.flank_spectra import write_flank_matrix

        if self.flank_spectra_result is None:
            self.calculate_flank_spectra()
        return write_flank_matrix(self.flank_spectra_result, output_file)

    def write_flank_spectra_comparisons(
        self, output_file: str, *, min_sites: int = 20
    ) -> str:
        """
        Write the per-sequence flank-context comparison stats, computing if needed.

        Parameters
        ----------
        output_file : str
            Destination path for the tab-separated comparison file.
        min_sites : int, optional
            Minimum site count on both sides for the chi-squared reliability flag
            (default: 20).

        Returns
        -------
        str
            The path written.
        """
        from derip2.stats.flank_spectra import write_flank_comparisons

        if self.flank_spectra_result is None:
            self.calculate_flank_spectra()
        return write_flank_comparisons(
            self.flank_spectra_result, output_file, min_sites=min_sites
        )

    def plot_flank_spectra(self, output_file: Optional[str] = None, **kwargs):
        """
        Draw the pooled flank-context bihistograms, computing spectra if needed.

        Parameters
        ----------
        output_file : str, optional
            Path to write the figure to. Use ``.svg`` or ``.pdf`` for
            publication output.
        **kwargs
            Forwarded to
            :func:`derip2.plotting.flank_spectra.plot_flank_bihistograms_pooled`
            (e.g. ``title``, ``percentage``, ``width``).

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        from derip2.plotting.flank_spectra import plot_flank_bihistograms_pooled

        if self.flank_spectra_result is None:
            self.calculate_flank_spectra()
        return plot_flank_bihistograms_pooled(
            self.flank_spectra_result, output_file, **kwargs
        )

    def write_html_report(
        self,
        output_file: str,
        title: Optional[str] = None,
        ambiguous: str = 'split',
        **kwargs,
    ) -> str:
        """
        Write a self-contained HTML report of the strand-bias analysis.

        The report embeds three strand-bias figures (RIP-like mutations,
        non-RIP deamination, and all deamination) as inline SVG, alongside the
        per-sequence statistics table. It has no external assets, so it can be
        emailed or archived as a single file.

        Parameters
        ----------
        output_file : str
            Destination path for the HTML file.
        title : str, optional
            Report heading.
        ambiguous : str, optional
            Ambiguity policy for RSI (default: ``'split'``).
        **kwargs
            Forwarded to each figure, e.g. ``scale``, ``xaxis``, ``columns``.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first.
        """
        from derip2.report import write_html_report

        self._require_rip('writing an HTML report')
        return write_html_report(
            self, output_file, title=title, ambiguous=ambiguous, **kwargs
        )

    def write_per_sequence_report(
        self,
        output_file: str,
        title: Optional[str] = None,
        ambiguous: str = 'split',
        max_seqs: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Write a single-file, interactive per-sequence HTML report.

        The report renders one panel per input sequence — the alignment row with
        RIP sites highlighted, a fixed-height per-sequence strand-bias strip, a
        per-sequence SBS-96 spectrum against the reconstructed ancestor, and that
        sequence's summary statistics — and lets the reader step between
        sequences with the arrow keys. Every figure is inline SVG, so the file is
        self-contained.

        Parameters
        ----------
        output_file : str
            Destination path for the HTML file.
        title : str, optional
            Report heading.
        ambiguous : str, optional
            Ambiguity policy for the per-sequence RSI statistics
            (default: ``'split'``).
        max_seqs : int, optional
            Cap the number of sequence panels; the strongest strand-bias
            sequences are kept. ``None`` (default) renders every sequence.
        **kwargs
            Forwarded to :func:`derip2.persequence_report.write_per_sequence_report`.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first.
        """
        from derip2.persequence_report import write_per_sequence_report

        self._require_rip('writing a per-sequence report')
        return write_per_sequence_report(
            self,
            output_file,
            title=title,
            ambiguous=ambiguous,
            max_seqs=max_seqs,
            **kwargs,
        )

    def calculate_dinucleotide_frequency(self, sequence):
        """
        Calculate the frequency of specific dinucleotides in a sequence.

        Parameters
        ----------
        sequence : str
            The DNA sequence to analyze.

        Returns
        -------
        dict
            A dictionary with dinucleotide counts.
        """
        # Convert to uppercase and remove gaps
        seq = sequence.upper().replace('-', '')

        # Count dinucleotides
        dinucleotides = {'TpA': 0, 'ApT': 0, 'CpA': 0, 'TpG': 0, 'ApC': 0, 'GpT': 0}

        for i in range(len(seq) - 1):
            di = seq[i : i + 2]
            if di == 'TA':
                dinucleotides['TpA'] += 1
            elif di == 'AT':
                dinucleotides['ApT'] += 1
            elif di == 'CA':
                dinucleotides['CpA'] += 1
            elif di == 'TG':
                dinucleotides['TpG'] += 1
            elif di == 'AC':
                dinucleotides['ApC'] += 1
            elif di == 'GT':
                dinucleotides['GpT'] += 1

        return dinucleotides

    def calculate_cri(self, sequence):
        """
        Calculate the Composite RIP Index (CRI) for a DNA sequence.

        Parameters
        ----------
        sequence : str
            The DNA sequence to analyze.

        Returns
        -------
        tuple
            (cri, pi, si) - Composite RIP Index, Product Index, and Substrate Index.
        """
        dinucleotides = self.calculate_dinucleotide_frequency(sequence)

        # Calculate RIP product index (PI) = TpA / ApT
        pi = (
            dinucleotides['TpA'] / dinucleotides['ApT']
            if dinucleotides['ApT'] != 0
            else 0
        )

        # Calculate RIP substrate index (SI) = (CpA + TpG) / (ApC + GpT)
        numerator = dinucleotides['CpA'] + dinucleotides['TpG']
        denominator = dinucleotides['ApC'] + dinucleotides['GpT']
        si = numerator / denominator if denominator != 0 else 0

        # Calculate composite RIP index (CRI) = PI - SI
        cri = pi - si

        return cri, pi, si

    def calculate_cri_for_all(self):
        """
        Calculate the Composite RIP Index (CRI) for each sequence in the alignment
        and assign CRI values as annotations to each sequence record.

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            The alignment with CRI metadata added to each record.

        Notes
        -----
        This method calculates:
        - Product Index (PI) = TpA / ApT
        - Substrate Index (SI) = (CpA + TpG) / (ApC + GpT)
        - Composite RIP Index (CRI) = PI - SI

        High CRI values indicate strong RIP activity.
        """
        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Process each sequence in the alignment
        for record in self.alignment:
            # Calculate CRI, PI, and SI for this sequence
            cri, pi, si = self.calculate_cri(str(record.seq))

            # Update the description to include CRI information
            if record.description == record.id:
                record.description = (
                    f'{record.id} CRI={cri:.4f} PI={pi:.4f} SI={si:.4f}'
                )
            else:
                record.description += f' CRI={cri:.4f} PI={pi:.4f} SI={si:.4f}'

            # Add CRI values as annotations
            if not hasattr(record, 'annotations'):
                record.annotations = {}

            record.annotations['CRI'] = cri
            record.annotations['PI'] = pi
            record.annotations['SI'] = si

        logger.info(f'Calculated CRI values for {len(self.alignment)} sequences')
        return self.alignment

    def get_cri_values(self):
        """
        Return a list of CRI values for all sequences in the alignment.

        If a sequence doesn't have a CRI value yet, calculate it first.

        Returns
        -------
        list of dict
            List of dictionaries containing CRI, PI, SI values and sequence ID,
            in the same order as sequences appear in the alignment.
        """
        if self.alignment is None:
            raise ValueError('No alignment loaded')

        cri_values = []

        # Process each sequence in the alignment
        for record in self.alignment:
            # Check if CRI is already calculated
            if not hasattr(record, 'annotations') or 'CRI' not in record.annotations:
                # Calculate CRI for this sequence
                cri, pi, si = self.calculate_cri(str(record.seq))

                # Store values in annotations
                if not hasattr(record, 'annotations'):
                    record.annotations = {}

                record.annotations['CRI'] = cri
                record.annotations['PI'] = pi
                record.annotations['SI'] = si

            # Add values to result list
            cri_values.append(
                {
                    'id': record.id,
                    'CRI': record.annotations['CRI'],
                    'PI': record.annotations['PI'],
                    'SI': record.annotations['SI'],
                }
            )

        return cri_values

    def sort_by_cri(self, descending=True, inplace=False):
        """
        Sort the alignment by CRI score.

        Parameters
        ----------
        descending : bool, optional
            If True, sort in descending order (highest CRI first). Default: True.
        inplace : bool, optional
            If True, replace the current alignment with the sorted alignment.
            If False, return a new alignment without modifying the original (default: False).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            A new alignment with sequences sorted by CRI score.
        """
        from Bio.Align import MultipleSeqAlignment

        # Ensure all sequences have CRI values
        self.get_cri_values()

        # Sort records by CRI score
        sorted_records = sorted(
            self.alignment,
            key=lambda record: record.annotations['CRI'],
            reverse=descending,
        )

        # Create a new alignment with the sorted records
        sorted_alignment = MultipleSeqAlignment(sorted_records)

        # Replace current alignment if inplace=True
        if inplace:
            self.alignment = sorted_alignment
            logger.info('Updated alignment in-place with CRI-sorted sequences')

            # Clear calculated results since alignment changed
            self._invalidate_results()

        return sorted_alignment

    def summarize_cri(self):
        """
        Generate a formatted table summarizing CRI values for all sequences.

        Returns
        -------
        str
            A formatted string containing the CRI summary table.
        """
        from io import StringIO

        import pandas as pd

        # Ensure all sequences have CRI values
        cri_data = self.get_cri_values()

        # Create DataFrame
        df = pd.DataFrame(cri_data)

        # Format floating point columns
        for col in ['CRI', 'PI', 'SI']:
            df[col] = df[col].map('{:.4f}'.format)

        # Use StringIO to capture formatted output
        buffer = StringIO()
        df.to_string(buffer, index=False)

        return buffer.getvalue()

    def filter_by_cri(self, min_cri=0.0, inplace=False):
        """
        Filter the alignment to remove sequences with CRI values below a threshold.

        Parameters
        ----------
        min_cri : float, optional
            Minimum CRI value to keep a sequence in the alignment (default: 0.0).
        inplace : bool, optional
            If True, replace the current alignment with the filtered alignment.
            If False, return a new alignment without modifying the original (default: False).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            A new alignment containing only sequences with CRI values >= min_cri.

        Raises
        ------
        ValueError
            If no alignment is loaded or if filtering would remove all sequences.
        Warning
            If fewer than 2 sequences remain after filtering.

        Notes
        -----
        CRI values will be calculated for sequences that don't already have them.
        If inplace=True, this will modify the original alignment in the DeRIP object.
        """
        import warnings

        from Bio.Align import MultipleSeqAlignment

        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have CRI values
        self.get_cri_values()

        # Filter sequences based on CRI threshold
        filtered_records = [
            record for record in self.alignment if record.annotations['CRI'] >= min_cri
        ]

        # Check if any sequences remain after filtering
        if not filtered_records:
            raise ValueError(
                f'No sequences remain after filtering with min_cri={min_cri}. '
                f'The highest CRI value in the alignment is {max([r.annotations["CRI"] for r in self.alignment]):.4f}'
            )

        # Warn if fewer than 2 sequences remain
        if len(filtered_records) < 2:
            passed_records = [(r.id, r.annotations['CRI']) for r in filtered_records]
            failed_records = [
                (r.id, r.annotations['CRI'])
                for r in self.alignment
                if r.id not in [rec.id for rec in filtered_records]
            ]

            print(
                f'DEBUG: Records that passed filter threshold {min_cri}: {len(passed_records)}',
                file=sys.stderr,
            )
            print(
                f'DEBUG: Records that failed filter threshold {min_cri}: {len(failed_records)}',
                file=sys.stderr,
            )

            warnings.warn(
                f'Only {len(filtered_records)} sequence remains after CRI filtering. DeRIP works best with multiple sequences.',
                stacklevel=2,
            )
        elif len(filtered_records) < len(self.alignment):
            logger.info(
                f'CRI filtering removed {len(self.alignment) - len(filtered_records)} sequences '
                f'({len(filtered_records)}/{len(self.alignment)} sequences remaining)'
            )

        # Create new alignment with filtered records
        filtered_alignment = MultipleSeqAlignment(filtered_records)

        # Replace current alignment if inplace=True
        if inplace:
            self.alignment = filtered_alignment
            logger.info('Updated alignment in-place with CRI-filtered sequences')

            # Clear calculated results since alignment changed
            self._invalidate_results()

        return filtered_alignment

    def keep_low_cri(self, n=2, inplace=False):
        """
        Retain only the n sequences with the lowest CRI values.

        Parameters
        ----------
        n : int, optional
            Number of sequences with lowest CRI values to keep (default: 2).
        inplace : bool, optional
            If True, replace the current alignment with the filtered alignment.
            If False, return a new alignment without modifying the original (default: False).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            A new alignment containing only the n sequences with lowest CRI values.

        Raises
        ------
        ValueError
            If no alignment is loaded.

        Notes
        -----
        CRI values will be calculated for sequences that don't already have them.
        If inplace=True, this will modify the original alignment in the DeRIP object.
        If n is greater than the number of sequences, no filtering occurs.
        If n is less than 2, no filtering occurs to ensure DeRIP has enough sequences to work with.
        """

        from Bio.Align import MultipleSeqAlignment

        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have CRI values
        self.get_cri_values()

        # Check if n exceeds alignment length
        if n >= len(self.alignment):
            logger.info(
                f'Requested to keep {n} sequences but alignment only has {len(self.alignment)}. No filtering performed.'
            )
            return self.alignment

        # Check if n is too small
        if n < 2:
            logger.warning(
                f'Cannot keep fewer than 2 sequences (requested {n}). DeRIP works best with multiple sequences. No filtering performed.'
            )
            return self.alignment

        # Sort records by CRI values (ascending order - lowest first)
        sorted_records = sorted(
            self.alignment, key=lambda record: record.annotations['CRI']
        )

        # Keep only the first n sequences with lowest CRI
        kept_records = sorted_records[:n]

        # Create new alignment with kept records
        kept_alignment = MultipleSeqAlignment(kept_records)

        # Log which sequences were kept
        kept_ids = [record.id for record in kept_records]
        cri_values = [record.annotations['CRI'] for record in kept_records]
        logger.info(
            f'Kept {n} sequences with lowest CRI values: {list(zip(kept_ids, cri_values))}'
        )
        logger.info(
            f'Removed {len(self.alignment) - n} sequences with higher CRI values'
        )

        # Replace current alignment if inplace=True
        if inplace:
            self.alignment = kept_alignment
            logger.info('Updated alignment in-place with low-CRI filtered sequences')

            # Clear calculated results since alignment changed
            self._invalidate_results()

        return kept_alignment

    def get_gc_content(self):
        """
        Calculate and return the GC content for all sequences in the alignment.

        Returns
        -------
        list of dict
            List of dictionaries containing sequence ID and GC content,
            in the same order as sequences appear in the alignment.

        Raises
        ------
        ValueError
            If no alignment is loaded.
        """
        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Import the gc_fraction function from Bio.SeqUtils
        from Bio.SeqUtils import gc_fraction

        gc_values = []

        # Process each sequence in the alignment
        for record in self.alignment:
            # Get sequence without gaps - using string replacement instead of ungap method
            seq_no_gaps = str(record.seq).replace('-', '')

            # Calculate GC content using Bio.SeqUtils.gc_fraction
            # This returns a value between 0 and 1, which is what we want
            gc_content = gc_fraction(seq_no_gaps)

            # Store GC content in annotations
            if not hasattr(record, 'annotations'):
                record.annotations = {}

            record.annotations['GC_content'] = gc_content

            # Update the description to include GC content if not already present
            if 'GC=' not in record.description:
                if record.description == record.id:
                    record.description = f'{record.id} GC={gc_content:.4f}'
                else:
                    record.description += f' GC={gc_content:.4f}'

            # Add GC content to result list
            gc_values.append({'id': record.id, 'GC_content': gc_content})

        return gc_values

    def filter_by_gc(self, min_gc=0.0, inplace=False):
        """
        Filter the alignment to remove sequences with GC content below a threshold.

        Parameters
        ----------
        min_gc : float, optional
            Minimum GC content to keep a sequence in the alignment (default: 0.0).
            Value should be between 0.0 and 1.0.
        inplace : bool, optional
            If True, replace the current alignment with the filtered alignment.
            If False, return a new alignment without modifying the original (default: False).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            A new alignment containing only sequences with GC content >= min_gc.

        Raises
        ------
        ValueError
            If no alignment is loaded or if filtering would remove all sequences.
        Warning
            If fewer than 2 sequences remain after filtering.

        Notes
        -----
        GC content will be calculated for sequences that don't already have it.
        If inplace=True, this will modify the original alignment in the DeRIP object.
        """
        import warnings

        from Bio.Align import MultipleSeqAlignment

        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have GC content values
        self.get_gc_content()

        # Validate min_gc is in valid range
        if not 0.0 <= min_gc <= 1.0:
            raise ValueError(f'min_gc must be between 0.0 and 1.0, got {min_gc}')

        # Filter sequences based on GC content threshold
        filtered_records = [
            record
            for record in self.alignment
            if record.annotations['GC_content'] >= min_gc
        ]

        # Check if any sequences remain after filtering
        if not filtered_records:
            max_gc = max([r.annotations['GC_content'] for r in self.alignment])
            raise ValueError(
                f'No sequences remain after filtering with min_gc={min_gc}. '
                f'The highest GC content in the alignment is {max_gc:.4f}'
            )

        # Warn if fewer than 2 sequences remain
        if len(filtered_records) < 2:
            warnings.warn(
                f'Only {len(filtered_records)} sequence remains after GC filtering. '
                f'DeRIP works best with multiple sequences.',
                stacklevel=2,
            )
        elif len(filtered_records) < len(self.alignment):
            logger.info(
                f'GC filtering removed {len(self.alignment) - len(filtered_records)} sequences '
                f'({len(filtered_records)}/{len(self.alignment)} sequences remaining)'
            )

        # Create new alignment with filtered records
        filtered_alignment = MultipleSeqAlignment(filtered_records)

        # Replace current alignment if inplace=True
        if inplace:
            self.alignment = filtered_alignment
            logger.info('Updated alignment in-place with GC-filtered sequences')

            # Clear calculated results since alignment changed
            self._invalidate_results()

        return filtered_alignment

    def keep_high_gc(self, n=2, inplace=False):
        """
        Retain only the n sequences with the highest GC content.

        Parameters
        ----------
        n : int, optional
            Number of sequences with highest GC content to keep (default: 2).
        inplace : bool, optional
            If True, replace the current alignment with the filtered alignment.
            If False, return a new alignment without modifying the original (default: False).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            A new alignment containing only the n sequences with highest GC content.

        Raises
        ------
        ValueError
            If no alignment is loaded.

        Notes
        -----
        GC content will be calculated for sequences that don't already have it.
        If inplace=True, this will modify the original alignment in the DeRIP object.
        If n is greater than the number of sequences, no filtering occurs.
        If n is less than 2, no filtering occurs to ensure DeRIP has enough sequences to work with.
        """

        from Bio.Align import MultipleSeqAlignment

        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have GC content values
        self.get_gc_content()

        # Check if n exceeds alignment length
        if n >= len(self.alignment):
            logger.info(
                f'Requested to keep {n} sequences but alignment only has {len(self.alignment)}. No filtering performed.'
            )
            return self.alignment

        # Check if n is too small
        if n < 2:
            logger.warning(
                f'Cannot keep fewer than 2 sequences (requested {n}). DeRIP works best with multiple sequences. No filtering performed.'
            )
            return self.alignment

        # Sort records by GC content values (descending order - highest first)
        sorted_records = sorted(
            self.alignment,
            key=lambda record: record.annotations['GC_content'],
            reverse=True,
        )

        # Keep only the first n sequences with highest GC content
        kept_records = sorted_records[:n]

        # Create new alignment with kept records
        kept_alignment = MultipleSeqAlignment(kept_records)

        # Log which sequences were kept
        kept_ids = [record.id for record in kept_records]
        gc_values = [record.annotations['GC_content'] for record in kept_records]
        logger.info(
            f'Kept {n} sequences with highest GC content: {list(zip(kept_ids, gc_values))}'
        )
        logger.info(
            f'Removed {len(self.alignment) - n} sequences with lower GC content'
        )

        # Replace current alignment if inplace=True
        if inplace:
            self.alignment = kept_alignment
            logger.info('Updated alignment in-place with high-GC filtered sequences')

            # Clear calculated results since alignment changed
            self._invalidate_results()

        return kept_alignment
