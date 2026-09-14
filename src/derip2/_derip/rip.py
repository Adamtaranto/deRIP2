"""RIP detection and consensus correction."""

import logging
import time
from typing import Dict, Optional

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

import derip2.aln_ops as ao

logger = logging.getLogger(__name__)


class RIPCorrectionMixin:
    """
    RIP detection and consensus correction.

    Runs the column classification and consensus build, tracks corrected
    positions, renders the ANSI-coloured views lazily, and writes the consensus
    and (masked) alignment.
    """

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

        # Create the consensus sequence once (gapped), then derive the degapped
        # record from it rather than walking the tracker a second time.
        gapped_consensus = ao.getDERIP(tracker, ID=label, deGAP=False)
        consensus = SeqRecord(
            Seq(str(gapped_consensus.seq).replace('-', '')),
            id=gapped_consensus.id,
            name=gapped_consensus.name,
            description=gapped_consensus.description,
        )

        # Store results in attributes
        self.masked_alignment = masked_alignment
        self.consensus = consensus
        self.gapped_consensus = gapped_consensus
        self.consensus_tracker = tracker
        self.rip_counts = rip_counts

        # The ANSI-coloured consensus/alignment views are rendered on first
        # access (see ``colored_consensus`` and friends): on a large alignment
        # they dominate this method's runtime yet are only ever printed for
        # small alignments.
        self._colored = {}
        _lap('fill', _t)
        logger.debug(f'calculate_rip: total {time.perf_counter() - _t0:.3f}s')

        # Log summary
        logger.info(
            f'RIP correction complete. Reference sequence used for filling: {ref_id}'
        )

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

    def _lazy_colored(self, key: str, render) -> Optional[str]:
        """
        Return a cached ANSI-coloured view, rendering it on first access.

        Parameters
        ----------
        key : str
            Cache key (``'consensus'``, ``'alignment'`` or ``'masked'``).
        render : callable
            Zero-argument function producing the string when results exist.

        Returns
        -------
        str or None
            The coloured text, or ``None`` before :meth:`calculate_rip` has run.
        """
        if self.markupdict is None or self.gapped_consensus is None:
            return None
        if self._colored.get(key) is None:
            self._colored[key] = render()
        return self._colored[key]

    @property
    def colored_consensus(self) -> Optional[str]:
        """
        Gapped consensus with deRIP-corrected positions in bold green (ANSI).

        Returns
        -------
        str or None
            ``None`` until :meth:`calculate_rip` has been run.
        """
        return self._lazy_colored('consensus', self._colorize_corrected_positions)

    @colored_consensus.setter
    def colored_consensus(self, value: Optional[str]) -> None:
        """
        Override (or with ``None``, discard) the cached coloured consensus.

        Parameters
        ----------
        value : str or None
            Replacement text; ``None`` forces a re-render on next access.
        """
        self._colored['consensus'] = value

    @property
    def colored_alignment(self) -> Optional[str]:
        """
        Input alignment with RIP sites coloured by category (ANSI).

        Returns
        -------
        str or None
            ``None`` until :meth:`calculate_rip` has been run.
        """
        return self._lazy_colored(
            'alignment', lambda: self._create_colored_alignment(self.alignment)
        )

    @colored_alignment.setter
    def colored_alignment(self, value: Optional[str]) -> None:
        """
        Override (or with ``None``, discard) the cached coloured alignment.

        Parameters
        ----------
        value : str or None
            Replacement text; ``None`` forces a re-render on next access.
        """
        self._colored['alignment'] = value

    @property
    def colored_masked_alignment(self) -> Optional[str]:
        """
        Masked alignment with RIP sites coloured by category (ANSI).

        Returns
        -------
        str or None
            ``None`` until :meth:`calculate_rip` has been run.
        """
        return self._lazy_colored(
            'masked', lambda: self._create_colored_alignment(self.masked_alignment)
        )

    @colored_masked_alignment.setter
    def colored_masked_alignment(self, value: Optional[str]) -> None:
        """
        Override (or with ``None``, discard) the cached coloured masked alignment.

        Parameters
        ----------
        value : str or None
            Replacement text; ``None`` forces a re-render on next access.
        """
        self._colored['masked'] = value

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

        return ''.join(seq_chars)

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

        # Group every marked position by row once, so each row only visits its
        # own markup (rather than scanning the whole markupdict per row). Category
        # order is preserved within a row, so overlapping marks resolve exactly as
        # before: a later category overwrites an earlier one.
        by_row: Dict[int, list] = {}
        for category, positions in self.markupdict.items():
            # Skip non_rip_deamination highlighting if reaminate is False
            if category == 'non_rip_deamination' and not self.reaminate:
                continue
            target_color = target_color_map[category]
            offset_color = offset_color_map[category]
            for pos in positions:
                by_row.setdefault(pos.rowIdx, []).append(
                    (pos.colIdx, pos.offset, target_color, offset_color)
                )

        # Create a colored representation of each sequence in the alignment
        lines = []

        # Process each sequence in the alignment
        for row_idx in range(len(alignment)):
            seq = alignment[row_idx].seq
            seq_id = alignment[row_idx].id
            n = len(seq)

            # Create list of characters for this sequence with their default coloring
            colored_chars = list(str(seq))

            for col_idx, offset, target_color, offset_color in by_row.get(row_idx, ()):
                # Apply bold+color formatting to the target base
                if 0 <= col_idx < n:
                    colored_chars[col_idx] = (
                        f'{target_color}{colored_chars[col_idx]}{RESET}'
                    )

                # Determine range of offset positions to color (but not bold)
                if offset is not None:
                    if offset > 0:
                        # Color bases to the right (excluding target)
                        start_col = col_idx + 1
                        end_col = min(col_idx + offset, n - 1)
                    else:  # offset < 0
                        # Color bases to the left (excluding target)
                        start_col = max(0, col_idx + offset)  # offset is negative
                        end_col = col_idx - 1

                    # Apply color-only formatting to the offset bases
                    for i in range(max(start_col, 0), min(end_col, n - 1) + 1):
                        colored_chars[i] = f'{offset_color}{colored_chars[i]}{RESET}'

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
