"""Per-sequence statistics: strand bias (RSI), CRI, GC and row selection."""

from io import StringIO
import logging
import math
from typing import Callable
import warnings

from Bio.Align import MultipleSeqAlignment
from Bio.SeqUtils import gc_fraction
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StatsMixin:
    """
    Per-sequence statistics: strand bias (RSI), CRI, GC and row selection.

    Everything that annotates, summarises, sorts or filters the alignment rows.
    """

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
        self.get_rsi_values()

        def _sort_key(record):
            rsi = record.annotations['RSI']
            # NaN sorts last in both directions.
            if math.isnan(rsi):
                return (1, 0.0)
            return (0, -rsi if descending else rsi)

        return self._sort_rows(_sort_key, inplace, 'RSI')

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
        # Convert to uppercase, drop gaps, and compare each base with its
        # right-hand neighbour vectorised rather than slicing per position.
        seq = np.frombuffer(sequence.upper().encode('ascii'), dtype='S1')
        seq = seq[seq != b'-']
        left, right = seq[:-1], seq[1:]
        is_a, is_c = left == b'A', left == b'C'
        is_g, is_t = left == b'G', left == b'T'
        next_a, next_c = right == b'A', right == b'C'
        next_g, next_t = right == b'G', right == b'T'

        return {
            'TpA': int(np.count_nonzero(is_t & next_a)),
            'ApT': int(np.count_nonzero(is_a & next_t)),
            'CpA': int(np.count_nonzero(is_c & next_a)),
            'TpG': int(np.count_nonzero(is_t & next_g)),
            'ApC': int(np.count_nonzero(is_a & next_c)),
            'GpT': int(np.count_nonzero(is_g & next_t)),
        }

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

    # -- shared row selection helpers --------------------------------------------
    def _sort_rows(self, key: Callable, inplace: bool, label: str):
        """
        Return the alignment sorted by ``key``; optionally adopt it in place.

        Parameters
        ----------
        key : callable
            Sort key taking a :class:`Bio.SeqRecord.SeqRecord`.
        inplace : bool
            Replace the current alignment (and discard results) when True.
        label : str
            Metric name used in the in-place log message (e.g. ``'CRI'``).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            The sorted alignment.
        """
        sorted_alignment = MultipleSeqAlignment(sorted(self.alignment, key=key))
        if inplace:
            self.alignment = sorted_alignment
            logger.info(f'Updated alignment in-place with {label}-sorted sequences')
            self._invalidate_results()
        return sorted_alignment

    def _filter_rows(
        self,
        annotation: str,
        threshold: float,
        inplace: bool,
        *,
        label: str,
        param: str,
        quantity: str,
    ):
        """
        Keep the records whose ``annotation`` value is at least ``threshold``.

        Parameters
        ----------
        annotation : str
            Record annotation key holding the metric (``'CRI'`` or
            ``'GC_content'``); every record must already carry it.
        threshold : float
            Minimum value to keep a record.
        inplace : bool
            Replace the current alignment (and discard results) when True.
        label : str
            Short metric name for log/warning text (``'CRI'``, ``'GC'``).
        param : str
            Name of the caller's threshold argument, for the error message.
        quantity : str
            Human phrase for the metric in the error message
            (``'CRI value'``, ``'GC content'``).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            The filtered alignment.

        Raises
        ------
        ValueError
            If no sequences pass the threshold.
        """
        kept = [r for r in self.alignment if r.annotations[annotation] >= threshold]

        if not kept:
            highest = max(r.annotations[annotation] for r in self.alignment)
            raise ValueError(
                f'No sequences remain after filtering with {param}={threshold}. '
                f'The highest {quantity} in the alignment is {highest:.4f}'
            )

        n_total = len(self.alignment)
        if len(kept) < 2:
            kept_ids = {r.id for r in kept}
            logger.debug(
                f'Records that passed filter threshold {threshold}: {len(kept)}; '
                f'failed: {n_total - len(kept_ids)}'
            )
            warnings.warn(
                f'Only {len(kept)} sequence remains after {label} filtering. '
                f'DeRIP works best with multiple sequences.',
                stacklevel=3,
            )
        elif len(kept) < n_total:
            logger.info(
                f'{label} filtering removed {n_total - len(kept)} sequences '
                f'({len(kept)}/{n_total} sequences remaining)'
            )

        filtered_alignment = MultipleSeqAlignment(kept)
        if inplace:
            self.alignment = filtered_alignment
            logger.info(f'Updated alignment in-place with {label}-filtered sequences')
            self._invalidate_results()
        return filtered_alignment

    def _keep_n_rows(
        self,
        annotation: str,
        n: int,
        inplace: bool,
        *,
        highest: bool,
        label: str,
        quantity: str,
    ):
        """
        Keep the ``n`` records with the lowest (or highest) ``annotation`` value.

        Parameters
        ----------
        annotation : str
            Record annotation key holding the metric; every record must carry it.
        n : int
            Number of records to keep. Must be at least 2 and fewer than the
            alignment size, otherwise the alignment is returned unchanged.
        inplace : bool
            Replace the current alignment (and discard results) when True.
        highest : bool
            Keep the highest values when True, the lowest when False.
        label : str
            Metric name for the in-place log message (``'low-CRI'``,
            ``'high-GC'``).
        quantity : str
            Human phrase for the log messages (``'CRI values'``,
            ``'GC content'``).

        Returns
        -------
        Bio.Align.MultipleSeqAlignment
            The reduced alignment (or the current one if no filtering applied).
        """
        n_total = len(self.alignment)
        if n >= n_total:
            logger.info(
                f'Requested to keep {n} sequences but alignment only has {n_total}. '
                'No filtering performed.'
            )
            return self.alignment
        if n < 2:
            logger.warning(
                f'Cannot keep fewer than 2 sequences (requested {n}). DeRIP works '
                'best with multiple sequences. No filtering performed.'
            )
            return self.alignment

        kept = sorted(
            self.alignment, key=lambda r: r.annotations[annotation], reverse=highest
        )[:n]
        kept_alignment = MultipleSeqAlignment(kept)

        which = 'highest' if highest else 'lowest'
        other = 'lower' if highest else 'higher'
        logger.info(
            f'Kept {n} sequences with {which} {quantity}: '
            f'{[(r.id, r.annotations[annotation]) for r in kept]}'
        )
        logger.info(f'Removed {n_total - n} sequences with {other} {quantity}')

        if inplace:
            self.alignment = kept_alignment
            logger.info(f'Updated alignment in-place with {label} filtered sequences')
            self._invalidate_results()
        return kept_alignment

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
        # Ensure all sequences have CRI values
        self.get_cri_values()
        sign = -1.0 if descending else 1.0
        return self._sort_rows(
            lambda record: sign * record.annotations['CRI'], inplace, 'CRI'
        )

    def summarize_cri(self):
        """
        Generate a formatted table summarizing CRI values for all sequences.

        Returns
        -------
        str
            A formatted string containing the CRI summary table.
        """
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
        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have CRI values
        self.get_cri_values()
        return self._filter_rows(
            'CRI', min_cri, inplace, label='CRI', param='min_cri', quantity='CRI value'
        )

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

        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have CRI values
        self.get_cri_values()
        return self._keep_n_rows(
            'CRI', n, inplace, highest=False, label='low-CRI', quantity='CRI values'
        )

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
        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have GC content values
        self.get_gc_content()

        # Validate min_gc is in valid range
        if not 0.0 <= min_gc <= 1.0:
            raise ValueError(f'min_gc must be between 0.0 and 1.0, got {min_gc}')
        return self._filter_rows(
            'GC_content',
            min_gc,
            inplace,
            label='GC',
            param='min_gc',
            quantity='GC content',
        )

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

        if self.alignment is None:
            raise ValueError('No alignment loaded')

        # Ensure all sequences have GC content values
        self.get_gc_content()
        return self._keep_n_rows(
            'GC_content',
            n,
            inplace,
            highest=True,
            label='high-GC',
            quantity='GC content',
        )
