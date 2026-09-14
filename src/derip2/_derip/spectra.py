"""Mutation spectra, flank-context spectra and maximum-RIP counterfactuals."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class SpectraMixin:
    """
    Mutation spectra, flank-context spectra and maximum-RIP counterfactuals.

    Thin facades over :mod:`derip2.stats`, :mod:`derip2.spectra` and
    :mod:`derip2.maxrip` that cache their results on the instance.
    """

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

    def calculate_flank_spectra(self, flank_length: int = 1):
        """
        Compute the flanking-context spectra of RIP-like sites.

        Classifies every RIP-like dinucleotide by the ``flank_length`` bases
        upstream and downstream (a ``2 + 2 * flank_length`` bp motif; a 4 bp motif
        for the default 1 bp flank). Surviving substrate sites (``CpA``/``TpG``
        anywhere in each sequence) and RIP product sites (``TpA`` in RIP-informative
        columns) are counted separately, one sample column per input sequence,
        folded onto ``CA``/``TA``-equivalent channels. The result is cached on
        :attr:`flank_spectra_result` and recomputed if a different ``flank_length``
        is requested.

        Parameters
        ----------
        flank_length : int, optional
            Number of flanking bases resolved on each side of the centre
            dinucleotide (default 1), giving ``4 ** (2 * flank_length)`` channels.

        Returns
        -------
        derip2.stats.flank_spectra.FlankSpectraResult
            The four ``(4 ** (2 * flank_length), n_rows)`` count matrices and
            per-state skipped counts.

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
        cached = self.flank_spectra_result
        if cached is not None and cached.flank_length == flank_length:
            return cached
        sample_names = [record.id for record in self.alignment]
        self.flank_spectra_result = compute_flank_spectra(
            self.column_classes, sample_names=sample_names, flank_length=flank_length
        )
        return self.flank_spectra_result

    def calculate_max_rip(self, variant: str = 'all'):
        """
        Build a maximally RIP-mutated variant of the deRIP'd consensus.

        The counterfactual complement of :meth:`calculate_rip`: instead of
        restoring the bases RIP removed, it mutates every RIP target the
        corrected sequence still carries. Results are cached per variant on
        :attr:`max_rip_results`.

        Parameters
        ----------
        variant : {'all', 'observed', 'all_plus_nonrip'}, optional
            Which sites to convert (default ``'all'``, every substrate site in
            the consensus). See :mod:`derip2.maxrip` for the full definitions.

        Returns
        -------
        derip2.maxrip.MaxRIPResult
            The mutated sequence and the positions that were changed.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first, or ``variant``
            is not one of :data:`derip2.maxrip.MAX_RIP_VARIANTS`.

        See Also
        --------
        derip2.maxrip.compute_max_rip : The calculation.
        """
        from derip2.maxrip import compute_max_rip

        self._require_rip('calculating the maximum-RIP sequence')
        if variant not in self.max_rip_results:
            self.max_rip_results[variant] = compute_max_rip(
                self.gapped_consensus, self.column_classes, variant=variant
            )
        return self.max_rip_results[variant]

    def get_max_rip_string(self, variant: str = 'all', gapped: bool = False) -> str:
        """
        Return a maximum-RIP sequence as a string, computing it if needed.

        Parameters
        ----------
        variant : {'all', 'observed', 'all_plus_nonrip'}, optional
            Which sites to convert (default ``'all'``).
        gapped : bool, optional
            Return the column-aligned sequence rather than the ungapped one
            (default ``False``).

        Returns
        -------
        str
            The maximally RIP-mutated sequence.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first, or ``variant``
            is unknown.
        """
        result = self.calculate_max_rip(variant)
        return result.gapped_seq if gapped else result.seq

    def get_max_rip_positions(
        self, variant: str = 'all', gapped: bool = False
    ) -> List[int]:
        """
        Return the positions converted by a maximum-RIP variant.

        Parameters
        ----------
        variant : {'all', 'observed', 'all_plus_nonrip'}, optional
            Which sites to convert (default ``'all'``).
        gapped : bool, optional
            Return alignment column indices rather than offsets into the
            ungapped sequence (default ``False``).

        Returns
        -------
        list of int
            Ascending zero-based positions of the converted sites.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first, or ``variant``
            is unknown.
        """
        result = self.calculate_max_rip(variant)
        indices = result.converted_cols if gapped else result.converted_positions
        return indices.tolist()

    def write_max_rip(
        self,
        output_file: str,
        variants=None,
        seq_id: str = 'maxRIPseq',
        gapped: bool = False,
    ) -> str:
        """
        Write maximum-RIP sequences to a FASTA file, computing them if needed.

        Parameters
        ----------
        output_file : str
            Destination path.
        variants : iterable of str, optional
            Which variants to write, in order; defaults to every variant in
            :data:`derip2.maxrip.MAX_RIP_VARIANTS`.
        seq_id : str, optional
            Base record id; each record is suffixed with its variant name
            (default ``'maxRIPseq'``).
        gapped : bool, optional
            Write column-aligned sequences rather than ungapped ones
            (default ``False``).

        Returns
        -------
        str
            ``output_file``, for chaining.

        Raises
        ------
        ValueError
            If :meth:`calculate_rip` has not been called first, or a requested
            variant is unknown.
        """
        from derip2.maxrip import MAX_RIP_VARIANTS, write_max_rip_fasta

        chosen = MAX_RIP_VARIANTS if variants is None else tuple(variants)
        results = [self.calculate_max_rip(variant) for variant in chosen]
        return write_max_rip_fasta(results, output_file, seq_id=seq_id, gapped=gapped)

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

    def plot_flank_spectra(
        self,
        output_file: Optional[str] = None,
        *,
        percentage: bool = False,
        **kwargs,
    ):
        """
        Draw the pooled flank-context bihistograms, computing spectra if needed.

        Parameters
        ----------
        output_file : str, optional
            Path to write the figure to. Use ``.svg`` or ``.pdf`` for
            publication output.
        percentage : bool, optional
            When ``True``, plot **count-normalised proportions** instead of raw
            counts: each state (substrate, product) is rescaled to sum to 100
            across the 16 motifs, so the two spectra are compared on equal footing
            regardless of how many substrate vs product sites there are (default:
            ``False``, raw counts).
        **kwargs
            Forwarded to
            :func:`derip2.plotting.flank_spectra.plot_flank_bihistograms_pooled`
            (e.g. ``title``, ``strands``, ``width``).

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        """
        from derip2.plotting.flank_spectra import plot_flank_bihistograms_pooled

        if self.flank_spectra_result is None:
            self.calculate_flank_spectra()
        return plot_flank_bihistograms_pooled(
            self.flank_spectra_result, output_file, percentage=percentage, **kwargs
        )

    def plot_flank_conversion_heatmap(
        self,
        output_file: Optional[str] = None,
        *,
        flank_length: int = 1,
        **kwargs,
    ):
        """
        Draw the pooled flank-context RIP-conversion heatmap, computing if needed.

        Each cell of the ``4 ** flank_length`` x ``4 ** flank_length`` grid shows
        the percentage of a RIP target CpA converted to TpA (the product share) as
        a joint function of the upstream (rows) and downstream (columns) flank
        bases.

        Parameters
        ----------
        output_file : str, optional
            Path to write the figure to (``.svg``/``.png``/``.pdf``).
        flank_length : int, optional
            Flank width; recomputes the spectra if it differs from the cached
            result (default 1).
        **kwargs
            Forwarded to
            :func:`derip2.plotting.flank_spectra.plot_flank_conversion_heatmap`
            (e.g. ``title``, ``bare``, ``flank_sort``). Notably ``cmap`` restyles
            the colour scale: pass a matplotlib colormap name such as
            ``'magma_r'`` or ``'viridis'``, a
            :class:`~matplotlib.colors.Colormap`, or a list of colours to
            interpolate between.

        Returns
        -------
        matplotlib.figure.Figure
            The heatmap figure.
        """
        from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

        result = self.calculate_flank_spectra(flank_length=flank_length)
        return plot_flank_conversion_heatmap(
            result, sample=None, outfile=output_file, **kwargs
        )
