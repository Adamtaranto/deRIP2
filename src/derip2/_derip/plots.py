"""Alignment/strand-bias figures and the HTML reports."""

import logging
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PlotsReportsMixin:
    """
    Alignment/strand-bias figures and the HTML reports.

    Facades over :mod:`derip2.plotting`, :mod:`derip2.report` and
    :mod:`derip2.persequence_report`; imports stay function-local to avoid
    import cycles and keep CLI start-up light.
    """

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
            Cap the number of sequence panels, keeping the first ``max_seqs``
            rows in alignment order. Sort or filter the alignment first (e.g.
            :meth:`sort_by_rsi`) to change which sequences that is. ``None``
            (default) renders every sequence.
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
