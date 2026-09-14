"""
DNA alignment visualization tool for generating overview images of sequence alignments.

This module provides functions to visualize DNA sequence alignments as color-coded
images, making it easier to identify patterns, gaps, and conserved regions. It is
derived from the CIAlign package (https://github.com/KatyBrown/CIAlign) with
modifications for the deRIP2 project.
"""

import logging
import os
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from Bio.Align import MultipleSeqAlignment
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np

from derip2.aln_ops import alignment_to_array

matplotlib.use('Agg')  # Use non-interactive backend for server environments

logger = logging.getLogger(__name__)

RIPPosition = NamedTuple(
    'RIPPosition', [('colIdx', int), ('rowIdx', int), ('base', str), ('offset', int)]
)


def get_color_palette(palette: str = 'colorblind') -> Dict[str, str]:
    """
    Get a color palette mapping DNA bases to hexadecimal color codes.

    This function provides access to predefined color schemes for visualizing
    DNA sequence alignments. Different palettes are optimized for various
    purposes including colorblind accessibility, high contrast, and specific
    visualization preferences.

    Parameters
    ----------
    palette : str, optional
        Name of the color palette to use. Options include:
        - 'colorblind': Colors chosen to be distinguishable by people with color vision deficiencies
        - 'bright': High-contrast vibrant colors
        - 'tetrimmer': Traditional nucleotide coloring scheme
        - 'basegrey': All bases colored in grey (for contrast with markup)
        - 'derip2': Default scheme for deRIP2 with bright, distinct colors
        Default is 'colorblind'.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping nucleotide characters to hexadecimal color codes.
        Keys include 'A', 'C', 'G', 'T', 'N', '-' (gap), and sometimes lowercase
        or additional variants.

    Notes
    -----
    The coloring schemes generally follow these conventions:
    - A: Green or Red
    - G: Yellow or Gray
    - T: Red or Green
    - C: Blue
    - N: Gray or Light Blue
    - Gaps (-): White

    Examples
    --------
    >>> palette = get_color_palette('derip2')
    >>> palette['A']
    '#ff3f3f'
    """
    # Define color palettes for different visualization preferences
    # Each palette maps DNA bases to their respective hexadecimal color codes
    color_palettes = {
        # Colorblind-friendly palette (default)
        'colorblind': {
            'A': '#56ae6c',  # Green
            'G': '#c9c433',  # Yellow
            'T': '#a22c49',  # Red
            'C': '#0038a2',  # Blue
            'N': '#6979d3',  # Light blue
            'n': '#6979d3',  # Light blue (lowercase)
            '-': '#FFFFFF',  # White (gap)
            'X': '#6979d3',  # Light blue (unknown)
        },
        # Bright color palette for high contrast
        'bright': {
            'A': '#f20707',  # Bright red
            'G': '#ffd500',  # Bright yellow
            'T': '#64bc3c',  # Bright green
            'C': '#0907f2',  # Bright blue
            'N': '#c7d1d0',  # Gray
            'n': '#c7d1d0',  # Gray (lowercase)
            '-': '#FFFFFF',  # White (gap)
            'X': '#c7d1d0',  # Gray (unknown)
        },
        # Traditional tetrimmer color scheme
        'tetrimmer': {
            'A': '#00CC00',  # Green
            'G': '#949494',  # Gray
            'T': '#FF6666',  # Pink/red
            'C': '#6161ff',  # Blue
            'N': '#c7d1d0',  # Light gray
            'n': '#c7d1d0',  # Light gray (lowercase)
            '-': '#FFFFFF',  # White (gap)
            'X': '#c7d1d0',  # Light gray (unknown)
        },
        # Grayscale palette for bases (useful for highlighting only markup)
        'basegrey': {
            'A': '#c7d1d0',  # Gray
            'G': '#c7d1d0',  # Gray
            'T': '#c7d1d0',  # Gray
            'C': '#c7d1d0',  # Gray
            'N': '#c7d1d0',  # Light gray
            'n': '#c7d1d0',  # Light gray (lowercase)
            '-': '#FFFFFF',  # White (gap)
            'X': '#c7d1d0',  # Light gray (unknown)
        },
        # DeRIP2 color scheme - optimized for the deRIP2 tool visualization
        'derip2': {
            'A': '#ff3f3f',  # Bright red
            'G': '#fbe216',  # Bright yellow
            'T': '#64bc3c',  # Bright green
            'C': '#55c1ed',  # Bright blue
            'N': '#c7d1d0',  # Gray
            '-': '#FFFFFF',  # White (gap)
        },
    }

    # Return the requested palette or default to colorblind if not found
    if palette not in color_palettes:
        logger.warning(f"Palette '{palette}' not found, using 'colorblind' instead")
        return color_palettes['colorblind']

    return color_palettes[palette]


#: Characters the alignment image can draw; anything else is shown as a gap.
VALID_CHARS = np.array([b'A', b'G', b'C', b'T', b'N', b'-'], dtype='S1')


def MSAToArray(
    alignment: MultipleSeqAlignment,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[int]]:
    """
    Convert a Biopython MultipleSeqAlignment object into a numpy array.

    This function is an alternative to FastaToArray that works directly with
    in-memory alignment objects rather than reading from files.

    Parameters
    ----------
    alignment : Bio.Align.MultipleSeqAlignment
        The multiple sequence alignment object.

    Returns
    -------
    arr : np.ndarray or None
        2D numpy array where each row represents a sequence and each column
        represents a position in the alignment. Returns None if only one
        sequence is found.
    nams : List[str] or None
        List of sequence names in the same order as in the input alignment.
        Returns None if only one sequence is found.
    seq_len : int or None
        Number of sequences in the alignment. Returns None if only one
        sequence is found.

    Raises
    ------
    ValueError
        If the alignment is empty or sequences have different lengths.
    """
    logger.debug(f'MSAToArray: alignment={alignment}')

    # Check if alignment is empty
    if not alignment or len(alignment) == 0:
        raise ValueError('Empty alignment provided')

    nams = [record.id for record in alignment]
    seq_len = len(nams)
    if seq_len <= 1:
        return None, None, None

    # Decode the whole alignment in one pass (bytes), fold case, and replace
    # anything outside the drawable alphabet with a gap, all vectorised, then
    # widen to one-character strings for the drawing code.
    arr = np.char.upper(alignment_to_array(alignment))
    arr[~np.isin(arr, VALID_CHARS)] = b'-'
    return arr.astype('U1'), nams, seq_len


def arrNumeric(
    arr: np.ndarray, palette: str = 'colorblind'
) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap]:
    """
    Convert sequence array into a numerical matrix with a color map for visualization.

    This function transforms the sequence data into a format that matplotlib
    can interpret as an image. The sequence array is flipped vertically so the
    output image has rows in the same order as the input alignment.

    Parameters
    ----------
    arr : np.ndarray
        The DNA sequence alignment stored as a numpy array.
    palette : str, optional
        Color palette to use. Options: 'colorblind' (default), 'bright', 'tetrimmer'.

    Returns
    -------
    arr2 : np.ndarray
        The flipped alignment as an array of integers where each integer represents
        a specific nucleotide.
    cmap : matplotlib.colors.ListedColormap
        A color map with colors corresponding to each nucleotide.
    """
    # Flip the array vertically so the output image matches input alignment order
    arr = np.flip(arr, axis=0)

    # Select the appropriate color pattern or default to colorblind
    color_pattern = get_color_palette(palette)

    # Get dimensions of the alignment
    ali_height, ali_width = np.shape(arr)

    # Build the mapping and color list for the nucleotides present in the
    # alignment, preserving the palette's key order.
    colours = []  # List of colors for the colormap
    arr2 = np.zeros((ali_height, ali_width))

    # One vectorised boolean assignment per present nucleotide replaces the
    # former per-cell nested Python loop over the whole array.
    i = 0
    for key in color_pattern.keys():
        matches = arr == key
        if matches.any():
            arr2[matches] = i
            colours.append(color_pattern[key])
            i += 1

    # Create the colormap for visualization
    cmap = matplotlib.colors.ListedColormap(colours)
    return arr2, cmap


def drawMiniAlignment(
    alignment: MultipleSeqAlignment,
    outfile: str,
    dpi: int = 300,
    title: Optional[str] = None,
    width: int = 20,
    height: int = 15,
    orig_nams: Optional[List[str]] = None,
    keep_numbers: bool = False,
    force_numbers: bool = False,
    palette: str = 'derip2',
    markupdict: Optional[Dict[str, List[RIPPosition]]] = None,
    column_ranges: Optional[List[Tuple[int, int, str, str]]] = None,
    annotation_track: Optional[List[Tuple[int, int, str, str, int]]] = None,
    cds_tracks: Optional[List[Tuple]] = None,
    show_chars: bool = False,
    draw_boxes: bool = False,
    consensus_seq: Optional[str] = None,
    corrected_positions: Optional[List[int]] = None,
    reaminate: bool = False,
    reference_seq_index: Optional[int] = None,
    show_rip: str = 'both',  # 'substrate', 'product', or 'both'
    highlight_corrected: bool = True,
    flag_corrected: bool = False,
    return_figure: bool = False,
) -> Union[str, bool, plt.Figure]:
    """
    Generate a visualization of a DNA sequence alignment with optional RIP markup.

    This function is an alternative to drawMiniAlignment that works directly with
    in-memory alignment objects rather than reading from files.

    Parameters
    ----------
    alignment : Bio.Align.MultipleSeqAlignment
        The multiple sequence alignment object to visualize.
    outfile : str
        Path to save the output image file.
    dpi : int, optional
        Resolution of the output image in dots per inch (default: 300).
    title : str, optional
        Title to display on the image (default: None).
    width : int, optional
        Width of the output image in inches (default: 20).
    height : int, optional
        Height of the output image in inches (default: 15).
    orig_nams : List[str], optional
        Original sequence names for label preservation (default: empty list).
    keep_numbers : bool, optional
        Whether to keep original sequence numbers (default: False).
    force_numbers : bool, optional
        Whether to force display of all sequence numbers (default: False).
    palette : str, optional
        Color palette to use: 'colorblind', 'bright', 'tetrimmer', 'basegrey', or 'derip2' (default: 'basegrey').
    markupdict : Dict[str, List[RIPPosition]], optional
        Dictionary with RIP categories as keys and lists of position tuples as values.
        Categories are 'rip_product', 'rip_substrate', and 'non_rip_deamination'.
        Each position is a named tuple with (colIdx, rowIdx, base, offset).
    column_ranges : List[Tuple[int, int, str, str]], optional
        List of column ranges to mark, each as (start_col, end_col, color, label).
    annotation_track : List[Tuple[int, int, str, str, int]], optional
        Stacked gene-annotation spans to draw below the alignment, each as
        (start_col, end_col, color, label, track_row). Columns are alignment
        columns (already adjusted for gaps by the caller); ``track_row`` (0 at
        the top) stacks spans of different annotation types into separate rows
        (default: None).
    cds_tracks : list of tuple, optional
        Rich per-gene CDS tracks, each ``(exon_spans, strand, stop_columns,
        label, colour)`` in alignment-column coordinates. When given, the
        annotation sub-plot draws rounded exon segments joined across introns by
        a coloured midline, a strand arrowhead, and a bold red ``*`` at each stop
        column; takes precedence over ``annotation_track`` (default: None).
    show_chars : bool, optional
        Whether to display sequence characters inside the colored cells (default: False).
    draw_boxes : bool, optional
        Whether to draw black borders around highlighted bases (default: False).
    consensus_seq : str, optional
        Consensus sequence to display in a separate subplot below the alignment (default: None).
    corrected_positions : List[int], optional
        List of column indices that were corrected during deRIP (default: None).
    reaminate : bool, optional
        Whether to highlight non-RIP deamination positions (default: False).
    reference_seq_index : int, optional
        Index of the reference sequence used to fill uncorrected positions (default: None).
    show_rip : str, optional
        Which RIP markup categories to include: 'substrate', 'product', or 'both' (default: 'both').
    highlight_corrected : bool, optional
        If True, only corrected positions in the consensus will be colored, all others will be gray (default: True).
    flag_corrected : bool, optional
        If True, corrected positions will be marked with a large asterisk above the consensus (default: False).
    return_figure : bool, optional
        If True, return the live :class:`matplotlib.figure.Figure` without writing
        a file or closing it (for callers that render it to inline SVG); the
        output format otherwise follows the ``outfile`` extension, defaulting to
        SVG (default: False).

    Returns
    -------
    Union[str, bool, matplotlib.figure.Figure]
        The output file path if written, the Figure if ``return_figure`` is True,
        or False if only one sequence was found.

    Notes
    -----
    The alignment is visualized with each nucleotide represented by a color-coded cell:
    - A: green
    - G: yellow
    - T: red
    - C: blue
    - N: light blue
    - Gaps (-): white

    When markupdict is provided:
    - All bases are dimmed with a gray overlay
    - RIP products are highlighted in red
    - RIP substrates are highlighted in blue
    - Non-RIP deamination events are highlighted in orange
    """
    # DEBUG: Print function parameters for troubleshooting
    logger.debug(
        f'drawMiniAlignment: outfile={outfile}, dpi={dpi}, title={title}, width={width}, height={height}, orig_nams={orig_nams}, keep_numbers={keep_numbers}, force_numbers={force_numbers}, palette={palette}, markupdict={markupdict}, column_ranges={column_ranges}, show_chars={show_chars}, consensus_seq={consensus_seq}, corrected_positions={corrected_positions}, reaminate={reaminate}, reference_seq_index={reference_seq_index}, show_rip={show_rip}, highlight_corrected={highlight_corrected}'
    )
    # Handle default value for orig_nams
    if orig_nams is None:
        orig_nams = []

    # Convert the MSA object to a numpy array
    arr, nams, seq_len = MSAToArray(alignment)

    # Return False if only one sequence was found
    if arr is None:
        return False

    # Adjust height for small alignments
    if seq_len <= 75:
        calculated_height = seq_len * 0.2
        # Ensure a minimum height of 5 inches to prevent title overlap
        height = max(calculated_height, 5)

    # Get alignment dimensions
    ali_height, ali_width = np.shape(arr)

    # Define plot styling parameters
    fontsize = 14

    # Determine tick interval based on the number of sequences
    if force_numbers:
        tickint = 1
    elif ali_height <= 10:
        tickint = 1
    elif ali_height <= 500:
        tickint = 10
    else:
        tickint = 100

    # The rest of the function is identical to drawMiniAlignment,
    # continuing with the same plotting logic

    # Calculate line weights based on alignment dimensions
    lineweight_h = 10 / ali_height  # Horizontal grid lines
    lineweight_v = 10 / ali_width  # Vertical grid lines

    # Calculate padding to add to figure dimensions
    width_padding = 0.2  # Add 0.2 inches of padding to width
    height_padding = 1  # Add 1 inch of padding to height

    # Create the figure. The alignment always occupies the top row (ratio 4); the
    # optional deRIP-consensus row sits directly below it, and the optional
    # gene-annotation track sits below the consensus (inside the deRIP-consensus
    # section) so a gene's exon/stop glyphs read against the corrected sequence
    # they annotate. Building them as separate axes (rather than drawing the track
    # onto the alignment axis) keeps the SVG chrome clean and lets the track carry
    # its own tooltips.
    f = plt.figure(figsize=(width + width_padding, height + height_padding), dpi=dpi)

    n_ann_rows = 0
    if cds_tracks:
        n_ann_rows = len(cds_tracks)
    elif annotation_track:
        n_ann_rows = max(row for *_rest, row in annotation_track) + 1
    ann_ratio = 0.5 * n_ann_rows if n_ann_rows else 0.0

    # Stack order top -> bottom: alignment, consensus, annotation track.
    ratios = [4.0]
    if consensus_seq is not None:
        ratios.append(1.0)
    if ann_ratio:
        ratios.append(ann_ratio)

    gs = f.add_gridspec(len(ratios), 1, height_ratios=ratios)
    a = f.add_subplot(gs[0])

    row_i = 1
    consensus_ax = None
    if consensus_seq is not None:
        consensus_ax = f.add_subplot(gs[row_i])
        row_i += 1

    ann_ax = None
    if ann_ratio:
        ann_ax = f.add_subplot(gs[row_i])

    # Position adjustments - keep the same relative positioning
    if consensus_ax is not None or ann_ax is not None:
        f.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.88, hspace=0.5)
    else:
        f.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.88)

    # On a wide alignment, rasterize the dense per-cell content (base grid, RIP
    # markup, the per-column consensus row) into the embedded image rather than
    # emitting thousands of tiny vector rectangles in the SVG. Text, tick labels
    # and the reference marker (zorder >= 200) stay crisp vector. Narrow
    # alignments keep everything vector, so they render sharply at any zoom. The
    # annotation axis is left untouched: it holds only a handful of gene glyphs,
    # which must stay vector so their tooltips survive in the SVG.
    if ali_width > 500:
        for _ax in (a, consensus_ax):
            if _ax is not None:
                _ax.set_rasterization_zorder(200)

    # Setup the alignment plot with normal limits
    a.set_xlim(-0.5, ali_width - 0.5)
    a.set_ylim(-0.5, ali_height - 0.5)

    # Convert alignment to numeric form and get color map
    arr2, cm = arrNumeric(arr, palette='basegrey')

    # Process markup if provided
    if markupdict:
        # Filter the markup dictionary based on show_rip parameter
        filtered_markup = {}

        # Always include non-RIP deamination if specified (controlled by reaminate parameter)
        if 'non_rip_deamination' in markupdict:
            filtered_markup['non_rip_deamination'] = markupdict['non_rip_deamination']

        # Include RIP substrates if requested
        if show_rip in ['substrate', 'both'] and 'rip_substrate' in markupdict:
            filtered_markup['rip_substrate'] = markupdict['rip_substrate']

        # Include RIP products if requested
        if show_rip in ['product', 'both'] and 'rip_product' in markupdict:
            filtered_markup['rip_product'] = markupdict['rip_product']

        # Get all positions that will be highlighted without drawing them
        positions_to_highlight = getHighlightedPositions(
            filtered_markup, ali_height, arr, reaminate
        )

        # Create a mask where highlighted positions are True
        mask = np.zeros_like(arr2, dtype=bool)
        for x, y in positions_to_highlight:
            if 0 <= x < ali_width and 0 <= y < ali_height:
                mask[y, x] = True

        # Create masked array where highlighted positions are transparent
        masked_arr2 = np.ma.array(arr2, mask=mask)

        # Draw the alignment with highlighted positions masked out
        a.imshow(
            masked_arr2, cmap=cm, aspect='auto', interpolation='nearest', zorder=10
        )

        # Draw the colored highlights on top
        highlighted_positions, target_positions = markupRIPBases(
            a, filtered_markup, ali_height, arr, reaminate, palette, draw_boxes
        )
    else:
        # No markup, just draw the regular alignment
        a.imshow(arr2, cmap=cm, aspect='auto', interpolation='nearest', zorder=10)
        _highlighted_positions = set()
        target_positions = set()

    # Continue with the rest of the plotting code from drawMiniAlignment...
    # (Including grid lines, reference marker, labels, text, etc.)

    # Add grid lines. The horizontal (per-row) lines are always cheap. The
    # vertical (per-column) lines are only drawn when columns are few enough to
    # be visible: on a wide alignment they are sub-pixel and, in SVG output,
    # would explode into thousands of invisible vector paths.
    a.hlines(
        np.arange(-0.5, ali_height),
        -0.5,
        ali_width,
        lw=lineweight_h,
        color='white',
        zorder=100,
    )
    if ali_width <= 500:
        a.vlines(
            np.arange(-0.5, ali_width),
            -0.5,
            ali_height,
            lw=lineweight_v,
            color='white',
            zorder=100,
        )

    # Mark the reference (fill) sequence with a black circle just past the end of
    # its row. Drawn in *data* coordinates (not baked figure coordinates) so it
    # tracks the axes transform: an annotation track that extends the y-limits
    # then no longer knocks the marker out of alignment with its row.
    if reference_seq_index is not None and 0 <= reference_seq_index < ali_height:
        ref_y = ali_height - reference_seq_index - 1  # rows drawn top-to-bottom
        marker_x = ali_width - 0.5 + max(0.6, 0.01 * ali_width)
        a.scatter(
            [marker_x],
            [ref_y],
            s=48,
            marker='o',
            facecolor='black',
            edgecolor='white',
            linewidth=1.5,
            clip_on=False,  # allowed to sit in the right-hand margin
            zorder=1000,
        )

    # Remove unnecessary spines
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_visible(False)

    # Add title if provided - position it higher to avoid overlap
    if title:
        f.suptitle(title, fontsize=fontsize * 1.5, y=0.98)

    # Set font size for x-axis tick labels
    for t in a.get_xticklabels():
        t.set_fontsize(fontsize)

    # Configure y-axis ticks and labels
    a.set_yticks(np.arange(ali_height - 1, -1, -tickint))

    # Set y-axis tick labels based on configuration
    x = 1
    if tickint == 1:
        if keep_numbers and orig_nams:
            # Use original sequence numbers
            labs = []
            for nam in orig_nams:
                if nam in nams:
                    labs.append(x)
                x += 1
            a.set_yticklabels(labs, fontsize=fontsize * 0.75)
        else:
            # Generate sequence numbers 1 through N
            a.set_yticklabels(
                np.arange(1, ali_height + 1, tickint), fontsize=fontsize * 0.75
            )
    else:
        # Use tick intervals for larger alignments
        a.set_yticklabels(np.arange(0, ali_height, tickint), fontsize=fontsize)

    # Add column range markers if provided
    if column_ranges:
        addColumnRangeMarkers(a, column_ranges, ali_height)

    # Add a gene-annotation track in its own sub-plot below the deRIP consensus
    # (within the consensus section). Rich CDS tracks (rounded exon segments joined across
    # introns by a coloured midline, a strand arrowhead, and a bold red '*' at
    # each stop codon in the projected reading frame) are preferred; a flat span
    # list is the fallback. The returned gid -> label map is attached to the
    # figure so the HTML report can inject hover tooltips onto the inline overview.
    if cds_tracks and ann_ax is not None:
        from derip2.plotting.persequence import _draw_annotation_tracks

        f.annotation_titles = _draw_annotation_tracks(
            ann_ax, cds_tracks, ali_width, width, show_labels=False
        )
    elif annotation_track and ann_ax is not None:
        f.annotation_titles = addAnnotationTrack(ann_ax, annotation_track, ali_width)
    else:
        f.annotation_titles = {}

    # Display sequence characters if requested and alignment isn't too large
    if show_chars and ali_width < 500:  # Limit for performance reasons
        # Increase font size for better visibility
        char_fontsize = min(
            14, 18000 / (ali_width * ali_height)
        )  # Adjusted for larger font

        # Don't show characters if they'll be too small
        if char_fontsize >= 4:
            for y in range(ali_height):
                for x in range(ali_width):
                    # Flip y-coordinate to match alignment orientation
                    flipped_y = ali_height - y - 1

                    # Get the character at this position
                    char = arr[y, x]

                    # Determine text color based on whether position is a target position
                    text_color = (
                        'black' if (x, flipped_y) in target_positions else '#777777'
                    )  # Lighter grey for non-target bases (including offsets)

                    # Add character as text annotation
                    a.text(
                        x,
                        flipped_y,
                        char,
                        ha='center',
                        va='center',
                        fontsize=char_fontsize,
                        color=text_color,
                        fontweight='bold',
                        zorder=200,  # Make sure characters are on top of everything
                    )

    # If consensus sequence is provided, add it to the second subplot
    if consensus_seq is not None and consensus_ax is not None:
        # Determine colors for each nucleotide
        nuc_colors = get_color_palette(palette)

        # If highlighting only corrected positions, ensure we have a valid list
        corrected_set = (
            set(corrected_positions)
            if corrected_positions and highlight_corrected
            else set()
        )

        # Set up the consensus subplot with extra space for asterisks
        consensus_ax.set_xlim(-0.5, len(consensus_seq) - 0.5)
        consensus_ax.set_ylim(
            -0.5,
            1.5,  # Increased vertical space to add area above sequence
        )
        consensus_ax.set_yticks([])
        consensus_ax.set_title('deRIP Consensus', fontsize=fontsize)

        # Hide spines
        for spine in consensus_ax.spines.values():
            spine.set_visible(False)

        # A full-width, invisible patch over the consensus row carrying a gid so
        # the HTML report can make the whole deRIP'd sequence clickable (a click
        # opens its FASTA popup). Drawn at a high zorder so it stays vector even
        # when a wide alignment rasterizes the consensus cells (zorder < 200) and
        # its gid group survives in the SVG. In file output it is inert and unseen.
        click_patch = matplotlib.patches.Rectangle(
            (-0.5, -0.5),
            len(consensus_seq),
            2.0,
            facecolor='none',
            edgecolor='none',
            zorder=250,
        )
        click_patch.set_gid('deripseq')
        consensus_ax.add_patch(click_patch)
        if isinstance(getattr(f, 'annotation_titles', None), dict):
            f.annotation_titles['deripseq'] = 'deRIP consensus — click for FASTA'

        # Add vertical grid lines (only when columns are few enough to see them;
        # see the alignment-axis note above).
        if len(consensus_seq) <= 500:
            consensus_ax.vlines(
                np.arange(-0.5, len(consensus_seq)),
                -0.5,
                1.5,  # Extended grid lines to cover the new space
                lw=lineweight_v,
                color='white',
                zorder=100,
            )

        # Plot each base in the consensus as a colored cell. The cells go into
        # one PatchCollection: adding thousands of individual patches to the
        # axes is slow (each add_patch updates the data limits and every patch
        # is transformed separately at draw time).
        cells = []
        for i, base in enumerate(consensus_seq):
            # Determine cell color based on whether this is a corrected position
            if highlight_corrected and i not in corrected_set:
                # Use gray for non-corrected positions when highlight_corrected is True
                color = '#c7d1d0'  # Standard gray color
            else:
                # Use the regular color palette for this base
                color = nuc_colors.get(
                    base.upper(), '#CCCCCC'
                )  # Default to gray for unknown bases
            cells.append(
                matplotlib.patches.Rectangle((i - 0.5, -0.5), 1, 1, color=color)
            )
        consensus_ax.add_collection(
            matplotlib.collections.PatchCollection(
                cells, match_original=True, zorder=10
            ),
            autolim=False,
        )

        for i, base in enumerate(consensus_seq):
            # Add the character as text with increased font size
            if show_chars:
                # Determine text color - use black for all characters for better readability
                text_color = 'black'

                consensus_ax.text(
                    i,
                    0,
                    base,
                    ha='center',
                    va='center',
                    fontsize=min(
                        18, 30 - len(consensus_seq) / 100
                    ),  # Further increased font size
                    color=text_color,
                    fontweight='bold',
                    zorder=20,
                )

        # Add markers for corrected positions if provided
        if corrected_positions and flag_corrected:
            for pos in corrected_positions:
                if 0 <= pos < len(consensus_seq):
                    # Calculate appropriate font size based on sequence length
                    # Scale inversely with sequence length to fit within cells
                    # Use a slightly larger size than the base characters to stand out
                    asterisk_fontsize = min(24, max(14, 40 - len(consensus_seq) / 50))

                    # Draw a large asterisk centered in the space above each corrected position
                    # Size now scales with the cell dimensions
                    consensus_ax.text(
                        pos,  # x position
                        1.0,  # y position (centered in new space above sequence)
                        '*',  # asterisk character
                        ha='center',  # horizontally centered
                        va='center',  # vertically centered
                        fontsize=asterisk_fontsize,  # dynamically scaled font size
                        color='red',  # red color
                        fontweight='bold',  # bold for emphasis
                        zorder=30,  # ensure it's on top
                    )

    # Hand the live figure back to the caller (e.g. the HTML report, which
    # renders it to inline SVG) without writing a file or closing it.
    if return_figure:
        del arr, arr2, nams
        return f

    # Otherwise save to disk, deriving the format from the output extension
    # (defaulting to SVG when there is none) so the alignment chrome stays vector.
    ext = os.path.splitext(outfile)[1].lower().lstrip('.')
    fmt = ext if ext else 'svg'
    f.savefig(outfile, format=fmt)

    # Clean up resources
    plt.close()
    del arr, arr2, nams

    return outfile


def _mark_cells(
    markupdict: Dict[str, List[RIPPosition]],
    ali_height: int,
    arr: Optional[np.ndarray],
    reaminate: bool,
):
    """
    Expand every drawn RIP mark into the alignment cells it covers, vectorised.

    Marks of the drawn categories are gathered, in their original order, into
    flat arrays; each is then expanded into the contiguous run of columns from
    the mark to ``col + offset`` (a single cell for offset 0/None). Context
    cells that fall outside the alignment or on a gap in that row are masked
    out, as the original per-cell loops did.

    Parameters
    ----------
    markupdict : dict of str to list of RIPPosition
        RIP positions by category (see :func:`markupRIPBases`).
    ali_height : int
        Number of alignment rows.
    arr : numpy.ndarray or None
        ``(ali_height, ali_width)`` alignment characters, used for bounds and
        gap checks; ``None`` skips both.
    reaminate : bool
        Whether ``'non_rip_deamination'`` marks are drawn.

    Returns
    -------
    tuple or None
        ``(cols, rows, ys, bases, single, span, valid)``: per-mark column, row,
        flipped y and base (``bases`` is a tuple of str), a boolean ``single``
        mask for zero-offset marks, and ``span``/``valid`` ``(n_marks, K)``
        arrays giving the column of every candidate cell and whether it is
        drawn. ``None`` when there is nothing to draw.
    """
    drawn = [
        positions
        for category, positions in markupdict.items()
        if positions and (category != 'non_rip_deamination' or reaminate)
    ]
    if not drawn:
        return None
    col_l, row_l, base_l, off_l = zip(
        *(pos for positions in drawn for pos in positions)
    )
    n_pos = len(col_l)
    cols = np.fromiter(col_l, dtype=np.int64, count=n_pos)
    rows = np.fromiter(row_l, dtype=np.int64, count=n_pos)
    offsets = np.fromiter(
        (0 if o is None else o for o in off_l), dtype=np.int64, count=n_pos
    )
    ys = ali_height - rows - 1

    single = offsets == 0
    length = np.abs(offsets) + 1
    start = np.where(offsets < 0, cols + offsets, cols)
    d = np.arange(int(length.max()))
    span = start[:, None] + d[None, :]
    valid = d[None, :] < length[:, None]
    if arr is not None:
        arr = np.asarray(arr)
        ali_width = arr.shape[1]
        valid &= (span >= 0) & (span < ali_width)
        ctx_valid = valid & ~single[:, None]
        rr = np.broadcast_to(rows[:, None], span.shape)
        gap = np.zeros(span.shape, dtype=bool)
        gap[ctx_valid] = arr[rr[ctx_valid], span[ctx_valid]] == '-'
        valid &= ~gap
    return cols, rows, ys, base_l, single, span, valid


def markupRIPBases(
    a: plt.Axes,
    markupdict: Dict[str, List[RIPPosition]],
    ali_height: int,
    arr: np.ndarray = None,
    reaminate: bool = False,
    palette: str = 'derip2',
    draw_boxes: bool = True,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Highlight RIP-related bases in the alignment plot with color coding and borders.

    This function visualizes different categories of RIP mutations by adding colored
    rectangles to the matplotlib axes. Target bases (primary mutation sites) are drawn
    with full opacity and black borders, while offset bases (context around mutations)
    are drawn with reduced opacity.

    Parameters
    ----------
    a : matplotlib.pyplot.Axes
        The matplotlib axes object where the alignment is being plotted.
    markupdict : Dict[str, List[RIPPosition]]
        Dictionary containing RIP positions to highlight, with categories as keys:
        - 'rip_product': Positions where RIP mutations have occurred (typically T from C→T)
        - 'rip_substrate': Positions with unmutated nucleotides in RIP context
        - 'non_rip_deamination': Positions with deamination events not in RIP context

        Each value is a list of RIPPosition named tuples with fields:
        - colIdx: column index in alignment (int)
        - rowIdx: row index in alignment (int)
        - base: nucleotide base at this position (str)
        - offset: context range around the mutation, negative=left, positive=right (int or None)
    ali_height : int
        Height of the alignment in rows (number of sequences).
    arr : np.ndarray, optional
        Original alignment array, needed to get base identities for offset positions.
        Shape should be (ali_height, alignment_width).
    reaminate : bool, optional
        Whether to include non-RIP deamination highlights (default: False).
    palette : str, optional
        Color palette to use for base highlighting (default: 'derip2').
    draw_boxes : bool, optional
        Whether to draw black borders around highlighted bases (default: True).

    Returns
    -------
    highlighted_positions : Set[Tuple[int, int]]
        Set of all (col_idx, y_coord) positions that received highlighting,
        including both target bases and offset positions.
    target_positions : Set[Tuple[int, int]]
        Set of only the primary mutation site (col_idx, y_coord) positions,
        excluding offset positions. Used for text coloring elsewhere.

    Notes
    -----
    - Target bases are drawn with full opacity and black borders
    - Offset bases (context) are drawn with 70% opacity
    - Text color is managed by the calling function based on target_positions
    - Coordinates in returned sets are in matplotlib coordinates, where y-axis
      is flipped compared to the alignment array (0 at bottom, increasing upward)
    """
    logger.debug(
        f'markupRIPBases: ali_height={ali_height}, reaminate={reaminate}, palette={palette}, draw_boxes={draw_boxes}'
    )

    highlighted_positions = set()
    target_positions = set()  # Track primary target positions separately
    border_thickness = 2.5  # Border thickness
    inset = 0.05  # Smaller inset for borders to reduce gap with grid lines

    # Define colors for nucleotide bases and precompute their RGBA tuples once.
    nuc_colors = get_color_palette(palette)
    rgba_cache = {
        base: matplotlib.colors.to_rgba(color) for base, color in nuc_colors.items()
    }
    default_rgba = matplotlib.colors.to_rgba('#CCCCCC')

    # Colored highlights are composited into a single transparent RGBA overlay
    # and drawn with one imshow, rather than adding one Rectangle patch per cell
    # (which does not scale to alignments with tens of thousands of markup cells).
    ali_width = arr.shape[1] if arr is not None else 0
    overlay = np.zeros((ali_height, ali_width, 4), dtype=float)

    cells = _mark_cells(markupdict, ali_height, arr, reaminate)
    if cells is None:
        return highlighted_positions, target_positions
    cols, rows, ys, base_l, single, span, valid = cells
    n_pos = cols.size

    # Every mark is a target and is highlighted, whatever its case below.
    target_positions.update(zip(cols.tolist(), ys.tolist()))
    highlighted_positions.update(target_positions)

    # RGBA lookup tables by character code: one for the marked base itself
    # (single-cell marks colour by the mark's base, and only if it is in the
    # palette) and one for context cells (coloured by the alignment base, grey
    # when unknown).
    base_lut = np.zeros((256, 4))
    base_known = np.zeros(256, dtype=bool)
    ctx_lut = np.tile(np.asarray(default_rgba), (256, 1))
    for base, rgba in rgba_cache.items():
        code = ord(base)
        if code < 256:
            base_lut[code] = rgba
            base_known[code] = True
            ctx_lut[code] = rgba
    base_codes = np.fromiter(
        (ord(b) if len(b) == 1 else 0 for b in base_l), dtype=np.int64, count=n_pos
    )
    base_codes[base_codes >= 256] = 0
    if arr is not None:
        arr_codes = np.asarray(arr).astype('U1').view(np.uint32).reshape(ali_height, -1)
        arr_codes = np.where(arr_codes < 256, arr_codes, 0).astype(np.int64)

    # Single-cell marks only draw when their base is in the palette.
    valid[single] &= base_known[base_codes[single]][:, None]

    # Context cells: every valid cell of a multi-cell mark, in (mark, cell) order.
    ctx_mask = valid & ~single[:, None]
    ctx_pos, ctx_d = np.nonzero(ctx_mask)
    ctx_x = span[ctx_pos, ctx_d]
    ctx_y = ys[ctx_pos]
    highlighted_positions.update(zip(ctx_x.tolist(), ctx_y.tolist()))

    # Colour/alpha per drawn cell, laid out in the same order as the old loop:
    # position-major, cell-minor. Single marks: the mark's base colour at full
    # opacity. Context marks: the alignment base colour, semi-transparent
    # except at the mark column itself.
    cell_pos, cell_d = np.nonzero(valid)
    cell_x = span[cell_pos, cell_d]
    cell_y = ys[cell_pos]
    cell_single = single[cell_pos]
    rgba = np.empty((cell_pos.size, 4))
    rgba[cell_single] = base_lut[base_codes[cell_pos[cell_single]]]
    if arr is not None:
        rgba[~cell_single] = ctx_lut[
            arr_codes[rows[cell_pos[~cell_single]], cell_x[~cell_single]]
        ]
    else:
        rgba[~cell_single] = base_lut[base_codes[cell_pos[~cell_single]]]
    alpha = np.where(cell_single | (cell_x == cols[cell_pos]), 1.0, 0.7)

    # Keep only in-bounds cells, then the LAST write to each cell.
    keep = (cell_x >= 0) & (cell_x < ali_width) & (cell_y >= 0) & (cell_y < ali_height)
    cell_x, cell_y, rgba, alpha = cell_x[keep], cell_y[keep], rgba[keep], alpha[keep]
    if cell_x.size:
        flat = cell_y * max(ali_width, 1) + cell_x
        _, last_rev = np.unique(flat[::-1], return_index=True)
        last = cell_x.size - 1 - last_rev
        overlay[cell_y[last], cell_x[last], :3] = rgba[last, :3]
        overlay[cell_y[last], cell_x[last], 3] = alpha[last]

    # Black borders: one rectangle per mark, spanning its drawn cells. Only
    # requested for small alignments, so a per-mark loop is fine here.
    if draw_boxes:
        for p in range(n_pos):
            drawn_cols = span[p][valid[p]]
            if drawn_cols.size == 0:
                continue
            x0, x1 = int(drawn_cols.min()), int(drawn_cols.max())
            a.add_patch(
                matplotlib.patches.Rectangle(
                    (x0 - 0.5 + inset, ys[p] - 0.5 + inset),
                    (x1 - x0 + 1) - 2 * inset,
                    1.0 - 2 * inset,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=border_thickness,
                    zorder=150,  # Above grid lines (100)
                )
            )

    # Draw all colored highlights in a single raster pass over the base image.
    if ali_width:
        a.imshow(
            overlay,
            aspect='auto',
            interpolation='nearest',
            zorder=50,  # Above base image (10), below grid lines (100)
        )

    return highlighted_positions, target_positions


def addColumnRangeMarkers(
    a: plt.Axes, ranges: List[Tuple[int, int, str, str]], ali_height: int
) -> None:
    """
    Add colored bars to mark column ranges in the alignment.

    Parameters
    ----------
    a : plt.Axes
        The matplotlib axes object containing the alignment.
    ranges : List[Tuple[int, int, str, str]]
        List of ranges to mark, each as (start_col, end_col, color, label).
    ali_height : int
        Height of the alignment (number of rows).

    Returns
    -------
    None
        Modifies the plot in-place.
    """
    # Set bar position and height
    bar_y = -2  # Below the alignment
    bar_height = 1

    for start_col, end_col, color, label in ranges:
        # Add colored bar
        a.add_patch(
            matplotlib.patches.Rectangle(
                (start_col - 0.5, bar_y),  # (x, y) bottom left corner
                end_col - start_col + 1,  # width
                bar_height,  # height
                color=color,  # fill color
                zorder=90,  # above most other elements
            )
        )

        # Add label if provided
        if label:
            mid_col = (start_col + end_col) / 2
            a.text(
                mid_col,  # x position (middle of range)
                bar_y - 0.5,  # y position (below bar)
                label,  # text
                ha='center',  # horizontal alignment
                va='top',  # vertical alignment
                fontsize=8,  # font size
                color='black',  # text color
            )


def addAnnotationTrack(
    ann_ax: plt.Axes, spans: List[Tuple[int, int, str, str, int]], ali_width: int
) -> Dict[str, str]:
    """
    Draw a stacked gene-annotation track in its own sub-plot below the alignment.

    Each span is a coloured bar at its ``track_row``; different annotation types
    occupy different rows so overlapping features (e.g. gene vs CDS vs exon) do
    not collide. No text labels are drawn on the plot itself — each bar instead
    carries a ``gid`` so the HTML report can attach a hover/click tooltip. The
    returned map lets the caller build those tooltips.

    Parameters
    ----------
    ann_ax : plt.Axes
        The dedicated annotation axes below the alignment.
    spans : List[Tuple[int, int, str, str, int]]
        Spans to draw, each as ``(start_col, end_col, color, label, track_row)``
        in alignment-column coordinates (0 at the top of the stack).
    ali_width : int
        Number of alignment columns, used to match the alignment axis x-range so
        the track lines up column-for-column.

    Returns
    -------
    Dict[str, str]
        Maps each bar's ``gid`` to its annotation label (for SVG tooltips).
    """
    # Match the alignment axis x-range so columns line up; invert y so row 0 is at
    # the top, and strip the axis chrome (the alignment axis carries the ticks).
    n_rows = (max(row for *_rest, row in spans) + 1) if spans else 1
    ann_ax.set_xlim(-0.5, ali_width - 0.5)
    ann_ax.set_ylim(n_rows, 0)
    ann_ax.set_xticks([])
    ann_ax.set_yticks([])
    for spine in ann_ax.spines.values():
        spine.set_visible(False)

    titles: Dict[str, str] = {}
    bar_height = 0.8
    gap = (1.0 - bar_height) / 2.0
    for i, (start_col, end_col, color, label, track_row) in enumerate(spans):
        rect = matplotlib.patches.Rectangle(
            (start_col - 0.5, track_row + gap),
            end_col - start_col + 1,
            bar_height,
            color=color,
            zorder=90,
        )
        gid = f'anntip{i}'
        rect.set_gid(gid)
        ann_ax.add_patch(rect)
        if label:
            titles[gid] = label
    return titles


def getHighlightedPositions(
    markupdict: Dict[str, List[RIPPosition]],
    ali_height: int,
    arr: np.ndarray = None,
    reaminate: bool = False,
) -> Set[Tuple[int, int]]:
    """
    Get all positions that should be highlighted based on the markup dictionary.

    Parameters
    ----------
    markupdict : Dict[str, List[RIPPosition]]
        Dictionary with categories as keys and lists of position tuples as values.
    ali_height : int
        Height of the alignment (number of rows).
    arr : np.ndarray, optional
        The original alignment array, used to check for gap positions.
    reaminate : bool, optional
        Whether to include non-RIP deamination positions.

    Returns
    -------
    Set[Tuple[int, int]]
        Set of (col_idx, flipped_y) tuples for all highlighted positions.
    """
    cells = _mark_cells(markupdict, ali_height, arr, reaminate)
    if cells is None:
        return set()
    cols, _rows, ys, _bases, single, span, valid = cells
    highlighted_positions = set(zip(cols.tolist(), ys.tolist()))
    ctx_pos, ctx_d = np.nonzero(valid & ~single[:, None])
    highlighted_positions.update(
        zip(span[ctx_pos, ctx_d].tolist(), ys[ctx_pos].tolist())
    )
    return highlighted_positions
