"""
DNA alignment visualization tool for generating overview images of sequence alignments.

This module provides functions to visualize DNA sequence alignments as color-coded
images, making it easier to identify patterns, gaps, and conserved regions. It is
derived from the CIAlign package (https://github.com/KatyBrown/CIAlign) with
modifications for the deRIP2 project.
"""

import logging
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from Bio.Align import MultipleSeqAlignment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')  # Use non-interactive backend for server environments

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
        logging.warning(f"Palette '{palette}' not found, using 'colorblind' instead")
        return color_palettes['colorblind']

    return color_palettes[palette]


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
    # DEBUG: Print function parameters for troubleshooting
    logging.debug(f'MSAToArray: alignment={alignment}')

    # Check if alignment is empty
    if not alignment or len(alignment) == 0:
        raise ValueError('Empty alignment provided')

    # Initialize lists to store sequence names and data
    nams = []
    seqs = []

    # Define valid nucleotide characters for DNA sequences
    valid_chars: Set[str] = {'A', 'G', 'C', 'T', 'N', '-'}

    # Extract sequences from the alignment object
    for record in alignment:
        nams.append(record.id)
        # Convert sequence to uppercase and replace invalid characters with gaps
        seq = [
            base if base.upper() in valid_chars else '-'
            for base in str(record.seq).upper()
        ]
        seqs.append(seq)

    # Check if we have enough sequences for an alignment
    seq_len = len(seqs)
    if seq_len <= 1:
        return None, None, None

    # Verify all sequences have the same length (proper alignment)
    # This should always be true for a Biopython MSA object, but check anyway
    seq_lengths = {len(seq) for seq in seqs}
    if len(seq_lengths) > 1:
        raise ValueError(
            'ERROR: The sequences in the alignment have different lengths. This should not happen with a MultipleSeqAlignment.'
        )

    # Convert list of sequences to numpy array
    arr = np.array(seqs)
    return arr, nams, seq_len


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

    # Create mapping from nucleotides to numeric values
    keys = list(color_pattern.keys())
    nD = {}  # Dictionary mapping nucleotides to integers
    colours = []  # List of colors for the colormap

    # Build the mapping and color list for the specific nucleotides in the alignment
    i = 0
    for key in keys:
        if key in arr:
            nD[key] = i
            colours.append(color_pattern[key])
            i += 1

    # Create the numeric representation of the alignment
    arr2 = np.empty([ali_height, ali_width])
    for x in range(ali_width):
        for y in range(ali_height):
            # Convert each nucleotide to its corresponding integer
            arr2[y, x] = nD[arr[y, x]]

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
    show_chars: bool = False,
    draw_boxes: bool = False,
    consensus_seq: Optional[str] = None,
    corrected_positions: Optional[List[int]] = None,
    reaminate: bool = False,
    reference_seq_index: Optional[int] = None,
    show_rip: str = 'both',  # 'substrate', 'product', or 'both'
    highlight_corrected: bool = True,
    flag_corrected: bool = False,
) -> Union[str, bool]:
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

    Returns
    -------
    Union[str, bool]
        Path to the output image file if successful, False if only one sequence was found.

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
    logging.debug(
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

    # Create the figure with subplots if consensus is provided
    if consensus_seq is not None:
        # Use GridSpec for more control over subplot sizes
        f = plt.figure(
            figsize=(width + width_padding, height + height_padding), dpi=dpi
        )
        gs = f.add_gridspec(
            2, 1, height_ratios=[4, 1]
        )  # 4:1 ratio for alignment:consensus

        # Create the alignment subplot
        a = f.add_subplot(gs[0])

        # Create the consensus subplot
        consensus_ax = f.add_subplot(gs[1])

        # Position adjustments - keep the same relative positioning
        f.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.88, hspace=0.5)
    else:
        # Create a single plot for alignment only
        f = plt.figure(
            figsize=(width + width_padding, height + height_padding), dpi=dpi
        )
        a = f.add_subplot(1, 1, 1)
        # Keep the same relative positioning
        f.subplots_adjust(top=0.88, bottom=0.15, left=0.12, right=0.88)
        consensus_ax = None  # No consensus subplot

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

    # Add grid lines
    a.hlines(
        np.arange(-0.5, ali_height),
        -0.5,
        ali_width,
        lw=lineweight_h,
        color='white',
        zorder=100,
    )
    a.vlines(
        np.arange(-0.5, ali_width),
        -0.5,
        ali_height,
        lw=lineweight_v,
        color='white',
        zorder=100,
    )

    # Mark reference sequence with a black circle at the end of the row if provided
    if reference_seq_index is not None and 0 <= reference_seq_index < ali_height:
        # Convert row index to matplotlib coordinates (flipped)
        ref_y = ali_height - reference_seq_index - 1

        # First, convert data coordinates to display coordinates
        # This finds where in the figure the end of the reference row is
        display_coords = a.transData.transform((ali_width - 0.5, ref_y))

        # Convert display coordinates to figure coordinates
        fig_coords = f.transFigure.inverted().transform(display_coords)

        # Add a smaller offset to place the circle closer to the alignment
        circle_x = fig_coords[0] + 0.015  # Reduced offset for closer positioning
        circle_y = fig_coords[1]  # Same vertical position

        # Get figure dimensions to calculate aspect ratio
        fig_width_inches, fig_height_inches = f.get_size_inches()
        aspect_ratio = fig_width_inches / fig_height_inches

        # Create a smaller ellipse that will appear as a circle by accounting for aspect ratio
        circle = matplotlib.patches.Ellipse(
            (circle_x, circle_y),  # Position in figure coordinates
            width=0.0075,  # X radius (horizontal)
            height=0.0075 * aspect_ratio,  # Y radius adjusted for aspect ratio
            facecolor='black',  # Black fill
            edgecolor='white',  # White border
            linewidth=1.5,  # Border thickness
            transform=f.transFigure,  # Use figure coordinates
            zorder=1000,  # Ensure it's on top
        )
        f.patches.append(circle)  # Add to figure patches

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

        # Add vertical grid lines
        consensus_ax.vlines(
            np.arange(-0.5, len(consensus_seq)),
            -0.5,
            1.5,  # Extended grid lines to cover the new space
            lw=lineweight_v,
            color='white',
            zorder=100,
        )

        # Plot each base in the consensus as a colored cell with character
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

            # Create colored rectangle for this base
            consensus_ax.add_patch(
                matplotlib.patches.Rectangle(
                    (i - 0.5, -0.5),  # bottom left corner
                    1,
                    1,  # width, height
                    color=color,
                    zorder=10,
                )
            )

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

    # Save the plot as a PNG image
    f.savefig(outfile, format='png')

    # Clean up resources
    plt.close()
    del arr, arr2, nams

    return outfile


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
    # DEBUG: Print function parameters for troubleshooting
    logging.debug(
        f'markupRIPBases: markupdict={markupdict}, ali_height={ali_height}, arr={arr}, reaminate={reaminate}, palette={palette}, draw_boxes={draw_boxes}'
    )

    highlighted_positions = set()
    target_positions = set()  # Track primary target positions separately
    border_thickness = 2.5  # Border thickness
    inset = 0.05  # Smaller inset for borders to reduce gap with grid lines

    # Define colors for nucleotide bases
    nuc_colors = get_color_palette(palette)

    # Count total positions to process for progress bar
    total_positions = sum(
        len(positions)
        for category, positions in markupdict.items()
        if category != 'non_rip_deamination' or reaminate
    )

    # Create one progress bar for all positions
    pbar = tqdm(
        total=total_positions,
        desc='Highlighting RIP positions',
        unit='pos',
        ncols=80,
        leave=False,
    )

    # Process all positions in the markup dictionary
    for category, positions in markupdict.items():
        # Skip non-RIP deamination if reaminate is False
        if category == 'non_rip_deamination' and not reaminate:
            continue

        # Update progress bar description to show current category
        pbar.set_description(f'Highlighting {category}')

        # Process each position with progress tracking
        for pos in positions:
            col_idx, row_idx, base, offset = pos
            y = ali_height - row_idx - 1
            highlighted_positions.add((col_idx, y))
            target_positions.add(
                (col_idx, y)
            )  # Add only the target position to target set

            # Case 1: Single base (no offset or offset=0)
            if offset is None or offset == 0:
                if base in nuc_colors:
                    color = nuc_colors[base]

                    # Draw the base with full-size colored rectangle - use EXACT cell dimensions
                    a.add_patch(
                        matplotlib.patches.Rectangle(
                            (col_idx - 0.5, y - 0.5),  # Exact cell boundaries
                            1.0,  # Full width
                            1.0,  # Full height
                            facecolor=color,
                            edgecolor='none',
                            linewidth=0,
                            zorder=50,  # Above base image (10), below grid lines (100)
                        )
                    )

                    # Draw black border with smaller inset and thinner line for cleaner appearance
                    if draw_boxes:
                        a.add_patch(
                            matplotlib.patches.Rectangle(
                                (
                                    col_idx - 0.5 + inset,
                                    y - 0.5 + inset,
                                ),  # Inset from cell edge
                                1.0 - 2 * inset,  # Width with minimal inset
                                1.0 - 2 * inset,  # Height with minimal inset
                                facecolor='none',
                                edgecolor='black',
                                linewidth=border_thickness,
                                zorder=150,  # Above grid lines (100)
                            )
                        )

            # Case 2: Multiple positions (with offset)
            elif offset != 0:
                # Process range and get valid cells as before
                if offset < 0:  # Positions to the left
                    start_idx = max(0, col_idx + offset)
                    end_idx = col_idx
                else:  # Positions to the right
                    start_idx = col_idx
                    end_idx = (
                        min(arr.shape[1] - 1, col_idx + offset)
                        if arr is not None
                        else col_idx + offset
                    )

                # Skip gaps and out-of-bounds positions
                valid_indices = []
                for i in range(start_idx, end_idx + 1):
                    if i < 0 or (
                        arr is not None
                        and (i >= arr.shape[1] or arr[ali_height - y - 1, i] == '-')
                    ):
                        continue
                    valid_indices.append(i)

                if not valid_indices:
                    pbar.update(1)  # Update progress bar even if skipping
                    continue

                # Fill cells with appropriate colors (full-size) - use EXACT cell dimensions
                for i in valid_indices:
                    # Add to highlighted positions
                    highlighted_positions.add((i, y))
                    # Note: We don't add offset positions to target_positions

                    # Get color for this base
                    cell_base = arr[ali_height - y - 1, i] if arr is not None else base
                    cell_color = nuc_colors.get(cell_base, '#CCCCCC')

                    # Determine alpha value based on whether this is the target cell or an offset cell
                    # For cells with offset > 0, make offset cells semi-transparent
                    cell_alpha = 1.0
                    if (
                        offset > 0 or offset < 0
                    ) and i != col_idx:  # This is an offset cell
                        cell_alpha = 0.7

                    # Draw full-size colored cell with appropriate transparency
                    a.add_patch(
                        matplotlib.patches.Rectangle(
                            (i - 0.5, y - 0.5),  # Exact cell boundaries
                            1.0,  # Full width
                            1.0,  # Full height
                            facecolor=cell_color,
                            edgecolor='none',
                            linewidth=0,
                            alpha=cell_alpha,  # Apply transparency to offset cells
                            zorder=50,  # Above base image, below grid
                        )
                    )

                # Draw border with smaller inset
                if valid_indices and draw_boxes:
                    start_i = min(valid_indices)
                    end_i = max(valid_indices)

                    a.add_patch(
                        matplotlib.patches.Rectangle(
                            (start_i - 0.5 + inset, y - 0.5 + inset),
                            (end_i - start_i + 1) - 2 * inset,
                            1.0 - 2 * inset,
                            facecolor='none',
                            edgecolor='black',
                            linewidth=border_thickness,
                            zorder=150,  # Above grid lines
                        )
                    )

            # Update progress bar
            pbar.update(1)

    # Close the progress bar
    pbar.close()

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
    highlighted_positions = set()

    for category, positions in markupdict.items():
        # Skip non-RIP deamination if reaminate is False
        if category == 'non_rip_deamination' and not reaminate:
            continue

        for pos in positions:
            col_idx, row_idx, base, offset = pos

            # Convert row index to matplotlib coordinates (flipped)
            y = ali_height - row_idx - 1

            # Add target position to highlighted set
            highlighted_positions.add((col_idx, y))

            # Handle offset positions
            if offset is not None:
                if offset < 0:
                    # Negative offset means positions to the left
                    for i in range(col_idx + offset, col_idx):
                        if i >= 0 and (
                            arr is None or arr[ali_height - y - 1, i] != '-'
                        ):
                            highlighted_positions.add((i, y))
                elif offset > 0:
                    # Positive offset means positions to the right
                    for i in range(col_idx + 1, col_idx + offset + 1):
                        if i <= arr.shape[1] - 1 and (
                            arr is None or arr[ali_height - y - 1, i] != '-'
                        ):
                            highlighted_positions.add((i, y))

    return highlighted_positions
