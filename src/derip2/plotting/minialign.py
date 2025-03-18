"""
DNA alignment visualization tool for generating overview images of sequence alignments.

This module provides functions to visualize DNA sequence alignments as color-coded
images, making it easier to identify patterns, gaps, and conserved regions. It is
derived from the CIAlign package (https://github.com/KatyBrown/CIAlign) with
modifications for the deRIP2 project.
"""

from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from Bio import SeqIO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for server environments

RIPPosition = NamedTuple(
    'RIPPosition', [('colIdx', int), ('rowIdx', int), ('base', str), ('offset', int)]
)


def FastaToArray(
    infile: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[int]]:
    """
    Convert a FASTA alignment file into a numpy array.

    Parameters
    ----------
    infile : str
        Path to input alignment file in FASTA format.

    Returns
    -------
    arr : np.ndarray or None
        2D numpy array where each row represents a sequence and each column
        represents a position in the alignment. Returns None if only one
        sequence is found.
    nams : List[str] or None
        List of sequence names in the same order as in the input file.
        Returns None if only one sequence is found.
    seq_len : int or None
        Number of sequences in the alignment. Returns None if only one
        sequence is found.

    Raises
    ------
    ValueError
        If the sequences in the file have different lengths (not properly aligned).
    """
    # Initialize lists to store sequence names and data
    nams = []
    seqs = []

    # Define valid nucleotide characters for DNA sequences
    valid_chars: Set[str] = {'A', 'G', 'C', 'T', 'N', '-'}

    # Parse FASTA file and extract sequences
    for record in SeqIO.parse(infile, 'fasta'):
        nams.append(record.id)
        # Convert sequence to uppercase and replace invalid characters with gaps
        seq = [base if base in valid_chars else '-' for base in str(record.seq).upper()]
        seqs.append(seq)

    # Check if we have enough sequences for an alignment
    seq_len = len(seqs)
    if seq_len <= 1:
        return None, None, None

    # Verify all sequences have the same length (proper alignment)
    seq_lengths = {len(seq) for seq in seqs}
    if len(seq_lengths) > 1:
        raise ValueError(
            'ERROR: The sequences provided may not be aligned - all the sequences are not the same length.'
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

    # Define color palettes for different visualization preferences
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
    }

    # Select the appropriate color pattern or default to colorblind
    if palette not in color_palettes:
        palette = 'colorblind'
    color_pattern = color_palettes[palette]

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
    input_file: str,
    outfile: str,
    dpi: int = 300,
    title: Optional[str] = None,
    width: int = 20,
    height: int = 15,
    orig_nams: Optional[List[str]] = None,
    keep_numbers: bool = False,
    force_numbers: bool = False,
    palette: str = 'colorblind',
    markupdict: Optional[Dict[str, List[RIPPosition]]] = None,
    column_ranges: Optional[List[Tuple[int, int, str, str]]] = None,
) -> Union[str, bool]:
    """
    Generate a visualization of a DNA sequence alignment with optional RIP markup.

    This function creates an image showing a color-coded representation of the
    entire alignment, making it easy to spot patterns, gaps, and conserved regions.
    It can highlight specific RIP-related bases and mark column ranges.

    Parameters
    ----------
    input_file : str
        Path to the input alignment file in FASTA format.
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
        Color palette to use: 'colorblind', 'bright', or 'tetrimmer' (default: 'colorblind').
    markupdict : Dict[str, List[RIPPosition]], optional
        Dictionary with RIP categories as keys and lists of position tuples as values.
        Categories are 'rip_product', 'rip_substrate', and 'non_rip_deamination'.
        Each position is a named tuple with (colIdx, rowIdx, base, offset).
    column_ranges : List[Tuple[int, int, str, str]], optional
        List of column ranges to mark, each as (start_col, end_col, color, label).

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
    # Handle default value for orig_nams
    if orig_nams is None:
        orig_nams = []

    # Convert the FASTA file to a numpy array
    arr, nams, seq_len = FastaToArray(input_file)

    # Return False if only one sequence was found
    if arr is None:
        return False

    # Adjust height for small alignments
    if seq_len <= 75:
        height = seq_len * 0.2

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

    # Calculate line weights based on alignment dimensions
    lineweight_h = 10 / ali_height  # Horizontal grid lines
    lineweight_v = 10 / ali_width  # Vertical grid lines

    # Create the figure and axis
    f = plt.figure(figsize=(width, height), dpi=dpi)
    a = f.add_subplot(1, 1, 1)
    a.set_xlim(-0.5, ali_width)
    a.set_ylim(-0.5, ali_height - 0.5)

    # Convert alignment to numeric form and get color map
    arr2, cm = arrNumeric(arr, palette=palette)

    # Plot the alignment as an image
    a.imshow(arr2, cmap=cm, aspect='auto', interpolation='nearest')

    # Adjust subplot positioning
    f.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95)

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

    # Remove unnecessary spines
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_visible(False)

    # Add title if provided
    if title:
        f.suptitle(title, fontsize=fontsize * 1.5, y=0.92)

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

    # Apply RIP markup if provided
    if markupdict:
        markupRIPBases(a, markupdict, ali_height)

    # Add column range markers if provided
    if column_ranges:
        addColumnRangeMarkers(a, column_ranges, ali_height)

    # Save the plot as a PNG image
    f.savefig(outfile, format='png', bbox_inches='tight')

    # Clean up resources
    plt.close()
    del arr, arr2, nams

    return outfile


def markupRIPBases(
    a: plt.Axes, markupdict: Dict[str, List[RIPPosition]], ali_height: int
) -> None:
    """
    Highlight RIP-related bases in the alignment with colored rectangles.

    Parameters
    ----------
    a : plt.Axes
        The matplotlib axes object containing the alignment.
    markupdict : Dict[str, List[RIPPosition]]
        Dictionary with categories as keys and lists of position tuples as values.
        Each position is a named tuple with (colIdx, rowIdx, offset).
    ali_height : int
        Height of the alignment (number of rows).

    Returns
    -------
    None
        Modifies the plot in-place.
    """
    # Define colors for each category
    colors = {
        'rip_product': '#FF0000',  # Red
        'rip_substrate': '#0000FF',  # Blue
        'non_rip_deamination': '#FFA500',  # Orange
    }

    # First add a semi-transparent grey overlay to dim the entire alignment
    a.add_patch(
        matplotlib.patches.Rectangle(
            (-0.5, -0.5),  # (x, y) bottom left corner
            a.get_xlim()[1] + 1,  # width (covering entire plot)
            ali_height,  # height
            color='gray',  # grey fill
            alpha=0.5,  # semi-transparent
            zorder=40,  # below markup
        )
    )

    # Add colored rectangles for each category
    for category, positions in markupdict.items():
        color = colors.get(
            category, '#CCCCCC'
        )  # Default to grey if category not recognized

        for pos in positions:
            col_idx, row_idx, base, offset = pos

            # Convert row index to matplotlib coordinates (flipped)
            y = ali_height - row_idx - 1

            # Calculate range based on offset
            start_col = col_idx
            end_col = col_idx
            if offset < 0:
                # Negative offset means highlight to the left
                start_col = max(0, col_idx + offset)
            elif offset > 0:
                # Positive offset means highlight to the right
                end_col = min(int(a.get_xlim()[1]), col_idx + offset)

            width = end_col - start_col + 1

            # Add colored rectangle
            a.add_patch(
                matplotlib.patches.Rectangle(
                    (start_col - 0.5, y - 0.5),  # (x, y) bottom left corner
                    width,  # width
                    1,  # height (one row)
                    color=color,  # fill color
                    alpha=0.8,  # slightly transparent
                    zorder=50,  # above grey background
                )
            )


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
