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
    num_threads: int = None,
    min_items_for_threading: int = 500,
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
    num_threads : int, optional
        Number of threads to use for parallel processing. If None, uses the number
        of CPU cores available (default: None).
    min_items_for_threading : int, optional
        Minimum number of cells/characters required to use parallel processing.
        For smaller alignments, sequential processing is used (default: 500).

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
    import concurrent.futures
    from multiprocessing import cpu_count
    import threading
    import time

    start_time = time.time()

    # Set number of threads to use if not specified
    if num_threads is None:
        num_threads = cpu_count()

    # If num_threads > available CPU cores, set to available cores
    if num_threads > cpu_count():
        logging.warning(
            f'Requested {num_threads} threads, but only {cpu_count()} available. Using {cpu_count()} threads.'
        )
        num_threads = cpu_count()

    # Log function call with important parameters
    logging.debug(
        f'drawMiniAlignment: outfile={outfile}, dpi={dpi}, title={title}, width={width}, height={height}, '
        f'show_chars={show_chars}, consensus_seq={"provided" if consensus_seq else "None"}, '
        f'corrected_positions={"provided" if corrected_positions else "None"}, '
        f'num_threads={num_threads}, min_items_for_threading={min_items_for_threading}'
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

    # Calculate total cells for threading decisions
    total_cells = ali_height * ali_width
    use_threading_for_chars = (
        show_chars
        and total_cells >= min_items_for_threading
        and num_threads > 1
        and ali_width < 500
    )

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
        markup_start_time = time.time()

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

        # Draw the colored highlights on top using the parallelized function
        highlighted_positions, target_positions = markupRIPBases(
            a,
            filtered_markup,
            ali_height,
            arr,
            reaminate,
            palette,
            draw_boxes,
            num_threads=num_threads,
            min_items_for_threading=min_items_for_threading,
        )

        markup_time = time.time() - markup_start_time
        logging.info(f'RIP markup processing took {markup_time:.2f} seconds')
    else:
        # No markup, just draw the regular alignment
        a.imshow(arr2, cmap=cm, aspect='auto', interpolation='nearest', zorder=10)
        _highlighted_positions = set()
        target_positions = set()

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
        chars_start_time = time.time()

        # Increase font size for better visibility
        char_fontsize = min(
            14, 18000 / (ali_width * ali_height)
        )  # Adjusted for larger font

        # Skip character rendering if they'll be too small
        if char_fontsize >= 4:
            # Use multithreaded character rendering for large alignments
            if use_threading_for_chars:
                logging.info(
                    f'Using {num_threads} threads to render {total_cells} characters'
                )

                # Text commands to be executed sequentially (for matplotlib thread safety)
                text_commands = []
                text_lock = threading.Lock()

                # Process a chunk of rows
                def process_rows(row_range):
                    nonlocal arr, ali_width, ali_height, target_positions
                    local_commands = []
                    for y in row_range:
                        for x in range(ali_width):
                            flipped_y = ali_height - y - 1
                            char = arr[y, x]
                            text_color = (
                                'black'
                                if (x, flipped_y) in target_positions
                                else '#777777'
                            )

                            # Store text parameters rather than rendering directly
                            local_commands.append((x, flipped_y, char, text_color))
                    return local_commands

                # Create chunks of rows for parallel processing
                chunk_size = max(
                    1, ali_height // (num_threads * 2)
                )  # Ensure at least 1 row per chunk
                row_chunks = [
                    range(i, min(i + chunk_size, ali_height))
                    for i in range(0, ali_height, chunk_size)
                ]

                # Process row chunks in parallel
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_threads
                ) as executor:
                    # Submit all row chunk processing jobs
                    future_to_chunk = {
                        executor.submit(process_rows, chunk): i
                        for i, chunk in enumerate(row_chunks)
                    }

                    # Process results as they complete
                    with tqdm(
                        total=len(row_chunks), desc='Rendering text', unit='chunk'
                    ) as pbar:
                        for future in concurrent.futures.as_completed(future_to_chunk):
                            try:
                                commands = future.result()
                                with text_lock:
                                    text_commands.extend(commands)
                                pbar.update(1)
                            except Exception as e:
                                logging.error(f'Error processing text chunk: {e}')

                # Render all text commands sequentially (for matplotlib thread safety)
                for x, y, char, color in text_commands:
                    a.text(
                        x,
                        y,
                        char,
                        ha='center',
                        va='center',
                        fontsize=char_fontsize,
                        color=color,
                        fontweight='bold',
                        zorder=200,
                    )
            else:
                # Sequential processing for smaller alignments
                for y in tqdm(range(ali_height), desc='Rendering text', unit='row'):
                    for x in range(ali_width):
                        flipped_y = ali_height - y - 1
                        char = arr[y, x]
                        text_color = (
                            'black' if (x, flipped_y) in target_positions else '#777777'
                        )

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

            chars_time = time.time() - chars_start_time
            logging.info(
                f'Character rendering took {chars_time:.2f} seconds for {total_cells} cells'
            )

    # If consensus sequence is provided, add it to the second subplot
    if consensus_seq is not None and consensus_ax is not None:
        consensus_start_time = time.time()

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

        # Determine if we should use threading for consensus visualization
        use_threading_for_consensus = (
            len(consensus_seq) >= min_items_for_threading and num_threads > 1
        )

        if use_threading_for_consensus:
            logging.info(
                f'Using {num_threads} threads to render consensus sequence ({len(consensus_seq)} bases)'
            )

            # Queue to collect drawing operations
            drawing_queue = []
            drawing_lock = threading.Lock()

            # Function to process a chunk of the consensus sequence
            def process_consensus_chunk(base_range):
                local_drawing_ops = []

                for i in base_range:
                    base = consensus_seq[i]

                    # Determine cell color based on whether this is a corrected position
                    if highlight_corrected and i not in corrected_set:
                        color = '#c7d1d0'  # Standard gray for non-corrected
                    else:
                        color = nuc_colors.get(
                            base.upper(), '#CCCCCC'
                        )  # Default to gray for unknown

                    # Add colored rectangle for this base
                    local_drawing_ops.append(
                        (
                            'rect',
                            i,
                            {
                                'xy': (i - 0.5, -0.5),
                                'width': 1,
                                'height': 1,
                                'color': color,
                                'zorder': 10,
                            },
                        )
                    )

                    # Add character as text if requested
                    if show_chars:
                        text_color = 'black'  # For better readability
                        fontsize = min(18, 30 - len(consensus_seq) / 100)

                        local_drawing_ops.append(
                            (
                                'text',
                                i,
                                {
                                    'x': i,
                                    'y': 0,
                                    'text': base,
                                    'ha': 'center',
                                    'va': 'center',
                                    'fontsize': fontsize,
                                    'color': text_color,
                                    'fontweight': 'bold',
                                    'zorder': 20,
                                },
                            )
                        )

                    # Add marker for corrected positions if requested
                    if corrected_positions and flag_corrected and i in corrected_set:
                        asterisk_fontsize = min(
                            24, max(14, 40 - len(consensus_seq) / 50)
                        )

                        local_drawing_ops.append(
                            (
                                'text',
                                i,
                                {
                                    'x': i,
                                    'y': 1.0,
                                    'text': '*',
                                    'ha': 'center',
                                    'va': 'center',
                                    'fontsize': asterisk_fontsize,
                                    'color': 'red',
                                    'fontweight': 'bold',
                                    'zorder': 30,
                                },
                            )
                        )

                return local_drawing_ops

            # Create chunks of consensus sequence bases for parallel processing
            chunk_size = max(1, len(consensus_seq) // (num_threads * 2))
            base_chunks = [
                range(i, min(i + chunk_size, len(consensus_seq)))
                for i in range(0, len(consensus_seq), chunk_size)
            ]

            # Process chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                future_to_chunk = {
                    executor.submit(process_consensus_chunk, chunk): i
                    for i, chunk in enumerate(base_chunks)
                }

                # Process results as they complete
                with tqdm(
                    total=len(base_chunks), desc='Rendering consensus', unit='chunk'
                ) as pbar:
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        try:
                            local_ops = future.result()
                            with drawing_lock:
                                drawing_queue.extend(local_ops)
                            pbar.update(1)
                        except Exception as e:
                            logging.error(f'Error processing consensus chunk: {e}')

            # Sort operations by index for consistent rendering
            drawing_queue.sort(key=lambda x: x[1])

            # Execute all drawing operations sequentially on the consensus axis
            for op_type, _, params in drawing_queue:
                if op_type == 'rect':
                    consensus_ax.add_patch(matplotlib.patches.Rectangle(**params))
                elif op_type == 'text':
                    consensus_ax.text(**params)
        else:
            # Sequential processing for consensus visualization
            for i in tqdm(
                range(len(consensus_seq)), desc='Rendering consensus', unit='base'
            ):
                base = consensus_seq[i]

                # Determine cell color
                if highlight_corrected and i not in corrected_set:
                    color = '#c7d1d0'  # Standard gray for non-corrected
                else:
                    color = nuc_colors.get(base.upper(), '#CCCCCC')

                # Add colored rectangle
                consensus_ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (i - 0.5, -0.5),  # bottom left corner
                        1,
                        1,  # width, height
                        color=color,
                        zorder=10,
                    )
                )

                # Add character text if requested
                if show_chars:
                    consensus_ax.text(
                        i,
                        0,
                        base,
                        ha='center',
                        va='center',
                        fontsize=min(18, 30 - len(consensus_seq) / 100),
                        color='black',
                        fontweight='bold',
                        zorder=20,
                    )

                # Add asterisk for corrected positions if requested
                if corrected_positions and flag_corrected and i in corrected_set:
                    asterisk_fontsize = min(24, max(14, 40 - len(consensus_seq) / 50))
                    consensus_ax.text(
                        i,
                        1.0,
                        '*',
                        ha='center',
                        va='center',
                        fontsize=asterisk_fontsize,
                        color='red',
                        fontweight='bold',
                        zorder=30,
                    )

        consensus_time = time.time() - consensus_start_time
        logging.info(
            f'Consensus visualization took {consensus_time:.2f} seconds for {len(consensus_seq)} bases'
        )

    # Save the plot as a PNG image
    logging.info(f'Saving figure to {outfile}')
    f.savefig(outfile, format='png')

    # Clean up resources
    plt.close()
    del arr, arr2, nams

    # Report total processing time
    total_time = time.time() - start_time
    logging.info(f'Total visualization time: {total_time:.2f} seconds')

    return outfile


def markupRIPBases(
    a: plt.Axes,
    markupdict: Dict[str, List[RIPPosition]],
    ali_height: int,
    arr: np.ndarray = None,
    reaminate: bool = False,
    palette: str = 'derip2',
    draw_boxes: bool = True,
    num_threads: int = None,
    min_items_for_threading: int = 100,
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
    num_threads : int, optional
        Number of threads to use for parallel processing. If None, uses the number
        of CPU cores available (default: None).
    min_items_for_threading : int, optional
        Minimum number of positions required to use parallel processing.
        For smaller datasets, sequential processing is used (default: 100).

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
    import concurrent.futures
    from multiprocessing import cpu_count
    import threading
    import time

    start_time = time.time()

    # DEBUG: Print function parameters for troubleshooting
    logging.debug(
        f'markupRIPBases: markupdict={markupdict}, ali_height={ali_height}, arr={arr}, '
        f'reaminate={reaminate}, palette={palette}, draw_boxes={draw_boxes}, '
        f'num_threads={num_threads}, min_items_for_threading={min_items_for_threading}'
    )

    # Initialize result sets with thread locks
    highlighted_positions = set()
    target_positions = set()
    highlighted_lock = threading.Lock()
    target_lock = threading.Lock()

    # Styling parameters
    border_thickness = 2.5  # Border thickness
    inset = 0.05  # Smaller inset for borders to reduce gap with grid lines

    # Define colors for nucleotide bases
    nuc_colors = get_color_palette(palette)

    # Count total positions to process and determine if threading should be used
    total_positions = sum(
        len(positions)
        for category, positions in markupdict.items()
        if category != 'non_rip_deamination' or reaminate
    )

    # Set number of threads to use
    if num_threads is None:
        num_threads = cpu_count()

    # If num_threads > available CPU cores, set to available cores
    if num_threads > cpu_count():
        logging.warning(
            f'Requested {num_threads} threads, but only {cpu_count()} available. Using {cpu_count()} threads.'
        )
        num_threads = cpu_count()

    # Determine if we should use threading based on dataset size
    use_threading = (total_positions >= min_items_for_threading) and (num_threads > 1)

    if use_threading:
        logging.info(
            f'Using {num_threads} threads to process {total_positions} RIP positions'
        )
    else:
        logging.info(f'Using sequential processing for {total_positions} RIP positions')

    # Create one progress bar for all positions
    pbar = tqdm(
        total=total_positions,
        desc='Highlighting RIP positions',
        unit='pos',
        ncols=80,
        leave=False,
    )

    # Store drawing instructions to handle matplotlib's thread-safety issues
    # We'll collect all drawing instructions and execute them sequentially
    drawing_queue = []
    drawing_lock = threading.Lock()

    def process_position(category, pos):
        """
        Process a single RIP position and generate drawing instructions.

        This function handles the visualization of a single RIP-related position,
        generating appropriate drawing instructions for both the target nucleotide
        and any context nucleotides (specified by offset). It creates rectangles
        with appropriate colors and, if requested, black borders around the highlighted
        regions.

        Parameters
        ----------
        category : str
            The category of RIP mutation ('rip_product', 'rip_substrate',
            or 'non_rip_deamination').
        pos : RIPPosition
            A named tuple containing:
            - colIdx : int
                Column index in the alignment matrix.
            - rowIdx : int
                Row index in the alignment matrix.
            - base : str
                Nucleotide base at this position.
            - offset : int or None
                Context range around the mutation, negative=left, positive=right.

        Returns
        -------
        tuple
            A tuple containing three elements:
            - local_highlighted : set
                Set of (x, y) coordinate tuples for all highlighted positions.
            - local_target : set
                Set of (x, y) coordinate tuples for primary mutation sites.
            - local_draw_instructions : list
                List of drawing instruction tuples, each containing:
                ('rect', {rectangle_parameters}) for matplotlib rendering.

        Notes
        -----
        This function accesses several variables from the outer scope:
        - ali_height : int
            Height of the alignment in rows.
        - arr : np.ndarray
            Original alignment array for accessing base information.
        - nuc_colors : dict
            Dictionary mapping nucleotides to color codes.
        - draw_boxes : bool
            Whether to draw borders around highlighted positions.
        - inset : float
            Inset amount for drawing borders.
        - border_thickness : float
            Thickness of the border lines.
        """
        col_idx, row_idx, base, offset = pos
        y = ali_height - row_idx - 1

        # Local results to collect
        local_highlighted = set()
        local_target = set()
        local_draw_instructions = []

        # Add target position to highlighted set
        local_highlighted.add((col_idx, y))
        local_target.add((col_idx, y))

        # Case 1: Single base (no offset or offset=0)
        if offset is None or offset == 0:
            if base in nuc_colors:
                color = nuc_colors[base]

                # Queue drawing instruction for the base rectangle
                local_draw_instructions.append(
                    (
                        'rect',
                        {
                            'xy': (col_idx - 0.5, y - 0.5),
                            'width': 1.0,
                            'height': 1.0,
                            'facecolor': color,
                            'edgecolor': 'none',
                            'linewidth': 0,
                            'zorder': 50,
                        },
                    )
                )

                # Queue drawing instruction for the border if needed
                if draw_boxes:
                    local_draw_instructions.append(
                        (
                            'rect',
                            {
                                'xy': (col_idx - 0.5 + inset, y - 0.5 + inset),
                                'width': 1.0 - 2 * inset,
                                'height': 1.0 - 2 * inset,
                                'facecolor': 'none',
                                'edgecolor': 'black',
                                'linewidth': border_thickness,
                                'zorder': 150,
                            },
                        )
                    )

        # Case 2: Multiple positions (with offset)
        elif offset != 0:
            # Process range and get valid cell indices
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

            # Collect valid indices (excluding gaps and out-of-bounds)
            valid_indices = []
            for i in range(start_idx, end_idx + 1):
                if i < 0 or (
                    arr is not None
                    and (i >= arr.shape[1] or arr[ali_height - y - 1, i] == '-')
                ):
                    continue
                valid_indices.append(i)

            if not valid_indices:
                return local_highlighted, local_target, local_draw_instructions

            # Queue drawing instructions for each valid cell
            for i in valid_indices:
                # Add position to highlighted set
                local_highlighted.add((i, y))

                # Get color for this base
                cell_base = arr[ali_height - y - 1, i] if arr is not None else base
                cell_color = nuc_colors.get(cell_base, '#CCCCCC')

                # Determine transparency based on whether it's target or context
                cell_alpha = 1.0
                if (offset > 0 or offset < 0) and i != col_idx:
                    cell_alpha = 0.7

                # Queue drawing instruction
                local_draw_instructions.append(
                    (
                        'rect',
                        {
                            'xy': (i - 0.5, y - 0.5),
                            'width': 1.0,
                            'height': 1.0,
                            'facecolor': cell_color,
                            'edgecolor': 'none',
                            'linewidth': 0,
                            'alpha': cell_alpha,
                            'zorder': 50,
                        },
                    )
                )

            # Queue drawing instruction for the border around the whole group
            if valid_indices and draw_boxes:
                start_i = min(valid_indices)
                end_i = max(valid_indices)

                local_draw_instructions.append(
                    (
                        'rect',
                        {
                            'xy': (start_i - 0.5 + inset, y - 0.5 + inset),
                            'width': (end_i - start_i + 1) - 2 * inset,
                            'height': 1.0 - 2 * inset,
                            'facecolor': 'none',
                            'edgecolor': 'black',
                            'linewidth': border_thickness,
                            'zorder': 150,
                        },
                    )
                )

        return local_highlighted, local_target, local_draw_instructions

    # Process positions either in parallel or sequentially
    if use_threading:
        # Prepare all tasks for processing
        all_tasks = []
        for category, positions in markupdict.items():
            # Skip non-RIP deamination if reaminate is False
            if category == 'non_rip_deamination' and not reaminate:
                continue
            for pos in positions:
                all_tasks.append((category, pos))

        # Process in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all position processing jobs
            future_results = {
                executor.submit(process_position, category, pos): (category, pos)
                for category, pos in all_tasks
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_results):
                try:
                    local_highlighted, local_target, local_draw = future.result()

                    # Synchronize results with locks
                    with highlighted_lock:
                        highlighted_positions.update(local_highlighted)
                    with target_lock:
                        target_positions.update(local_target)
                    with drawing_lock:
                        drawing_queue.extend(local_draw)

                    # Update progress bar
                    pbar.update(1)

                except Exception as e:
                    logging.error(f'Error processing position: {e}')
    else:
        # Sequential processing
        for category, positions in markupdict.items():
            # Skip non-RIP deamination if reaminate is False
            if category == 'non_rip_deamination' and not reaminate:
                continue

            # Update progress bar description to show current category
            pbar.set_description(f'Highlighting {category}')

            # Process each position sequentially
            for pos in positions:
                local_highlighted, local_target, local_draw = process_position(
                    category, pos
                )
                highlighted_positions.update(local_highlighted)
                target_positions.update(local_target)
                drawing_queue.extend(local_draw)
                pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Execute all drawing operations sequentially (for matplotlib thread safety)
    for draw_type, params in drawing_queue:
        if draw_type == 'rect':
            a.add_patch(matplotlib.patches.Rectangle(**params))

    # Report processing time
    elapsed_time = time.time() - start_time
    logging.info(
        f'RIP highlighting completed in {elapsed_time:.2f} seconds ({total_positions} positions)'
    )
    logging.info(
        f'Highlighted {len(highlighted_positions)} total positions, {len(target_positions)} target positions'
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


def getHighlightedPositions(
    markupdict: Dict[str, List[RIPPosition]],
    ali_height: int,
    arr: np.ndarray = None,
    reaminate: bool = False,
    num_threads: int = None,
    min_items_for_threading: int = 1000,
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
    num_threads : int, optional
        Number of threads to use for parallel processing. If None, uses the number
        of CPU cores available (default: None).
    min_items_for_threading : int, optional
        Minimum number of positions required to use parallel processing.
        For smaller datasets, sequential processing is used (default: 100).

    Returns
    -------
    Set[Tuple[int, int]]
        Set of (col_idx, flipped_y) tuples for all highlighted positions.
    """
    import concurrent.futures
    from multiprocessing import cpu_count
    import threading
    import time

    start_time = time.time()

    # Shared set for collecting results with thread safety
    highlighted_positions = set()
    positions_lock = threading.Lock()

    # Set number of threads to use if not specified
    if num_threads is None:
        num_threads = cpu_count()

    # If num_threads > available CPU cores, set to available cores
    if num_threads > cpu_count():
        logging.warning(
            f'Requested {num_threads} threads, but only {cpu_count()} available. Using {cpu_count()} threads.'
        )
        num_threads = cpu_count()

    # Count total positions to determine if threading should be used
    total_positions = sum(
        len(positions)
        for category, positions in markupdict.items()
        if category != 'non_rip_deamination' or reaminate
    )

    # Determine if threading should be used
    use_threading = total_positions >= min_items_for_threading and num_threads > 1

    # Function to process a single markup position
    def process_position(pos):
        """
        Process a single position and collect its coordinates for highlighting.

        This function takes a RIP position and determines all coordinates that should
        be highlighted in the visualization. It handles both the target position itself
        and any context positions specified by the offset parameter, taking into account
        alignment boundaries and gaps.

        Parameters
        ----------
        pos : RIPPosition
            A named tuple containing:
            - colIdx : int
                Column index in the alignment matrix.
            - rowIdx : int
                Row index in the alignment matrix.
            - base : str
                Nucleotide base at this position.
            - offset : int or None
                Context range around the mutation, negative=left, positive=right.

        Returns
        -------
        set
            Set of (col_idx, y) coordinate tuples for all positions that should be
            highlighted, where y is the flipped row index in matplotlib coordinates.

        Notes
        -----
        This function accesses several variables from the outer scope:
        - ali_height : int
            Height of the alignment in rows.
        - arr : np.ndarray
            Original alignment array for checking gap positions.
        """
        col_idx, row_idx, base, offset = pos
        local_positions = set()

        # Convert row index to matplotlib coordinates (flipped)
        y = ali_height - row_idx - 1

        # Add target position to highlighted set
        local_positions.add((col_idx, y))

        # Handle offset positions
        if offset is not None:
            if offset < 0:
                # Negative offset means positions to the left
                for i in range(col_idx + offset, col_idx):
                    if i >= 0 and (arr is None or arr[ali_height - y - 1, i] != '-'):
                        local_positions.add((i, y))
            elif offset > 0:
                # Positive offset means positions to the right
                for i in range(col_idx + 1, col_idx + offset + 1):
                    if arr is None or (
                        i < arr.shape[1] and arr[ali_height - y - 1, i] != '-'
                    ):
                        local_positions.add((i, y))

        return local_positions

    if use_threading:
        logging.debug(
            f'Using {num_threads} threads to process {total_positions} positions for highlighting'
        )

        # Collect all positions to process
        positions_to_process = []
        for category, positions in markupdict.items():
            # Skip non-RIP deamination if reaminate is False
            if category != 'non_rip_deamination' or reaminate:
                positions_to_process.extend(positions)

        # Process positions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create progress bar for parallel processing
            with tqdm(
                total=total_positions, desc='Finding highlight positions', unit='pos'
            ) as pbar:
                # Submit all positions for processing
                future_to_pos = {
                    executor.submit(process_position, pos): pos
                    for pos in positions_to_process
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_pos):
                    try:
                        local_positions = future.result()
                        # Synchronize results
                        with positions_lock:
                            highlighted_positions.update(local_positions)
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f'Error processing position: {e}')
    else:
        # Sequential processing
        for category, positions in markupdict.items():
            # Skip non-RIP deamination if reaminate is False
            if category == 'non_rip_deamination' and not reaminate:
                continue

            # Process each position sequentially with progress tracking
            with tqdm(
                total=len(positions), desc=f'Finding {category} positions', unit='pos'
            ) as pbar:
                for pos in positions:
                    local_positions = process_position(pos)
                    highlighted_positions.update(local_positions)
                    pbar.update(1)

    elapsed_time = time.time() - start_time
    logging.info(
        f'Position highlighting calculation completed in {elapsed_time:.2f} seconds'
    )
    logging.info(f'Found {len(highlighted_positions)} positions to highlight')

    return highlighted_positions
