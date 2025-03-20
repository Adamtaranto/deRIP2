"""
Alignment operations for deRIP2.

This module provides functions for manipulating and analyzing DNA sequence alignments,
with a focus on detecting and correcting RIP (Repeat-Induced Point mutation) mutations.
It includes utilities for loading alignments, tracking RIP-like mutations, building
consensus sequences, and outputting corrected sequences in various formats.
"""

from collections import Counter, namedtuple
from copy import deepcopy

# import defaultdict
from io import StringIO
import logging
from operator import itemgetter
import sys
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from Bio import AlignIO, SeqIO
from Bio.Align import AlignInfo, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction
from tqdm import tqdm

from derip2.utils.checks import isfile

RIPPosition = NamedTuple(
    'RIPPosition', [('colIdx', int), ('rowIdx', int), ('base', str), ('offset', int)]
)


def checkUniqueID(align: MultipleSeqAlignment) -> None:
    """
    Validate that all sequence IDs in an alignment are unique.

    This function checks if there are any duplicate sequence IDs in the alignment.
    If duplicates are found, it logs a warning message with the duplicate IDs and
    terminates program execution.

    Parameters
    ----------
    align : MultipleSeqAlignment
        The sequence alignment to check for unique IDs.

    Returns
    -------
    None
        Function doesn't return any value if all IDs are unique.

        Raises
    ------
    SystemExit
        If any duplicate sequence IDs are found in the alignment.
    """
    # Extract all sequence IDs from the alignment
    rowIDs = [list(align)[x].id for x in range(align.__len__())]

    # Count occurrences of each ID
    IDcounts = Counter(rowIDs)

    # Identify IDs that occur more than once
    duplicates = [k for k, v in IDcounts.items() if v > 1]

    # If duplicates were found, log warning and exit
    if duplicates:
        logging.error('Sequence IDs not unique. Quiting.')
        logging.info(f'Non-unique IDs: {duplicates}')
        sys.exit(1)


def checkLen(align: MultipleSeqAlignment) -> None:
    """
    Validate that an alignment contains at least two sequences.

    This function checks if the alignment has a minimum of two sequences.
    If fewer than two sequences are found, it logs an error message and
    terminates program execution.

    Parameters
    ----------
    align : MultipleSeqAlignment
        The sequence alignment to check.

    Returns
    -------
    None
        Function doesn't return any value if alignment length is valid.

    Raises
    ------
    SystemExit
        If the alignment contains fewer than two sequences.
    """
    # Check if alignment has fewer than 2 sequences
    if align.__len__() < 2:
        # Log error and terminate execution if requirement not met
        logging.error('Alignment contains < 2 sequences. Quiting.')
        sys.exit(1)
    # If alignment has 2 or more sequences, function returns implicitly


def loadAlign(file: str, alnFormat: str = 'fasta') -> 'AlignIO.MultipleSeqAlignment':
    """
    Import an alignment file and validate its contents.

    This function loads a sequence alignment from a file, verifies the file exists,
    and performs basic validation checks such as ensuring the alignment has at least
    two sequences and all sequence IDs are unique. Supports both plain text and
    gzipped alignment files.

    Parameters
    ----------
    file : str
        Path to the alignment file to be loaded. Can be a regular file or a gzipped file.
    alnFormat : str, optional
        Format of the alignment file (default: 'fasta').
        Must be a format supported by Biopython's AlignIO.

    Returns
    -------
    Bio.Align.MultipleSeqAlignment
        Alignment object containing the loaded sequences.

    Raises
    ------
    SystemExit
        If alignment contains fewer than 2 sequences or contains duplicate sequence IDs.
    FileNotFoundError
        If the specified file does not exist.
    """
    # Add import for gzip support
    import gzip

    # Verify the input file exists and return its path
    path = isfile(file)

    # Check if file is gzipped based on extension
    is_gzipped = path.lower().endswith(('.gz', '.gzip'))

    # Log loading information with compression status
    logging.info(
        f'Loading{"" if not is_gzipped else " gzipped"} alignment from file: {file}'
    )

    # Load the alignment from file using Biopython's AlignIO
    # Use gzip.open for compressed files, regular open for uncompressed files
    if is_gzipped:
        with gzip.open(path, 'rt') as f:  # "rt" mode = read text
            align = AlignIO.read(f, alnFormat)
    else:
        align = AlignIO.read(path, alnFormat)

    # Validate alignment has at least 2 sequences (exits if not)
    checkLen(align)

    # Validate all sequences have unique IDs (exits if duplicates found)
    checkUniqueID(align)

    return align


def alignSummary(align: 'AlignIO.MultipleSeqAlignment') -> None:
    """
    Log a summary of an alignment's dimensions and sequence IDs.

    This function logs information about the alignment including the number of
    sequences (rows), alignment length (columns), and a table mapping row indices
    to sequence IDs. This is useful for debugging and tracking alignment processing.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to summarize.

    Returns
    -------
    None
        This function only logs information and doesn't return a value.
    """
    logging.debug('Generating alignment summary...')
    aln_summary_msg = []

    # Log the dimensions of the alignment
    aln_summary_msg.append(
        f'\nAlignment has {align.__len__()} rows and {align.get_alignment_length()} columns.'
    )

    # Create a header for the row index to sequence ID mapping table
    aln_summary_msg.append('Row\tID')

    # Log each sequence's index and ID as a table row
    for x in range(align.__len__()):
        aln_summary_msg.append(f'{x}:\t{align[x].id}')

    # Join all messages into a single string and log it
    logging.info('\n'.join(aln_summary_msg))


def checkrow(align: 'AlignIO.MultipleSeqAlignment', idx: Optional[int] = None) -> None:
    """
    Validate that a row index is within the range of an alignment.

    This function checks if the provided row index is valid for the given alignment.
    If the index is out of range, it logs a warning message and terminates program
    execution.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to check against.
    idx : int, optional
        The row index to validate. If None, no validation is performed.

    Returns
    -------
    None
        Function doesn't return any value if the row index is valid.

    Raises
    ------
    SystemExit
        If the provided index is outside the valid range of rows in the alignment
    """
    # Check if index is provided and is outside the range of the alignment
    if idx not in range(align.__len__()):
        # Log warning and terminate execution if index is invalid
        logging.warning(f'Row index {idx} is outside range. Quitting.')
        sys.exit(1)
    # If index is valid, function returns implicitly


def initTracker(align: 'AlignIO.MultipleSeqAlignment') -> dict:
    """
    Initialize a tracking dictionary for storing the consensus deRIPed sequence.

    Creates a dictionary where keys are column indices (0-based) and values are
    named tuples containing the column index and a base value (initially None).
    This tracker is used to build up the deRIPed consensus sequence as the
    algorithm progresses.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The input sequence alignment.

    Returns
    -------
    dict
        Dictionary mapping column indices to namedtuples with fields:
        - idx: int, the column index
        - base: str or None, the nucleotide base (initially None).
    """
    logging.debug('Initializing consensus sequence tracker...')

    # Create empty dictionary to track consensus sequence
    tracker = {}

    # Define namedtuple type for storing column position and base information
    colItem = namedtuple('colPosition', ['idx', 'base'])

    # Initialize tracker with one entry per alignment column, all bases set to None
    for x in range(align.get_alignment_length()):
        tracker[x] = colItem(idx=x, base=None)

    return tracker


def initRIPCounter(align: 'AlignIO.MultipleSeqAlignment') -> Dict[int, NamedTuple]:
    """
    Initialize a counter to track RIP mutations observed in each sequence.

    Creates a dictionary where keys are row indices (0-based) and values are
    named tuples containing sequence information and counters for tracking
    different types of RIP mutations.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The input sequence alignment.

    Returns
    -------
    Dict[int, NamedTuple]
        Dictionary mapping row indices to namedtuples with fields:
        - idx: int, the row index.
        - SeqID: str, the sequence identifier.
        - revRIPcount: int, counter for reverse strand RIP mutations.
        - RIPcount: int, counter for forward strand RIP mutations.
        - nonRIPcount: int, counter for non-RIP C→T or G→A mutations.
        - GC: float, GC content percentage of the sequence.
    """

    logging.debug('Initializing RIP mutation counter...')

    # Create empty dictionary to track RIP mutations for each sequence
    RIPcounts = {}

    # Define namedtuple type for storing sequence information and RIP counters
    rowItem = namedtuple(
        'RIPtracker', ['idx', 'SeqID', 'revRIPcount', 'RIPcount', 'nonRIPcount', 'GC']
    )

    # Initialize counter for each sequence in the alignment
    for x in range(align.__len__()):
        # Create entry with sequence info, zero-initialized counters, and calculated GC content
        RIPcounts[x] = rowItem(
            idx=x,  # Row index in alignment
            SeqID=align[x].id,  # Sequence identifier
            revRIPcount=0,  # Counter for reverse strand RIP mutations
            RIPcount=0,  # Counter for forward strand RIP mutations
            nonRIPcount=0,  # Counter for non-RIP mutations
            GC=gc_fraction(align[x].seq) * 100,  # GC content as percentage
        )

    return RIPcounts


def updateTracker(
    idx: int, newChar: str, tracker: Dict[int, NamedTuple], force: bool = False
) -> Dict[int, NamedTuple]:
    """
    Update a position in the consensus sequence tracker.

    Updates the base value for a specific column index in the tracker dictionary.
    By default, only updates if the position has not been assigned a base (is None).
    With force=True, will overwrite any existing base value.

    Parameters
    ----------
    idx : int
        Column index to update in the alignment.
    newChar : str
        The new character/base to assign at this position.
    tracker : dict
        Dictionary mapping column indices to namedtuples with fields idx and base.
    force : bool, optional
        If True, overwrite existing base values; if False, only update if current value is None
        (default: False).

    Returns
    -------
    dict
        Updated tracker dictionary.
    """
    # If position already has a value and force=True, overwrite it
    if tracker[idx].base and force:
        # Log that we're overwriting an existing base
        logging.info(
            f"Overwriting base at position {idx}: '{tracker[idx].base}' → '{newChar}'"
        )

        # Overwrite the existing base with the new value
        tracker[idx] = tracker[idx]._replace(base=newChar)
    # If position has no value yet, update it
    elif not tracker[idx].base:
        # Set the base at this position to the new value
        tracker[idx] = tracker[idx]._replace(base=newChar)

    # Return the updated tracker
    return tracker


def updateRIPCount(
    idx: int,
    RIPtracker: Dict[int, NamedTuple],
    addRev: int = 0,
    addFwd: int = 0,
    addNonRIP: int = 0,
) -> Dict[int, NamedTuple]:
    """
    Update counters for observed RIP mutations in a specific sequence.

    This function increments the counters for different types of RIP mutations
    in the RIP tracking dictionary for a specified sequence (row index).

    Parameters
    ----------
    idx : int
        Row index of the sequence to update counters for.
    RIPtracker : Dict[int, NamedTuple]
        Dictionary tracking RIP mutation counts for each sequence.
    addRev : int, optional
        Number of reverse strand RIP mutations to add (default: 0).
    addFwd : int, optional
        Number of forward strand RIP mutations to add (default: 0).
    addNonRIP : int, optional
        Number of non-RIP deamination events to add (default: 0).

    Returns
    -------
    Dict[int, NamedTuple]
        Updated RIPtracker dictionary with incremented counters.
    """
    # Calculate new totals by adding increments to current values
    TallyRev = RIPtracker[idx].revRIPcount + addRev  # Reverse strand RIP count
    TallyFwd = RIPtracker[idx].RIPcount + addFwd  # Forward strand RIP count
    TallyNonRIP = RIPtracker[idx].nonRIPcount + addNonRIP  # Non-RIP deamination count

    # Update the namedtuple with new counter values while preserving other fields
    RIPtracker[idx] = RIPtracker[idx]._replace(
        revRIPcount=TallyRev,  # Update reverse strand RIP counter
        RIPcount=TallyFwd,  # Update forward strand RIP counter
        nonRIPcount=TallyNonRIP,  # Update non-RIP deamination counter
    )

    return RIPtracker


def fillConserved(
    align: 'AlignIO.MultipleSeqAlignment',
    tracker: Dict[int, NamedTuple],
    maxGaps: float = 0.7,
) -> Dict[int, NamedTuple]:
    """
    Update tracker with bases from invariant or highly gapped alignment columns.

    This function examines each column in the alignment and updates the tracker
    with bases from positions that are:
    1. Completely invariant (all non-gap positions have the same base)
    2. Invariant except for gaps (all non-gap positions have the same base)
    3. Highly gapped columns (gap proportion exceeds maxGaps threshold)

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The input sequence alignment to analyze.
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    maxGaps : float, optional
        Maximum proportion of gaps allowed in a column before considering
        it a gap column in the consensus (default: 0.7).

    Returns
    -------
    Dict[int, NamedTuple]
        Updated tracker dictionary with bases filled in for conserved positions.
    """
    logging.debug('Filling conserved positions in the consensus sequence...')
    # Create deep copy of tracker to avoid modifying the original
    tracker = deepcopy(tracker)

    # Process each column in alignment
    for idx in range(align.get_alignment_length()):
        # Get frequencies for DNA bases + gaps in this column
        colProps = AlignInfo.SummaryInfo(align)._get_letter_freqs(
            idx, align, ['A', 'T', 'G', 'C', '-'], []
        )

        # Case 1: If column is completely invariant, use that base
        # (frequency of 1.0 means all positions have this base)
        for base in [k for k, v in colProps.items() if v == 1]:
            tracker = updateTracker(idx, base, tracker, force=False)

        # Case 2: If non-gap positions are invariant (base + gap = 100%)
        for base in [k for k, v in colProps.items() if v + colProps['-'] == 1]:
            # Exclude gap character as potential base
            # Only update if gap proportion is below threshold
            if base != '-' and colProps['-'] < maxGaps:
                tracker = updateTracker(idx, base, tracker, force=False)

        # Case 3: If column has more gaps than threshold, use gap character
        if itemgetter('-')(colProps) >= maxGaps:
            # Set this position to gap in the tracker
            tracker = updateTracker(idx, '-', tracker, force=False)

    return tracker


def nextBase(
    align: 'AlignIO.MultipleSeqAlignment', colID: int, motif: str
) -> Tuple[List[int], List[int]]:
    """
    Find rows where a base is followed by a specific nucleotide in the next non-gap position.

    This function identifies all rows in an alignment where the column at index colID
    contains the first base of a specified dinucleotide motif, and the next non-gap
    position contains the second base of the motif.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to analyze.
    colID : int
        Column index to check for the first base of the motif.
    motif : str
        Dinucleotide motif (e.g., 'CA' or 'TG').

    Returns
    -------
    Tuple[List[int], List[int]]
        A tuple containing:
        - List of row indices where the specified pattern was found.
        - List of corresponding offsets (distance to the next non-gap position).

    Raises
    ------
    ValueError
        If the row indices and offsets lists have different lengths.

    Examples
    --------
    >>> rows, offsets = nextBase(alignment, 5, 'CA')
    >>> print(f"Found CA motif at rows {rows} with offsets {offsets}")
    """
    # Find all rows where colID base matches first base of motif
    # Note: Column IDs are indexed from zero
    rowsX = find(align[:, colID], motif[0])

    # Initialize output list to store matching rows
    rowsXY = []
    # Initialize list to store offsets for each row
    # (distance to next non-gap position after motif base)
    offsets = []

    # For each row where starting col matches first base of motif
    for rowID in rowsX:
        offset = 0
        # Loop through all positions to the right of starting col
        # From position to immediate right of X to end of seq
        for base in align[rowID].seq[colID + 1 :]:
            offset += 1
            # For first non-gap position encountered
            if base != '-':
                # Check if base matches motif position two
                if base == motif[1]:
                    # If base is a match, add row ID to result list
                    rowsXY.append(rowID)
                    # Add offset to list
                    offsets.append(offset)
                # If first non-gap position is not a match, end loop for this row
                break
            # Else if position is a gap, continue to the next base

    # Check that rowsXY and offsets are the same length
    if len(rowsXY) != len(offsets):
        raise ValueError('Row indices and offsets are not the same length.')

    return rowsXY, offsets


def lastBase(
    align: 'AlignIO.MultipleSeqAlignment', colID: int, motif: str
) -> Tuple[List[int], List[int]]:
    """
    Find rows where a base is preceded by a specific nucleotide in the previous non-gap position.

    This function identifies all rows in an alignment where the column at index colID
    contains the second base of a specified dinucleotide motif, and the previous non-gap
    position contains the first base of the motif.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to analyze.
    colID : int
        Column index to check for the second base of the motif.
    motif : str
        Dinucleotide motif (e.g., 'CA' or 'TG').

    Returns
    -------
    Tuple[List[int], List[int]]
        A tuple containing:
        - List of row indices where the specified pattern was found.
        - List of corresponding offsets (distance to the previous non-gap position).

    Raises
    ------
    ValueError
        If the row indices and offsets lists have different lengths.
    """
    # Find all rows where colID base matches second base of motif
    rowsY = find(align[:, colID], motif[1])

    # Initialize output list to store matching rows
    rowsXY = []
    # Initialize list to store offsets for each row
    # (distance to first non-gap position preceding motif base)
    offsets = []

    # For each row where current col matches second base of motif
    for rowID in rowsY:
        offset = 0
        # From position to immediate left of Y to beginning of seq, reversed
        for base in align[rowID].seq[colID - 1 :: -1]:
            offset -= 1
            # For first non-gap position encountered
            if base != '-':
                # Check if base matches motif position one
                if base == motif[0]:
                    # If it is a match, add row ID to result list
                    rowsXY.append(rowID)
                    # Add offset to list
                    offsets.append(offset)
                # If first non-gap position is not a match, end loop for this row
                break
            # Else if position is a gap, continue to the previous base

    # Check that rowsXY and offsets are the same length
    if len(rowsXY) != len(offsets):
        raise ValueError('Row indices and offsets are not the same length.')

    return rowsXY, offsets


def find(lst: List[str], a: Union[str, List[str], Set[str]]) -> List[int]:
    """
    Find indices of elements in a list that match specified characters.

    Parameters
    ----------
    lst : List[str]
        List or sequence of characters to search through.
    a : Union[str, List[str], Set[str]]
        Character or collection of characters to find in the list.

    Returns
    -------
    List[int]
        List of indices where matching characters were found.

    Examples
    --------
    >>> find(['A', 'T', 'G', 'C', 'A'], 'A')
    [0, 4]
    >>> find(['A', 'T', 'G', 'C', 'A'], ['A', 'T'])
    [0, 1, 4]
    """
    # Convert search target to a set for efficient lookup
    search_set = set(a)

    # Return indices where list items are in the search set using list comprehension
    return [i for i, x in enumerate(lst) if x in search_set]


def hasBoth(lst: List[str], a: str, b: str) -> bool:
    """
    Check if a list contains at least one instance of each of two characters.

    Parameters
    ----------
    lst : List[str]
        List or sequence of characters to search through.
    a : str
        First character to find.
    b : str
        Second character to find.

    Returns
    -------
    bool
        True if both characters are present, False otherwise.

    Examples
    --------
    >>> hasBoth(['A', 'T', 'G', 'C'], 'A', 'T')
    True
    >>> hasBoth(['A', 'T', 'G', 'C'], 'A', 'N')
    False
    """
    # Find indices of first character
    hasA = find(lst, a)

    # Find indices of second character
    hasB = find(lst, b)

    # Return True if both characters were found (both lists are non-empty)
    return bool(hasA and hasB)


def replaceBase(
    align: 'AlignIO.MultipleSeqAlignment',
    targetCol: int,
    targetRows: List[int],
    newbase: str,
) -> 'AlignIO.MultipleSeqAlignment':
    """
    Replace bases in specific positions of a multiple sequence alignment.

    This function modifies an alignment by replacing bases at a specific column
    for multiple rows with a new base character (e.g., for masking).

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to modify.
    targetCol : int
        Column index where bases should be replaced.
    targetRows : List[int]
        List of row indices identifying sequences to modify.
    newbase : str
        New base character to insert (can be an IUPAC ambiguity code).

    Returns
    -------
    Bio.Align.MultipleSeqAlignment
        Modified alignment with replaced bases.
    """
    # For each target row in the alignment
    for row in targetRows:
        # Convert sequence to list for modification
        seqList = list(align[row].seq)

        # Replace the base at the target column
        seqList[targetCol] = newbase

        # Convert back to a Seq object and update the alignment
        # Note: No longer using deprecated Gapped(IUPAC.ambiguous_dna) alphabet
        align[row].seq = Seq(''.join(seqList))

    return align


def correctRIP(
    align: 'AlignIO.MultipleSeqAlignment',
    tracker: Dict[int, NamedTuple],
    RIPcounts: Dict[int, NamedTuple],
    maxSNPnoise: float = 0.5,
    minRIPlike: float = 0.1,
    reaminate: bool = True,
    mask: bool = False,
) -> Tuple[
    Dict[int, NamedTuple],
    Dict[int, NamedTuple],
    'AlignIO.MultipleSeqAlignment',
    List[int],
    Dict[str, List[RIPPosition]],
]:
    """
    Scan alignment for RIP-like mutations and correct them in the consensus sequence.

    This function analyzes each column of the alignment for patterns consistent with
    Repeat-Induced Point (RIP) mutations, which typically involve C→T transitions in
    specific dinucleotide contexts. For each identified RIP site, it:
    1. Logs the RIP event in the RIPcounts tracker
    2. Updates the consensus sequence tracker with the ancestral (pre-RIP) base
    3. Optionally masks the corrected positions in the output alignment

    RIP signatures as observed in the + sense strand, with RIP targeting CpA
    motifs on either the +/- strand:

    Target strand:    ++  --
    Wild type:     5' CA--TG 3'
    RIP mutated:   5' TA--TA 3'
    Consensus:        YA--TR

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The input sequence alignment to analyze.
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    RIPcounts : Dict[int, NamedTuple]
        Dictionary tracking RIP mutation counts for each sequence.
    maxSNPnoise : float, optional
        Minimum proportion of positions in a column that must be C/T or G/A to be
        considered for RIP correction (default: 0.5).
    minRIPlike : float, optional
        Minimum proportion of C→T or G→A transitions that must be in a RIP-like
        context to trigger correction (default: 0.1).
    reaminate : bool, optional
        If True, also correct C→T or G→A transitions not in RIP context (default: True).
    mask : bool, optional
        If True, mask corrected positions in the alignment output (default: False).

    Returns
    -------
    Tuple[Dict[int, NamedTuple], Dict[int, NamedTuple], Bio.Align.MultipleSeqAlignment, List[int], Dict[str, List[RIPPosition]]]
        A tuple containing:
        - Updated tracker dictionary with corrected bases
        - Updated RIPcounts dictionary with observed RIP events
        - Masked alignment (if mask=True) showing positions that were corrected
        - List of column indices that were corrected in the consensus
        - Dictionary mapping RIP mutation categories to positions for visualization:
          'rip_product': Positions containing RIP mutation products (e.g., T from C→T)
          'rip_substrate': Positions containing unmutated nucleotides in RIP context
          'non_rip_deamination': Positions with C→T or G→A outside of RIP context
    """
    logging.debug('Correcting RIP-like mutations in the consensus sequence...')
    # Create deep copies of input objects to avoid modifying the originals
    tracker = deepcopy(tracker)
    RIPcounts = deepcopy(RIPcounts)
    maskedAlign = deepcopy(align)

    # Store colIdx for each position that was corrected in the tracker
    corrected_positions = []

    # markupdict : Dict[str, List[RIPPosition]], optional
    #    Dictionary with RIP categories as keys and lists of position tuples as values.
    #    Categories are 'rip_product', 'rip_substrate', and 'non_rip_deamination'.
    #    Each position is a named tuple with (colIdx, rowIdx, offset).
    #    i.e. RIPPosition = NamedTuple('RIPPosition', [('colIdx', int), ('rowIdx', int), ('base',str),('offset', int)])

    # Initialize dictionary to store RIP categories for each position
    markupdict = {'rip_product': [], 'rip_substrate': [], 'non_rip_deamination': []}

    # Process each column in the alignment with progress bar
    for colIdx in tqdm(
        range(align.get_alignment_length()),
        desc='Scanning for RIP mutations',
        unit='column',
        ncols=80,
    ):
        # Track if we revert T→C or A→G in this column
        modC = False
        modG = False

        # Count total number of nucleotide bases (excluding gaps)
        baseCount = len(find(align[:, colIdx], ['A', 'T', 'G', 'C']))

        # Skip columns with no bases
        if baseCount:
            # Identify rows containing C or T in this column
            CTinCol = find(align[:, colIdx], ['C', 'T'])
            # Identify rows containing G or A in this column
            GAinCol = find(align[:, colIdx], ['G', 'A'])

            # Calculate proportion of C/T and G/A bases
            CTprop = len(CTinCol) / baseCount
            GAprop = len(GAinCol) / baseCount

            # FORWARD STRAND RIP DETECTION (C→T)
            # Check if column has sufficient C/T content
            if CTprop >= maxSNPnoise:
                # Find rows where C is followed by A (RIP substrate)
                # Even if whole column is C, we can still have RIP substrate
                CArows, _CA_nextbase_offsets = nextBase(align, colIdx, motif='CA')
                # Record forward strand RIP substrate for CA rows
                for rowCA, offset in zip(CArows, _CA_nextbase_offsets):
                    markupdict = updateMarkupDict(
                        'rip_substrate',
                        markupdict,
                        colIdx,
                        base='C',
                        row_idx=rowCA,
                        offset=offset,
                    )
                # Check if C/T content is higher than G/A content and both C and T are present
                if CTprop > GAprop and hasBoth(align[:, colIdx], 'C', 'T'):
                    # Find rows where C/T is followed by A (potential RIP context)
                    TArows, _TA_nextbase_offsets = nextBase(
                        align, colIdx, motif='TA'
                    )  # T followed by A (mutated)
                    CArows, _CA_nextbase_offsets = nextBase(
                        align, colIdx, motif='CA'
                    )  # C followed by A (ancestral)

                    # Get rows with T in this column
                    TinCol = find(align[:, colIdx], ['T'])

                    # If we have both CA and TA context (indicating RIP transition)
                    if CArows and TArows:
                        # Calculate proportion of C/T positions in a RIP-like context
                        propRIPlike = (len(TArows) + len(CArows)) / len(CTinCol)

                        # Record forward strand RIP substrate for CA rows
                        for rowCA, offset in zip(CArows, _CA_nextbase_offsets):
                            markupdict = updateMarkupDict(
                                'rip_substrate',
                                markupdict,
                                colIdx,
                                base='C',
                                row_idx=rowCA,
                                offset=offset,
                            )

                        # Record forward strand RIP events for TA rows
                        for rowTA in set(TArows):
                            RIPcounts = updateRIPCount(rowTA, RIPcounts, addFwd=1)

                        for rowTA, offset in zip(TArows, _TA_nextbase_offsets):
                            markupdict = updateMarkupDict(
                                'rip_product',
                                markupdict,
                                colIdx,
                                base='T',
                                row_idx=rowTA,
                                offset=offset,
                            )

                        # Record non-RIP deamination for T's not in TA context
                        for TnonRIP in {x for x in TinCol if x not in TArows}:
                            RIPcounts = updateRIPCount(TnonRIP, RIPcounts, addNonRIP=1)
                            markupdict = updateMarkupDict(
                                'non_rip_deamination',
                                markupdict,
                                colIdx,
                                base='T',
                                row_idx=TnonRIP,
                                offset=0,
                            )

                        # If sufficient mutations are in RIP context, correct to ancestral C
                        if propRIPlike >= minRIPlike:
                            tracker = updateTracker(colIdx, 'C', tracker, force=False)
                            modC = True
                            # Log corrected position in tracker
                            corrected_positions.append(colIdx)
                        # Otherwise correct if reaminate option is enabled
                        elif reaminate:
                            tracker = updateTracker(colIdx, 'C', tracker, force=False)
                            modC = True
                            # Log corrected position in tracker
                            corrected_positions.append(colIdx)

                    # If C and T present but not in RIP context
                    else:
                        # If reaminate flag is on, correct to C anyway
                        if reaminate:
                            tracker = updateTracker(colIdx, 'C', tracker, force=False)
                            modC = True
                            # Log corrected position in tracker
                            corrected_positions.append(colIdx)

                        # Log all T's as non-RIP deamination events
                        for TnonRIP in TinCol:
                            RIPcounts = updateRIPCount(TnonRIP, RIPcounts, addNonRIP=1)
                            markupdict = updateMarkupDict(
                                'non_rip_deamination',
                                markupdict,
                                colIdx,
                                base='T',
                                row_idx=TnonRIP,
                                offset=0,
                            )

            # REVERSE STRAND RIP DETECTION (G→A)
            # Check if column has sufficient G/A content and both G and A are present
            if GAprop >= maxSNPnoise:
                # Find rows where G is followed by T (RIP substrate)
                # Even if whole column is G, we can still have RIP substrate
                TGrows, _TG_lastbase_offsets = lastBase(align, colIdx, motif='TG')
                # Record forward strand RIP substrate for TG rows
                for rowTG, offset in zip(TGrows, _TG_lastbase_offsets):
                    markupdict = updateMarkupDict(
                        'rip_substrate',
                        markupdict,
                        colIdx,
                        base='G',
                        row_idx=rowTG,
                        offset=offset,
                    )
                # Check if G/A content is higher than C/T content and both G and A are present
                if GAprop > CTprop and hasBoth(align[:, colIdx], 'G', 'A'):
                    # Find rows where G/A is preceded by T (potential RIP context)
                    TGrows, _TG_lastbase_offsets = lastBase(
                        align, colIdx, motif='TG'
                    )  # T followed by G (ancestral)
                    TArows, _TA_lastbase_offsets = lastBase(
                        align, colIdx, motif='TA'
                    )  # T followed by A (mutated)

                    # Get rows with A in this column
                    AinCol = find(align[:, colIdx], ['A'])

                    # If we have both TG and TA context (indicating RIP transition)
                    if TGrows and TArows:
                        # Calculate proportion of G/A positions in a RIP-like context
                        propRIPlike = (len(TGrows) + len(TArows)) / len(GAinCol)

                        # Record forward strand RIP substrate for TG rows
                        for rowTG, offset in zip(TGrows, _TG_lastbase_offsets):
                            markupdict = updateMarkupDict(
                                'rip_substrate',
                                markupdict,
                                colIdx,
                                base='G',
                                row_idx=rowTG,
                                offset=offset,
                            )

                        # Record reverse strand RIP events for TA rows
                        for rowTA in set(TArows):
                            RIPcounts = updateRIPCount(rowTA, RIPcounts, addRev=1)

                        for rowTA, offset in zip(TArows, _TA_lastbase_offsets):
                            markupdict = updateMarkupDict(
                                'rip_product',
                                markupdict,
                                colIdx,
                                base='A',
                                row_idx=rowTA,
                                offset=offset,
                            )

                        # Record non-RIP deamination for A's not in TA context
                        for AnonRIP in {x for x in AinCol if x not in TArows}:
                            RIPcounts = updateRIPCount(AnonRIP, RIPcounts, addNonRIP=1)
                            markupdict = updateMarkupDict(
                                'non_rip_deamination',
                                markupdict,
                                colIdx,
                                base='A',
                                row_idx=AnonRIP,
                                offset=0,
                            )

                        # If sufficient mutations are in RIP context, correct to ancestral G
                        if propRIPlike >= minRIPlike:
                            tracker = updateTracker(colIdx, 'G', tracker, force=False)
                            modG = True
                            # Log corrected position in tracker
                            corrected_positions.append(colIdx)
                        # Otherwise correct if reaminate option is enabled
                        elif reaminate:
                            tracker = updateTracker(colIdx, 'G', tracker, force=False)
                            modG = True
                            # Log corrected position in tracker
                            corrected_positions.append(colIdx)

                    # If G and A present but not in RIP context
                    else:
                        # If reaminate flag is on, correct to G anyway
                        if reaminate:
                            tracker = updateTracker(colIdx, 'G', tracker, force=False)
                            modG = True
                            # Log corrected position in tracker
                            corrected_positions.append(colIdx)

                        # Log all A's as non-RIP deamination events
                        for AnonRIP in AinCol:
                            RIPcounts = updateRIPCount(AnonRIP, RIPcounts, addNonRIP=1)
                            markupdict = updateMarkupDict(
                                'non_rip_deamination',
                                markupdict,
                                colIdx,
                                base='A',
                                row_idx=AnonRIP,
                                offset=0,
                            )

            # Apply masking for C→T corrections if requested
            if modC:
                if reaminate:
                    # If reaminating all C→T transitions, mask all T positions in column
                    targetRows = find(align[:, colIdx], ['T'])
                else:
                    # Otherwise only mask 'T' positions in TpA context where C→T occurred
                    targetRows = TArows
                    # substrate_rows = CArows

                # Replace target positions with IUPAC ambiguity code Y (C or T)
                maskedAlign = replaceBase(maskedAlign, colIdx, targetRows, 'Y')

            # Apply masking for G→A corrections if requested
            if modG:
                if reaminate:
                    # If reaminating all G→A transitions, mask all positions
                    targetRows = find(align[:, colIdx], ['A'])
                else:
                    # Otherwise only mask 'A' positions in TpA context where G→A occurred
                    targetRows = TArows
                    # substrate_rows = TGrows

                # Replace target positions with IUPAC ambiguity code R (G or A)
                maskedAlign = replaceBase(maskedAlign, colIdx, targetRows, 'R')

    return (tracker, RIPcounts, maskedAlign, corrected_positions, markupdict)


def updateMarkupDict(
    category: str,
    markupdict: Dict[str, List[RIPPosition]],
    colIdx: int,
    base: str,
    row_idx: int,
    offset: int,
) -> Dict[str, List[RIPPosition]]:
    """
    Update a dictionary tracking RIP mutation categories with new position data.

    This function adds a new RIP position entry to the specified category in the markup
    dictionary. Each position contains column index, row index, nucleotide base, and
    an offset value indicating context position.

    Parameters
    ----------
    category : str
        Category of the RIP mutation, typically one of:
        'rip_product' - Position containing a RIP mutation product (e.g., T from C→T)
        'rip_substrate' - Position containing an unmutated nucleotide in RIP context
        'non_rip_deamination' - Position with C→T or G→A outside of RIP context.
    markupdict : Dict[str, List[RIPPosition]]
        Dictionary containing categories as keys and lists of RIP positions as values.
    colIdx : int
        Column index in the alignment (0-based).
    base : str
        Nucleotide base at this position ('A', 'C', 'G', 'T').
    row_idx : int
        Row index in the alignment (0-based).
    offset : int or None
        Distance to contextual base that forms RIP dinucleotide context:
        - Positive value: offset positions to the right
        - Negative value: offset positions to the left
        - None: no specific dinucleotide context.

    Returns
    -------
    Dict[str, List[RIPPosition]]
        Updated markup dictionary with the new position added to the specified category.

    Notes
    -----
    The function creates a new RIPPosition namedtuple and appends it to the
    list in markupdict under the specified category.

    This markup can be used for visualization highlighting of RIP patterns in
    the alignment.

    Examples
    --------
    >>> markupdict = {'rip_product': [], 'rip_substrate': [], 'non_rip_deamination': []}
    >>> markupdict = updateMarkupDict('rip_product', markupdict, colIdx=15, base='T',
    ...                               row_idx=3, offset=1)
    """
    # Create new namedtuple with position data
    newpos = RIPPosition(colIdx=colIdx, rowIdx=row_idx, base=base, offset=offset)

    # Check if this position already exists in the list
    if newpos not in markupdict[category]:
        # Only append if it's not already in the list
        markupdict[category].append(newpos)

    return markupdict


def summarizeRIP(RIPcounts: Dict[int, NamedTuple]) -> str:
    """
    Generate a summary of RIP mutation counts and GC content for each sequence.

    This function generates a well-formatted tabular report showing the frequency
    of RIP-like mutations detected in each sequence of the alignment, along with
    their GC content.

    Parameters
    ----------
    RIPcounts : Dict[int, NamedTuple]
        Dictionary tracking RIP mutation counts for each sequence.

    Returns
    -------
    str
        A formatted string containing the RIP summary table.
    """
    logging.debug('Summarizing RIP mutation counts...')

    from io import StringIO

    import pandas as pd

    # Create data for pandas DataFrame
    data = []
    for x in range(len(RIPcounts)):
        total_rip = RIPcounts[x].revRIPcount + RIPcounts[x].RIPcount
        data.append(
            {
                'Index': RIPcounts[x].idx,
                'ID': RIPcounts[x].SeqID,
                'RIP events': total_rip,
                'Non-RIP-deamination': RIPcounts[x].nonRIPcount,
                'GC %': round(RIPcounts[x].GC, 2),
            }
        )

    # Create DataFrame and format it
    df = pd.DataFrame(data)

    # Use StringIO to capture formatted output
    buffer = StringIO()
    df.to_string(buffer, index=False)

    # Return the formatted string
    return buffer.getvalue()


def setRefSeq(
    align: 'AlignIO.MultipleSeqAlignment',
    RIPcounter: Optional[Dict[int, NamedTuple]] = None,
    getMinRIP: bool = True,
    getMaxGC: bool = False,
) -> int:
    """
    Determine the optimal reference sequence for filling remaining positions.

    This function selects the best reference sequence based on either:
    1. Sequence with fewest RIP mutations (if getMinRIP is True)
    2. Sequence with highest GC content (if getMaxGC is True or no RIP data available)

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to analyze.
    RIPcounter : Dict[int, NamedTuple], optional
        Dictionary tracking RIP mutation counts for each sequence (default: None).
    getMinRIP : bool, optional
        If True, select sequence with fewest RIP mutations (default: True).
    getMaxGC : bool, optional
        If True, select sequence with highest GC content regardless of RIP counts (default: False).

    Returns
    -------
    int
        Row index of the best reference sequence.
    """
    logging.debug('Selecting reference sequence for filling remaining positions...')

    # Ignore RIP sorting if getMaxGC is set
    if getMaxGC:
        getMinRIP = False

    # Case 1: Use RIP counter and select sequence with fewest mutations
    if RIPcounter and getMinRIP:
        # Sort ascending for RIP count then descending
        # for GC content within duplicate RIP values
        refIdx = sorted(
            RIPcounter.values(), key=lambda x: (x.RIPcount + x.revRIPcount, -x.GC)
        )[0].idx
        logging.info(
            f'Selecting reference sequence with fewest RIP mutations: {refIdx}: {align[refIdx].id}'
        )

    # Case 2: Use RIP counter but select based on GC content
    elif RIPcounter:
        # Select row with highest GC content
        refIdx = sorted(RIPcounter.values(), key=lambda x: (-x.GC))[0].idx
        logging.info(
            f'Selecting reference sequence with highest GC content: {refIdx}: {align[refIdx].id}'
        )

    # Case 3: No RIP counter provided, select based on GC content
    else:
        # Calculate GC content for all sequences in the alignment
        GClist = []
        for x in range(align.__len__()):
            GClist.append((x, gc_fraction(align[x].seq) * 100))

        # Select sequence with highest GC content
        refIdx = sorted(GClist, key=lambda x: (-x[1]))[0][0]
        logging.info(
            f'No RIP data available, selecting reference sequence with highest GC content: {refIdx}: {align[refIdx].id}'
        )
    return refIdx


def fillRemainder(
    align: 'AlignIO.MultipleSeqAlignment',
    fromSeqID: int,
    tracker: Dict[int, NamedTuple],
) -> Dict[int, NamedTuple]:
    """
    Fill all remaining unset positions in the consensus sequence from a reference sequence.

    This function takes bases from a specified reference sequence to fill any positions
    in the consensus tracker that haven't been assigned a value yet.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to draw bases from.
    fromSeqID : int
        Row index of the reference sequence to use for filling.
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.

    Returns
    -------
    Dict[int, NamedTuple]
        Updated tracker dictionary with all positions filled.
    """
    # Log which sequence is being used as reference
    logging.info(
        'Filling uncorrected positions from: Row index %s: %s'
        % (str(fromSeqID), str(align[fromSeqID].id))
    )

    # Create a deep copy to avoid modifying the original tracker
    tracker = deepcopy(tracker)

    # Go through each position in the alignment
    for x in range(align.get_alignment_length()):
        # Get the base from the reference sequence at this position
        newBase = align[fromSeqID].seq[x]

        # Update the tracker (force=False means only positions with None will be updated)
        tracker = updateTracker(x, newBase, tracker, force=False)

    return tracker


def getDERIP(
    tracker: Dict[int, NamedTuple], ID: str = 'deRIPseq', deGAP: bool = True
) -> SeqRecord:
    """
    Convert consensus tracker to a SeqRecord object.

    This function converts the tracker dictionary containing the deRIPed consensus
    sequence into a Biopython SeqRecord object, optionally removing gaps.

    Parameters
    ----------
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    ID : str, optional
        Identifier for the output sequence (default: 'deRIPseq').
    deGAP : bool, optional
        If True, remove all gap characters ('-') from the output sequence (default: True).

    Returns
    -------
    Bio.SeqRecord.SeqRecord
        SeqRecord object containing the deRIPed consensus sequence.
    """
    logging.debug('Generating deRIPed sequence...')

    # Check that all positions have been filled
    if None in [x.base for x in tracker.values()]:
        raise ValueError('Not all positions have been filled in the tracker!')

    # Join all bases in the tracker, ordering by column index
    deRIPstr = ''.join([y.base for y in sorted(tracker.values(), key=lambda x: (x[0]))])

    # Remove gap characters if requested
    if deGAP:
        deRIPstr = deRIPstr.replace('-', '')

    # Create a SeqRecord object from the string
    deRIPseq = SeqRecord(
        Seq(deRIPstr),
        id=ID,
        name=ID,
        description='Hypothetical ancestral sequence produced by deRIP2',
    )

    return deRIPseq


def writeDERIP(
    tracker: Dict[int, NamedTuple], outPathFile: str, ID: str = 'deRIPseq'
) -> None:
    """
    Write the deRIPed consensus sequence to a file in FASTA format.

    This function creates a SeqRecord object from the consensus tracker,
    removes gaps, and writes it to the specified output file.

    Parameters
    ----------
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    outPathFile : str
        Path to the output FASTA file.
    ID : str, optional
        Identifier for the output sequence (default: 'deRIPseq').

    Returns
    -------
    None
        This function writes to a file but doesn't return a value.
    """
    # Generate the deRIPed sequence as a SeqRecord object (with gaps removed)
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=True)

    # Write the sequence to the specified file in FASTA format
    with open(outPathFile, 'w') as f:
        SeqIO.write(deRIPseq, f, 'fasta')


def writeDERIP2stdout(tracker: Dict[int, NamedTuple], ID: str = 'deRIPseq') -> None:
    """
    Write the deRIPed consensus sequence to standard output in FASTA format.

    This function creates a SeqRecord object from the consensus tracker,
    removes gaps, and prints it to standard output.

    Parameters
    ----------
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    ID : str, optional
        Identifier for the output sequence (default: 'deRIPseq').

    Returns
    -------
    None
        This function prints to stdout but doesn't return a value.
    """
    # Generate the deRIPed sequence as a SeqRecord object (with gaps removed)
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=True)

    # Create a string buffer to hold the FASTA-formatted sequence
    output = StringIO()

    # Write the sequence to the string buffer in FASTA format
    SeqIO.write(deRIPseq, output, 'fasta')

    # Get the formatted string from the buffer
    fasta_string = output.getvalue()

    # Close the buffer
    output.close()

    # Print the FASTA-formatted sequence to standard output
    print(fasta_string)


def writeAlign(
    tracker: Dict[int, NamedTuple],
    align: 'AlignIO.MultipleSeqAlignment',
    outPathAln: str,
    ID: str = 'deRIPseq',
    outAlnFormat: str = 'fasta',
    noappend: bool = False,
) -> None:
    """
    Write all sequences including the deRIPed consensus to an alignment file.

    This function creates a SeqRecord object from the consensus tracker (keeping gaps),
    optionally appends it to the alignment, and writes the resulting alignment to a file.

    Parameters
    ----------
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    align : Bio.Align.MultipleSeqAlignment
        The sequence alignment to write (possibly with deRIPed sequence added)
        Note: If noappend=False, this object will be modified.
    outPathAln : str
        Path to the output alignment file.
    ID : str, optional
        Identifier for the deRIPed sequence (default: 'deRIPseq').
    outAlnFormat : str, optional
        Format for the output alignment file (default: 'fasta').
    noappend : bool, optional
        If True, don't append the deRIPed sequence to the alignment (default: False).

    Returns
    -------
    None
        This function writes to a file but doesn't return a value.
    """
    # Generate the deRIPed sequence as a SeqRecord object (preserving gaps)
    logging.debug('Generating deRIPed sequence...')
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=False)

    # Create a working copy if we're going to append to it
    output_align = align
    if not noappend:
        output_align = deepcopy(align)
        output_align.append(deRIPseq)

    # Write the alignment to the specified file in the requested format
    with open(outPathAln, 'w') as f:
        AlignIO.write(output_align, f, outAlnFormat)
