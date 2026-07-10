"""
Alignment operations for deRIP2.

This module provides functions for manipulating and analyzing DNA sequence alignments,
with a focus on detecting and correcting RIP (Repeat-Induced Point mutation) mutations.
It includes utilities for loading alignments, tracking RIP-like mutations, building
consensus sequences, and outputting corrected sequences in various formats.
"""

from collections import Counter, namedtuple
from copy import deepcopy
from dataclasses import dataclass

# import defaultdict
from io import StringIO
import logging
import sys
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction
import numpy as np
from tqdm import tqdm

from derip2.utils.checks import isfile

logger = logging.getLogger(__name__)

RIPPosition = NamedTuple(
    'RIPPosition', [('colIdx', int), ('rowIdx', int), ('base', str), ('offset', int)]
)


def alignment_to_array(align: 'AlignIO.MultipleSeqAlignment') -> np.ndarray:
    """
    Decode a Biopython alignment into a 2D array of single-byte characters.

    Each row of the returned array corresponds to a sequence and each column to
    an alignment position. Characters are preserved exactly (no case folding or
    substitution) so that downstream queries match the original per-column
    string comparisons.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The alignment to convert.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, n_columns)`` with dtype ``'S1'``.
    """
    n_rows = len(align)
    n_cols = align.get_alignment_length()
    arr = np.empty((n_rows, n_cols), dtype='S1')
    for i in range(n_rows):
        arr[i] = np.frombuffer(str(align[i].seq).encode('ascii'), dtype='S1')
    return arr


def _array_to_alignment(
    arr: np.ndarray, template: 'AlignIO.MultipleSeqAlignment'
) -> 'AlignIO.MultipleSeqAlignment':
    """
    Rebuild a MultipleSeqAlignment from a byte array, reusing template metadata.

    Parameters
    ----------
    arr : numpy.ndarray
        Array of shape ``(n_sequences, n_columns)`` with dtype ``'S1'``.
    template : Bio.Align.MultipleSeqAlignment
        Alignment providing the sequence ``id``/``name``/``description`` for
        each row (must have the same number of rows as ``arr``).

    Returns
    -------
    Bio.Align.MultipleSeqAlignment
        Alignment carrying the characters from ``arr``.
    """
    records = []
    for i in range(arr.shape[0]):
        seq_str = arr[i].tobytes().decode('ascii')
        records.append(
            SeqRecord(
                Seq(seq_str),
                id=template[i].id,
                name=template[i].name,
                description=template[i].description,
            )
        )
    return MultipleSeqAlignment(records)


def _nongap_neighbors(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the next and previous non-gap column index for every cell.

    For each cell ``(row, col)`` this returns the column index of the closest
    non-gap character strictly to the right (``next_idx``) and strictly to the
    left (``prev_idx``); ``-1`` indicates that no non-gap base exists in that
    direction. This vectorises the per-row gap-skipping that ``nextBase`` and
    ``lastBase`` performed by scanning the sequence.

    Parameters
    ----------
    arr : numpy.ndarray
        Byte array of shape ``(n_sequences, n_columns)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(next_idx, prev_idx)``, each of shape ``(n_sequences, n_columns)``.
    """
    n_rows, n_cols = arr.shape
    nongap = arr != b'-'

    next_idx = np.full((n_rows, n_cols), -1, dtype=np.int64)
    nxt = np.full(n_rows, -1, dtype=np.int64)
    for j in range(n_cols - 1, -1, -1):
        next_idx[:, j] = nxt
        nxt = np.where(nongap[:, j], j, nxt)

    prev_idx = np.full((n_rows, n_cols), -1, dtype=np.int64)
    prv = np.full(n_rows, -1, dtype=np.int64)
    for j in range(n_cols):
        prev_idx[:, j] = prv
        prv = np.where(nongap[:, j], j, prv)

    return next_idx, prev_idx


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
        logger.error('Sequence IDs not unique. Quiting.')
        logger.info(f'Non-unique IDs: {duplicates}')
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
        logger.error('Alignment contains < 2 sequences. Quiting.')
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
    logger.info(
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
    logger.debug('Generating alignment summary...')

    # Log the dimensions of the alignment
    logger.info(
        f'Alignment has {align.__len__()} rows and {align.get_alignment_length()} columns.'
    )

    # Create a header for the row index to sequence ID mapping table
    aln_records_msg = []
    aln_records_msg.append('Row\tID')

    # Log each sequence's index and ID as a table row
    for x in range(align.__len__()):
        aln_records_msg.append(f'{x}:\t{align[x].id}')

    # Join all messages into a single string and log it
    logger.debug('\n'.join(aln_records_msg))


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
        logger.warning(f'Row index {idx} is outside range. Quitting.')
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
    logger.debug('Initializing consensus sequence tracker...')

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

    logger.debug('Initializing RIP mutation counter...')

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
        logger.info(
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
    max_gaps: float = 0.7,
) -> Dict[int, NamedTuple]:
    """
    Update tracker with bases from invariant or highly gapped alignment columns.

    This function examines each column in the alignment and updates the tracker
    with bases from positions that are:
    1. Completely invariant (all non-gap positions have the same base)
    2. Invariant except for gaps (all non-gap positions have the same base)
    3. Highly gapped columns (gap proportion exceeds max_gaps threshold)

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The input sequence alignment to analyze.
    tracker : Dict[int, NamedTuple]
        Dictionary tracking the consensus sequence state for each column.
    max_gaps : float, optional
        Maximum proportion of gaps allowed in a column before considering
        it a gap column in the consensus (default: 0.7).

    Returns
    -------
    Dict[int, NamedTuple]
        Updated tracker dictionary with bases filled in for conserved positions.
    """
    logger.debug('Filling conserved positions in the consensus sequence...')
    # Create deep copy of tracker to avoid modifying the original
    tracker = deepcopy(tracker)

    # Decode once and precompute per-column base/gap counts with vectorised
    # reductions instead of slicing + Counter for every column.
    arr = alignment_to_array(align)
    total = arr.shape[0]
    bases = ['A', 'T', 'G', 'C', '-']
    col_counts = {base: (arr == base.encode('ascii')).sum(axis=0) for base in bases}

    # Process each column in alignment
    for idx in range(arr.shape[1]):
        # Integer base/gap counts for this column (order: A, T, G, C, -)
        counts = {base: int(col_counts[base][idx]) for base in bases}
        gap = counts['-']
        gapProp = gap / total

        # Case 1: If column is completely invariant, use that base
        # (count == total means all positions have this base)
        for base, c in counts.items():
            if c == total:
                tracker = updateTracker(idx, base, tracker, force=False)

        # Case 2: If non-gap positions are invariant (base + gap = 100%)
        for base, c in counts.items():
            # Exclude gap character; only update if gap proportion below threshold
            if c + gap == total and base != '-' and gapProp < max_gaps:
                tracker = updateTracker(idx, base, tracker, force=False)

        # Case 3: If column has more gaps than threshold, use gap character
        if gapProp >= max_gaps:
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


@dataclass(frozen=True)
class ColumnClassification:
    """
    Per-cell and per-column classification of RIP context across an alignment.

    This is the single source of truth for "which cells are RIP substrate,
    product, or non-RIP deamination, and on which strand". Both the consensus
    correction (:func:`apply_classification`) and the strand-bias statistics
    consume it, so the two can never disagree.

    RIP deaminates the C of a CpA dinucleotide. Read on the forward strand a
    reverse-strand CpA appears as TpG, so RIP on either strand yields a forward
    strand TpA::

        Target strand:    ++  --
        Wild type:     5' CA--TG 3'
        RIP mutated:   5' TA--TA 3'

    Dinucleotides are defined per row over the nearest *non-gap* neighbour, so
    a ``C-A`` spanning a gap column is still a CpA substrate.

    Attributes
    ----------
    arr : numpy.ndarray
        ``(n_rows, n_cols)`` byte array of the alignment, dtype ``'S1'``.
    next_idx, prev_idx : numpy.ndarray
        ``(n_rows, n_cols)`` int arrays giving the column index of the closest
        non-gap base to the right / left of each cell (``-1`` if none).
    ca, ta, tg, ta2 : numpy.ndarray
        ``(n_rows, n_cols)`` boolean masks of the four dinucleotide contexts:
        ``ca`` = C followed by A (forward substrate), ``ta`` = T followed by A
        (forward product candidate), ``tg`` = G preceded by T (reverse
        substrate), ``ta2`` = A preceded by T (reverse product candidate).
    ct_ok, ga_ok : numpy.ndarray
        ``(n_cols,)`` boolean. Column has enough C/T (or G/A) content to be
        assessed, i.e. proportion of non-gap bases >= ``max_snp_noise``.
    fwd_block, rev_block : numpy.ndarray
        ``(n_cols,)`` boolean. Column is a candidate for forward (reverse)
        correction: gate passed, strand is the majority, and both the substrate
        and product bases occur somewhere in the column.
    fwd_col, rev_col : numpy.ndarray
        ``(n_cols,)`` boolean. Column shows *both* an unmutated substrate
        dinucleotide and a product dinucleotide, so a product observed here can
        be attributed to RIP. These are the "RIP columns" used by the
        strand-bias statistics.
    modC, modG : numpy.ndarray
        ``(n_cols,)`` boolean. Column's consensus base is corrected to the
        ancestral C (G).
    base_counts : numpy.ndarray
        ``(n_cols, 5)`` int64 counts of ``A, C, G, T, -`` per column.
    reaminate : bool
        Whether non-RIP-context deaminations are also corrected.

    Notes
    -----
    ``fwd_col`` requires at least one surviving ``CA`` somewhere in the column.
    A column in which *every* row has been converted to ``TA`` therefore cannot
    be recognised as a RIP column: with no ancestral C left in any sequence,
    the alignment carries no evidence that the column was ever CpA. RIP is only
    visible where at least one sibling sequence escaped it.
    """

    arr: np.ndarray
    next_idx: np.ndarray
    prev_idx: np.ndarray
    ca: np.ndarray
    ta: np.ndarray
    tg: np.ndarray
    ta2: np.ndarray
    ct_ok: np.ndarray
    ga_ok: np.ndarray
    fwd_block: np.ndarray
    rev_block: np.ndarray
    fwd_col: np.ndarray
    rev_col: np.ndarray
    modC: np.ndarray
    modG: np.ndarray
    base_counts: np.ndarray
    reaminate: bool

    # -- per-column base counts -------------------------------------------------
    @property
    def nA(self) -> np.ndarray:
        """
        Per-column count of A bases.

        Returns
        -------
        numpy.ndarray
            ``(n_cols,)`` int array.
        """
        return self.base_counts[:, 0]

    @property
    def nC(self) -> np.ndarray:
        """
        Per-column count of C bases.

        Returns
        -------
        numpy.ndarray
            ``(n_cols,)`` int array.
        """
        return self.base_counts[:, 1]

    @property
    def nG(self) -> np.ndarray:
        """
        Per-column count of G bases.

        Returns
        -------
        numpy.ndarray
            ``(n_cols,)`` int array.
        """
        return self.base_counts[:, 2]

    @property
    def nT(self) -> np.ndarray:
        """
        Per-column count of T bases.

        Returns
        -------
        numpy.ndarray
            ``(n_cols,)`` int array.
        """
        return self.base_counts[:, 3]

    @property
    def n_gap(self) -> np.ndarray:
        """
        Per-column count of gap characters.

        Returns
        -------
        numpy.ndarray
            ``(n_cols,)`` int array.
        """
        return self.base_counts[:, 4]

    @property
    def base_count(self) -> np.ndarray:
        """
        Per-column count of unambiguous ACGT bases.

        Returns
        -------
        numpy.ndarray
            ``(n_cols,)`` int array. IUPAC ambiguity codes are excluded.
        """
        return self.base_counts[:, :4].sum(axis=1)

    # -- derived cell masks -----------------------------------------------------
    @property
    def sub_fwd(self) -> np.ndarray:
        """
        Forward RIP substrate cells: C in CpA context, in assessable columns.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        return self.ca & self.ct_ok

    @property
    def sub_rev(self) -> np.ndarray:
        """
        Reverse RIP substrate cells: G in TpG context, in assessable columns.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        return self.tg & self.ga_ok

    @property
    def prod_fwd(self) -> np.ndarray:
        """
        Forward RIP product cells: T in TpA context, in forward RIP columns.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        return self.ta & self.fwd_col

    @property
    def prod_rev(self) -> np.ndarray:
        """
        Reverse RIP product cells: A in TpA context, in reverse RIP columns.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        return self.ta2 & self.rev_col

    @property
    def nonrip_fwd(self) -> np.ndarray:
        """
        T cells in a forward candidate column that are not RIP products.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        return self.fwd_block & (self.arr == b'T') & ~(self.fwd_col & self.ta)

    @property
    def nonrip_rev(self) -> np.ndarray:
        """
        A cells in a reverse candidate column that are not RIP products.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        return self.rev_block & (self.arr == b'A') & ~(self.rev_col & self.ta2)

    @property
    def mask_Y(self) -> np.ndarray:
        """
        Mask of cells overwritten with the IUPAC code Y (C/T) in the masked alignment.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        targets = (self.arr == b'T') if self.reaminate else self.ta
        return self.modC & targets

    @property
    def mask_R(self) -> np.ndarray:
        """
        Mask of cells overwritten with the IUPAC code R (A/G) in the masked alignment.

        Returns
        -------
        numpy.ndarray
            ``(n_rows, n_cols)`` boolean mask.
        """
        targets = (self.arr == b'A') if self.reaminate else self.ta2
        return self.modG & targets

    # -- per-row tallies --------------------------------------------------------
    @property
    def add_fwd(self) -> np.ndarray:
        """
        Per-row count of forward-strand RIP events.

        Returns
        -------
        numpy.ndarray
            ``(n_rows,)`` int array.
        """
        return self.prod_fwd.sum(axis=1)

    @property
    def add_rev(self) -> np.ndarray:
        """
        Per-row count of reverse-strand RIP events.

        Returns
        -------
        numpy.ndarray
            ``(n_rows,)`` int array.
        """
        return self.prod_rev.sum(axis=1)

    @property
    def add_nonrip(self) -> np.ndarray:
        """
        Per-row count of non-RIP deamination events.

        Returns
        -------
        numpy.ndarray
            ``(n_rows,)`` int array.
        """
        return self.nonrip_fwd.sum(axis=1) + self.nonrip_rev.sum(axis=1)

    @property
    def corrected_positions(self) -> List[int]:
        """
        Column indices whose consensus base was corrected.

        Returns
        -------
        list of int
            Ascending column indices.
        """
        return np.where(self.modC | self.modG)[0].tolist()


def _default_block_size(
    n_rows: int, n_cols: int, budget_bytes: int = 64 * 1024 * 1024
) -> int:
    """
    Choose a column-block width that keeps transient allocations under a budget.

    The dominant transient cost per column is the pair of integer index planes
    used to gather neighbouring bases, plus a handful of byte-wide planes.

    Parameters
    ----------
    n_rows : int
        Number of sequences in the alignment.
    n_cols : int
        Number of alignment columns.
    budget_bytes : int, optional
        Approximate ceiling on transient allocation (default: 64 MiB).

    Returns
    -------
    int
        Block width in columns, at least 1 and at most ``n_cols``.
    """
    per_col = max(1, n_rows * 24)
    return max(1, min(n_cols, budget_bytes // per_col))


def classify_columns(
    arr: np.ndarray,
    next_idx: np.ndarray,
    prev_idx: np.ndarray,
    max_snp_noise: float = 0.5,
    min_rip_like: float = 0.1,
    reaminate: bool = False,
    block_size: Optional[int] = None,
    progress: bool = True,
) -> ColumnClassification:
    """
    Classify every cell and column of an alignment by RIP context.

    This is a vectorised reformulation of the per-column scan that
    :func:`correctRIP` used to perform, and reproduces its decisions exactly.
    Forward-strand RIP (C→T in CpA context) and reverse-strand RIP (G→A in TpG
    context) are detected independently.

    A column is assessed on the forward strand when its C+T bases make up at
    least ``max_snp_noise`` of the non-gap bases, and is a *correction*
    candidate only when C/T is the strict majority over G/A. Because every
    non-gap base falls in exactly one of the C/T and G/A pairs, the two
    proportions sum to one, so the strict inequality makes forward and reverse
    correction mutually exclusive. A column can never be corrected on both
    strands, and the Y/R masks can never collide.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(n_rows, n_cols)`` byte array of the alignment, dtype ``'S1'``, as
        produced by :func:`alignment_to_array`.
    next_idx, prev_idx : numpy.ndarray
        Non-gap neighbour indices from :func:`_nongap_neighbors`.
    max_snp_noise : float, optional
        Minimum proportion of a column's non-gap bases that must be C/T (or
        G/A) for that strand to be assessed (default: 0.5).
    min_rip_like : float, optional
        Minimum proportion of a column's C/T (or G/A) bases that must sit in
        RIP dinucleotide context before the column is corrected (default: 0.1).
    reaminate : bool, optional
        If True, correct C→T and G→A transitions outside RIP context too
        (default: False).
    block_size : int, optional
        Number of columns processed per block. Blocking bounds peak memory and
        is bit-identical to processing the whole array at once, because every
        reduction is within a single column and neighbour gathers index the
        full array. Defaults to a width chosen from a 64 MiB budget.
    progress : bool, optional
        Show a progress bar when more than one block is processed
        (default: True).

    Returns
    -------
    ColumnClassification
        Cell masks, column flags, and per-column base counts.

    Notes
    -----
    Cell classification (substrate / product / non-RIP) and the per-row tallies
    depend only on ``max_snp_noise``; ``min_rip_like`` and ``reaminate`` affect
    only whether a column's consensus base is corrected and masked.
    """
    n_rows, n_cols = arr.shape

    bA, bT, bG, bC = b'A', b'T', b'G', b'C'

    # Per-column base composition. Only unambiguous ACGT count toward baseCount,
    # so IUPAC ambiguity codes are excluded from the strand proportions.
    base_counts = np.empty((n_cols, 5), dtype=np.int64)
    for k, b in enumerate((bA, bC, bG, bT, b'-')):
        base_counts[:, k] = (arr == b).sum(axis=0)

    ca = np.zeros(arr.shape, dtype=bool)
    ta = np.zeros(arr.shape, dtype=bool)
    tg = np.zeros(arr.shape, dtype=bool)
    ta2 = np.zeros(arr.shape, dtype=bool)

    if n_cols:
        if block_size is None:
            block_size = _default_block_size(n_rows, n_cols)
        block_size = max(1, int(block_size))

        rows = np.arange(n_rows)[:, None]
        has_next = next_idx >= 0
        has_prev = prev_idx >= 0

        starts = range(0, n_cols, block_size)
        blocks = tqdm(
            starts,
            desc='Scanning for RIP mutations',
            unit='block',
            ncols=80,
            disable=not progress or n_cols <= block_size,
        )
        for c0 in blocks:
            sl = slice(c0, min(c0 + block_size, n_cols))
            blk = arr[:, sl]
            hn, hp = has_next[:, sl], has_prev[:, sl]

            # Gather the nearest non-gap neighbour of every cell in the block.
            # The stored indices are absolute and may point outside the block,
            # so both gathers read the full array.
            nb = arr[rows, np.where(hn, next_idx[:, sl], 0)]
            pb = arr[rows, np.where(hp, prev_idx[:, sl], 0)]

            ca[:, sl] = (blk == bC) & hn & (nb == bA)
            ta[:, sl] = (blk == bT) & hn & (nb == bA)
            tg[:, sl] = (blk == bG) & hp & (pb == bT)
            ta2[:, sl] = (blk == bA) & hp & (pb == bT)

    # Column-level strand proportions.
    base_count = base_counts[:, :4].sum(axis=1)
    has_bases = base_count > 0
    ct_count = base_counts[:, 1] + base_counts[:, 3]  # C + T
    ga_count = base_counts[:, 0] + base_counts[:, 2]  # A + G

    ct_prop = np.divide(ct_count, base_count, out=np.zeros(n_cols), where=has_bases)
    ga_prop = np.divide(ga_count, base_count, out=np.zeros(n_cols), where=has_bases)

    ct_ok = (ct_prop >= max_snp_noise) & has_bases
    ga_ok = (ga_prop >= max_snp_noise) & has_bases

    n_ca, n_ta = ca.sum(axis=0), ta.sum(axis=0)
    n_tg, n_ta2 = tg.sum(axis=0), ta2.sum(axis=0)

    # A correction candidate needs the gate, strict strand majority, and both
    # the ancestral and derived base present somewhere in the column.
    fwd_block = (
        ct_ok & (ct_prop > ga_prop) & (base_counts[:, 1] > 0) & (base_counts[:, 3] > 0)
    )
    rev_block = (
        ga_ok & (ga_prop > ct_prop) & (base_counts[:, 2] > 0) & (base_counts[:, 0] > 0)
    )

    # A RIP column additionally shows an unmutated substrate dinucleotide
    # alongside a product dinucleotide.
    fwd_col = fwd_block & (n_ca > 0) & (n_ta > 0)
    rev_col = rev_block & (n_tg > 0) & (n_ta2 > 0)

    prip_f = np.divide(n_ta + n_ca, ct_count, out=np.zeros(n_cols), where=ct_count > 0)
    prip_r = np.divide(n_ta2 + n_tg, ga_count, out=np.zeros(n_cols), where=ga_count > 0)

    # Correct when the column is RIP-like enough, or unconditionally under
    # reaminate. Columns with C/T variation but no RIP context are corrected
    # only under reaminate.
    modC = fwd_block & (
        (fwd_col & ((prip_f >= min_rip_like) | reaminate)) | (~fwd_col & reaminate)
    )
    modG = rev_block & (
        (rev_col & ((prip_r >= min_rip_like) | reaminate)) | (~rev_col & reaminate)
    )

    return ColumnClassification(
        arr=arr,
        next_idx=next_idx,
        prev_idx=prev_idx,
        ca=ca,
        ta=ta,
        tg=tg,
        ta2=ta2,
        ct_ok=ct_ok,
        ga_ok=ga_ok,
        fwd_block=fwd_block,
        rev_block=rev_block,
        fwd_col=fwd_col,
        rev_col=rev_col,
        modC=modC,
        modG=modG,
        base_counts=base_counts,
        reaminate=reaminate,
    )


def classify_alignment(
    align: 'AlignIO.MultipleSeqAlignment',
    max_snp_noise: float = 0.5,
    min_rip_like: float = 0.1,
    reaminate: bool = False,
    block_size: Optional[int] = None,
    progress: bool = True,
) -> ColumnClassification:
    """
    Convenience wrapper: decode an alignment and classify its RIP context.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The alignment to classify.
    max_snp_noise : float, optional
        See :func:`classify_columns` (default: 0.5).
    min_rip_like : float, optional
        See :func:`classify_columns` (default: 0.1).
    reaminate : bool, optional
        See :func:`classify_columns` (default: False).
    block_size : int, optional
        See :func:`classify_columns`.
    progress : bool, optional
        See :func:`classify_columns` (default: True).

    Returns
    -------
    ColumnClassification
        Classification of the alignment.
    """
    arr = alignment_to_array(align)
    next_idx, prev_idx = _nongap_neighbors(arr)
    return classify_columns(
        arr,
        next_idx,
        prev_idx,
        max_snp_noise=max_snp_noise,
        min_rip_like=min_rip_like,
        reaminate=reaminate,
        block_size=block_size,
        progress=progress,
    )


def _build_markupdict(cls: ColumnClassification) -> Dict[str, List[RIPPosition]]:
    """
    Convert a classification into the per-category position lists used for markup.

    Parameters
    ----------
    cls : ColumnClassification
        Classification produced by :func:`classify_columns`.

    Returns
    -------
    Dict[str, List[RIPPosition]]
        Keys ``'rip_product'``, ``'rip_substrate'``, ``'non_rip_deamination'``.
        Positions are ordered by column then row; ordering within a category is
        not otherwise significant.

    Notes
    -----
    No deduplication is required. Within each category the forward and reverse
    entries carry different bases (C/G for substrate, T/A for product and
    non-RIP), and a cell holds one base, so forward and reverse contributions
    are always disjoint.
    """
    n_rows = cls.arr.shape[0]

    def _cells(mask, partner_idx):
        """Cell coordinates in column-major order, with dinucleotide offsets."""
        # Transposing makes np.where emit indices ordered by column then row,
        # which is the order the markup lists are expected in.
        cols, rows = np.where(mask.T)
        if partner_idx is None:
            offs = np.zeros(cols.size, dtype=np.int64)
        else:
            offs = partner_idx[rows, cols] - cols
        return cols, rows, offs

    def _merge(fwd, rev, fwd_base, rev_base):
        f_cols, f_rows, f_offs = fwd
        r_cols, r_rows, r_offs = rev
        cols = np.concatenate((f_cols, r_cols))
        rows = np.concatenate((f_rows, r_rows))
        offs = np.concatenate((f_offs, r_offs))
        bases = [fwd_base] * f_cols.size + [rev_base] * r_cols.size

        # Stable sort on a single composite key: (col, row) is unique per cell.
        order = np.argsort(cols * n_rows + rows, kind='stable')
        return [
            RIPPosition(int(cols[i]), int(rows[i]), bases[i], int(offs[i]))
            for i in order.tolist()
        ]

    # Forward strand: dinucleotide partner lies to the right (positive offset).
    # Reverse strand: partner lies to the left (negative offset).
    return {
        'rip_substrate': _merge(
            _cells(cls.sub_fwd, cls.next_idx),
            _cells(cls.sub_rev, cls.prev_idx),
            'C',
            'G',
        ),
        'rip_product': _merge(
            _cells(cls.prod_fwd, cls.next_idx),
            _cells(cls.prod_rev, cls.prev_idx),
            'T',
            'A',
        ),
        'non_rip_deamination': _merge(
            _cells(cls.nonrip_fwd, None),
            _cells(cls.nonrip_rev, None),
            'T',
            'A',
        ),
    }


def apply_classification(
    align: 'AlignIO.MultipleSeqAlignment',
    tracker: Dict[int, NamedTuple],
    RIPcounts: Dict[int, NamedTuple],
    cls: ColumnClassification,
) -> Tuple[
    Dict[int, NamedTuple],
    Dict[int, NamedTuple],
    'AlignIO.MultipleSeqAlignment',
    List[int],
    Dict[str, List[RIPPosition]],
]:
    """
    Apply a column classification to the consensus tracker, counters and mask.

    Parameters
    ----------
    align : Bio.Align.MultipleSeqAlignment
        The alignment the classification was computed from; supplies record
        metadata for the rebuilt masked alignment.
    tracker : Dict[int, NamedTuple]
        Consensus tracker keyed by column index. Not mutated.
    RIPcounts : Dict[int, NamedTuple]
        Per-sequence RIP counters keyed by row index. Not mutated.
    cls : ColumnClassification
        Classification produced by :func:`classify_columns`.

    Returns
    -------
    Tuple
        ``(tracker, RIPcounts, maskedAlign, corrected_positions, markupdict)``.
    """
    tracker = deepcopy(tracker)
    RIPcounts = deepcopy(RIPcounts)

    # Masked output starts as the original characters; corrected cells are
    # overwritten in place with IUPAC codes.
    maskedArr = cls.arr.copy()
    maskedArr[cls.mask_Y] = b'Y'
    maskedArr[cls.mask_R] = b'R'

    for col in np.where(cls.modC)[0].tolist():
        tracker = updateTracker(col, 'C', tracker, force=False)
    for col in np.where(cls.modG)[0].tolist():
        tracker = updateTracker(col, 'G', tracker, force=False)

    add_fwd, add_rev, add_nonrip = cls.add_fwd, cls.add_rev, cls.add_nonrip
    nz_rows = np.where((add_fwd != 0) | (add_rev != 0) | (add_nonrip != 0))[0]
    for r in nz_rows.tolist():
        RIPcounts = updateRIPCount(
            r,
            RIPcounts,
            addRev=int(add_rev[r]),
            addFwd=int(add_fwd[r]),
            addNonRIP=int(add_nonrip[r]),
        )

    maskedAlign = _array_to_alignment(maskedArr, align)

    return (
        tracker,
        RIPcounts,
        maskedAlign,
        cls.corrected_positions,
        _build_markupdict(cls),
    )


def correctRIP(
    align: 'AlignIO.MultipleSeqAlignment',
    tracker: Dict[int, NamedTuple],
    RIPcounts: Dict[int, NamedTuple],
    max_snp_noise: float = 0.5,
    min_rip_like: float = 0.1,
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
    max_snp_noise : float, optional
        Minimum proportion of positions in a column that must be C/T or G/A to be
        considered for RIP correction (default: 0.5).
    min_rip_like : float, optional
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
    logger.debug('Correcting RIP-like mutations in the consensus sequence...')

    cls = classify_alignment(
        align,
        max_snp_noise=max_snp_noise,
        min_rip_like=min_rip_like,
        reaminate=reaminate,
    )
    return apply_classification(align, tracker, RIPcounts, cls)


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
    logger.debug('Summarizing RIP mutation counts...')

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
    logger.debug('Selecting reference sequence for filling remaining positions...')

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
        logger.info(
            f'Selecting reference sequence with fewest RIP mutations: {refIdx}: {align[refIdx].id}'
        )

    # Case 2: Use RIP counter but select based on GC content
    elif RIPcounter:
        # Select row with highest GC content
        refIdx = sorted(RIPcounter.values(), key=lambda x: -x.GC)[0].idx
        logger.info(
            f'Selecting reference sequence with highest GC content: {refIdx}: {align[refIdx].id}'
        )

    # Case 3: No RIP counter provided, select based on GC content
    else:
        # Calculate GC content for all sequences in the alignment
        GClist = []
        for x in range(align.__len__()):
            GClist.append((x, gc_fraction(align[x].seq) * 100))

        # Select sequence with highest GC content
        refIdx = sorted(GClist, key=lambda x: -x[1])[0][0]
        logger.info(
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
    logger.info(
        'Filling uncorrected positions from: Row index %s: %s'
        % (str(fromSeqID), str(align[fromSeqID].id))
    )

    # A shallow copy is sufficient to avoid modifying the caller's dict: the
    # values are immutable namedtuples and each update below rebinds a key to a
    # newly built tuple (via ._replace), never mutating the shared originals.
    # This avoids a costly deepcopy of one namedtuple per column.
    tracker = dict(tracker)

    # Materialise the reference sequence as a string once rather than indexing
    # the Biopython Seq one character at a time.
    refSeq = str(align[fromSeqID].seq)

    # Fill only the positions still unset, inlining the update to skip the
    # per-column updateTracker call overhead. Tracker keys are 0..cols-1 and
    # len(refSeq) == cols, so refSeq[x] is always valid.
    for x, item in tracker.items():
        if item.base is None:
            tracker[x] = item._replace(base=refSeq[x])

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
    logger.debug('Generating deRIPed sequence...')

    # Check that all positions have been filled
    if None in [x.base for x in tracker.values()]:
        raise ValueError('Not all positions have been filled in the tracker!')

    # Join all bases in the tracker, ordering by column index
    deRIPstr = ''.join([y.base for y in sorted(tracker.values(), key=lambda x: x[0])])

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
    logger.debug('Generating deRIPed sequence...')
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=False)

    # Create a working copy if we're going to append to it
    output_align = align
    if not noappend:
        output_align = deepcopy(align)
        output_align.append(deRIPseq)

    # Write the alignment to the specified file in the requested format
    with open(outPathAln, 'w') as f:
        AlignIO.write(output_align, f, outAlnFormat)
