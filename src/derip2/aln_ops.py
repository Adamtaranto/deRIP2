from collections import Counter
from collections import namedtuple
from copy import deepcopy
from io import StringIO
from operator import itemgetter
import logging
import sys

from Bio import AlignIO
from Bio import SeqIO
from Bio.Align import AlignInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction

from derip2.utils import isfile


def checkUniqueID(align):
    rowIDs = [list(align)[x].id for x in range(align.__len__())]
    IDcounts = Counter(rowIDs)
    duplicates = [k for k, v in IDcounts.items() if v > 1]
    if duplicates:
        logging.warning("Sequence IDs not unique. Quiting.")
        logging.info(f"Non-unique IDs: {duplicates}")
        sys.exit(1)
    else:
        pass


def checkLen(align):
    if align.__len__() < 2:
        logging.warning("Alignment contains < 2 sequences. Quiting.")
        sys.exit(1)
    else:
        pass


def loadAlign(file, alnFormat="fasta"):
    """
    Import alignment Check at least 2 rows in alignment
    and all names are unique.
    """
    # Check input file exists
    path = isfile(file)
    # Load file
    logging.info("Loading alignment from file: %s" % file)
    align = AlignIO.read(path, alnFormat)
    # Check alignment has at least 2 rows
    checkLen(align)
    # Check that all sequences have unique IDs
    checkUniqueID(align)
    # Return alignment object
    return align


def alignSummary(align):
    logging.info(
        "Alignment has %s rows and %s columns."
        % (str(align.__len__()), str(align.get_alignment_length()))
    )
    logging.info("Row index:\tSequence ID")
    for x in range(align.__len__()):
        logging.info("%s:\t%s" % (str(x), str(align[x].id)))
    pass


def checkrow(align, idx=None):
    if idx not in range(align.__len__()):
        logging.warning("Row index %s is outside range. Quitting." % str(idx))
        sys.exit(1)
    else:
        pass


def initTracker(align):
    """
    Initialise object to compose final deRIP'd sequence.
    List of tuples (colIdx,base). Base default to None.
    """
    tracker = dict()
    colItem = namedtuple("colPosition", ["idx", "base"])
    for x in range(align.get_alignment_length()):
        tracker[x] = colItem(idx=x, base=None)
    return tracker


def initRIPCounter(align):
    """
    For each row create dict key for seq name,
    assign named tuple (revRIPcount,RIPcount).
    """
    RIPcounts = dict()
    rowItem = namedtuple(
        "RIPtracker", ["idx", "SeqID", "revRIPcount", "RIPcount", "nonRIPcount", "GC"]
    )
    for x in range(align.__len__()):
        RIPcounts[x] = rowItem(
            idx=x,
            SeqID=align[x].id,
            revRIPcount=0,
            RIPcount=0,
            nonRIPcount=0,
            GC=gc_fraction(align[x].seq) * 100,
        )
    return RIPcounts


def updateTracker(idx, newChar, tracker, force=False):
    """
    Set final sequence value by column index if 'None'.
    Optionally force overwrite of previously updated base.
    """
    if tracker[idx].base and force:
        # Note: Add alert when overwriting a previously set base
        tracker[idx] = tracker[idx]._replace(base=newChar)
    elif not tracker[idx].base:
        tracker[idx] = tracker[idx]._replace(base=newChar)
    return tracker


def updateRIPCount(idx, RIPtracker, addRev=0, addFwd=0, addNonRIP=0):
    """
    Add observed RIP events to tracker by row.
    """
    TallyRev = RIPtracker[idx].revRIPcount + addRev
    TallyFwd = RIPtracker[idx].RIPcount + addFwd
    TallyNonRIP = RIPtracker[idx].nonRIPcount + addNonRIP
    RIPtracker[idx] = RIPtracker[idx]._replace(
        revRIPcount=TallyRev, RIPcount=TallyFwd, nonRIPcount=TallyNonRIP
    )
    return RIPtracker


def fillConserved(align, tracker, maxGaps=0.7):
    """
    Update positions in tracker object which are invariant
    OR excessively gapped.
    """
    tracker = deepcopy(tracker)
    # For each column in alignment
    for idx in range(align.get_alignment_length()):
        # Get frequencies for DNA bases + gaps
        colProps = AlignInfo.SummaryInfo(align)._get_letter_freqs(
            idx, align, ["A", "T", "G", "C", "-"], []
        )
        # If column is invariant, tracker inherits state
        for base in [k for k, v in colProps.items() if v == 1]:
            tracker = updateTracker(idx, base, tracker, force=False)
        # If non-gap rows are invariant AND proportion of gaps is < threshold set base
        for base in [k for k, v in colProps.items() if v + colProps["-"] == 1]:
            # Exclude '-' in case column is 50:50 somebase:gap
            # Check that proportion of gaps < threshold
            if base != "-" and colProps["-"] < maxGaps:
                tracker = updateTracker(idx, base, tracker, force=False)
        # If column contains more gaps than threshold,
        # force gap regardless of identity of remaining rows
        if itemgetter("-")(colProps) >= maxGaps:
            # If value not already set update to "-"
            tracker = updateTracker(idx, "-", tracker, force=False)
    # baseKeys = ["A","T","G","C"]
    # baseProp = sum(itemgetter(*baseKeys)(colProps))
    # gapProp = itemgetter("-")(colProps)
    return tracker


def nextBase(align, colID, motif):
    """
    For colIdx, and dinucleotide motif XY return list of
    rowIdx values where col=X and is followed by a Y in the
    next non-gap column.
    """
    # Find all rows where colID base matches first base of motif
    # Note: Column IDs are indexed from zero
    rowsX = find(align[:, colID], motif[0])
    # Init output list
    rowsXY = list()
    # For each row where starting col matches first base of motif
    for rowID in rowsX:
        # Loop through all positions to the right of starting col
        # From position to immediate right of X to end of seq
        for base in align[rowID].seq[colID + 1 :]:
            # For first non-gap position encountered
            if base != "-":
                # Check if base matches motif position two
                if base == motif[1]:
                    # If it is a match log row ID
                    rowsXY.append(rowID)
                # If first non-gap position is not a match end loop
                break
            # Else if position is a gap continue on to the next base
    return rowsXY


def lastBase(align, colID, motif):
    """
    For colIdx, and dinucleotide motif XY return list of
    rowIdx values where col=Y and is preceeded by an X in
    the previous non-gap column.
    """
    rowsY = find(align[:, colID], motif[1])
    rowsXY = list()
    for rowID in rowsY:
        # From position to immediate left of Y to begining of seq, reversed
        for base in align[rowID].seq[colID - 1 :: -1]:
            if base != "-":
                if base == motif[0]:
                    rowsXY.append(rowID)
                break
    return rowsXY


def find(lst, a):
    """
    Return list of indices for positions in list which
    contain a character in set 'a'.
    """
    return [i for i, x in enumerate(lst) if x in set(a)]


def hasBoth(lst, a, b):
    """
    Return "True" if list contains at least one instance
    of characters 'a' and 'b'.
    """
    hasA = find(lst, a)
    hasB = find(lst, b)
    if hasA and hasB:
        return True
    else:
        return False


def replaceBase(align, targetCol, targetRows, newbase):
    """
    Given an alignment object extract seqRecord objects from
    a list of target rows and replace bases in a target column
    with a specified new base. Replace rows and return updated
    alignment object.
    """
    for row in targetRows:
        seqList = list(align[row].seq)
        seqList[targetCol] = newbase
        # Replace seq record in alignment.
        # align[row].seq = Seq(''.join(seqList), Gapped(IUPAC.ambiguous_dna))
        align[row].seq = Seq("".join(seqList))
    return align


def correctRIP(
    align,
    tracker,
    RIPcounts,
    maxSNPnoise=0.5,
    minRIPlike=0.1,
    reaminate=True,
    mask=False,
):
    """
    Scan alignment for RIP-like dinucleotide shifts log RIP events by row in
    'RIPcounts', if position is unset in deRIP tracker (not logged as
    excessivley gapped) update with corrected base, optionally correct tracker
    for Cytosine deamination events in non-RIP contexts.
    Return updated deRIP tracker and RIPcounts objects.
    RIP signatures as observed in the + sense strand, with RIP targeting CpA
    motifs on either the +/- strand
    Target strand:    ++  --
    WT:            5' CA--TG 3'
    RIP:           5' TA--TA 3'
    Cons:             YA--TR
    """
    tracker = deepcopy(tracker)
    RIPcounts = deepcopy(RIPcounts)
    maskedAlign = deepcopy(align)
    # For each column in alignment
    for colIdx in range(align.get_alignment_length()):
        modC = False
        modG = False
        # Count total number of bases
        baseCount = len(find(align[:, colIdx], ["A", "T", "G", "C"]))
        # If total bases not zero
        if baseCount:
            # Sum C/T counts
            CTinCol = find(align[:, colIdx], ["C", "T"])
            # Sum G/A counts
            GAinCol = find(align[:, colIdx], ["G", "A"])
            # Get proportion C/T
            CTprop = len(CTinCol) / baseCount
            # Get proportion G/A
            GAprop = len(GAinCol) / baseCount
            # If proportion of C+T non-gap positions is > miscSNP threshold, AND bases are majority 'CT', AND both 'C' and 'T' are present
            if (
                CTprop >= maxSNPnoise
                and CTprop > GAprop
                and hasBoth(align[:, colIdx], "C", "T")
            ):
                # Get list of rowIdxs for which C/T in colIdx is followed by an 'A'
                TArows = nextBase(align, colIdx, motif="TA")
                CArows = nextBase(align, colIdx, motif="CA")
                # Get list of rowIdxs with value "T"
                TinCol = find(align[:, colIdx], ["T"])
                if CArows and TArows:
                    # Calc proportion of C/T positions in column followed by an 'A'
                    propRIPlike = (len(TArows) + len(CArows)) / len(CTinCol)
                    # For rows with 'TA' log RIP event
                    for rowTA in set(TArows):
                        RIPcounts = updateRIPCount(rowTA, RIPcounts, addFwd=1)
                    # For non-TA T's log non-RIP deamination event
                    for TnonRIP in set([x for x in TinCol if x not in TArows]):
                        RIPcounts = updateRIPCount(TnonRIP, RIPcounts, addNonRIP=1)
                    # If critical number of deamination events were in RIP context, update deRIP tracker
                    if propRIPlike >= minRIPlike:
                        tracker = updateTracker(colIdx, "C", tracker, force=False)
                        modC = True
                    # Else if in reaminate mode update deRIP tracker
                    elif reaminate:
                        tracker = updateTracker(colIdx, "C", tracker, force=False)
                        modC = True
                else:
                    # If C and T in col but not at least one CA and TA
                    # If in reaminate mode update deRIP tracker to C
                    if reaminate:
                        tracker = updateTracker(colIdx, "C", tracker, force=False)
                        modC = True
                    # Log all T's as non-RIP deamination events
                    for TnonRIP in TinCol:
                        RIPcounts = updateRIPCount(TnonRIP, RIPcounts, addNonRIP=1)
            # If proportion of G+A non-gap positions is > miscSNP threshold, AND bases are majority 'CT', AND both 'G' and 'A' are present
            elif (
                GAprop >= maxSNPnoise
                and GAprop > CTprop
                and hasBoth(align[:, colIdx], "G", "A")
            ):
                # Get list of rowIdxs for which G/A in colIdx is preceeded by a 'T'
                TGrows = lastBase(align, colIdx, motif="TG")
                TArows = lastBase(align, colIdx, motif="TA")
                # Get list of rowIdxs with value "A"
                AinCol = find(align[:, colIdx], ["A"])
                if TGrows and TArows:
                    # Calc proportion of G/A positions in column preceeded by a 'T'
                    propRIPlike = (len(TGrows) + len(TArows)) / len(GAinCol)
                    # For rows with 'TA' log revRIP event
                    for rowTA in set(TArows):
                        RIPcounts = updateRIPCount(rowTA, RIPcounts, addRev=1)
                    # For non-TA A's log non-RIP deamination event
                    for AnonRIP in set([x for x in AinCol if x not in TArows]):
                        RIPcounts = updateRIPCount(AnonRIP, RIPcounts, addNonRIP=1)
                    # If critical number of deamination events were in RIP context, update deRIP tracker
                    if propRIPlike >= minRIPlike:
                        tracker = updateTracker(colIdx, "G", tracker, force=False)
                        modG = True
                    # Else if in reaminate mode update deRIP tracker
                    elif reaminate:
                        tracker = updateTracker(colIdx, "G", tracker, force=False)
                        modG = True
                else:
                    # If G and A in col but not at least one TG and TA
                    # If in reaminate mode update deRIP tracker to G
                    if reaminate:
                        tracker = updateTracker(colIdx, "G", tracker, force=False)
                        modG = True
                    # Log all A's as non-RIP deamination events
                    for AnonRIP in AinCol:
                        RIPcounts = updateRIPCount(AnonRIP, RIPcounts, addNonRIP=1)
            # Mask T -> C corrections made in col
            if modC:
                if reaminate:
                    # Correct all rows with C or T in col
                    targetRows = CTinCol
                else:
                    # Row index list for CpA or TpA in col
                    targetRows = TArows + CArows
                # Replace target C/T positions with Y in colIdx
                maskedAlign = replaceBase(maskedAlign, colIdx, targetRows, "Y")
            # Mask A -> G corrections made in col
            if modG:
                if reaminate:
                    # Correct all rows with G or A in col
                    targetRows = GAinCol
                else:
                    # Row index list TpG or TpA in col
                    targetRows = TArows + TGrows
                # Replace target G/A positions with R in colIdx
                maskedAlign = replaceBase(maskedAlign, colIdx, targetRows, "R")
    return (tracker, RIPcounts, maskedAlign)


def summarizeRIP(RIPcounts):
    """
    Print summary of RIP counts and GC content calculated
    for each sequence in alignment.
    """
    logging.info("Summarizing RIP")
    print("Index:\tID\tRIP\tNon-RIP-deamination\tGC", file=sys.stderr)
    for x in range(len(RIPcounts)):
        print(
            "%s:\t%s\t%s\t%s\t%s"
            % (
                str(RIPcounts[x].idx),
                str(RIPcounts[x].SeqID),
                str(RIPcounts[x].revRIPcount + RIPcounts[x].RIPcount),
                str(RIPcounts[x].nonRIPcount),
                str(round(RIPcounts[x].GC, 2)),
            ),
            file=sys.stderr,
        )
    pass


def setRefSeq(align, RIPcounter=None, getMinRIP=True, getMaxGC=False):
    """
    Get row index of sequence with fewest RIP observations
    or highest GC if no RIP data.
    """
    # Ignore RIP sorting if getMaxGC is set
    if getMaxGC:
        getMinRIP = False
    if RIPcounter and getMinRIP:
        # Sort ascending for RIP count then descending
        # for GC content within duplicate RIP values
        refIdx = sorted(
            RIPcounter.values(), key=lambda x: (x.RIPcount + x.revRIPcount, -x.GC)
        )[0].idx
    elif RIPcounter:
        # Select row with highest GC contect
        refIdx = sorted(RIPcounter.values(), key=lambda x: (-x.GC))[0].idx
    else:
        # If no counter object get GC values from alignment
        GClist = list()
        for x in range(align.__len__()):
            GClist.append((x, gc_fraction(align[x].seq) * 100))
        refIdx = sorted(GClist, key=lambda x: (-x[1]))[0][0]
    return refIdx


def fillRemainder(align, fromSeqID, tracker):
    """
    Fill all remaining positions from least RIP effected row.
    """
    logging.info(
        "Filling uncorrected positions from: Row index %s: %s"
        % (str(fromSeqID), str(align[fromSeqID].id))
    )
    tracker = deepcopy(tracker)
    for x in range(align.get_alignment_length()):
        newBase = align[fromSeqID].seq[x]
        tracker = updateTracker(x, newBase, tracker, force=False)
    return tracker


def getDERIP(tracker, ID="deRIPseq", deGAP=True):
    """
    Write tracker object to sequence string.
    Requires that all base values are strings.
    """
    deRIPstr = "".join([y.base for y in sorted(tracker.values(), key=lambda x: (x[0]))])
    if deGAP:
        deRIPstr = deRIPstr.replace("-", "")
    # deRIPseq = SeqRecord(Seq(deRIPstr, Gapped(IUPAC.unambiguous_dna)),
    deRIPseq = SeqRecord(
        Seq(deRIPstr),
        id=ID,
        name=ID,
        description="Hypothetical ancestral sequence produced by deRIP2",
    )
    return deRIPseq


def writeDERIP(tracker, outPathFile, ID="deRIPseq"):
    """
    Call getDERIP, scrub gaps and Null positions.
    """
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=True)
    with open(outPathFile, "w") as f:
        SeqIO.write(deRIPseq, f, "fasta")


def writeDERIP2stdout(tracker, ID="deRIPseq"):
    """
    Call getDERIP, scrub gaps and Null positions.
    """
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=True)
    output = StringIO()
    SeqIO.write(deRIPseq, output, "fasta")
    fasta_string = output.getvalue()
    output.close()
    print(fasta_string)


def writeAlign(
    tracker, align, outPathAln, ID="deRIPseq", outAlnFormat="fasta", noappend=False
):
    """
    Assemble deRIPed sequence, append all seqs in
    ascending order of RIP events logged.
    """
    deRIPseq = getDERIP(tracker, ID=ID, deGAP=False)
    if not noappend:
        align.append(deRIPseq)
    with open(outPathAln, "w") as f:
        AlignIO.write(align, f, outAlnFormat)
