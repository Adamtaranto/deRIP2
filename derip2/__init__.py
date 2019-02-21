#!/usr/bin/env python
#python 3
#requires biopython
#Contact, Adam Taranto, adam.taranto@anu.edu.au

##################################################################################################################
# Takes a multi-sequence DNA alignment and estimates a progenitor sequence by correcting for RIP-like mutations. #
# deRIP2 searches all available sequences for evidence of un-RIP'd precursor states at each aligned position,    #
# allowing for improved RIP-correction across large repeat families in which members are variably RIP'd.         #
##################################################################################################################

import sys
import os
from collections import Counter
from collections import namedtuple
from copy import deepcopy
from operator import itemgetter
from Bio import SeqIO
from Bio import AlignIO
from Bio.SeqUtils import GC
from Bio.Align import AlignInfo #,MultipleSeqAlignment
from Bio.Alphabet import IUPAC, Gapped #,generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

__version__ = "0.0.1"

class Error (Exception): pass

def dochecks(usrOutDir):
	"""Make outDir if does not exist else set to current dir."""
	if usrOutDir:
		absOutDir = os.path.abspath(usrOutDir)
		if not os.path.isdir(absOutDir):
			os.makedirs(absOutDir)
		outDir = usrOutDir
	else:
		outDir = os.getcwd() 
	return outDir

def checkUniqueID(align):
	rowIDs = [list(align)[x].id for x in range(align.__len__())]
	IDcounts = Counter(rowIDs)
	duplicates = [k for k, v in IDcounts.items() if v > 1]
	if duplicates:
		print("Sequence IDs not unique. Quiting.")
		print(duplicates)
		sys.exit(1)
	else:
		pass

def checkLen(align):
	if align.__len__() < 2:
		print("Alignment contains < 2 sequences. Quiting.")
		sys.exit(1)
	else:
		pass

def loadAlign(file,alnFormat="fasta"):
	"""Import alignment Check at least 2 rows in alignment and all names are unique."""
	align = AlignIO.read(file,alnFormat)
	checkLen(align)
	checkUniqueID(align)
	return align

def initTracker(align):
	"""Initialise object to compose final deRIP'd sequence. 
	List of tuples (colIdx,base). Base default to None."""
	tracker = dict()
	colItem = namedtuple("colPosition",['idx','base'])
	for x in range(align.get_alignment_length()):
		tracker[x] = colItem(idx=x,base=None)
	return tracker

def initRIPCounter(align):
	"""For each row create dict key for seq name, 
	assign named tuple (revRIPcount,RIPcount)."""
	RIPcounts = dict()
	rowItem = namedtuple("RIPtracker",['idx','SeqID','revRIPcount','RIPcount','GC'])
	for x in range(align.__len__()):
		RIPcounts[x] = rowItem(idx=x,SeqID=align[x].id,revRIPcount=0,RIPcount=0,GC=GC(align[x].seq))
	return RIPcounts

def updateTracker(idx,newChar,tracker,force=False):
	"""Set final sequence value by column index if 'None'. 
	Optionally force overwrite of previously updated base."""
	if tracker[idx].base and force:
		tracker[idx] = tracker[idx]._replace(base=newChar)
	elif not tracker[idx].base:
		tracker[idx] = tracker[idx]._replace(base=newChar)
	return tracker

def updateRIPCount(idx,RIPtracker,addRev=0,addFwd=0):
	"""Add observed RIP events to tracker by row."""
	TallyRev = RIPtracker[idx].revRIPcount + addRev
	TallyFwd = RIPtracker[idx].RIPcount + addFwd
	RIPtracker[idx] = RIPtracker[idx]._replace(revRIPcount=TallyRev,RIPcount=TallyFwd)
	return RIPtracker

def fillConserved(align,tracker,maxGaps=0.7):
	"""Update positions in tracker object which are invariant OR excessively gapped."""
	tracker = deepcopy(tracker)
	# For each column in alignment
	for idx in range(align.get_alignment_length()):
		# Get frequencies for DNA bases + gaps
		colProps = AlignInfo.SummaryInfo(align)._get_letter_freqs(idx,align,["A","T","G","C","-"],[])
		# If column is invariant, tracker inherits state
		for base in [k for k, v in colProps.items() if v == 1]:
			tracker = updateTracker(idx,base,tracker,force=False)
		# If non-gap rows are invariant AND proportion of gaps is < threshold set base
		for base in [k for k, v in colProps.items() if v+colProps['-'] == 1]:
			if base != '-' and colProps['-'] < maxGaps:
				tracker = updateTracker(idx,base,tracker,force=False)
		# If column contains more gaps than threshold, force gap regardless of identity of remaining rows
		if itemgetter("-")(colProps) >= maxGaps:
			# If value not already set update to "-"
			tracker = updateTracker(idx,"-",tracker,force=False)
	#baseKeys = ["A","T","G","C"]
	#baseProp = sum(itemgetter(*baseKeys)(colProps))
	#gapProp = itemgetter("-")(colProps)
	return tracker

def nextBase(align,colID,motif):
	"""For colIdx, and dinucleotide motif XY return list of rowIdx values where col=X 
	and is followed by a Y in the next non-gap column."""
	rowsX = find(align[:,colID],motif[0])
	rowsXY = list()
	for rowID in rowsX:
		for base in align[rowID].seq[colID+1:]: #From position to immediate right of X to end of seq
			if base != "-":
				if base == motif[1]:
					rowsXY.append(rowID)
				break
	return rowsXY

def lastBase(align,colID,motif):
	"""For colIdx, and dinucleotide motif XY return list of rowIdx values where col=Y 
	and is preceeded by an X in the previous non-gap column."""
	rowsY = find(align[:,colID],motif[1])
	rowsXY = list()
	for rowID in rowsY:
		for base in align[rowID].seq[colID-1::-1]: #From position to immediate left of Y to begining of seq, reversed
			if base != "-":
				if base == motif[0]:
					rowsXY.append(rowID)
				break
	return rowsXY

def find(lst, a):
	"""Return list of indicies for positions in list which contain a character in set 'a'."""
	return [i for i, x in enumerate(lst) if x in set(a)]

def hasBoth(lst,a,b):
	"""Return "True" if list contains at least one instance of characters 'a' and 'b'."""
	hasA = find(lst, a)
	hasB = find(lst, b)
	if hasA and hasB:
		return True
	else:
		return False

def correctRIP(align,tracker,RIPcounts,maxSNPnoise=0.5,minRIPlike=0.1,reaminate=True):
	"""
	Scan alignment for RIP-like dinucleotide shifts log RIP events by row in 'RIPcounts',
	if position is unset in deRIP tracker (not logged as excessivley gapped) update with corrected base,
	optionally correct tracker for Cytosine deamination events in non-RIP contexts.
	Return updated deRIP tracker and RIPcounts objects.
	RIP signatures as observed in the + sense strand, with RIP targeting CpA motifs on either the +/- strand
	Target strand:    ++  --
	WT:            5' CA--TG 3' 
	RIP:           5' TA--TA 3' 
	Cons:             YA--TR
	"""
	tracker = deepcopy(tracker)
	RIPcounts = deepcopy(RIPcounts)
	for colIdx in range(align.get_alignment_length()):
		baseCount = len(find(align[:,colIdx],["A","T","G","C"]))
		if baseCount:
			CTinCol = find(align[:,colIdx],["C","T"])
			GAinCol = find(align[:,colIdx],["G","A"])
			CTprop = len(CTinCol) / baseCount
			GAprop = len(GAinCol) / baseCount
			# If proportion of C+T non-gap positions is > miscSNP threshold, AND bases are majority 'CT', AND both 'C' and 'T' are present
			if CTprop >= maxSNPnoise and CTprop > GAprop and hasBoth(align[:,colIdx],"C","T"):
				# Get list of rowIdxs for which C/T in colIdx is followed by an 'A'
				TArows = nextBase(align,colIdx, motif='TA')
				CArows = nextBase(align,colIdx, motif='CA')
				# Calc proportion of C/T positions in column followed by an 'A'
				propRIPlike = len(TArows) + len(CArows) / len(CTinCol)
				# For rows with 'TA' log RIP event
				for rowTA in set(TArows):
					RIPcounts = updateRIPCount(rowTA, RIPcounts, addFwd=1)
				# If critical number of deamination events were in RIP context, update deRIP tracker	
				if propRIPlike >= minRIPlike:
					tracker = updateTracker(colIdx, 'C', tracker, force=False)
				# Else if in reaminate mode update deRIP tracker
				elif reaminate:
					tracker = updateTracker(colIdx, 'C', tracker, force=False)
			# If proportion of G+A non-gap positions is > miscSNP threshold AND both 'G' and 'A' are present
			elif GAprop >= maxSNPnoise and hasBoth(align[:,colIdx],"G","A"):
				# Get list of rowIdxs for which G/A in colIdx is preceeded by a 'T'
				TGrows = lastBase(align,colIdx, motif='TG')
				TArows = lastBase(align,colIdx, motif='TA')
				# Calc proportion of G/A positions in column preceeded by a 'T'
				propRIPlike = len(TGrows) + len(TArows) / len(GAinCol)
				# For rows with 'TA' log revRIP event
				for rowTA in set(TArows):
					RIPcounts = updateRIPCount(rowTA, RIPcounts, addRev=1)
				# If critical number of deamination events were in RIP context, update deRIP tracker
				if propRIPlike >= minRIPlike:
					tracker = updateTracker(colIdx, 'G', tracker, force=False)
				# Else if in reaminate mode update deRIP tracker
				elif reaminate:
					tracker = updateTracker(colIdx, 'G', tracker, force=False)
	return (tracker,RIPcounts)

def setRefSeq(align, RIPcounter=None, getMinRIP=True):
	"""Get row index of sequence with fewest RIP observations or highest GC if no RIP data."""
	if RIPcounter and getMinRIP:
		refIdx = sorted(RIPcounter.values(), key = lambda x: (x.RIPcount + x.revRIPcount, -x.GC))[0].idx
	elif RIPcounter:
		refIdx = sorted(RIPcounter.values(), key = lambda x: (-x.GC))[0].idx
	else:
		GClist = list()
		for x in range(align.__len__()):
			GClist.append((x,GC(align[x].seq)))
		refIdx = sorted(GClist, key = lambda x: (-x[1]))[0][0]
	return refIdx

def fillRemainder(align,fromSeqID,tracker):
	"""Fill all remaining positions from least RIP effected row."""
	tracker = deepcopy(tracker)
	for x in range(align.get_alignment_length()):
		newBase = align[fromSeqID].seq[x]
		tracker = updateTracker(x,newBase,tracker,force=False)
	return tracker

def getDERIP(tracker,ID="deRIPseq",deGAP=True):
	"""Write tracker object to sequence string.
	   Requires that all base values are strings."""
	deRIPstr = ''.join([y.base for y in sorted(tracker.values(), key = lambda x: (x[0]))])
	if deGAP:
		deRIPstr = deRIPstr.replace("-", "")
	deRIPseq = SeqRecord(Seq(deRIPstr, Gapped(IUPAC.unambiguous_dna)), id=ID, name=ID, description="Hypothetical ancestral sequence produced by deRIP2")
	return deRIPseq

def writeDERIP(tracker,outPathFile,ID="deRIPseq"):
	"""Call getDERIP, scrub gaps and Null positions."""
	deRIPseq = getDERIP(tracker,ID=ID,deGAP=True)
	with open(outPathFile, "w") as f:
		SeqIO.write(deRIPseq, f, "fasta")

def writeAlign(tracker,align,outPathAln,ID="deRIPseq",outAlnFormat="fasta"):
	"""Assemble deRIPed sequence, append all seqs in ascending order of RIP events logged."""
	deRIPseq = getDERIP(tracker,ID=ID,deGAP=False)
	align.append(deRIPseq)
	with open(outPathAln, "w") as f:
		AlignIO.write(align, f, outAlnFormat)
