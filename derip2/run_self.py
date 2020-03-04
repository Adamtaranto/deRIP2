#!/usr/bin/env python

from __future__ import print_function
from derip2 import __version__
#from derip2 import log
import os
#import sys
#import glob
#import shutil
import derip2
import argparse

def mainArgs():
	'''Parse command line arguments.'''
	parser = argparse.ArgumentParser(
			description	=	'Predict ancestral sequence of fungal repeat elements by correcting for RIP-like mutations or cytosine deamination in multi-sequence DNA alignments. Optionally, mask corrected positions in alignment.',
			prog		=	'derip2'
			)
	parser.add_argument('--version', action='version',version='%(prog)s {version}'.format(version=__version__))
	# Inputs
	parser.add_argument("-i","--inAln",required=True,type=str,default= None,help="Multiple sequence alignment.")
	parser.add_argument('--format',default="fasta",choices=["clustal","emboss","fasta","fasta-m10","ig","nexus","phylip","phylip-sequential","phylip-relaxed","stockholm"],help='Format of input alignment. Default: fasta')
	# Options	
	parser.add_argument("-g","--maxGaps",type=float,default=0.7, help="Maximum proportion of gapped positions in column to be tolerated before forcing a gap in final deRIP sequence. Default: 0.7")
	parser.add_argument("-a","--reaminate",action="store_true",default=False, help="Correct all deamination events independent of RIP context. Default: False")
	parser.add_argument("--maxSNPnoise",type=float,default=0.5,help="Maximum proportion of conflicting SNPs permitted before excluding column from RIP/deamination assessment. i.e. By default a column with >= 0.5 'C/T' bases will have 'TpA' positions logged as RIP events. Default: 0.5")
	parser.add_argument("--minRIPlike",type=float,default=0.1,help="Minimum proportion of deamination events in RIP context (5' CpA 3' --> 5' TpA 3') required for column to deRIP'd in final sequence. Note: If 'reaminate' option is set all deamination events will be corrected. Default 0.1 ")
	parser.add_argument("--fillmaxgc",action="store_true",default=False, help="By default uncorrected positions in the output sequence are filled from the sequence with the lowest RIP count. If this option is set remaining positions are filled from the sequence with the highest G/C content. Default: False")
	parser.add_argument("--fillindex",type=int,default=None,help="Force selection of alignment row to fill uncorrected positions from by row index number (indexed from 0). Note: Will override '--fillmaxgc' option.")		
	parser.add_argument('--mask',default=False,action='store_true',help='Mask corrected positions in alignment with degenerate IUPAC codes.')
	parser.add_argument('--noappend',default=False,action='store_true',help="If set, do not append deRIP'd sequence to output alignment.")
	# Outputs
	parser.add_argument("-d","--outDir",type=str,default=None,help="Directory for deRIP'd sequence files to be written to.")
	#parser.add_argument("-o","--outName",type=str,default= "deRIP_output.fa", help="Write deRIP sequence to this file. Default: ./deRIP_output.fa")
	parser.add_argument('--outAlnName',default=None,help='Optional: If set write alignment including deRIP corrected sequence to this file.')
	parser.add_argument('--outAlnFormat',default="fasta",choices=["fasta","nexus"],help='Optional: Write alignment including deRIP sequence to file of format X. Default: fasta')
	parser.add_argument('--label',default="deRIPseq",help="Use label as name for deRIP'd sequence in output files.")
	args = parser.parse_args()
	return args

def main():
	'''Do the work.'''
	# Get cmd line args
	args = mainArgs()
	# Check for output directory, create if required, else set to cwd
	outDir = derip2.dochecks(args.outDir)
	# Set output file paths
	# New default: write deRIP'd seq to stdout
	#outPathFile = os.path.join(outDir,args.outName)
	if args.outAlnName:
		outPathAln = os.path.join(outDir,args.outAlnName)
	# Read in alignment file, check at least 2 sequences present and names are unique
	align = derip2.loadAlign(args.inAln,args.format)
	# Report alignment summary
	derip2.alignSummary(align)
	# Initialise object to assemble deRIP'd sequence
	tracker = derip2.initTracker(align)
	# Initialise object to track RIP observations and GC content by row
	RIPcounts = derip2.initRIPCounter(align)
	# Preset invariant or highly gapped positions in final sequence
	tracker = derip2.fillConserved(align,tracker,args.maxGaps)
	# Correct / tally RIP + optionally correct C->T / G->A transitions
	tracker,RIPcounts,maskedAlign = derip2.correctRIP(align,tracker,RIPcounts,maxSNPnoise=args.maxSNPnoise,minRIPlike=args.minRIPlike,reaminate=args.reaminate,mask=args.mask)
	# Report RIP counts per sequence
	derip2.summarizeRIP(RIPcounts)
	# Set reference sequence to fill remaining uncorrected positions from
	if not args.fillindex:
		# Select least RIP'd sequence (or most GC-rich if no RIP or tied for min-RIP) in alignment to inherit remaining unset positions from
		refID = derip2.setRefSeq(align, RIPcounts, getMinRIP=True, getMaxGC=args.fillmaxgc)
	else:
		# Check row index exists
		derip2.checkrow(align,idx=args.fillindex)
		# Set ref sequence to user specified row
		refID = args.fillindex
	# Fill remaining unset positions from reference sequence
	tracker = derip2.fillRemainder(align,refID,tracker)
	# Write ungapped deRIP to file [old default behaviour]
	# derip2.writeDERIP(tracker,outPathFile,ID=args.label)
	derip2.writeDERIP2stdout(tracker,ID=args.label)
	# Write updated alignment (including gapped deRIP'd sequence) to file. Optional.
	if args.outAlnName:
		derip2.log("Preparing output alignment.")
		# Log if deRIP'd sequence will be appended to alignment.
		if not args.noappend:
			derip2.log("Appending corrected sequence to alignment with ID: %s" % args.label)
		# Use masked alignment if mask option set.
		if args.mask:
			derip2.log("Masking alignment columns with detected mutations.")
			outputAlign = maskedAlign
		else:
			outputAlign = align
		derip2.log("Writing modified alignment to path: %s" % outPathAln)
		derip2.writeAlign(tracker,outputAlign,outPathAln,ID=args.label,outAlnFormat=args.outAlnFormat,noappend=args.noappend)
	