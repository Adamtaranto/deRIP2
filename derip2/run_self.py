#!/usr/bin/env python

from __future__ import print_function
from derip2 import __version__
import os
import sys
#import glob
#import shutil
import derip2
import argparse

def log(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def mainArgs():
	'''Parse command line arguments.'''
	parser = argparse.ArgumentParser(
			description	=	'Predict ancestral sequence of fungal repeat elements by correcting for RIP-like mutations in multi-sequence DNA alignments.',
			prog		=	'derip2'
			)
	parser.add_argument('--version', action='version',version='%(prog)s {version}'.format(version=__version__))
	parser.add_argument("-i","--inAln",required=True,type=str,default= None,help="Multiple sequence alignment.")
	parser.add_argument('--format',default="fasta",choices=["clustal","emboss","fasta","fasta-m10","ig","nexus","phylip","phylip-sequential","phylip-relaxed","stockholm"],help='Format of input alignment.')
	parser.add_argument('--outAlnFormat',default="fasta",choices=["clustal","emboss","fasta","fasta-m10","ig","nexus","phylip","phylip-sequential","phylip-relaxed","stockholm"],help='Optional: Write alignment including deRIP sequence to file of format X.')
	parser.add_argument("-g","--maxGaps",type=float,default=0.7, help="Maximum proportion of gapped positions in column to be tolerated before \forcing a gap in final deRIP sequence.")
	parser.add_argument("-a","--reaminate",action="store_true",default=False, help="Correct deamination events in non-RIP contexts.")
	parser.add_argument("--maxSNPnoise",type=float,default=0.5,help="Maximum proportion of conflicting SNPs permitted before excluding column \from RIP/deamination assessment. i.e. By default a column with >= 0.5 'C/T' bases \will have 'TpA' positions logged as RIP events.")
	parser.add_argument("--minRIPlike",type=float,default=0.1,help="Minimum proportion of deamination events in RIP context (5' CpA 3' --> 5' TpA 3') \required for column to deRIP'd in final sequence. Note: If 'reaminate' option is \set all deamination events will be corrected ")
	parser.add_argument("-o","--outName",type=str,default= "deRIP_output.fa", help="Write deRIP sequence to this file.")
	parser.add_argument('--outAlnName',default=None,help='Optional: If set write alignment including deRIP sequence to this file.')
	parser.add_argument('--label',default="deRIPseq",help="Use label as name for deRIP'd sequence in output files.")
	parser.add_argument("-d","--outDir",type=str,default= None,help="Directory for deRIP'd sequence files to be written to.")
	args = parser.parse_args()
	return args

def main():
	'''Do the work.'''
	# Get cmd line args
	args = mainArgs()

	# Check for output directories
	outDir = derip2.dochecks(args.outDir)
	outPathFile = os.path.join(outDir,args.outName)
	outPathAln = os.path.join(outDir,args.outAlnName)
	# Read in alignment file, check at least 2 sequences present and names are unique
	align = derip2.loadAlign(args.inAln,args.format)
	# Initialise object to assemble deRIP'd sequence
	tracker = derip2.initTracker(align)
	# Initialise object to track RIP observations and GC content by row
	RIPcounts = derip2.initRIPCounter(align)
	# Preset invariant or highly gapped positions in final sequence
	tracker = derip2.fillConserved(align,tracker,args.maxGaps)
	# Correct / tally RIP + correct C->T / G->A conversions
	tracker,RIPcounts = derip2.correctRIP(align,tracker,RIPcounts,maxSNPnoise=args.maxSNPnoise,minRIPlike=args.minRIPlike,reaminate=args.reaminate)
	# Select least RIP'd / most GC-rich sequence in alignment to inherit remaining unset positions from
	refID = derip2.setRefSeq(align, RIPcounts, getMinRIP=True)
	# Fill remaining unset positions from min RIP / max GC original sequence
	tracker = derip2.fillRemainder(align,refID,tracker)
	# Write ungapped deRIP to file
	derip2.writeDERIP(tracker,outPathFile,ID=args.label)
	# Write updated alignment (including gapped deRIP) to file. Optional.
	derip2.writeAlign(tracker,align,outPathAln,ID=args.label,outAlnFormat=args.outAlnFormat)
	