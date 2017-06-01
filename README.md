# deRIP2 v0.1.0

Predict progenitor sequence of fungal repeat families by correcting for RIP-like mutations (CpA --> TpA) and cytosine deamination (C --> T) events.

## Usage

```bash
./deRIP2 --inAln myalignment.fa --format fasta \
--maxGaps 0.7 \
--maxSNPnoise 0.5 \
--minRIPlike 0.1 \
--outName deRIPed_sequence.fa \
--outAlnName aligment_with_deRIP.aln \
--outAlnFormat clustal \
--outDir results \
--reaminate
```

## Options  

**-h, --help**  

  - Show this help message and exit.  

**-v, --version**  

  - Show program's version number and exit.  

**-i,--inAln**  

  - Multiple sequence alignment.  

**--format**  

  - Format of input alignment.  

  - Accepted formats: {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}  
                      
**--outAlnFormat**  

  - Optional: Write alignment including deRIP sequence to file of format X.  

  - Accepted formats: {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}  

**-g,--maxGaps**  

  - Maximum proportion of gapped positions in column to be tolerated before forcing a gap in final deRIP sequence.  

**-a, --reaminate**  

  - Correct deamination events in non-RIP contexts.  

**--maxSNPnoise**  

  - Maximum proportion of conflicting SNPs permitted before excluding column from RIP/deamination assessment. 
  i.e. By default a column with >= 50 'C/T' bases will have 'TpA' positions logged as RIP events.  

**--minRIPlike**  
  
  - Minimum proportion of deamination events in RIP context (5' CpA 3' --> 5' TpA 3') required for column to deRIP'd in final sequence. Note: If 'reaminate' option is set all deamination events will be corrected.  

**-o,--outName**  
  
  - Write deRIP sequence to this file.  

**--outAlnName**  

  - Optional: If set write alignment including deRIP sequence to this file.  

**-d,--outDir**  

  - Directory for deRIP'd sequence files to be written to.  


