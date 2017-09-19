# deRIP2

Predict progenitor sequence of fungal repeat families by correcting for RIP-like mutations 
(CpA --> TpA) and cytosine deamination (C --> T) events.

# Table of contents

* [Algorithm overview](#algorithm-overview)
* [Options and usage](#options-and-usage)
    * [Installing deRIP2](#installing-derip2)
    * [Example usage](#example-usage)
    * [Standard options](#standard-options)
* [License](#license)

# Algorithm overview
# Options and usage

## Installing deRIP2

Clone from this repository:

```bash
git clone https://github.com/Adamtaranto/deRIP2.git && cd deRIP2
```
## Example usage

For aligned sequences in 'myalignment.fa':
  - Any column >= 50% non RIP/cytosine deamination mutations is not corrected.
  - Any column >= 70% gap positions is not corrected.
  - Make RIP corrections if column is >= 10% RIP context.
  - Correct Cytosine-deamination mutations outside of RIP context.
  - Inherit all remaining uncorrected positions from least RIP'd sequence.

```bash
./deRIP2.py --inAln myalignment.fa --format fasta \
--maxGaps 0.7 \
--maxSNPnoise 0.5 \
--minRIPlike 0.1 \
--outName deRIPed_sequence.fa \
--outAlnName aligment_with_deRIP.aln \
--outAlnFormat clustal \
--label deRIPseqName \
--outDir results \
--reaminate
```

**Output:**  
  - results/aligment_with_deRIP.aln 

## Standard options

```
Usage: ./deRIP2.py [-h] -i INALN
                   [--format {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}]
                   [--outAlnFormat {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}]
                   [-g MAXGAPS] [-a] [--maxSNPnoise MAXSNPNOISE]
                   [--minRIPlike MINRIPLIKE] [-o OUTNAME] [--outAlnName OUTALNNAME]
                   [--label LABEL] [-d OUTDIR]

Takes a multi-sequence DNA alignment and estimates a progenitor sequence by
correcting for RIP-like mutations.

Optional arguments:

# Info: 

-h, --help       Show this help message and exit.  

# Input: 

-i, --inAln      Multiple sequence alignment.
                  (Required) 
--format         Format of the input alignment.
                  Accepted formats: 
                  {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}

# Output:

--label          Use label as name for deRIP'd sequence in output files.
-o, --outName    Write deRIP sequence to this file.         
-d, --outDir     Directory for deRIP'd sequence files to be written to.  
--outAlnName     If set writes alignment including deRIP sequence to this file. 
--outAlnFormat   Write alignment including deRIP sequence to file of format X.
                  Accepted formats: 
                  {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm} 

# deRIP settings:

-g, --maxGaps    The maximum proportion of gapped positions in a column to be 
                  tolerated before forcing a gap in the final deRIP sequence.  
-a, --reaminate  Correct deamination events in non-RIP contexts.  
--maxSNPnoise    The maximum proportion of conflicting SNPs permitted before 
                  excluding a column from RIP/deamination assessment. 
                  i.e. By default a column with >= 50 'C/T' bases will have 
                  'TpA' positions logged as RIP events.  
--minRIPlike     Minimum proportion of deamination events in RIP context 
                  (5' CpA 3' --> 5' TpA 3') required for column to deRIP'd in 
                  final sequence. Note: If 'reaminate' option is set all 
                  deamination events will be corrected.  
```

# License

Software provided under MIT license.