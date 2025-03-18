import os
import sys
import tempfile

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pytest

from derip2.aln_ops import (
    checkLen,
    checkrow,
    checkUniqueID,
    correctRIP,
    fillConserved,
    fillRemainder,
    find,
    getDERIP,
    hasBoth,
    initRIPCounter,
    initTracker,
    lastBase,
    loadAlign,
    nextBase,
    setRefSeq,
    updateRIPCount,
    updateTracker,
    writeAlign,
    writeDERIP,
    writeDERIP2stdout,
)


# Helper functions to create test data
def create_test_alignment():
    """Create a test alignment with 3 sequences"""
    records = [
        SeqRecord(
            Seq('ACGTACGTAC'), id='seq1', name='seq1', description='test sequence 1'
        ),
        SeqRecord(
            Seq('ACGTATGTAC'), id='seq2', name='seq2', description='test sequence 2'
        ),
        SeqRecord(
            Seq('ACGTAAGTAC'), id='seq3', name='seq3', description='test sequence 3'
        ),
    ]
    return MultipleSeqAlignment(records)


def create_rip_alignment():
    """Create an alignment with RIP-like mutations (C→T and G→A in RIP contexts)"""
    records = [
        # Ancestral sequence (no RIP)
        SeqRecord(
            Seq('ACGTACAACGTGTACT--G'),
            id='ancestral',
            name='ancestral',
            description='no RIP',
        ),
        # Forward strand RIP (CA→TA)
        SeqRecord(
            Seq(
                'ACGTATAACGTGTATT--G'
            ),  # 6th base is mutated from C to T, 15th is C to T non-RIP deamination
            id='fwd_rip',
            name='fwd_rip',
            description='forward RIP',
        ),
        # Reverse strand RIP (TG→TA)
        SeqRecord(
            Seq('ACGTACAACGTATACT--G'),  # 12th base is mutated from G to A
            id='rev_rip',
            name='rev_rip',
            description='reverse RIP',
        ),
        # Sequence with deamination not in RIP context
        SeqRecord(
            Seq('ACGTACAACATGTATT--G'),  # 10th G to A, 15th C to T,
            id='deamin',
            name='deamin',
            description='non-RIP deamination',
        ),
    ]
    return MultipleSeqAlignment(records)


def create_duplicate_id_alignment():
    """Create an alignment with duplicate sequence IDs"""
    records = [
        SeqRecord(
            Seq('ACGTACGTAC'), id='seq1', name='seq1', description='test sequence 1'
        ),
        SeqRecord(
            Seq('ACGTATGTAC'),
            id='seq1',
            name='seq1',
            description='test sequence 1 duplicate',
        ),
        SeqRecord(
            Seq('ACGTAAGTAC'), id='seq3', name='seq3', description='test sequence 3'
        ),
    ]
    return MultipleSeqAlignment(records)


def create_single_sequence_alignment():
    """Create an alignment with only one sequence"""
    records = [
        SeqRecord(
            Seq('ACGTACGTAC'), id='seq1', name='seq1', description='test sequence 1'
        ),
    ]
    return MultipleSeqAlignment(records)


def create_gapped_alignment():
    """Create an alignment with gaps"""
    records = [
        SeqRecord(
            Seq('ACGT-CGTAC'),
            id='seq1',
            name='seq1',
            description='test sequence 1 with gap',
        ),
        SeqRecord(
            Seq('ACGTA-GTAC'),
            id='seq2',
            name='seq2',
            description='test sequence 2 with gap',
        ),
        SeqRecord(
            Seq('ACGTA--TAC'),
            id='seq3',
            name='seq3',
            description='test sequence 3 with gaps',
        ),
    ]
    return MultipleSeqAlignment(records)


def save_alignment_to_fasta(alignment, filename):
    """Save an alignment to a FASTA file"""
    with open(filename, 'w') as f:
        AlignIO.write(alignment, f, 'fasta')


# Test fixtures
@pytest.fixture
def test_alignment():
    return create_test_alignment()


@pytest.fixture
def rip_alignment():
    return create_rip_alignment()


@pytest.fixture
def duplicate_id_alignment():
    return create_duplicate_id_alignment()


@pytest.fixture
def single_sequence_alignment():
    return create_single_sequence_alignment()


@pytest.fixture
def gapped_alignment():
    return create_gapped_alignment()


@pytest.fixture
def temp_fasta_file():
    """Create a temporary FASTA file with test alignment"""
    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.fasta', delete=False) as temp:
        alignment = create_test_alignment()
        AlignIO.write(alignment, temp, 'fasta')
        temp_name = temp.name

    yield temp_name

    # Clean up the file after the test
    if os.path.exists(temp_name):
        os.unlink(temp_name)


@pytest.fixture
def rip_fasta_file():
    """Create a temporary FASTA file with RIP alignment"""
    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.fasta', delete=False) as temp:
        alignment = create_rip_alignment()
        AlignIO.write(alignment, temp, 'fasta')
        temp_name = temp.name

    yield temp_name

    # Clean up the file after the test
    if os.path.exists(temp_name):
        os.unlink(temp_name)


# Tests for validation functions
def test_check_unique_id(test_alignment, duplicate_id_alignment, caplog):
    """Test checkUniqueID function with valid and invalid alignments"""
    # Should not raise any error with unique IDs
    checkUniqueID(test_alignment)

    # Should log error and exit with duplicate IDs
    with pytest.raises(SystemExit):
        checkUniqueID(duplicate_id_alignment)
        assert 'Sequence IDs not unique' in caplog.text
        assert "Non-unique IDs: ['seq1']" in caplog.text


def test_check_len(test_alignment, single_sequence_alignment, caplog):
    """Test checkLen function with valid and invalid alignments"""
    # Should not raise any error with >= 2 sequences
    checkLen(test_alignment)

    # Should log error and exit with < 2 sequences
    with pytest.raises(SystemExit):
        checkLen(single_sequence_alignment)
        assert 'Alignment contains < 2 sequences' in caplog.text


def test_load_align(temp_fasta_file):
    """Test loadAlign function with a valid FASTA file"""
    # Should successfully load the alignment
    alignment = loadAlign(temp_fasta_file)
    assert alignment.__len__() == 3
    assert alignment[0].id == 'seq1'
    assert alignment[1].id == 'seq2'
    assert alignment[2].id == 'seq3'


def test_checkrow(test_alignment):
    """Test checkrow function"""
    # Valid index should not raise error
    checkrow(test_alignment, 0)
    checkrow(test_alignment, 2)

    # Invalid index should raise SystemExit
    with pytest.raises(SystemExit):
        checkrow(test_alignment, 3)


# Tests for initialization functions
def test_init_tracker(test_alignment):
    """Test initTracker function"""
    tracker = initTracker(test_alignment)

    # Should create a dictionary with keys for each column
    assert len(tracker) == test_alignment.get_alignment_length()

    # Each value should be a namedtuple with idx and base=None
    for i in range(test_alignment.get_alignment_length()):
        assert tracker[i].idx == i
        assert tracker[i].base is None


def test_init_rip_counter(test_alignment):
    """Test initRIPCounter function"""
    rip_counts = initRIPCounter(test_alignment)

    # Should create a dictionary with keys for each row
    assert len(rip_counts) == test_alignment.__len__()

    # Each value should be a namedtuple with the correct fields
    for i in range(test_alignment.__len__()):
        assert rip_counts[i].idx == i
        assert rip_counts[i].SeqID == test_alignment[i].id
        assert rip_counts[i].revRIPcount == 0
        assert rip_counts[i].RIPcount == 0
        assert rip_counts[i].nonRIPcount == 0
        assert isinstance(rip_counts[i].GC, float)


# Tests for update functions
def test_update_tracker(test_alignment):
    """Test updateTracker function"""
    # Initialize a tracker
    tracker = initTracker(test_alignment)

    # Update a position
    updated_tracker = updateTracker(0, 'A', tracker)
    assert updated_tracker[0].base == 'A'

    # Try to update the same position without force
    updated_tracker = updateTracker(0, 'G', updated_tracker)
    assert updated_tracker[0].base == 'A'  # Should not change

    # Update with force=True
    updated_tracker = updateTracker(0, 'G', updated_tracker, force=True)
    assert updated_tracker[0].base == 'G'  # Should change


def test_update_rip_count(test_alignment):
    """Test updateRIPCount function"""
    # Initialize RIP counter
    rip_counts = initRIPCounter(test_alignment)

    # Update counts for row 0
    updated_counts = updateRIPCount(0, rip_counts, addFwd=1, addRev=2, addNonRIP=3)

    assert updated_counts[0].RIPcount == 1
    assert updated_counts[0].revRIPcount == 2
    assert updated_counts[0].nonRIPcount == 3

    # Update again to test cumulative counting
    updated_counts = updateRIPCount(0, updated_counts, addFwd=1)
    assert updated_counts[0].RIPcount == 2
    assert updated_counts[0].revRIPcount == 2
    assert updated_counts[0].nonRIPcount == 3


# Tests for fill functions
def test_fill_conserved(gapped_alignment):
    """Test fillConserved function"""
    # Initialize a tracker
    tracker = initTracker(gapped_alignment)

    # Fill conserved positions
    updated_tracker = fillConserved(gapped_alignment, tracker, maxGaps=0.5)

    # First 4 positions are invariant 'ACGT'
    assert updated_tracker[0].base == 'A'
    assert updated_tracker[1].base == 'C'
    assert updated_tracker[2].base == 'G'
    assert updated_tracker[3].base == 'T'

    # Position 5 has 1/3 gaps but < maxGaps, should have base 'A'
    assert updated_tracker[4].base == 'A'

    # Position 6-8 should be conserved except position 6 which has 2/3 gaps > maxGaps
    assert updated_tracker[5].base == '-'
    assert updated_tracker[6].base == 'G'  # 1/3 gaps
    assert updated_tracker[7].base == 'T'
    assert updated_tracker[8].base == 'A'
    assert updated_tracker[9].base == 'C'


def test_fill_remainder(test_alignment):
    """Test fillRemainder function"""
    # Initialize a tracker
    tracker = initTracker(test_alignment)

    # Fill from the first sequence
    updated_tracker = fillRemainder(test_alignment, 0, tracker)

    # All positions should match the first sequence
    for i in range(test_alignment.get_alignment_length()):
        assert updated_tracker[i].base == test_alignment[0].seq[i]


# Tests for motif search functions
def test_next_base(rip_alignment):
    """Test nextBase function"""
    # Find CA motifs at idx position 5 (6th base is a C followed by A)
    ca_rows, _CA_nextbase_offsets = nextBase(rip_alignment, 5, 'CA')
    assert 0 in ca_rows  # ancestral sequence has CA at idx pos 5-6
    assert 2 in ca_rows  # rev_rip sequence has CA at idx pos 5-6
    assert 3 in ca_rows  # deamin sequence has CA at idx pos 5-6
    assert 1 not in ca_rows  # fwd_rip has TA instead at idx pos 5-6

    # Find TA motifs at idx position 5 (6th base is T followed by A)
    ta_rows, _TA_nextbase_offsets = nextBase(rip_alignment, 5, 'TA')
    assert 1 in ta_rows  # fwd_rip has TA at idx pos 5-6
    assert 0 not in ta_rows  # ancestral has CA instead at idx pos 5-6
    assert 2 not in ta_rows  # rev_rip has CA instead at idx pos 5-6
    assert 3 not in ta_rows  # deamin sequence has CA at idx pos 5-6


def test_last_base(rip_alignment):
    """Test lastBase function"""
    # Find TG motifs ending at idx position 11 (base is a G preceeded by a T)
    tg_rows, _TG_nextbase_offsets = lastBase(rip_alignment, 11, 'TG')
    assert 0 in tg_rows  # ancestral sequence has TG at pos 10-11
    assert 1 in tg_rows  # fwd_rip has TG at pos 10-11
    assert 3 in tg_rows  # deamin has TG at pos 10-11
    assert 2 not in tg_rows  # rev_rip has TA instead

    # Find TG motifs ending at idx position 18 (base is a G preceeded by a T)
    # Offset should be 18
    tg_rows, _TG_nextbase_offsets = lastBase(rip_alignment, 18, 'TG')
    print(tg_rows, file=sys.stderr)
    assert -3 == _TG_nextbase_offsets[0]

    # Find TA motifs ending at idx position 11 (base is an A preceeded by a T)
    ta_rows, _TA_nextbase_offsets = lastBase(rip_alignment, 11, 'TA')
    assert 2 in ta_rows  # rev_rip has TA at in pos 10-11
    assert 0 not in ta_rows  # ancestral has TG instead
    assert 1 not in ta_rows  # fwd_rip has TG instead
    assert 3 not in ta_rows  # deamin has TG instead


def test_find():
    """Test find function"""
    # Test with a single character
    result = find(['A', 'C', 'G', 'T', 'A'], 'A')
    assert result == [0, 4]

    # Test with a list of characters
    result = find(['A', 'C', 'G', 'T', 'A'], ['A', 'G'])
    assert result == [0, 2, 4]

    # Test with a set of characters
    result = find(['A', 'C', 'G', 'T', 'A'], {'A', 'T'})
    assert result == [0, 3, 4]


def test_has_both():
    """Test hasBoth function"""
    # Test with both characters present
    assert hasBoth(['A', 'C', 'G', 'T'], 'A', 'T') is True

    # Test with one character missing
    assert hasBoth(['A', 'C', 'G'], 'A', 'T') is False

    # Test with both characters missing
    assert hasBoth(['C', 'G'], 'A', 'T') is False


# Tests for RIP correction functions
def test_correct_rip(rip_alignment):
    """Test correctRIP function"""
    # Initialize tracker and RIP counter
    tracker = initTracker(rip_alignment)
    rip_counts = initRIPCounter(rip_alignment)

    # Apply RIP correction
    updated_tracker, updated_counts, masked_align, _corrected_positions, _markupdict = (
        correctRIP(
            rip_alignment,
            tracker,
            rip_counts,
            maxSNPnoise=0.5,
            minRIPlike=0.1,
            reaminate=False,
            mask=True,
        )
    )

    # Check forward RIP correction (CA→TA): Idx position 5 (6th base) should be corrected to C
    assert updated_tracker[5].base == 'C'

    # Check reverse RIP correction (TG→TA): Idx position 11 (12th base) should be corrected to G
    assert updated_tracker[11].base == 'G'

    # Check non-RIP deamination (C→T): Idx position 14 (15th base) should not be corrected to C
    assert updated_tracker[14].base is None  # Should not be corrected

    # Check non-RIP deamination (G→A): Idx position 9 (10th base) should not be corrected to G
    assert updated_tracker[9].base is None  # Should not be corrected

    # Check RIP counts - sequence 1 should have forward RIP count
    assert updated_counts[1].RIPcount == 1

    # Check RIP counts - sequence 2 should have reverse RIP count
    assert updated_counts[2].revRIPcount == 1

    # Check masking - RIP-corrected positions should be masked
    assert masked_align[1].seq[5] == 'Y'  # Y = C/T ambiguity code
    assert masked_align[1].seq[14] == 'T'  # not Y = C/T ambiguity code
    assert masked_align[2].seq[11] == 'R'  # R = G/A ambiguity code


# Tests for RIP correction functions
def test_correct_rip_with_deamination(rip_alignment):
    """Test correctRIP function"""
    # Initialize tracker and RIP counter
    tracker = initTracker(rip_alignment)
    rip_counts = initRIPCounter(rip_alignment)

    # Apply RIP correction
    updated_tracker, updated_counts, masked_align, _corrected_positions, _markupdict = (
        correctRIP(
            rip_alignment,
            tracker,
            rip_counts,
            maxSNPnoise=0.5,
            minRIPlike=0.1,
            reaminate=True,
            mask=True,
        )
    )

    # 9=G, 14=C
    # Check forward RIP correction (CA→TA): Idx position 5 (6th base) should be corrected to C
    assert updated_tracker[5].base == 'C'

    # Check reverse RIP correction (TG→TA): Idx position 11 (12th base) should be corrected to G
    assert updated_tracker[11].base == 'G'

    # Check non-RIP deamination (C→T): Idx position 14 (15th base) should be corrected to C
    assert updated_tracker[14].base == 'C'

    # Check non-RIP deamination (G→A): Idx position 9 (10th base) should be corrected to G
    assert updated_tracker[9].base == 'G'

    # Check RIP counts - sequence 1 should have forward RIP count and 1 non-RIP deamination
    assert updated_counts[1].RIPcount == 1
    assert updated_counts[1].nonRIPcount == 1

    # Check RIP counts - sequence 2 should have reverse RIP count
    assert updated_counts[2].revRIPcount == 1

    # Check RIP counts - sequence 3 should have 2 non-RIP deaminations
    assert updated_counts[3].nonRIPcount == 2

    # Check masking - RIP-corrected positions should be masked
    assert masked_align[1].seq[5] == 'Y'  # Y = C/T ambiguity code
    assert masked_align[1].seq[14] == 'Y'  # Y = C/T ambiguity code
    assert masked_align[2].seq[11] == 'R'  # R = G/A ambiguity code


def test_set_ref_seq(rip_alignment):
    """Test setRefSeq function"""
    # Initialize RIP counter
    rip_counts = initRIPCounter(rip_alignment)

    # Update RIP counts for testing
    rip_counts = updateRIPCount(1, rip_counts, addFwd=1)
    rip_counts = updateRIPCount(2, rip_counts, addRev=1)

    # Test selection by minimum RIP count
    ref_idx = setRefSeq(rip_alignment, rip_counts, getMinRIP=True)
    assert ref_idx == 0 or ref_idx == 3  # Should select sequence with 0 RIP count

    # Test selection by maximum GC content
    ref_idx = setRefSeq(rip_alignment, rip_counts, getMaxGC=True)
    # Should select sequence with highest GC content
    highest_gc = max(rip_counts.values(), key=lambda x: x.GC).idx
    assert ref_idx == highest_gc


# Tests for output functions
def test_get_derip(test_alignment):
    """Test getDERIP function"""
    # Initialize and fill tracker
    tracker = initTracker(test_alignment)
    for i in range(test_alignment.get_alignment_length()):
        tracker = updateTracker(i, test_alignment[0].seq[i], tracker)

    # Get deRIP sequence
    derip_seq = getDERIP(tracker, ID='test_derip', deGAP=True)

    # Check sequence
    assert str(derip_seq.seq) == str(test_alignment[0].seq)
    assert derip_seq.id == 'test_derip'


def test_write_derip(test_alignment, tmp_path):
    """Test writeDERIP function"""
    # Initialize and fill tracker
    tracker = initTracker(test_alignment)
    for i in range(test_alignment.get_alignment_length()):
        tracker = updateTracker(i, test_alignment[0].seq[i], tracker)

    # Write to file
    out_file = tmp_path / 'derip_output.fa'
    writeDERIP(tracker, str(out_file), ID='test_derip')

    # Check file was created
    assert os.path.exists(out_file)

    # Check file contents
    with open(out_file) as f:
        content = f.read()
        assert '>test_derip' in content
        assert str(test_alignment[0].seq) in content


def test_write_align(test_alignment, tmp_path):
    """Test writeAlign function"""
    # Initialize and fill tracker
    tracker = initTracker(test_alignment)
    for i in range(test_alignment.get_alignment_length()):
        # Update tracker with bases from the first sequence
        tracker = updateTracker(i, test_alignment[0].seq[i], tracker)

    # Write alignment to file with deRIP sequence
    out_file = tmp_path / 'derip_alignment.fa'
    writeAlign(tracker, test_alignment, str(out_file), ID='test_derip', noappend=False)

    # Check file was created
    assert os.path.exists(out_file)

    # Read the alignment back and check
    new_align = AlignIO.read(out_file, 'fasta')

    # Should have one more sequence than the original
    assert new_align.__len__() == test_alignment.__len__() + 1
    assert new_align[-1].id == 'test_derip'


def test_write_derip_to_stdout(test_alignment, capsys):
    """Test writeDERIP2stdout function"""
    # Initialize and fill tracker
    tracker = initTracker(test_alignment)
    for i in range(test_alignment.get_alignment_length()):
        tracker = updateTracker(i, test_alignment[0].seq[i], tracker)

    # Write to stdout
    writeDERIP2stdout(tracker, ID='test_derip')

    # Capture stdout
    captured = capsys.readouterr()

    # Check output
    assert '>test_derip' in captured.out
    assert str(test_alignment[0].seq) in captured.out
