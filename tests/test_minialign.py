"""
Tests for the minialign module which provides functions for visualizing DNA sequence alignments.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.colors
import numpy as np
import pytest

from derip2.plotting.minialign import (
    FastaToArray,
    RIPPosition,
    addColumnRangeMarkers,
    arrNumeric,
    drawMiniAlignment,
    markupRIPBases,
)

# --- Fixtures ---


@pytest.fixture
def simple_fasta_file():
    """Create a simple FASTA alignment file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as f:
        f.write('>seq1\nAGCT\n>seq2\nAGCT\n>seq3\nAGCT\n')
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def misaligned_fasta_file():
    """Create a FASTA file with sequences of different lengths."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as f:
        f.write('>seq1\nAGCT\n>seq2\nAGCTTA\n>seq3\nAG\n')
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def single_seq_fasta_file():
    """Create a FASTA file with only one sequence."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as f:
        f.write('>seq1\nAGCT\n')
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def complex_fasta_file():
    """Create a more complex FASTA alignment with varied nucleotides."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as f:
        f.write('>seq1\nAGCT-NX\n>seq2\nAGAT-NX\n>seq3\nTGCTGNA\n')
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def rip_positions():
    """Create sample RIP positions for markup."""
    return {
        'rip_product': [RIPPosition(1, 0, 0), RIPPosition(2, 1, 2)],
        'rip_substrate': [RIPPosition(3, 2, -1)],
        'non_rip_deamination': [RIPPosition(4, 0, 0)],
    }


@pytest.fixture
def column_ranges():
    """Create sample column ranges for marking."""
    return [
        (1, 3, 'red', 'Region 1'),
        (5, 6, 'blue', 'Region 2'),
        (4, 4, 'green', ''),  # Single column with no label
    ]


# --- FastaToArray Tests ---


def test_FastaToArray_valid_alignment(simple_fasta_file):
    """Test FastaToArray with a valid alignment file."""
    arr, names, seq_len = FastaToArray(simple_fasta_file)

    assert arr is not None
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 4)  # 3 sequences of length 4
    assert names == ['seq1', 'seq2', 'seq3']
    assert seq_len == 3


def test_FastaToArray_single_sequence(single_seq_fasta_file):
    """Test FastaToArray with a file containing only one sequence."""
    result = FastaToArray(single_seq_fasta_file)

    assert result == (None, None, None)


def test_FastaToArray_misaligned_sequences(misaligned_fasta_file):
    """Test FastaToArray with misaligned sequences."""
    with pytest.raises(ValueError) as excinfo:
        FastaToArray(misaligned_fasta_file)

    assert 'ERROR: The sequences provided may not be aligned' in str(excinfo.value)


def test_FastaToArray_complex_alignment(complex_fasta_file):
    """Test FastaToArray with a more complex alignment containing gaps and special characters."""
    arr, names, seq_len = FastaToArray(complex_fasta_file)

    assert arr is not None
    assert arr.shape == (3, 7)  # 3 sequences of length 7

    # Check that invalid characters are replaced with gaps
    assert arr[0, 6] == '-'  # 'X' should be replaced with '-'
    assert arr[2, 5] == 'N'  # 'N' should remain 'N'


def test_FastaToArray_nonexistent_file():
    """Test FastaToArray with a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        FastaToArray('nonexistent_file.fa')


# --- arrNumeric Tests ---


def test_arrNumeric_basic():
    """Test basic functionality of arrNumeric."""
    # Create a simple alignment array
    arr = np.array([['A', 'G', 'C', 'T'], ['A', 'G', 'C', '-'], ['T', 'G', 'C', 'A']])

    numeric_arr, cmap = arrNumeric(arr)

    # Check that the array was flipped vertically
    assert numeric_arr.shape == (3, 4)

    # Check that we have the correct number of colors in the colormap
    # In this case, we expect 5 colors (A, G, C, T, and -)
    assert len(cmap.colors) == 5


def test_arrNumeric_palettes():
    """Test arrNumeric with different color palettes."""
    arr = np.array([['A', 'G', 'C', 'T', 'N', '-'], ['A', 'G', 'C', 'T', 'N', '-']])

    # Test each available palette
    for palette in ['colorblind', 'bright', 'tetrimmer']:
        numeric_arr, cmap = arrNumeric(arr, palette)
        assert len(cmap.colors) == 6  # A, G, C, T, N, -

    # Test with invalid palette (should default to colorblind)
    numeric_arr, cmap = arrNumeric(arr, 'invalid_palette')
    assert len(cmap.colors) == 6


# --- drawMiniAlignment Tests ---


@patch(
    'matplotlib.figure.Figure.savefig'
)  # Fix: patch Figure.savefig instead of pyplot.savefig
def test_drawMiniAlignment_basic(mock_savefig, simple_fasta_file):
    """Test basic functionality of drawMiniAlignment."""
    outfile = 'test_output.png'

    result = drawMiniAlignment(simple_fasta_file, outfile)

    assert result == outfile
    mock_savefig.assert_called_once_with(outfile, format='png', bbox_inches='tight')


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
def test_drawMiniAlignment_with_title(mock_savefig, simple_fasta_file):
    """Test drawMiniAlignment with a title."""
    outfile = 'test_output.png'
    title = 'Test Alignment'

    with patch('matplotlib.figure.Figure.suptitle') as mock_suptitle:
        result = drawMiniAlignment(simple_fasta_file, outfile, title=title)

    assert result == outfile
    mock_savefig.assert_called_once()
    mock_suptitle.assert_called_once()


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
def test_drawMiniAlignment_single_sequence(mock_savefig, single_seq_fasta_file):
    """Test drawMiniAlignment with a single sequence file."""
    outfile = 'test_output.png'

    result = drawMiniAlignment(single_seq_fasta_file, outfile)

    assert result is False
    mock_savefig.assert_not_called()


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
@patch('derip2.plotting.minialign.markupRIPBases')
def test_drawMiniAlignment_with_markup(
    mock_markup, mock_savefig, simple_fasta_file, rip_positions
):
    """Test drawMiniAlignment with RIP markup."""
    outfile = 'test_output.png'

    result = drawMiniAlignment(simple_fasta_file, outfile, markupdict=rip_positions)

    assert result == outfile
    mock_markup.assert_called_once()
    mock_savefig.assert_called_once()


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
@patch('derip2.plotting.minialign.addColumnRangeMarkers')
def test_drawMiniAlignment_with_column_ranges(
    mock_ranges, mock_savefig, simple_fasta_file, column_ranges
):
    """Test drawMiniAlignment with column range markers."""
    outfile = 'test_output.png'

    result = drawMiniAlignment(simple_fasta_file, outfile, column_ranges=column_ranges)

    assert result == outfile
    mock_ranges.assert_called_once()
    mock_savefig.assert_called_once()


def test_drawMiniAlignment_with_custom_dimensions(simple_fasta_file):
    """Test drawMiniAlignment with custom width and height."""
    outfile = 'test_output.png'
    width = 10
    height = 8

    # Create a mock figure with a mock savefig method
    mock_fig = MagicMock()
    mock_savefig = mock_fig.savefig

    with patch('matplotlib.pyplot.figure', return_value=mock_fig) as mock_figure:
        with patch('derip2.plotting.minialign.FastaToArray') as mock_fasta_to_array:
            # Mock the array data
            mock_arr = np.full((76, 4), 'A')
            mock_fasta_to_array.return_value = (mock_arr, ['seq1', 'seq2', 'seq3'], 76)

            # Also mock arrNumeric to avoid string/numeric conversion issues
            with patch('derip2.plotting.minialign.arrNumeric') as mock_arrnumeric:
                mock_arrnumeric.return_value = (
                    np.zeros((76, 4)),
                    matplotlib.colors.ListedColormap(['#000000', '#FFFFFF']),
                )

                result = drawMiniAlignment(
                    simple_fasta_file, outfile, width=width, height=height
                )

    assert result == outfile
    mock_figure.assert_called_once_with(figsize=(width, height), dpi=300)
    mock_savefig.assert_called_once()


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
def test_drawMiniAlignment_with_orig_names(mock_savefig, simple_fasta_file):
    """Test drawMiniAlignment with original sequence names."""
    outfile = 'test_output.png'
    orig_nams = ['original_seq1', 'original_seq2', 'original_seq3']

    result = drawMiniAlignment(
        simple_fasta_file, outfile, orig_nams=orig_nams, keep_numbers=True
    )

    assert result == outfile
    mock_savefig.assert_called_once()


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
def test_drawMiniAlignment_force_numbers(mock_savefig, complex_fasta_file):
    """Test drawMiniAlignment with force_numbers=True."""
    outfile = 'test_output.png'

    # Create mock axis
    mock_axis = MagicMock()

    with patch('matplotlib.pyplot.figure') as mock_figure:
        mock_figure.return_value.add_subplot.return_value = mock_axis
        result = drawMiniAlignment(complex_fasta_file, outfile, force_numbers=True)

    # Verify tick interval was set to 1
    mock_axis.set_yticks.assert_called_once()
    assert result == outfile


@patch('matplotlib.figure.Figure.savefig')  # Fix: patch Figure.savefig
def test_drawMiniAlignment_different_palettes(mock_savefig, simple_fasta_file):
    """Test drawMiniAlignment with different color palettes."""
    outfile = 'test_output.png'

    for palette in ['colorblind', 'bright', 'tetrimmer']:
        # Fix: Create a proper mock for arrNumeric that returns valid objects
        with patch('derip2.plotting.minialign.arrNumeric') as mock_arrNumeric:
            # Create a simple numeric array and a valid colormap
            numeric_arr = np.zeros((3, 4))
            cmap = matplotlib.colors.ListedColormap(['#ffffff', '#000000'])
            mock_arrNumeric.return_value = (numeric_arr, cmap)

            # Also patch the FastaToArray method to avoid file access
            with patch('derip2.plotting.minialign.FastaToArray') as mock_fasta_to_array:
                mock_arr = np.zeros((3, 4))
                mock_fasta_to_array.return_value = (
                    mock_arr,
                    ['seq1', 'seq2', 'seq3'],
                    3,
                )

                result = drawMiniAlignment(simple_fasta_file, outfile, palette=palette)
                assert result == outfile
                mock_arrNumeric.assert_called_once_with(mock_arr, palette=palette)


# --- markupRIPBases Tests ---


def test_markupRIPBases():
    """Test basic functionality of markupRIPBases."""
    mock_ax = MagicMock()
    mock_ax.get_xlim.return_value = [0, 10]

    markupdict = {
        'rip_product': [RIPPosition(1, 0, 0)],
        'rip_substrate': [RIPPosition(2, 1, 0)],
        'non_rip_deamination': [RIPPosition(3, 2, 0)],
    }

    ali_height = 3

    markupRIPBases(mock_ax, markupdict, ali_height)

    # First call should be the grey overlay
    assert mock_ax.add_patch.call_count == 4  # 1 overlay + 3 category markers


def test_markupRIPBases_with_offsets():
    """Test markupRIPBases with different offsets."""
    mock_ax = MagicMock()
    mock_ax.get_xlim.return_value = [0, 10]

    markupdict = {
        'rip_product': [RIPPosition(5, 0, -2)],  # Highlight 2 positions to the left
        'rip_substrate': [RIPPosition(5, 1, 3)],  # Highlight 3 positions to the right
    }

    ali_height = 3

    markupRIPBases(mock_ax, markupdict, ali_height)

    assert mock_ax.add_patch.call_count == 3  # 1 overlay + 2 category markers


def test_markupRIPBases_unknown_category():
    """Test markupRIPBases with an unknown category."""
    mock_ax = MagicMock()
    mock_ax.get_xlim.return_value = [0, 10]

    markupdict = {'unknown_category': [RIPPosition(1, 0, 0)]}

    ali_height = 3

    markupRIPBases(mock_ax, markupdict, ali_height)

    # Should still add patches with default color
    assert mock_ax.add_patch.call_count == 2  # 1 overlay + 1 marker


# --- addColumnRangeMarkers Tests ---


def test_addColumnRangeMarkers():
    """Test basic functionality of addColumnRangeMarkers."""
    mock_ax = MagicMock()

    ranges = [(1, 3, 'red', 'Region 1'), (5, 8, 'blue', 'Region 2')]

    ali_height = 3

    addColumnRangeMarkers(mock_ax, ranges, ali_height)

    # Should add 2 patches and 2 text labels
    assert mock_ax.add_patch.call_count == 2
    assert mock_ax.text.call_count == 2


def test_addColumnRangeMarkers_no_labels():
    """Test addColumnRangeMarkers with empty labels."""
    mock_ax = MagicMock()

    ranges = [(1, 3, 'red', ''), (5, 8, 'blue', None)]

    ali_height = 3

    addColumnRangeMarkers(mock_ax, ranges, ali_height)

    # Should add 2 patches but no text labels
    assert mock_ax.add_patch.call_count == 2
    assert mock_ax.text.call_count == 0


def test_addColumnRangeMarkers_single_column():
    """Test addColumnRangeMarkers with single-column ranges."""
    mock_ax = MagicMock()

    ranges = [(1, 1, 'red', 'Point 1'), (5, 5, 'blue', 'Point 2')]

    ali_height = 3

    addColumnRangeMarkers(mock_ax, ranges, ali_height)

    # Should add 2 patches and 2 text labels
    assert mock_ax.add_patch.call_count == 2
    assert mock_ax.text.call_count == 2
