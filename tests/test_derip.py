import os
import tempfile

import pytest

from derip2.derip import DeRIP

"""
Test module for DeRIP class in derip2.derip module.

This module contains tests for CRI and GC content-related methods
in the DeRIP class using pytest.
"""


# Import the DeRIP class from parent module using relative import


class TestDeRIPMethods:
    """Test class for DeRIP methods."""

    @pytest.fixture
    def mock_alignment_file(self):
        """Create a temporary mock alignment file for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=False) as f:
            # Create sequences with different CRI values:
            # 1. High CRI - lots of TpA dinucleotides
            # 2. Medium CRI - balanced dinucleotides
            # 3. Low CRI - more CpA/TpG than TpA
            # 4. Another with different CRI value
            f.write(
                '>seq1\nACGTACGTACGTACGT\n'  # Balanced dinucleotides
                '>seq2\nACGACGACGACGACGA\n'  # No TpA
                '>seq3\nTAGATAGATAGATAGA\n'  # High TpA, fewer other dinucleotides
                '>seq4\nTATATATATATATATA\n'  # Very high TpA/ApT ratio
                '>seq5\nCACACACACACATGTG\n'  # High CpA/TpG, different CRI
            )
            temp_filename = f.name

        yield temp_filename

        # Cleanup the temporary file after test
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

    @pytest.fixture
    def derip_instance(self, mock_alignment_file):
        """Create a DeRIP instance for testing."""
        return DeRIP(mock_alignment_file)

    @pytest.fixture
    def seq_with_known_dinucleotides(self):
        """Create a sequence with known dinucleotide counts."""
        return 'TACAGTGCAT'

    def test_calculate_dinucleotide_frequency(
        self, derip_instance, seq_with_known_dinucleotides
    ):
        """Test that dinucleotide frequencies are calculated correctly."""
        result = derip_instance.calculate_dinucleotide_frequency(
            seq_with_known_dinucleotides
        )

        # Check expected counts
        assert result['TpA'] == 1
        assert result['ApT'] == 1
        assert result['CpA'] == 2
        assert result['TpG'] == 1
        assert result['ApC'] == 1
        assert result['GpT'] == 1

    def test_calculate_dinucleotide_frequency_with_gaps(self, derip_instance):
        """Test dinucleotide counting with a sequence containing gaps."""
        # After removing gaps, this becomes "TACAGT"
        sequence = 'TA-CA-GT'
        result = derip_instance.calculate_dinucleotide_frequency(sequence)

        assert result['TpA'] == 1
        assert result['ApT'] == 0
        assert result['CpA'] == 1
        assert result['TpG'] == 0
        assert result['ApC'] == 1
        assert result['GpT'] == 1

    def test_calculate_dinucleotide_frequency_case_insensitive(self, derip_instance):
        """Test that dinucleotide counting is case-insensitive."""
        lower_seq = 'tacagtgcat'
        result = derip_instance.calculate_dinucleotide_frequency(lower_seq)

        # Should match the same counts as the uppercase version
        assert result['TpA'] == 1
        assert result['ApT'] == 1
        assert result['CpA'] == 2
        assert result['TpG'] == 1
        assert result['ApC'] == 1
        assert result['GpT'] == 1

    def test_calculate_dinucleotide_frequency_empty_sequence(self, derip_instance):
        """Test dinucleotide counting with an empty sequence."""
        result = derip_instance.calculate_dinucleotide_frequency('')

        # All counts should be zero
        assert sum(result.values()) == 0

    def test_calculate_cri(self, derip_instance, seq_with_known_dinucleotides):
        """Test that CRI is calculated correctly."""
        cri, pi, si = derip_instance.calculate_cri(seq_with_known_dinucleotides)

        # PI = TpA / ApT = 1 / 1 = 1.0
        # SI = (CpA + TpG) / (ApC + GpT) = (2 + 1) / (1 + 1) = 3 / 2 = 1.5
        # CRI = PI - SI = 1 - 1.5 = -0.5
        assert pi == 1.0
        assert si == 1.5
        assert cri == -0.5

    def test_calculate_cri_for_all(self, derip_instance):
        """Test that CRI values are calculated for all sequences."""
        alignment = derip_instance.calculate_cri_for_all()

        # Check that all sequences have CRI annotations
        for record in alignment:
            assert hasattr(record, 'annotations')
            assert 'CRI' in record.annotations
            assert 'PI' in record.annotations
            assert 'SI' in record.annotations

            # Verify the descriptions were updated
            assert f'CRI={record.annotations["CRI"]:.4f}' in record.description

    def test_get_cri_values(self, derip_instance):
        """Test retrieving CRI values for all sequences."""
        # Get CRI values (should calculate them if not done already)
        cri_values = derip_instance.get_cri_values()

        # Check the structure and content of the returned data
        assert isinstance(cri_values, list)
        assert (
            len(cri_values) == 5
        )  # Should have 5 sequences - updated to match mock_alignment_file

        # Each item should be a dictionary with id, CRI, PI, and SI
        for item in cri_values:
            assert set(item.keys()) == {'id', 'CRI', 'PI', 'SI'}

        # Check that annotations were added to records
        for record in derip_instance.alignment:
            assert 'CRI' in record.annotations

    def test_sort_by_cri(self, derip_instance):
        """Test sorting alignment by CRI values."""
        # Calculate CRI values first
        derip_instance.calculate_cri_for_all()

        # Get original CRI order
        original_cri_values = [
            record.annotations['CRI'] for record in derip_instance.alignment
        ]

        # Sort by CRI (descending by default)
        sorted_alignment = derip_instance.sort_by_cri()

        # Check that the alignment is sorted correctly
        sorted_cri_values = [record.annotations['CRI'] for record in sorted_alignment]
        assert sorted_cri_values == sorted(original_cri_values, reverse=True)

        # Test ascending order
        sorted_alignment_asc = derip_instance.sort_by_cri(descending=False)
        sorted_cri_values_asc = [
            record.annotations['CRI'] for record in sorted_alignment_asc
        ]
        assert sorted_cri_values_asc == sorted(original_cri_values)

    def test_summarize_cri(self, derip_instance):
        """Test CRI summary table generation."""
        # Calculate CRI values first
        derip_instance.calculate_cri_for_all()

        # Get summary
        summary = derip_instance.summarize_cri()

        # Check that the summary is a non-empty string
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Check that it contains key headers
        assert 'id' in summary
        assert 'CRI' in summary
        assert 'PI' in summary
        assert 'SI' in summary

    def test_filter_by_cri(self, derip_instance):
        """Test filtering alignment by CRI threshold."""
        import sys

        # Calculate CRI values first
        derip_instance.calculate_cri_for_all()

        # Get CRI values
        cri_values = [record.annotations['CRI'] for record in derip_instance.alignment]
        max_cri = max(cri_values)
        min_cri = min(cri_values)

        print('\nDEBUG: All CRI values:', cri_values, file=sys.stderr)
        print('DEBUG: min_cri:', min_cri, ', max_cri:', max_cri, file=sys.stderr)

        # Print sequence IDs with their CRI values
        for idx, record in enumerate(derip_instance.alignment):
            print(
                f'DEBUG: Seq {idx} ({record.id}): CRI={record.annotations["CRI"]}',
                file=sys.stderr,
            )

        # Test keeping all sequences
        filtered = derip_instance.filter_by_cri(min_cri=min_cri)
        assert len(filtered) == len(derip_instance.alignment)

        # Test filtering out all but the highest CRI sequence
        threshold = max_cri - 0.0001
        with pytest.warns(
            UserWarning, match='Only 1 sequence remains after CRI filtering'
        ):
            filtered = derip_instance.filter_by_cri(min_cri=threshold)
        assert len(filtered) >= 1
        assert all(r.annotations['CRI'] >= threshold for r in filtered)

        # Test filtering out all sequences should raise ValueError
        with pytest.raises(ValueError, match='No sequences remain'):
            derip_instance.filter_by_cri(min_cri=max_cri + 0.1)

        # Test in-place filtering - use a lower threshold to ensure at least 2 sequences remain
        original_len = len(derip_instance.alignment)

        # Sort CRI values to find a threshold that keeps at least 2 sequences
        sorted_cri = sorted(cri_values)
        print('\nDEBUG: Sorted CRI values:', sorted_cri, file=sys.stderr)

        if len(sorted_cri) >= 3:
            # Use the second lowest CRI value as threshold, which ensures at least 2 sequences remain
            threshold = sorted_cri[1] + 0.0001
            print(
                f'DEBUG: Using second lowest CRI {sorted_cri[1]} + 0.0001 as threshold',
                file=sys.stderr,
            )
        else:
            # If we have only 2 sequences, use the minimum CRI
            threshold = min_cri
            print(f'DEBUG: Using minimum CRI {threshold} as threshold', file=sys.stderr)

        # Count how many sequences would remain with this threshold
        would_remain = sum(1 for v in cri_values if v >= threshold)
        print(
            f'DEBUG: {would_remain} sequences would remain with threshold {threshold}',
            file=sys.stderr,
        )

        # Apply the filter not in-place
        filtered = derip_instance.filter_by_cri(min_cri=threshold, inplace=False)
        print(
            f'DEBUG: Non-inplace filter would keep {len(filtered)} sequences',
            file=sys.stderr,
        )

        # Apply the filter in-place
        derip_instance.filter_by_cri(min_cri=threshold, inplace=True)

        # Check how many actually remained
        remaining = len(derip_instance.alignment)
        print(
            f'DEBUG: {remaining} sequences actually remained after filtering',
            file=sys.stderr,
        )

        # Check that the number of sequences remaining is the same with both in-place and non-inplace filters
        assert len(filtered) == remaining == 2

        # If only 1 sequence remains, this is strange - check the threshold again
        if remaining == 1:
            for idx, record in enumerate(derip_instance.alignment):
                print(
                    f'DEBUG: After filter - Seq {idx} CRI={record.annotations["CRI"]}',
                    file=sys.stderr,
                )

        assert len(derip_instance.alignment) < original_len
        assert len(derip_instance.alignment) == 2
        assert all(r.annotations['CRI'] >= threshold for r in derip_instance.alignment)

    def test_get_gc_content(self, derip_instance):
        """Test GC content calculation."""
        # Ensure the method exists
        assert hasattr(derip_instance, 'get_gc_content'), (
            'get_gc_content method is missing'
        )

        gc_values = derip_instance.get_gc_content()

        # Check that the result has the expected structure
        assert isinstance(gc_values, list)
        assert len(gc_values) == 5  # Now we have 5 sequences

        for item in gc_values:
            assert set(item.keys()) == {'id', 'GC_content'}
            assert 0 <= item['GC_content'] <= 1  # GC content should be between 0 and 1

        # Check that annotations were added to sequences
        for record in derip_instance.alignment:
            assert hasattr(record, 'annotations')
            assert 'GC_content' in record.annotations

        # Verify exact GC content for first sequence only
        assert (
            derip_instance.alignment[0].annotations['GC_content'] == 0.5
        )  # seq1 is ACGTACGTACGTACGT (50% GC)

    def test_filter_by_gc(self, derip_instance):
        """Test filtering alignment by GC content threshold."""
        # Ensure the method exists
        assert hasattr(derip_instance, 'filter_by_gc'), 'filter_by_gc method is missing'

        # Calculate GC content first to ensure annotations exist
        derip_instance.get_gc_content()

        # Print the actual GC values for debugging
        import sys

        for idx, record in enumerate(derip_instance.alignment):
            print(
                f'DEBUG: Seq {idx} ({record.id}): GC={record.annotations["GC_content"]}',
                file=sys.stderr,
            )

        # Test with threshold that should keep all sequences with GC ≥ 0.25
        filtered = derip_instance.filter_by_gc(min_gc=0.25)
        assert len(filtered) == 4  # 4 sequences have GC ≥ 0.25 (seq1, seq2, seq3, seq5)

        # Test with threshold that should filter out sequences with GC < 0.4
        filtered = derip_instance.filter_by_gc(min_gc=0.4)
        assert len(filtered) == 3  # 3 sequences have GC ≥ 0.4 (seq1, seq2, seq5)
        assert all(r.annotations['GC_content'] >= 0.4 for r in filtered)

        # Test with threshold too high (beyond any sequence's GC)
        with pytest.raises(ValueError, match='No sequences remain'):
            derip_instance.filter_by_gc(min_gc=0.7)  # Use 0.7 since highest GC is 0.5

        # Test in-place filter
        derip_instance.filter_by_gc(min_gc=0.4, inplace=True)
        assert len(derip_instance.alignment) == 3  # Should keep seq1, seq2, seq5
        assert all(r.annotations['GC_content'] >= 0.4 for r in derip_instance.alignment)

        # Test invalid min_gc values
        with pytest.raises(ValueError, match='min_gc must be between 0.0 and 1.0'):
            derip_instance.filter_by_gc(min_gc=-0.1)

        with pytest.raises(ValueError, match='min_gc must be between 0.0 and 1.0'):
            derip_instance.filter_by_gc(min_gc=1.1)

    def test_warnings_with_few_sequences(self, derip_instance):
        """Test warnings when filtering results in fewer than 2 sequences."""
        # Calculate needed values
        derip_instance.get_gc_content()
        derip_instance.get_cri_values()

        # Find the highest GC content sequence
        records = list(derip_instance.alignment)
        highest_gc = max(records, key=lambda r: r.annotations['GC_content'])
        highest_gc_value = highest_gc.annotations['GC_content']

        # Filter to keep only one sequence and check warning
        with pytest.warns(UserWarning, match='Only 1 sequence remains'):
            filtered = derip_instance.filter_by_gc(min_gc=highest_gc_value)
            assert len(filtered) == 1

        # Find the highest CRI sequence
        highest_cri = max(records, key=lambda r: r.annotations['CRI'])
        highest_cri_value = highest_cri.annotations['CRI']

        # Filter to keep only one sequence and check warning
        with pytest.warns(UserWarning, match='Only 1 sequence remains'):
            filtered = derip_instance.filter_by_cri(min_cri=highest_cri_value)
            assert len(filtered) == 1

    def test_edge_cases(self, derip_instance):
        """Test edge cases for CRI and GC content methods."""
        # Test CRI calculation with a sequence that would cause division by zero
        # (no ApT dinucleotides and/or no ApC+GpT dinucleotides)

        # Sequence with no ApT dinucleotides (PI denominator = 0)
        no_apt_seq = 'CCCCGGGGCCCCGGGG'
        cri, pi, si = derip_instance.calculate_cri(no_apt_seq)
        assert pi == 0  # Should handle division by zero gracefully

        # Sequence with no ApC or GpT dinucleotides (SI denominator = 0)
        no_apc_gpt_seq = 'TATATATATATATAT'
        cri, pi, si = derip_instance.calculate_cri(no_apc_gpt_seq)
        assert si == 0  # Should handle division by zero gracefully

        # Empty sequence
        empty_seq = ''
        cri, pi, si = derip_instance.calculate_cri(empty_seq)
        assert (
            cri == 0 and pi == 0 and si == 0
        )  # Should handle empty sequence gracefully

        # Single nucleotide sequence (can't have dinucleotides)
        single_base = 'A'
        cri, pi, si = derip_instance.calculate_cri(single_base)
        assert cri == 0 and pi == 0 and si == 0  # Should handle single base gracefully
