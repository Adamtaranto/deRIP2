"""
Tests for SBS / downstream matrix I/O and the metadata sidecar.

These pin the round-trip of the ``MutationType`` tab-separated matrix files, the
kind/context compatibility guard (so a downstream matrix can never be written
under an SBS-96 label), and the JSON provenance sidecar.
"""

import json
import logging
import os

import numpy as np
import pytest

from derip2.spectra.channels import DOWNSTREAM_CHANNELS, SBS96_CHANNELS
from derip2.spectra.matrix_io import (
    read_sbs_matrix,
    write_matrix_metadata,
    write_sbs_matrix,
)
from derip2.stats import compute_spectra

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')


def _classes(seqs):
    """Wrap sequence strings as an object exposing ``arr`` like a classification."""
    from types import SimpleNamespace

    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from derip2.aln_ops import alignment_to_array

    aln = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )
    return SimpleNamespace(arr=alignment_to_array(aln))


def _downstream_result():
    """A small downstream spectra result with at least one event."""
    return compute_spectra(_classes(['TCAGT', 'TTAGT']), 'TCAGT', context='downstream')


def test_downstream_matrix_round_trip(tmp_path):
    """A downstream matrix writes with its own labels and reads back identically."""
    result = _downstream_result()
    path = str(tmp_path / 'ds.DSC96.txt')
    write_sbs_matrix(result, path, kind='downstream')

    channels, sample_names, matrix = read_sbs_matrix(path)
    assert channels == DOWNSTREAM_CHANNELS
    assert sample_names == result.sample_names
    assert np.array_equal(matrix, result.sbs96)


def test_downstream_labels_are_not_sbs96(tmp_path):
    """The written downstream labels are the distinct [REF>ALT]d1d2 form."""
    result = _downstream_result()
    path = str(tmp_path / 'ds.DSC96.txt')
    write_sbs_matrix(result, path, kind='downstream')
    channels, _names, _matrix = read_sbs_matrix(path)
    assert channels[0].startswith('[')
    assert not set(channels) & set(SBS96_CHANNELS)


def test_downstream_result_rejects_sbs96_kind(tmp_path):
    """A downstream result cannot be written as an SBS-96/192 matrix."""
    result = _downstream_result()
    with pytest.raises(ValueError):
        write_sbs_matrix(result, str(tmp_path / 'x.txt'), kind='96')
    with pytest.raises(ValueError):
        write_sbs_matrix(result, str(tmp_path / 'x.txt'), kind='192')


def test_trinucleotide_result_rejects_downstream_kind(tmp_path):
    """A trinucleotide result cannot be written under the downstream kind."""
    result = compute_spectra(_classes(['ACGTA', 'ATGTA']), 'ACGTA')
    with pytest.raises(ValueError):
        write_sbs_matrix(result, str(tmp_path / 'x.txt'), kind='downstream')


def test_metadata_sidecar_contents(tmp_path):
    """The sidecar records the context, method, kind, channel count and samples."""
    result = _downstream_result()
    path = str(tmp_path / 'ds.DSC96.meta.json')
    write_matrix_metadata(result, path, kind='downstream')
    with open(path) as handle:
        meta = json.load(handle)
    assert meta == {
        'context': 'downstream',
        'method': 'baseline',
        'kind': 'downstream',
        'n_channels': 96,
        'sample_names': list(result.sample_names),
    }
