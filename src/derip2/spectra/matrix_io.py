"""
Read and write SigProfiler-compliant SBS mutation matrices.

A matrix file is tab-separated. The first column is headed ``MutationType`` and
holds the channel labels in canonical order (``A[C>A]A`` and so on); every
remaining column is one sample's counts. This is exactly the format
``sigProfilerPlotting.plotSBS`` and ``SigProfilerAssignment`` expect, so the
files drop straight into those tools when they are installed, while deRIP2 itself
keeps no dependency on them.
"""

import json
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from derip2.spectra.channels import (
    DOWNSTREAM_CHANNELS,
    SBS96_CHANNELS,
    SBS192_CHANNELS,
)

if TYPE_CHECKING:  # pragma: no cover
    from derip2.stats.mutation_spectra import SpectraResult

logger = logging.getLogger(__name__)

# Map a requested matrix kind to its canonical channel order.
_CHANNELS_BY_KIND: Dict[str, List[str]] = {
    '96': SBS96_CHANNELS,
    '192': SBS192_CHANNELS,
    'downstream': DOWNSTREAM_CHANNELS,
}

# Which matrix kinds each spectra context may be written as. The trinucleotide
# context owns the SBS-96/192 kinds; the downstream context owns its single folded
# 96-channel kind. This prevents a downstream matrix being mislabelled as SBS-96.
_KINDS_BY_CONTEXT: Dict[str, Tuple[str, ...]] = {
    'trinucleotide': ('96', '192'),
    'downstream': ('downstream',),
}


def _check_kind_context(kind: str, context: str) -> None:
    """
    Validate that a matrix kind is compatible with a spectra context.

    Parameters
    ----------
    kind : str
        The requested matrix kind (``'96'``, ``'192'`` or ``'downstream'``).
    context : str
        The spectra context (``'trinucleotide'`` or ``'downstream'``).

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or is not valid for ``context``.
    """
    allowed = _KINDS_BY_CONTEXT.get(context)
    if allowed is None:
        raise ValueError(f'Unknown spectra context: {context!r}')
    if kind not in allowed:
        raise ValueError(
            f'kind {kind!r} is not valid for the {context!r} context; '
            f'valid kinds are {allowed}'
        )


def _matrix_for_kind(result: 'SpectraResult', kind: str) -> np.ndarray:
    """
    Select the count matrix for a matrix kind.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    kind : {'96', '192'}
        Which matrix to return.

    Returns
    -------
    numpy.ndarray
        The ``(n_channels, n_samples)`` count matrix.

    Raises
    ------
    ValueError
        If ``kind`` is not ``'96'``, ``'192'`` or ``'downstream'``.
    """
    if kind == '96':
        return result.sbs96
    if kind == '192':
        return result.sbs192
    if kind == 'downstream':
        # The downstream matrix is stored in the sbs96 field (see SpectraResult).
        return result.sbs96
    raise ValueError(f"kind must be '96', '192' or 'downstream', got {kind!r}")


def write_sbs_matrix(result: 'SpectraResult', path: str, kind: str = '96') -> str:
    """
    Write a spectra result to a SigProfiler-compliant SBS matrix file.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra to serialise.
    path : str
        Output file path.
    kind : {'96', '192', 'downstream'}, optional
        Which matrix to write (default: ``'96'``). Must be compatible with
        ``result.context`` (``'96'``/``'192'`` for the trinucleotide context,
        ``'downstream'`` for the downstream context).

    Returns
    -------
    str
        The path written, for convenience.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or is not valid for ``result.context``.
    """
    channels = _CHANNELS_BY_KIND.get(kind)
    if channels is None:
        raise ValueError(f"kind must be '96', '192' or 'downstream', got {kind!r}")
    # Guard against a downstream matrix being written under an SBS-96 label (or
    # vice versa): the kind must match the context that produced the result.
    _check_kind_context(kind, getattr(result, 'context', 'trinucleotide'))
    matrix = _matrix_for_kind(result, kind)

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('MutationType\t' + '\t'.join(result.sample_names) + '\n')
        for row, label in enumerate(channels):
            counts = matrix[row]
            # Integer counts are written without a trailing ``.0`` so the files
            # match hand-checked fixtures; fractional (probability-weighted)
            # counts keep six significant figures.
            cells = [
                str(int(v)) if float(v).is_integer() else f'{v:.6g}' for v in counts
            ]
            handle.write(label + '\t' + '\t'.join(cells) + '\n')

    logger.info('Wrote %s matrix (%d samples) to %s', kind, matrix.shape[1], path)
    return path


def write_matrix_metadata(result: 'SpectraResult', path: str, kind: str = '96') -> str:
    """
    Write a JSON sidecar describing how a matrix file was produced.

    The SBS matrix files are kept as clean ``MutationType`` tab-separated tables so
    third-party tools (e.g. SigProfilerPlotting / SigProfilerAssignment) can read
    them directly -- those tools reject in-file comment lines, so the provenance
    (which sequence context and calling method produced the matrix) is written to a
    companion JSON file instead of into the matrix itself.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra the matrix was written from.
    path : str
        Output path for the JSON sidecar.
    kind : {'96', '192', 'downstream'}, optional
        The matrix kind the sidecar documents (default: ``'96'``).

    Returns
    -------
    str
        The path written, for convenience.
    """
    matrix = _matrix_for_kind(result, kind)
    meta = {
        'context': getattr(result, 'context', 'trinucleotide'),
        'method': result.method,
        'kind': kind,
        'n_channels': int(matrix.shape[0]),
        'sample_names': list(result.sample_names),
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    logger.info('Wrote matrix metadata sidecar to %s', path)
    return path


def read_sbs_matrix(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Read a SigProfiler-compliant SBS matrix file.

    Parameters
    ----------
    path : str
        Path to a tab-separated matrix file with a ``MutationType`` first column.

    Returns
    -------
    tuple
        ``(channels, sample_names, matrix)`` where ``channels`` is the list of
        row labels, ``sample_names`` the count-column headers and ``matrix`` a
        ``(n_channels, n_samples)`` float array.
    """
    channels: List[str] = []
    values: List[List[float]] = []
    with open(path, encoding='utf-8') as handle:
        header = handle.readline().rstrip('\n').split('\t')
        sample_names = header[1:]
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            channels.append(fields[0])
            values.append([float(v) for v in fields[1:]])
    matrix = np.asarray(values, dtype=np.float64) if values else np.zeros((0, 0))
    return channels, sample_names, matrix
