"""
Read and write SigProfiler-compliant SBS mutation matrices.

A matrix file is tab-separated. The first column is headed ``MutationType`` and
holds the channel labels in canonical order (``A[C>A]A`` and so on); every
remaining column is one sample's counts. This is exactly the format
``sigProfilerPlotting.plotSBS`` and ``SigProfilerAssignment`` expect, so the
files drop straight into those tools when they are installed, while deRIP2 itself
keeps no dependency on them.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from derip2.spectra.channels import SBS96_CHANNELS, SBS192_CHANNELS

if TYPE_CHECKING:  # pragma: no cover
    from derip2.stats.mutation_spectra import SpectraResult

logger = logging.getLogger(__name__)

# Map a requested matrix kind to its canonical channel order.
_CHANNELS_BY_KIND: Dict[str, List[str]] = {
    '96': SBS96_CHANNELS,
    '192': SBS192_CHANNELS,
}


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
        If ``kind`` is not ``'96'`` or ``'192'``.
    """
    if kind == '96':
        return result.sbs96
    if kind == '192':
        return result.sbs192
    raise ValueError(f"kind must be '96' or '192', got {kind!r}")


def write_sbs_matrix(result: 'SpectraResult', path: str, kind: str = '96') -> str:
    """
    Write a spectra result to a SigProfiler-compliant SBS matrix file.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra to serialise.
    path : str
        Output file path.
    kind : {'96', '192'}, optional
        Which matrix to write (default: ``'96'``).

    Returns
    -------
    str
        The path written, for convenience.

    Raises
    ------
    ValueError
        If ``kind`` is not ``'96'`` or ``'192'``.
    """
    channels = _CHANNELS_BY_KIND.get(kind)
    if channels is None:
        raise ValueError(f"kind must be '96' or '192', got {kind!r}")
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

    logger.info('Wrote SBS-%s matrix (%d samples) to %s', kind, matrix.shape[1], path)
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
