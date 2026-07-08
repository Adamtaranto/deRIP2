"""
Golden regression tests for the deRIP2 pipeline.

These tests lock the *semantic* output of the pipeline (consensus, masked
alignment, RIP counts, corrected positions, and markup) so that the NumPy
vectorisation refactor can be verified to produce equivalent results.

Golden files live under ``tests/data/golden/``. To (re)generate them after an
intentional, reviewed behaviour change, run::

    DERIP_REGEN=1 pytest tests/test_regression.py

Ordering inside ``markupdict`` lists is not significant, so entries are sorted
before comparison.
"""

import json
import logging
import os

import pytest

from derip2.derip import DeRIP

# Silence logging noise during tests
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
GOLDEN_DIR = os.path.join(HERE, 'data', 'golden')
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')

# Each case is (name, DeRIP kwargs). mintest.fa is small (6x35) and exercises
# the core correction paths under both reaminate settings.
CASES = [
    ('mintest_default', {}),
    ('mintest_reaminate', {'reaminate': True}),
    ('mintest_maxgc', {'fill_max_gc': True}),
]


def _serialize(d: DeRIP) -> dict:
    """Produce a canonical, order-insensitive dict of a completed DeRIP run."""
    # Masked alignment as (id, sequence) pairs
    masked = [(rec.id, str(rec.seq)) for rec in d.masked_alignment]

    # RIP counts collapsed to the semantically meaningful totals per row
    rip_counts = [
        {
            'id': d.rip_counts[i].SeqID,
            'rip': d.rip_counts[i].RIPcount + d.rip_counts[i].revRIPcount,
            'nonrip': d.rip_counts[i].nonRIPcount,
            'gc': round(d.rip_counts[i].GC, 4),
        }
        for i in range(len(d.rip_counts))
    ]

    # markup: sorted tuples per category (order within a category is irrelevant)
    markup = {
        cat: sorted(tuple(pos) for pos in positions)
        for cat, positions in d.markupdict.items()
    }

    return {
        'consensus': str(d.consensus.seq),
        'gapped_consensus': str(d.gapped_consensus.seq),
        'masked_alignment': masked,
        'rip_counts': rip_counts,
        'corrected_positions': sorted(d.corrected_positions.keys()),
        'markup': markup,
        'fill_index': d.fill_index,
    }


def _run(kwargs: dict) -> dict:
    d = DeRIP(MINTEST, **kwargs)
    d.calculate_rip()
    return _serialize(d)


@pytest.mark.parametrize('name,kwargs', CASES)
def test_pipeline_matches_golden(name, kwargs):
    """The pipeline output must match the committed golden reference."""
    result = _run(kwargs)
    golden_path = os.path.join(GOLDEN_DIR, f'{name}.json')

    if os.environ.get('DERIP_REGEN') or not os.path.exists(golden_path):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, 'w') as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        pytest.skip(f'Regenerated golden file: {golden_path}')

    with open(golden_path) as fh:
        golden = json.load(fh)

    # JSON round-trips tuples to lists; normalise result the same way
    result = json.loads(json.dumps(result))
    assert result == golden
