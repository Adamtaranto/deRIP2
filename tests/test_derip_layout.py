"""Guards for the DeRIP mixin composition and the documented API surface."""

from pathlib import Path
import re

from derip2._derip.base import DeRIPBase
from derip2._derip.plots import PlotsReportsMixin
from derip2._derip.rip import RIPCorrectionMixin
from derip2._derip.spectra import SpectraMixin
from derip2._derip.stats import StatsMixin
from derip2.derip import DeRIP


def test_derip_composes_every_mixin():
    """DeRIP is the public facade over the five concern classes."""
    for cls in (
        RIPCorrectionMixin,
        StatsMixin,
        SpectraMixin,
        PlotsReportsMixin,
        DeRIPBase,
    ):
        assert cls in DeRIP.__mro__
    # The base class carries the shared state and comes last so mixins can
    # rely on its guards.
    assert DeRIP.__mro__.index(DeRIPBase) > DeRIP.__mro__.index(StatsMixin)


def test_documented_members_exist():
    """Every member listed for the API page resolves on the composed class."""
    doc = Path(__file__).resolve().parents[1] / 'docs' / 'api-docs' / 'deRIP_API.md'
    text = doc.read_text(encoding='utf-8')
    block = text.split('members:', 1)[1].split(':::', 1)[0]
    names = re.findall(r'^\s+- (\w+)\s*$', block, re.M)
    assert len(names) >= 30
    missing = [n for n in names if not hasattr(DeRIP, n)]
    assert not missing, missing


def test_colored_views_are_lazy_properties_on_the_composed_class():
    """The ANSI views stay properties (with setters) after the split."""
    for name in ('colored_consensus', 'colored_alignment', 'colored_masked_alignment'):
        prop = getattr(DeRIP, name)
        assert isinstance(prop, property) and prop.fset is not None
