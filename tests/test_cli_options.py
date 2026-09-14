"""The shared click option groups reach both commands' --help and defaults."""

from click.testing import CliRunner
import pytest

from derip2.app import main as derip2_main
from derip2.app_spectra import main as spectra_main
from derip2.utils.cli import HELP_CONTEXT

SHARED_FLAGS = (
    '--input',
    '--max-gaps',
    '--reaminate',
    '--max-snp-noise',
    '--min-rip-like',
    '--fill-max-gc',
    '--fill-index',
    '--out-dir',
    '--prefix',
    '--loglevel',
    '--logfile',
)


@pytest.mark.parametrize('command', [derip2_main, spectra_main])
def test_help_lists_every_option(command):
    """Every declared option (shared or not) appears in --help."""
    result = CliRunner().invoke(command, ['--help'])
    assert result.exit_code == 0, result.output
    declared = [
        opt for param in command.params for opt in param.opts if opt.startswith('--')
    ]
    assert set(SHARED_FLAGS) <= set(declared)
    missing = [opt for opt in declared if opt not in result.output]
    assert not missing, missing
    # -h works as an alias for --help on both commands.
    assert command.context_settings == HELP_CONTEXT
    assert CliRunner().invoke(command, ['-h']).exit_code == 0


def test_shared_options_agree_between_commands():
    """Shared options have identical type and default, except the prefix."""
    by_name = {}
    for command in (derip2_main, spectra_main):
        for param in command.params:
            by_name.setdefault(param.name, []).append(param)
    for flag in SHARED_FLAGS:
        name = flag.lstrip('-').replace('-', '_')
        a, b = by_name[name]
        assert a.opts == b.opts
        assert type(a.type) is type(b.type)
        if name != 'prefix':
            assert a.default == b.default, name


def test_prefix_defaults_differ_per_command():
    """derip2 keeps 'deRIPseq' and derip2-spectra keeps 'deRIPspectra'."""
    defaults = {}
    for label, command in (('derip2', derip2_main), ('spectra', spectra_main)):
        defaults[label] = next(p for p in command.params if p.name == 'prefix').default
    assert defaults == {'derip2': 'deRIPseq', 'spectra': 'deRIPspectra'}
    # --reference-tag (spectra only) still defaults to the derip2 prefix.
    ref = next(p for p in spectra_main.params if p.name == 'reference_tag')
    assert ref.default == 'deRIPseq'
