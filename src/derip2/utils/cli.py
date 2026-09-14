"""
Click option groups shared by the ``derip2`` and ``derip2-spectra`` commands.

Both commands take the same alignment input, deRIP algorithm parameters,
output location and logging options. Defining each group once here keeps
their flags, types, defaults and help text in step; each command still
chooses where in its ``--help`` listing a group appears by where it applies
the decorator.
"""

import sys

import click

from derip2.utils.checks import dochecks
from derip2.utils.logs import colored, init_logging

#: ``context_settings`` shared by both commands (``-h`` as well as ``--help``).
HELP_CONTEXT = {'help_option_names': ['-h', '--help']}


def _stack(*decorators):
    """
    Combine several click decorators into one, preserving their order.

    Parameters
    ----------
    *decorators : callable
        Decorators in the order they would be written top to bottom.

    Returns
    -------
    callable
        A decorator applying them all.
    """

    def apply(func):
        """
        Apply the stacked decorators to ``func``.

        Parameters
        ----------
        func : callable
            The click command callback.

        Returns
        -------
        callable
            The decorated callback.
        """
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return apply


#: ``-i/--input``: the multiple sequence alignment.
alignment_input_option = click.option(
    '-i',
    '--input',
    required=True,
    type=str,
    help='Multiple sequence alignment (FASTA, optionally gzipped).',
)

#: The six deRIP algorithm parameters, in the order ``derip2 --help`` lists them.
derip_parameter_options = _stack(
    click.option(
        '-g',
        '--max-gaps',
        type=float,
        default=0.7,
        show_default=True,
        help='Maximum proportion of gapped positions in a column to be tolerated '
        'before forcing a gap in the final deRIP sequence.',
    ),
    click.option(
        '-a',
        '--reaminate',
        is_flag=True,
        default=False,
        show_default=True,
        help='Correct all deamination events independent of RIP context.',
    ),
    click.option(
        '--max-snp-noise',
        type=float,
        default=0.5,
        show_default=True,
        help='Maximum proportion of conflicting SNPs permitted before excluding a '
        'column from RIP/deamination assessment. By default a column with at '
        "least 0.5 'C/T' bases will have 'TpA' positions logged as RIP events.",
    ),
    click.option(
        '--min-rip-like',
        type=float,
        default=0.1,
        show_default=True,
        help='Minimum proportion of deamination events in RIP context '
        "(5' CpA 3' to 5' TpA 3') required for a column to be deRIP'd in the "
        "final sequence. If '--reaminate' is set all deamination events are "
        'corrected.',
    ),
    click.option(
        '--fill-max-gc',
        is_flag=True,
        default=False,
        show_default=True,
        help='By default uncorrected positions in the output sequence are filled '
        'from the sequence with the lowest RIP count. If set, remaining '
        'positions are filled from the sequence with the highest G/C content.',
    ),
    click.option(
        '--fill-index',
        type=int,
        default=None,
        help='Force selection of the alignment row to fill uncorrected positions '
        "from, by row index (from 0). Overrides '--fill-max-gc'.",
    ),
)


def output_options(prefix_default):
    """
    Build the ``-d/--out-dir`` and ``-p/--prefix`` options.

    Parameters
    ----------
    prefix_default : str
        Default output prefix for the command (``'deRIPseq'`` for ``derip2``,
        ``'deRIPspectra'`` for ``derip2-spectra``).

    Returns
    -------
    callable
        A decorator adding both options.
    """
    return _stack(
        click.option(
            '-d',
            '--out-dir',
            type=str,
            default=None,
            help='Directory for output files (default: current directory).',
        ),
        click.option(
            '-p',
            '--prefix',
            default=prefix_default,
            show_default=True,
            help='Prefix for output file names.',
        ),
    )


#: ``--loglevel`` and ``--logfile``.
logging_options = _stack(
    click.option(
        '--loglevel',
        type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
        default='INFO',
        show_default=True,
        help='Set logging level.',
    ),
    click.option('--logfile', default=None, help='Log file path.'),
)


def start_cli(out_dir, logfile, loglevel):
    """
    Run the preamble every command shares.

    Prints the full command line, checks/creates the output directory and log
    file path, and configures logging.

    Parameters
    ----------
    out_dir : str or None
        Requested output directory.
    logfile : str or None
        Requested log file path.
    loglevel : str
        Logging level name.

    Returns
    -------
    tuple of str
        ``(out_dir, logfile)`` as resolved by :func:`derip2.utils.checks.dochecks`.
    """
    print(f'Command line call: {colored.green(" ".join(sys.argv))}\n')
    out_dir, logfile = dochecks(out_dir, logfile)
    init_logging(loglevel=loglevel, logfile=logfile)
    return out_dir, logfile
