"""Test helper: the settings that ``pdu run`` builds from a command line."""

from pdukit.cli.arguments import parse_run


def parse_args(argv):
    """Return the `Options` for these ``pdu run`` arguments."""
    return parse_run(argv)[0]
