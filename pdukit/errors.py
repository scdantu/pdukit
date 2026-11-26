"""The errors pdukit raises.  Catch ``PduError`` to catch them all."""


class PduError(Exception):
    """Base class of every error raised on purpose by pdukit."""


class ConfigError(PduError, ValueError):
    """A setting is unknown, malformed or inconsistent."""


class PpfError(ConfigError):
    """A parameter file could not be read."""


class ResourceError(PduError):
    """The run needs more memory (RAM or GPU) than this machine has free."""
