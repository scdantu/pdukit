"""Allow ``python -m pdukit``."""

import sys

from .cli import main

sys.exit(main())
