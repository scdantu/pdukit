"""Allow ``python -m pdukit``."""

import sys

from pdukit.cli.main import main

sys.exit(main())
