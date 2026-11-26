"""What a run writes (ResultsWriter) and reads back (ResultsReader)."""

# Layout of results.json.  Raise it when a key changes meaning or goes away
# (adding keys does not need it).  Files written before it was recorded have
# no ``schema_version`` key and read as version 0.
SCHEMA_VERSION = 1
