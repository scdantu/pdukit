"""PDU (Protein Dynamical Unit, the *dynamical* concept).

See the top-level README.

extract/    per-domain/trajectory -> PDU candidates (the D9 pipeline)
analyze/    cross-domain -> does the PDU vocabulary stay finite? (K3)
extractor.py  PDUExtractor  -- class facade over extract/
analyzer.py   PDUAnalyzer   -- class facade over analyze/
cli.py        argv dispatcher used by ../../bin/pdu
"""
