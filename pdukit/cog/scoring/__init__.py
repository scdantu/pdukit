"""Interface-cost / flux-dimension diagnostics for candidate PDUs (v3.0 line).

WHAT THIS MODULE IS FOR
-----------------------
The v3.0 reference asks a different question from the established extractor.
The extractor builds candidates by rotating global tICA modes into a localised
basis.  v3.0 Section 10 step 2 explicitly forbids that route for *discovery*
("Do not obtain them by rotating global slow modes").  This module therefore
never touches a global mode.  It starts from residues and asks, for a proposed
residue set S:

    given the unit's own present state, how many INDEPENDENT CHANNELS does the
    rest of the protein need in order to say anything further about the unit's
    future?

That number -- not the strength of the coupling -- is the proposed defining
property of a PDU.  A unit may be strongly coupled to its environment and still
be a unit, provided the coupling arrives through few channels.  Requiring the
coupling to be *small* instead would reduce the criterion to near-independence,
i.e. to the iVAMPnets assumption that v3.0 deliberately drops.

RELATION TO THE OPERATOR PICTURE
--------------------------------
Section 3 of the reference splits the transfer operator at lag tau into

    A = P K P,  B = P K R,  C = R K P,  D = R K R

with P the projection onto resolved (unit) observables.  The environment's
back-action on the unit is the self-energy  B (zI - D)^{-1} C.  Lyna's
boundary-flux proposal is ||(I - P) L P||, i.e. the NORM of the cross block.

In the linear-Gaussian case the two pictures coincide exactly, and the
rank-resolved version of that norm is computable without ever forming a large
operator: the singular values of the cross block, after whitening, ARE the
canonical correlations between

    Ytil = unit future  residualised on unit present
    Ztil = environment present residualised on unit present

So `flux_spectrum` below returns the singular-value spectrum of the boundary
flux.  ||.||  is its first singular value; what we report instead is how many
singular values there are.

WHAT THIS MODULE IS NOT
-----------------------
* Not a validity threshold.  Every number here is a descriptor with a stated
  null; nothing is pass/fail.
* Not causal.  Canonical correlations are observational.  A channel is a
  predictive channel, not a demonstrated mechanical force.
* Not nonlinear.  The estimator is linear-Gaussian by choice, because it is
  exactly solvable, has an analytic null, and cannot silently absorb the whole
  protein into a flexible encoder.  A nonlinear version is a later step and
  must be charged for its capacity.
* Not a slow-mode estimator.  At mdCATH's 1 ns spacing nothing faster than
  ~10 ns is resolved, and rare transitions are not sampled.  Everything here is
  a statement at the declared resolution R = (dt, tau, features).

Numpy only, deliberately: no scipy, no torch, matching the repo's existing
policy of not depending on mdtraj/biopython in the extract layer.
"""
