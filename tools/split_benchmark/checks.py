"""Scientific sanity checks for the offline comparison."""
import numpy as np
from benchmark import cohesion, tree

from pdukit.cog.units.cohesion import cohesion as production

rng=np.random.default_rng(5)
R=rng.uniform(size=(12,12));np.fill_diagonal(R,0)
assert abs(cohesion(R,[0,1,2])[2]-production(R,[0,1,2])['cohesion'])<1e-12
for method in ['spectral_ncut','directed_modularity','markov_1','markov_4','markov_16']:
    R=np.full((12,12),.1);R[:6,:6]=.8;R[6:,6:]=.8;np.fill_diagonal(R,0)
    t=tree(R,method,nnull=39)
    assert sorted(map(len,t['leaves']))==[6,6],method
    R=np.full((12,12),.4);np.fill_diagonal(R,0)
    assert len(tree(R,method,nnull=39)['leaves'])==1,method
print('Passed: production AUC agreement; two-block recovery and homogeneous no-split for all five variants.')
