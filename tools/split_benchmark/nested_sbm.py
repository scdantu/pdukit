"""Run the published graph-tool nested weighted SBM, separately from the venv."""
import argparse
import json
import os
import time
from pathlib import Path

import graph_tool.all as gt
import numpy as np

gt.openmp_set_num_threads(1)
ap=argparse.ArgumentParser();ap.add_argument('--binary',action='store_true');args=ap.parse_args()
OUT=Path(os.environ.get('PDU_SPLIT_OUT',str(Path(__file__).resolve().parent)))
datasets=json.loads((OUT/'datasets.json').read_text())
results=[]
for i,ds in enumerate(datasets):
    R=np.array(ds['rho']);n=len(R)
    for distribution in (['binary_theta'] if args.binary else ['real-normal','real-exponential']):
        started=time.perf_counter(); best=None; trials=[]
        for seed in range(3):
            gt.seed_rng(seed+77)
            g=gt.Graph(directed=True);g.add_vertex(n)
            w=g.new_ep('double')
            rows,cols=np.where((R>ds['theta']) & ~np.eye(n,dtype=bool) if args.binary else ~np.eye(n,dtype=bool))
            # rho[target,source]; graph edge source -> target.
            if args.binary:
                g.add_edge_list(np.c_[cols,rows])
                state=gt.minimize_nested_blockmodel_dl(g,base_state_args=dict(deg_corr=True))
            else:
                g.add_edge_list(np.c_[cols,rows,R[rows,cols]],eprops=[w])
                state=gt.minimize_nested_blockmodel_dl(g,base_state=gt.WeightedBlockState,
                    base_state_args=dict(rec=[w],rec_types=[distribution],deg_corr=False))
            for _ in range(5):state.multiflip_mcmc_sweep(niter=5,beta=np.inf)
            dl=float(state.entropy());trials.append(dl)
            if best is None or dl<best['description_length']:
                levels=[]
                for j in range(len(state.get_levels())):
                    labels=np.asarray(state.project_level(j).get_blocks().a).astype(int)
                    groups=[np.where(labels==k)[0].tolist() for k in np.unique(labels)]
                    if not levels or groups!=levels[-1]:levels.append(groups)
                    if len(groups)==1:break
                best=dict(description_length=dl,levels=levels,leaves=levels[0],
                          nodes=[{'residues':g} for level in levels for g in level])
        results.append(dict(dataset=ds['id'],kind=ds['kind'],method='nested_'+distribution,
                         seconds=time.perf_counter()-started,trials=trials,**best))
    (OUT/'sbm_results.json').write_text(json.dumps(results))
    print(i+1,ds['id'],[(r['method'],sorted(map(len,r['leaves']),reverse=True)) for r in results[-2:]],flush=True)
