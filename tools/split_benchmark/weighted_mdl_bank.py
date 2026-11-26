"""Restricted nested weighted SBM search using graph-tool's exact entropy.

The native weighted move optimizer crashes on this host. This evaluates an
explicit candidate bank, NOT the published unrestricted optimizer.
"""
import json
import os
import time
from pathlib import Path

import graph_tool as gt
import numpy as np
from graph_tool.inference import NestedBlockState, WeightedBlockState

OUT=Path(os.environ.get('PDU_SPLIT_OUT',str(Path(__file__).resolve().parent)))
datasets=json.loads((OUT/'datasets.json').read_text())
spectral=json.loads((OUT/'spectral_results.json').read_text())
spectral+=json.loads((OUT/'proposal_trees.json').read_text())
def encode(groups,n):
    labels=np.empty(n,dtype=int)
    for k,g in enumerate(groups):labels[g]=k
    return labels

def bank(tr,n):
    bysupport={tuple(sorted(r['residues'])):r for r in tr['nodes']}
    def at(S,depth):
        node=bysupport.get(tuple(sorted(S)),{})
        if depth==0 or not node.get('split'):return [S]
        return sum([at(ch,depth-1) for ch in node['children']],[])
    maxdepth=max(r.get('depth',0) for r in tr['nodes'])
    levels=[at(list(range(n)),d) for d in range(maxdepth+1)]
    for finest in range(1,len(levels)):
        chosen=levels[:finest+1][::-1];bs=[encode(chosen[0],n)]
        prev=chosen[0]
        for groups in chosen[1:]:
            labels=encode(groups,n)
            b=np.array([labels[g[0]] for g in prev],dtype=int)
            if len(set(b))<len(prev):bs.append(b)
            prev=groups
        yield bs,chosen

results=[]
for i,ds in enumerate(datasets):
    candidates=[r for r in spectral if r['dataset']==ds['id']]
    if not candidates:continue
    R=np.array(ds['rho']);n=len(R);g=gt.Graph(directed=True);g.add_vertex(n)
    w=g.new_ep('double');rows,cols=np.where(~np.eye(n,dtype=bool))
    g.add_edge_list(np.c_[cols,rows,R[rows,cols]],eprops=[w])
    candidate_bank=[('no_split',[np.zeros(n,dtype=int)],[[list(range(n))]])]
    seen=set()
    for tr in candidates:
        for bs,levels in bank(tr,n):
            key=str([b.tolist() for b in bs])
            if key not in seen:
                candidate_bank.append((tr['method'],bs,levels));seen.add(key)
    start=time.perf_counter();best=None;trials=[]
    for distribution in ['real-normal','real-exponential']:
        for name,bs,levels in candidate_bank:
            state=NestedBlockState(g,bs=bs,base_state=WeightedBlockState,
                 base_state_args=dict(rec=[w],rec_types=[distribution],deg_corr=False))
            dl=float(state.entropy())
            trials.append(dict(proposer=name,distribution=distribution,description_length=dl,groups=len(levels[0])))
            if best is None or dl<best['description_length']:
                best=dict(description_length=dl,selected_proposer=name,distribution=distribution,
                      leaves=levels[0],levels=levels,nodes=[{'residues':s} for lev in levels for s in lev])
    results.append(dict(dataset=ds['id'],kind=ds['kind'],method='nested_weighted_mdl_bank',
                        seconds=time.perf_counter()-start,trials=trials,**best))
    (OUT/'weighted_mdl_results.json').write_text(json.dumps(results))
    print(i+1,ds['id'],best['selected_proposer'],sorted(map(len,best['leaves']),reverse=True),flush=True)
