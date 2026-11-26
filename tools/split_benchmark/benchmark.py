"""Offline split-method comparison; no production detection changes.

Graph resampling below is a conditional graph null, NOT temporal MD uncertainty.
See README/report for scope and the minimum child size choice.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, expm
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

candidate_repo=Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get('PDU_REPO',str(candidate_repo if (candidate_repo/'pdukit').exists() else Path('/Users/sarathdantu/Projects/vsprojects/mdpdu/pdu-extractor'))))
DATA = REPO.parent / 'isambard'
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get('PDU_SPLIT_OUT',str(Path(__file__).resolve().parent)))
OUT.mkdir(parents=True,exist_ok=True)
MIN_CHILD = 3

def auc(x, y):
    x, y = np.ravel(x), np.ravel(y)
    if not len(x) or not len(y): return float('nan')
    ordered=np.sort(y)
    counts=np.searchsorted(ordered,x,side='left')+np.searchsorted(ordered,x,side='right')
    return float(counts.mean()/(2*len(y)))

def cohesion(R, S, U=None):
    U = list(range(len(R))) if U is None else list(U)
    S = list(S); E = sorted(set(U)-set(S))
    if len(S)<2 or not E: return [None,None,None]
    I = R[np.ix_(S,S)][~np.eye(len(S),dtype=bool)]
    incoming=auc(I,R[np.ix_(S,E)])
    outgoing=auc(I,R[np.ix_(E,S)])
    return [incoming,outgoing,(incoming+outgoing)/2]

def modularity_matrix(W):
    total=W.sum()
    return W-np.outer(W.sum(1),W.sum(0))/max(total,1e-30)

def proposal(W, method):
    """Return a local bisection and its method-specific quality."""
    n=len(W)
    if n<2*MIN_CHILD: return None
    H=(W+W.T)/2
    if method=='spectral_ncut':
        d=H.sum(1); inv=1/np.sqrt(np.maximum(d,1e-30))
        L=np.eye(n)-inv[:,None]*H*inv[None,:]
        _,V=eigh(L,subset_by_index=[0,1]); v=inv*V[:,1]
        order=np.argsort(v); best=None
        for k in range(MIN_CHILD,n-MIN_CHILD+1):
            a,b=order[:k],order[k:]; cut=H[np.ix_(a,b)].sum()
            score=-cut*(1/max(d[a].sum(),1e-30)+1/max(d[b].sum(),1e-30))
            if best is None or score>best[2]: best=(a,b,float(score))
        return best
    if method=='directed_modularity':
        B=modularity_matrix(W); B=(B+B.T)/2
    elif method.startswith('markov_'):
        t=float(method.split('_')[1])
        P=W.T/np.maximum(W.sum(0)[:,None],1e-30)
        # Tiny numerical teleportation ensures a unique stationary law.
        P=(1-1e-10)*P+1e-10/n
        ev,V=np.linalg.eig(P.T); pi=np.real(V[:,np.argmin(abs(ev-1))]); pi/=pi.sum()
        M=pi[:,None]*expm(t*(P-np.eye(n)))-np.outer(pi,pi)
        B=(M+M.T)/2
    else: raise ValueError(method)
    _,V=eigh(B,subset_by_index=[n-1,n-1]); order=np.argsort(V[:,0]); best=None
    for k in range(MIN_CHILD,n-MIN_CHILD+1):
        a,b=order[:k],order[k:]; z=-np.ones(n); z[a]=1
        q=float(z@B@z)
        if best is None or q>best[2]: best=(a,b,q)
    # Local node moves refine the spectral proposal without changing objective.
    a,b,q=best; z=-np.ones(n); z[a]=1
    for _ in range(n):
        gains=-4*z*(B@z)+4*np.diag(B)
        gains[(z>0) & ((z>0).sum()<=MIN_CHILD)]=-np.inf
        gains[(z<0) & ((z<0).sum()<=MIN_CHILD)]=-np.inf
        j=int(np.argmax(gains))
        if gains[j]<=1e-12: break
        z[j]*=-1
    return np.where(z>0)[0],np.where(z<0)[0],float(z@B@z)

def dyad_null(W,rng):
    """Preserves weight distribution and reciprocal pairs, not node strengths."""
    n=len(W); i,j=np.triu_indices(n,1); p=rng.permutation(len(i))
    V=np.zeros_like(W); V[i,j]=W[i[p],j[p]];V[j,i]=W[j[p],i[p]]
    return V

def cut_stat(W,a,b):
    # Require BOTH children and BOTH predictive directions to show a boundary.
    ca=cohesion(W,a); cb=cohesion(W,b)
    return min(ca[0],ca[1],cb[0],cb[1])

def tree(R, method, nnull=0, alpha=.05, seed=0, validation=None):
    rng=np.random.default_rng(seed); nodes=[]; leaves=[]
    def visit(S,depth):
        W=R[np.ix_(S,S)]; p=proposal(W,method)
        row={'residues':list(map(int,S)), 'depth':depth,'split':False}
        nodes.append(row)
        if p is None: leaves.append(S);return
        a,b,q=p; statistic=cut_stat(W,a,b)
        row.update(quality=q,cut_min_auc=statistic,
                   children=[list(map(int,np.asarray(S)[a])),list(map(int,np.asarray(S)[b]))])
        accepted=statistic>.5 and (method=='spectral_ncut' or q>1e-12)
        if nnull:
            values=[]
            for _ in range(nnull):
                V=dyad_null(W,rng); cut=proposal(V,method)
                values.append(cut_stat(V,cut[0],cut[1]))
            pv=(1+sum(v>=statistic for v in values))/(nnull+1)
            row['graph_null_p']=pv
            # Depth-wise alpha spending; exploratory conditional graph test.
            accepted &= pv<=alpha/(2**depth)
        if validation is not None:
            V=validation[np.ix_(S,S)]
            row['validation_cut_min_auc']=cut_stat(V,a,b)
        row['split']=bool(accepted)
        if accepted:
            visit(np.asarray(S)[a].tolist(),depth+1);visit(np.asarray(S)[b].tolist(),depth+1)
        else: leaves.append(S)
    visit(list(range(len(R))),0)
    return {'nodes':nodes,'leaves':leaves}

def covariance_rho(feats,segments):
    from pdukit.cog.graph import pair_rho
    xs=[];ys=[]
    for a,b in segments:
        f=np.asarray(feats[a:b],float);f-=f.mean(0)
        xs.append(f[:-1].reshape(len(f)-1,-1));ys.append(f[1:].reshape(len(f)-1,-1))
    X=np.concatenate(xs);Y=np.concatenate(ys);X-=X.mean(0);Y-=Y.mean(0)
    return pair_rho(X.T@X/len(X),X.T@Y/len(X),Y.T@Y/len(X),feats.shape[1],feats.shape[2])

def temporal_theta(feats,segments,seed):
    from pdukit.cog.graph import pair_rho
    rng=np.random.default_rng(seed); xs=[];ys=[];blocks=[]
    for a,b in segments:
        f=np.asarray(feats[a:b],float);f-=f.mean(0)
        x=f[:-1].reshape(len(f)-1,-1);y=f[1:].reshape(len(f)-1,-1)
        xs.append(x);ys.append(y);blocks.append((x,y))
    X=np.concatenate(xs);Y=np.concatenate(ys);X-=X.mean(0);Y-=Y.mean(0)
    c00=X.T@X/len(X);c01=X.T@Y/len(X);c11=Y.T@Y/len(X);values=[]
    for _ in range(8):
        Z=np.concatenate([np.roll(x,int(rng.integers(100,len(x)-100)),axis=0) for x,y in blocks])
        Z-=Z.mean(0)
        V=pair_rho(c00,c01,c11,feats.shape[1],feats.shape[2],X00=Z.T@X/len(X),X01=Z.T@Y/len(X))
        values.extend(V[~np.eye(len(V),dtype=bool)])
    return float(np.quantile(values,.95))

def prepare():
    datasets=[];seen=set()
    for f in sorted(DATA.rglob('results.json')):
        try: d=json.loads(f.read_text())
        except Exception: continue
        if not d.get('args',{}).get('planted') or 'rho' not in d.get('predictive_graph',{}):continue
        h=hashlib.sha256(str(d['predictive_graph']['rho']).encode()).hexdigest()
        if h in seen:continue
        seen.add(h)
        R=np.asarray(d['predictive_graph']['rho'])
        datasets.append({'id':f.parent.name,'kind':'cached_planted','source':str(f),
                         'rho':R.tolist(),'theta':d['predictive_graph']['theta'],'truth':d['truth']['home_units'],
                         'faces':d['truth'].get('face_residues',[])})
    from pdukit.bench.planted import planted_system
    for design in ['single_unit','nested_domains']:
        for seed in range(6):
            b=planted_system(n_units=1 if design=='single_unit' else 6,
                    unit_size=36 if design=='single_unit' else 6,m=4,
                    n_frames=4000,n_replicas=2,face_residues=0,seed=100+seed)
            f=b['feats'].astype(float);domains=[]
            if design=='nested_domains':
                rng=np.random.default_rng(900+seed)
                domains=[list(range(18)),list(range(18,36))]
                for domain in domains:
                    z=np.zeros((len(f),2)); noise=rng.normal(size=z.shape)
                    for a,e in b['segments']:
                        for t in range(a+1,e):z[t]=.92*z[t-1]+.35*noise[t]
                    # Common domain motion layered on private core dynamics.
                    for i in domain:f[:,i]+= .8*z@rng.normal(size=(2,4))
            train=[];test=[]
            for a,e in b['segments']:
                mid=(a+e)//2;train.append((a,mid));test.append((mid,e))
            R=covariance_rho(f,train);V=covariance_rho(f,test)
            datasets.append({'id':f'{design}_s{seed}','kind':design,
                    'rho':R.tolist(),'validation_rho':V.tolist(),'theta':temporal_theta(f,train,seed+901),
                    'truth':b['truth']['home_units'],'domains':domains,'faces':[]})
    root=DATA/'by_commit/2026-10-16_3c6758a_main'
    for f in sorted(root.glob('*/results.json')):
        d=json.loads(f.read_text());R=np.array(d['predictive_graph']['rho'])
        protein='T4L' if 'lyso' in str(f) else 'SYT'
        datasets.append({'id':protein+'_whole','kind':'protein','source':str(f),
                         'rho':R.tolist(),'theta':d['predictive_graph']['theta'],'support':list(range(len(R)))})
        for u in d['incoming_growth']['units']:
            S=u['residues']
            if len(S)<6:continue
            datasets.append({'id':protein+f"_growth{u['unit']}",'kind':'protein_growth',
                  'source':str(f),'rho':R[np.ix_(S,S)].tolist(),'theta':d['predictive_graph']['theta'],'support':S})
    (OUT/'datasets.json').write_text(json.dumps(datasets))
    print('Prepared',len(datasets),'datasets',flush=True)

def metrics(ds, groups, nodes):
    result={'groups':len(groups),'sizes':sorted(map(len,groups),reverse=True)}
    R=np.array(ds['rho']);U=list(range(len(R)))
    cs=[cohesion(R,g)[2] for g in groups if 2<=len(g)<len(R)]
    result['median_whole_cohesion']=float(np.median(cs)) if cs else None
    if 'truth' in ds:
        truth=ds['truth'];sets=[set(g) for g in groups]
        jac=[max(len(set(t)&g)/len(set(t)|g) for g in sets) for t in truth]
        allsets=[set(n['residues']) for n in nodes]+sets
        treejac=[max(len(set(t)&g)/len(set(t)|g) for g in allsets) for t in truth]
        lab=np.empty(len(R),int);true=np.empty(len(R),int)
        for k,g in enumerate(groups):lab[g]=k
        for k,g in enumerate(truth):true[g]=k
        result.update(mean_jaccard=float(np.mean(jac)),recovery_067=float(np.mean(np.array(jac)>=.67)),
                tree_jaccard=float(np.mean(treejac)),ari=float(adjusted_rand_score(true,lab)))
        if ds.get('domains'):
            result['domain_tree_jaccard']=float(np.mean([max(len(set(t)&g)/len(set(t)|g) for g in allsets) for t in ds['domains']]))
    return result

def run(nnull):
    datasets=json.loads((OUT/'datasets.json').read_text())
    saved=OUT/'spectral_results.json'
    results=json.loads(saved.read_text()) if saved.exists() else []
    done={r['dataset'] for r in results}
    for index,ds in enumerate(datasets):
        if ds['id'] in done:continue
        R=np.array(ds['rho']);V=np.array(ds['validation_rho']) if 'validation_rho' in ds else None
        for method in ['spectral_ncut','directed_modularity','markov_1','markov_4','markov_16']:
            start=time.perf_counter()
            tr=tree(R,method,nnull=nnull,seed=77,validation=V)
            results.append({'dataset':ds['id'],'kind':ds['kind'],'method':method,
                           'seconds':time.perf_counter()-start,**metrics(ds,tr['leaves'],tr['nodes']),**tr})
        print(index+1,ds['id'],[(r['method'],r['sizes']) for r in results[-5:]],flush=True)
        (OUT/'spectral_results.json').write_text(json.dumps(results))

def raw_proposals():
    out=[]
    for ds in json.loads((OUT/'datasets.json').read_text()):
        R=np.array(ds['rho'])
        for method in ['spectral_ncut','directed_modularity','markov_1','markov_4','markov_16']:
            nodes=[]
            def visit(S,depth):
                p=proposal(R[np.ix_(S,S)],method)
                node=dict(residues=S,depth=depth,split=p is not None);nodes.append(node)
                if p is not None:
                    node['children']=[np.array(S)[p[0]].tolist(),np.array(S)[p[1]].tolist()]
                    for child in node['children']:visit(child,depth+1)
            visit(list(range(len(R))),0)
            out.append(dict(dataset=ds['id'],method='raw_'+method,nodes=nodes))
    (OUT/'proposal_trees.json').write_text(json.dumps(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','raw']);ap.add_argument('--nulls',type=int,default=199)
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='raw':raw_proposals()
    else:run(args.nulls)
