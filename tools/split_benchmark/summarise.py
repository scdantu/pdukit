"""Write the comparison tables, residue-level cut diagnostics and plots."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import OUT, auc, cohesion, metrics

D={d['id']:d for d in json.loads((OUT/'datasets.json').read_text())}
results=json.loads((OUT/'spectral_results.json').read_text())
for filename in ['sbm_results.json','weighted_mdl_results.json']:
    extra=json.loads((OUT/filename).read_text())
    assert len(extra)==len(D),(filename,len(extra),len(D))
    for r in extra:r.update(metrics(D[r['dataset']],r['leaves'],r['nodes']))
    results+=extra
assert len(results)==7*len(D)
assert len({(r['dataset'],r['method']) for r in results})==len(results)
for r in results:
    n=len(D[r['dataset']]['rho'])
    members=[int(i) for g in r['leaves'] for i in g]
    assert sorted(members)==list(range(n)),(r['dataset'],r['method'],'invalid partition')
    assert all(set(node['residues'])<=set(range(n)) for node in r['nodes'])
    if r['method']=='nested_weighted_mdl_bank':
        assert np.isfinite(r['description_length'])
        assert abs(r['description_length']-min(t['description_length'] for t in r['trials']))<1e-9
(OUT/'combined_results.json').write_text(json.dumps(results))
methods=['spectral_ncut','directed_modularity','markov_1','markov_4','markov_16','nested_binary_theta','nested_weighted_mdl_bank']
labels={'spectral_ncut':'Recursive spectral (normalised cut)',
        'directed_modularity':'Directed modularity','markov_1':'Markov Stability, t=1',
        'markov_4':'Markov Stability, t=4','markov_16':'Markov Stability, t=16',
        'nested_binary_theta':'Nested SBM, θ-binary',
        'nested_weighted_mdl_bank':'Nested weighted SBM, restricted MDL search'}
def mean(rows,key):return float(np.mean([r[key] for r in rows]))
summary=[]
for method in methods:
    r=[x for x in results if x['method']==method]
    base=[x for x in r if x['kind']=='cached_planted']
    nested=[x for x in r if x['kind']=='nested_domains']
    single=[x for x in r if x['kind']=='single_unit']
    val=[n['validation_cut_min_auc'] for x in nested for n in x['nodes'] if n.get('split') and 'validation_cut_min_auc' in n]
    summary.append(dict(method=method,planted_recovery=mean(base,'recovery_067'),
         planted_jaccard=mean(base,'mean_jaccard'),planted_ari=mean(base,'ari'),
         single_false_splits=sum(x['groups']>1 for x in single),
         nested_core_recovery=mean(nested,'recovery_067'),
         nested_domain_tree_jaccard=mean(nested,'domain_tree_jaccard'),
         median_seconds=float(np.median([x['seconds'] for x in r])),
         heldout_boundary_fraction=sum(v>.5 for v in val)/len(val) if val else None))
with (OUT/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
flat=[];residue_rows=[];cut_rows=[]
for r in results:
    ds=D[r['dataset']];R=np.array(ds['rho'])
    row={k:r.get(k) for k in ['dataset','kind','method','groups','mean_jaccard','recovery_067','tree_jaccard','ari','domain_tree_jaccard','seconds']}
    row['sizes']=' '.join(map(str,r['sizes']));flat.append(row)
    if ds['kind'] not in ('protein','protein_growth'):continue
    support=np.array(ds['support']);whole=D[r['dataset'].split('_')[0]+'_whole'];W=np.array(whole['rho'])
    for groupid,group in enumerate(r['leaves']):
        actual=support[group]
        cut_rows.append(dict(dataset=r['dataset'],method=r['method'],leaf=groupid,size=len(group),
                 whole_protein_cohesion=cohesion(W,actual)[2],
                 residues_feature_index_1based=' '.join(str(i+1) for i in actual)))
    for cutid,node in enumerate(r['nodes']):
        if not node.get('split'):continue
        a,b=node['children']
        for side,inside,outside in [('A',a,b),('B',b,a)]:
            for i in inside:
                others=[j for j in inside if j!=i]
                incoming=auc(R[i,others],R[i,outside]);outgoing=auc(R[others,i],R[outside,i])
                residue_rows.append(dict(dataset=r['dataset'],method=r['method'],cut=cutid,side=side,
                    residue_feature_index_1based=int(support[i])+1,
                    incoming_within_vs_sibling_auc=incoming,outgoing_within_vs_sibling_auc=outgoing,
                    cross_preference=bool(incoming<=.5 or outgoing<=.5)))
for filename,rows in [('per_dataset.csv',flat),('protein_leaves.csv',cut_rows),('cut_residue_diagnostics.csv',residue_rows)]:
    with (OUT/filename).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

report=['# Offline comparison of PDU splitting methods — 17 October 2026','',
'## Scope and conclusion','',
'This is a comparison of split proposals and stopping rules on existing predictive matrices, not a new production PDU run. The incoming-only growth and the distance-derived protein features were unchanged. No replica-convergence requirement or biological acceptance criterion was imposed.','',
'Recursive normalised-cut spectral splitting is the strongest baseline on the cached planted systems in this comparison. This does not establish a universal winner: the shared-domain-motion controls expose a substantial unresolved core-separation problem. The unrestricted weighted nested-SBM optimizer could not be compared because its native library crashed.','',
'## Data','',
'- 30 distinct cached planted matrices: base, scattered, weak, strong and no-faces; six seeds each. Ninety saved files contained only 30 distinct matrices and were deduplicated. Base/scattered designs are related and are not independent experimental replicates.',
'- Six single-unit time-series controls: one cohesive 36-residue unit, where subdivision is unwanted.',
'- Six constructed nested-domain controls: six 6-residue private cores inside two 18-residue domains. Added domain-shared motion overlays the existing planted dynamics. These are diagnostic synthetic systems, not extra protein simulations. First and second halves of each synthetic trajectory supply separate estimated matrices.',
'- T4L and Syt whole-protein matrices, plus all saved incoming-grown supports of size at least six: 29 protein/parent cases. Parents are the original growth supports, not the later cohesion-first MC groups.',
'- Total: 71 cases per variant; seven variants (497 comparisons).','',
'## What was implemented','',
'1. **Recursive spectral:** symmetrise the directed weights for normalised-Laplacian proposal generation; sweep the Fiedler ordering for the minimum normalised cut. Direction is retained in the subsequent cohesion assessment. This is a specific recursive spectral adaptation, not every algorithm covered by Li et al.',
'2. **Directed modularity:** use the directed strength-preserving modularity matrix, spectral sweep and improving node moves. Recompute the objective inside each parent.',
'3. **Markov Stability:** directed row-stochastic walk with a stationary-law covariance objective, at graph diffusion times 1, 4 and 16; spectral sweep and improving moves. These graph times are not the MD lag τ. Walks are rebuilt inside each parent.',
'4. **Nested θ-binary SBM:** graph-tool 3.9, directed degree-corrected nested SBM; edges are exactly rho > the existing graph θ. Three optimisation starts and additional improving sweeps. This discards weight magnitudes and is explicitly not the weighted counterpart of the other methods.',
'5. **Nested weighted SBM, restricted search:** the native weighted optimizer crashed even on a 12-node diagnostic. As an additional experiment, evaluate graph-tool’s nested weighted model description length on a bank of hierarchical cuts from the other methods, including raw trees without the cohesion stopping rule. Compare real-normal and real-exponential weight models on the same untransformed weights. This uses the published model objective but a restricted candidate search; it is not an independent or unrestricted fit.','',
'For variants 1–3, the common boundary statistic is the minimum of the incoming/outgoing within-parent AUCs of both children. Require all four to exceed 0.5, plus positive objective gain for modularity/stability. Child size is at least three, inherited as an explicit algorithmic choice rather than a biological definition.','',
'The exploratory graph-null check uses 199 reciprocal-dyad permutations per proposed cut and reruns the proposal search in every null. It preserves the weight distribution and reciprocal pairs, but not each residue’s strength. Its p-value is conditional on this graph-exchangeability assumption; it is not a temporal-bootstrap significance result for MD. A depth-spending cutoff 0.05/2^depth is used. With 199 nulls, depth ≥4 cannot pass this cutoff; no positive-boundary case at such a depth affected the present results. Nested SBM variants use their model-selection stopping rules instead, so this compares whole procedures rather than isolated proposal algorithms.','',
'## Recovery and false splitting','',
'Recovery means the fraction of planted home supports with best terminal-leaf Jaccard ≥0.67, matching the existing benchmark convention. Overlapping face ownership is not scored as a hard partition. The table also gives mean best-leaf Jaccard. These are descriptive comparisons, not independent-trial confidence claims.','',
'| Method | Cached recovery | Mean Jaccard | Single-unit false splits | Nested core recovery | Domain best-tree Jaccard |',
'|---|---:|---:|---:|---:|---:|']
for s in summary:
    report.append(f"| {labels[s['method']]} | {100*s['planted_recovery']:.1f}% | {s['planted_jaccard']:.3f} | {s['single_false_splits']}/6 | {100*s['nested_core_recovery']:.1f}% | {s['nested_domain_tree_jaccard']:.3f} |")
report+=['','## By planted design','','| Method | Base | No faces | Scattered | Strong coupling | Weak coupling |','|---|---:|---:|---:|---:|---:|']
for m in methods:
    vals=[]
    for design in ['base','nofaces','scattered','strong','weak']:
        sub=[r for r in results if r['kind']=='cached_planted' and r['method']==m and r['dataset'].split('__')[0]==design]
        vals.append(f"{100*mean(sub,'recovery_067'):.1f}%")
    report.append('| '+labels[m]+' | '+' | '.join(vals)+' |')
report+=['','## Whole-protein exploratory splits','','These are partitions of the saved whole-protein ρ matrices, not PDU counts. They do not establish biological domains.','','| Method | T4L leaf sizes | Syt leaf sizes |','|---|---|---|']
for m in methods:
    a=next(r for r in results if r['method']==m and r['dataset']=='T4L_whole')
    b=next(r for r in results if r['method']==m and r['dataset']=='SYT_whole')
    report.append(f"| {labels[m]} | {', '.join(map(str,a['sizes']))} | {', '.join(map(str,b['sizes']))} |")
report+=['','## Splitting the actual incoming-grown parents','','| Method | T4L parents split / tested | Syt parents split / tested |','|---|---:|---:|']
for m in methods:
    cells=[]
    for protein in ['T4L','SYT']:
        sub=[r for r in results if r['method']==m and r['kind']=='protein_growth' and r['dataset'].startswith(protein)]
        cells.append(f"{sum(r['groups']>1 for r in sub)}/{len(sub)}")
    report.append('| '+labels[m]+' | '+' | '.join(cells)+' |')
report+=['','For the leading spectral procedure, the accepted parent cuts are:','']
for r in results:
    if r['method']=='spectral_ncut' and r['kind']=='protein_growth' and r['groups']>1:
        parent=D[r['dataset']];R=np.array(parent['rho'])
        cs=[cohesion(R,g)[2] for g in r['leaves']]
        report.append(f"- **{r['dataset']}**: {len(R)} → {' + '.join(map(str,r['sizes']))} residues; child within-parent cohesion {', '.join(f'{v:.3f}' for v in cs)}. Root graph-null p={r['nodes'][0]['graph_null_p']:.3f}, with the graph-null scope stated above.")
report+=['','## Interpretation and next decision','',
'- Use recursive spectral splitting as the leading baseline to carry forward, retaining directed modularity as a comparator. Markov time 1 performs poorly on the cached controls; larger graph diffusion times recover more structure but add a resolution parameter.',
'- Do not select a method from protein group counts or whole-protein cohesion alone. The known-answer controls provide the recovery comparison; protein results show the consequences of each choice.',
'- The nested-domain control is the key unresolved case. Recovering a domain boundary is substantially easier here than recovering all cores beneath the shared motion. This comparison does not demonstrate a complete solution to mutually coupled core separation.',
'- Restricted weighted MDL recovers more nested cores (38.9%) than spectral splitting (13.9%), but falsely subdivides 3/6 single-unit controls. It therefore does not dominate the spectral procedure. Its protein partitions are much finer and have not passed the common cohesion/graph-null stopping rule.',
'- The binary SBM results must not be taken as evidence against weighted SBM. Binary thresholding discards informative weight contrasts, and the native weighted optimizer was unavailable. The restricted weighted MDL experiment is useful model-selection evidence within its candidate bank, not an unrestricted result.',
'- The synthetic validation matrices assess whether selected boundaries retain directional within-versus-cross preference in the second halves of the sampled synthetic trajectories. They do not establish MD convergence. No holdout matrix was available for the cached or protein datasets.',
'- New child r_in/r_out channel counts cannot be recomputed from pairwise rho alone. That requires the covariance workspace or original features, whose saved paths point to the cluster. Existing parent channel scores remain available; do not inherit them for children.',
'- Before turning this into production detection, the temporal uncertainty/stopping step needs a workspace-based test on the selected cuts. The present graph-null test is deliberately labelled as exploratory.','',
'## Outputs and numbering','',
'`summary.csv` and `per_dataset.csv` contain the numerical comparison. `protein_leaves.csv` lists every protein support. `cut_residue_diagnostics.csv` gives each residue’s incoming and outgoing preference for its side versus its sibling at every accepted cut. A value ≤0.5 is marked cross-preferring for inspection; it is not a calibrated boundary/face classification. Residues are **one-based positions in the saved feature array**, not asserted PDB residue numbers.',
'','## References','',
'- Li et al. (2022), Hierarchical Community Detection by Recursive Partitioning. https://doi.org/10.1080/01621459.2020.1833888',
'- Leicht & Newman (2008), Community Structure in Directed Networks. https://doi.org/10.1103/PhysRevLett.100.118703',
'- Delvenne, Yaliraki & Barahona (2010), Stability of graph communities across time scales. https://doi.org/10.1073/pnas.0903215107',
'- Peixoto (2014), Hierarchical Block Structures and High-resolution Model Selection in Large Networks. https://doi.org/10.1103/PhysRevX.4.011047',
'- Peixoto (2017), Nonparametric Bayesian inference of the microcanonical stochastic block model. https://doi.org/10.1103/PhysRevE.95.012317',
'- Graph-tool weighted-network inference documentation: https://graph-tool.skewed.de/static/docs/stable/demos/inference/inference.html',
'- Fortunato & Barthélemy (2007), Resolution limit in community detection. https://doi.org/10.1073/pnas.0605965104','']
(OUT/'REPORT.md').write_text('\n'.join(report))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
names=['Spectral','Directed\nmodularity','Markov\nt=1','Markov\nt=4','Markov\nt=16','Nested\nbinary','Weighted MDL\nrestricted']
x=np.arange(len(methods))
for ax,key,title in [(axes[0],'planted_recovery','Cached planted units'),(axes[1],'nested_core_recovery','Cores under shared domain motion')]:
    ax.bar(x,[100*s[key] for s in summary],color=['#2878a5','#8095a8','#a4b9c6','#a4b9c6','#a4b9c6','#d0a270','#688e74'])
    ax.set(ylim=(0,100),ylabel='Recovered planted supports (%)',title=title,xticks=x,xticklabels=names)
    ax.tick_params(axis='x',labelrotation=35)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Offline split-method comparison: recovery depends on the planted design')
fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
print('\n'.join(report[report.index('## Recovery and false splitting'):report.index('## By planted design')]))
