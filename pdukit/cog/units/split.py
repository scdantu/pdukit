"""Recursive within-parent splitting of incoming-grown supports.

Proposal methods are interchangeable. Incoming/outgoing interface spectra
are measured after splitting and never used to veto a cohesion boundary.
The graph null is an exploratory reciprocal-dyad exchangeability model,
not a temporal null. Bootstrap intervals concern the selected fixed cut.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, expm

from pdukit.cog.units.grow import _measurement


def auc(inside, outside):
    """Probability inside exceeds outside, with half credit for ties."""
    inside, outside = np.ravel(inside), np.sort(np.ravel(outside))
    if not inside.size or not outside.size:
        return None
    ranks = np.searchsorted(outside, inside, side="left") + np.searchsorted(
        outside, inside, side="right"
    )
    return float(ranks.mean() / (2 * len(outside)))


def parent_cohesion(rho, support, parent=None):
    """Keep both directional AUCs and their mean within a declared parent."""
    support = np.asarray(sorted(support), dtype=int)
    parent = np.arange(len(rho)) if parent is None else np.asarray(parent)
    outside = np.setdiff1d(parent, support)
    if len(support) < 2 or not len(outside):
        return dict(auc_listener=None, auc_driver=None, cohesion=None)
    block = rho[np.ix_(support, support)]
    inside = block[~np.eye(len(support), dtype=bool)]
    incoming = auc(inside, rho[np.ix_(support, outside)])
    outgoing = auc(inside, rho[np.ix_(outside, support)])
    return dict(
        auc_listener=incoming,
        auc_driver=outgoing,
        cohesion=(incoming + outgoing) / 2,
    )


def _spectral(weight, min_size, **kwargs):
    n = len(weight)
    symmetric = (weight + weight.T) / 2
    degree = symmetric.sum(1)
    inv = 1 / np.sqrt(np.maximum(degree, 1e-30))
    laplacian = np.eye(n) - inv[:, None] * symmetric * inv[None, :]
    _, vectors = eigh(laplacian, subset_by_index=[0, 1])
    order = np.argsort(inv * vectors[:, 1], kind="stable")
    best = None
    for k in range(min_size, n - min_size + 1):
        a, b = order[:k], order[k:]
        cut = symmetric[np.ix_(a, b)].sum()
        quality = -cut * (
            1 / max(degree[a].sum(), 1e-30) + 1 / max(degree[b].sum(), 1e-30)
        )
        if best is None or quality > best[2]:
            best = a, b, float(quality)
    return best


def _matrix_split(matrix, min_size):
    n = len(matrix)
    _, vectors = eigh(matrix, subset_by_index=[n - 1, n - 1])
    order = np.argsort(vectors[:, 0], kind="stable")
    best = None
    for k in range(min_size, n - min_size + 1):
        signs = -np.ones(n)
        signs[order[:k]] = 1
        quality = float(signs @ matrix @ signs)
        if best is None or quality > best[1]:
            best = signs, quality
    signs = best[0]
    for _ in range(n):
        gains = -4 * signs * (matrix @ signs) + 4 * np.diag(matrix)
        if (signs > 0).sum() <= min_size:
            gains[signs > 0] = -np.inf
        if (signs < 0).sum() <= min_size:
            gains[signs < 0] = -np.inf
        index = int(np.argmax(gains))
        if gains[index] <= 1e-12:
            break
        signs[index] *= -1
    return (
        np.flatnonzero(signs > 0),
        np.flatnonzero(signs < 0),
        float(signs @ matrix @ signs),
    )


def _modularity(weight, min_size, **kwargs):
    expected = np.outer(weight.sum(1), weight.sum(0))
    matrix = weight - expected / max(weight.sum(), 1e-30)
    return _matrix_split((matrix + matrix.T) / 2, min_size)


def _markov(weight, min_size, markov_time=4.0, **kwargs):
    n = len(weight)
    # rho[target, source] -> walk[source, target].
    walk = weight.T.copy()
    degree = walk.sum(1)
    walk /= np.maximum(degree[:, None], 1e-30)
    walk[degree == 0] = 1 / n
    walk = (1 - 1e-10) * walk + 1e-10 / n
    values, vectors = np.linalg.eig(walk.T)
    stationary = np.real(vectors[:, np.argmin(abs(values - 1))])
    stationary /= stationary.sum()
    covariance = stationary[:, None] * expm(
        markov_time * (walk - np.eye(n))
    ) - np.outer(stationary, stationary)
    return _matrix_split((covariance + covariance.T) / 2, min_size)


SPLITTERS = {
    "recursive-spectral": _spectral,
    "directed-modularity": _modularity,
    "markov-stability": _markov,
}


def propose_cut(rho, method="recursive-spectral", min_size=3, markov_time=4.0):
    """Return local indices for a cut, or None when children are too small.

    A callable with the same signature as the registered proposals may also
    be passed by Python callers; the CLI exposes the registered methods.
    """
    if len(rho) < 2 * min_size:
        return None
    function = method if callable(method) else SPLITTERS[method]
    return function(rho, min_size=min_size, markov_time=markov_time)


def cut_statistic(rho, a, b):
    """Minimum directional cohesion across both children."""
    ca, cb = parent_cohesion(rho, a), parent_cohesion(rho, b)
    return min(
        ca["auc_listener"],
        ca["auc_driver"],
        cb["auc_listener"],
        cb["auc_driver"],
    )


def _null_graph(rho, rng):
    i, j = np.triu_indices(len(rho), 1)
    permutation = rng.permutation(len(i))
    result = np.zeros_like(rho)
    result[i, j] = rho[i[permutation], j[permutation]]
    result[j, i] = rho[j[permutation], i[permutation]]
    return result


def merge_overlapping(candidates, min_overlap=0.5, scorer=None, tol=1e-9):
    """Group overlapping incoming-grown supports into families; pick each
    family's parent by the incoming criterion.

    Different seeds often grow different versions of the same region.

    Families (no single-linkage chaining).  Supports are visited best first
    (lowest nu_in; ties by index).  A support joins the family whose MAJORITY
    CORE (residues in at least half of its members so far) it overlaps most,
    if that overlap is at least `min_overlap` of the smaller of the two
    (overlap coefficient); otherwise it starts a new family.  Joining is
    judged against the family's consensus, not against any one member, so a
    chain of loosely related versions cannot pull unrelated regions together.

    Parent (nu_in decides).  The union of a family's members is scored with
    `scorer`.  The union is the parent only if its nu_in is not higher than
    that of the family's best member (lowest nu_in); otherwise the best member
    is the parent and the other versions are reported as variants.  Without
    a scorer or member nu_in values the union is used.  Every parent is passed
    to the recursive split, which re-tests it.  Per-residue membership
    frequency across the family is reported (1.0 = in every version).
    """
    if not 0 < min_overlap <= 1:
        raise ValueError("merge overlap must be in (0, 1]")
    sets, nus = [], []
    for c in candidates:
        sets.append(
            frozenset(
                int(x) for x in (c["residues"] if isinstance(c, dict) else c)
            )
        )
        v = c.get("nu_in") if isinstance(c, dict) else None
        nus.append(
            float(v) if v is not None and np.isfinite(float(v)) else None
        )
    n = len(sets)
    have_nu = all(v is not None for v in nus)
    order = (
        sorted(range(n), key=lambda i: (nus[i], i))
        if have_nu
        else list(range(n))
    )

    def majority(members):
        return {
            r
            for r in set().union(*(sets[i] for i in members))
            if 2 * sum(r in sets[i] for i in members) >= len(members)
        }

    groups, joins = [], []
    for i in order:
        best, best_ov = None, 0.0
        for g, members in enumerate(groups):
            core = majority(members)
            small = min(len(sets[i]), len(core))
            ov = len(sets[i] & core) / small if small else 0.0
            if ov > best_ov:
                best, best_ov = g, ov
        if best is not None and best_ov >= min_overlap:
            groups[best].append(i)
            joins.append((i, best, best_ov))
        else:
            groups.append([i])

    unions = [sorted(set().union(*(sets[i] for i in m))) for m in groups]
    union_nu = [None] * len(groups)
    if scorer is not None and have_nu and groups:
        need = [g for g, m in enumerate(groups) if len(m) > 1]
        if need:
            supports = [unions[g] for g in need]
            values = (
                scorer.score_batch(supports)
                if hasattr(scorer, "score_batch")
                else [scorer(x) for x in supports]
            )
            for g, val in zip(need, values):
                union_nu[g] = float(_measurement(val)["nu_in"])
    families = []
    for g, members in enumerate(groups):
        union = unions[g]
        freq = {
            r: sum(r in sets[i] for i in members) / len(members) for r in union
        }
        best_member = members[0]  # visited best first
        best_nu = nus[best_member]
        if len(members) == 1:
            decision, parent = "single", sorted(sets[best_member])
        elif union_nu[g] is None:
            decision, parent = "union (no incoming comparison)", union
        elif union_nu[g] <= best_nu + tol:
            decision, parent = "union", union
        else:
            decision, parent = "best_member", sorted(sets[best_member])
        families.append(
            dict(
                family=g,
                residues=parent,
                size=len(parent),
                decision=decision,
                members=sorted(members),
                member_sizes=[len(sets[i]) for i in sorted(members)],
                member_nu_in=[nus[i] for i in sorted(members)],
                best_member=best_member,
                best_member_nu_in=best_nu,
                union=union,
                union_size=len(union),
                union_nu_in=union_nu[g],
                core=[r for r in union if freq[r] == 1.0],
                majority=sorted(r for r in union if freq[r] >= 0.5),
                residue_frequency=freq,
            )
        )
    return dict(
        rule=(
            "best-first; join the family whose majority core overlaps "
            ">= min_overlap of the smaller set; parent = union only if "
            "nu_in(union) <= nu_in(best member), else best member"
        ),
        min_overlap=float(min_overlap),
        n_candidates=n,
        n_families=len(families),
        joins=joins,
        families=families,
    )


def split_candidates(
    rho,
    candidates,
    scorer=None,
    *,
    method="recursive-spectral",
    min_size=3,
    markov_time=4.0,
    test="both",
    n_null=199,
    alpha=0.05,
    bootstrap=None,
    max_depth=8,
    seed=0,
    on_log=None,
):
    """Split each incoming-grown parent, preserving nesting and overlaps.

    With test='both', require an exploratory search-adjusted graph-null
    comparison and a positive bootstrap lower bound for the selected cut.
    No interface score is used for selection. Identical terminal supports
    are deduplicated, with every source node retained as provenance.
    """
    rho = np.asarray(rho, dtype=float)
    if (
        rho.ndim != 2
        or rho.shape[0] != rho.shape[1]
        or not np.isfinite(rho).all()
        or (rho < 0).any()
    ):
        raise ValueError("rho must be a finite nonnegative square matrix")
    if not callable(method) and method not in SPLITTERS:
        raise ValueError(f"unknown splitter: {method}")
    if min_size < 2 or max_depth < 0 or not 0 < alpha < 1:
        raise ValueError(
            "require min_size >= 2, max_depth >= 0, 0 < alpha < 1"
        )
    if test not in {"both", "bootstrap", "graph-null", "none"}:
        raise ValueError("unknown split test")
    if not np.isfinite(markov_time) or markov_time <= 0:
        raise ValueError("Markov time must be finite and positive")
    if test in {"both", "graph-null"} and n_null < 1:
        raise ValueError("graph-null test requires at least one null")
    bootstrap = [] if bootstrap is None else list(bootstrap)
    if test in {"both", "bootstrap"} and len(bootstrap) < 2:
        raise ValueError(
            "bootstrap split test requires resampled rho matrices"
        )
    for matrix in bootstrap:
        if np.shape(matrix) != rho.shape or not np.isfinite(matrix).all():
            raise ValueError("invalid bootstrap rho matrix")
    rng = np.random.default_rng(seed)
    nodes, boundary_rows, leaves = [], [], []
    log = on_log if on_log is not None else (lambda *_: None)

    def visit(support, parent_id, root_id, depth):
        support = sorted(set(int(x) for x in support))
        if not support or support[0] < 0 or support[-1] >= len(rho):
            raise ValueError("candidate support out of range")
        node_id = len(nodes)
        node = dict(
            node=node_id,
            parent=parent_id,
            root=root_id,
            depth=depth,
            residues=support,
            size=len(support),
            split=False,
            whole_protein=parent_cohesion(rho, support),
        )
        node["within_parent"] = (
            parent_cohesion(rho, support, nodes[parent_id]["residues"])
            if parent_id is not None
            else None
        )
        nodes.append(node)
        weight = rho[np.ix_(support, support)].copy()
        np.fill_diagonal(weight, 0)
        proposal = (
            propose_cut(weight, method, min_size, markov_time)
            if depth < max_depth
            else None
        )
        pad = "  " * (depth + 1)
        wp = node["within_parent"] or {}
        coh_txt = (
            f"cohesion {node['whole_protein']['cohesion']:.3f}"
            if node["whole_protein"]["cohesion"] is not None
            else "cohesion n/a"
        )
        if wp.get("cohesion") is not None:
            coh_txt += f", within parent {wp['cohesion']:.3f}"
        if proposal is None:
            node["stop_reason"] = (
                "max_depth" if depth >= max_depth else "minimum_child_size"
            )
            leaves.append(node_id)
            log(
                f"{pad}node {node_id} root {root_id} depth {depth}  size "
                f"{len(support)}  {coh_txt}  leaf ({node['stop_reason']})"
            )
            return node_id
        a, b, quality = proposal
        statistic = cut_statistic(weight, a, b)
        local_alpha = alpha / (2**depth)
        node.update(
            proposal_quality=float(quality),
            cut_min_directional_auc=float(statistic),
            alpha=local_alpha,
            proposed_children=[
                np.asarray(support)[a].tolist(),
                np.asarray(support)[b].tolist(),
            ],
        )
        accepted = statistic > 0.5
        if method != "recursive-spectral" and not callable(method):
            accepted &= quality > 1e-12
        if test in {"both", "graph-null"}:
            null_statistics = []
            for _ in range(n_null):
                null = _null_graph(weight, rng)
                ca, cb, _ = propose_cut(null, method, min_size, markov_time)
                null_statistics.append(cut_statistic(null, ca, cb))
            pvalue = (1 + sum(x >= statistic for x in null_statistics)) / (
                n_null + 1
            )
            node["graph_null_p"] = float(pvalue)
            node["graph_null_statistic_quantiles"] = {
                "median": float(np.median(null_statistics)),
                "q95": float(np.quantile(null_statistics, 0.95)),
                "maximum": float(max(null_statistics)),
            }
            node["graph_null_resolution_limited"] = (
                1 / (n_null + 1) > local_alpha
            )
            accepted &= pvalue <= local_alpha
        if test in {"both", "bootstrap"}:
            values = [
                cut_statistic(
                    np.asarray(matrix)[np.ix_(support, support)], a, b
                )
                for matrix in bootstrap
            ]
            lower = float(np.quantile(values, local_alpha))
            node["bootstrap_cut_lower_bound"] = lower
            node["bootstrap_cut_statistics"] = values
            accepted &= lower > 0.5
        node["split"] = bool(accepted)
        tests = [f"weakest directional AUC {statistic:.3f}"]
        if "graph_null_p" in node:
            tests.append(
                f"graph-null p {node['graph_null_p']:.3f} "
                f"(alpha {local_alpha:.4f})"
            )
        if "bootstrap_cut_lower_bound" in node:
            tests.append(
                "bootstrap lower bound "
                f"{node['bootstrap_cut_lower_bound']:.3f}"
            )
        log(
            f"{pad}node {node_id} root {root_id} depth {depth}  size "
            f"{len(support)}  {coh_txt}  cut {len(a)}+{len(b)}  "
            + "; ".join(tests)
            + ("  SPLIT" if accepted else "  leaf (boundary not supported)")
        )
        if not accepted:
            node["stop_reason"] = "boundary_not_supported"
            leaves.append(node_id)
            return node_id
        for side, inside, outside in (("A", a, b), ("B", b, a)):
            for i in inside:
                others = [j for j in inside if i != j]
                boundary_rows.append(
                    dict(
                        node=node_id,
                        side=side,
                        residue=int(support[i]),
                        incoming_within_vs_sibling_auc=auc(
                            weight[i, others], weight[i, outside]
                        ),
                        outgoing_within_vs_sibling_auc=auc(
                            weight[others, i], weight[outside, i]
                        ),
                    )
                )
        node["children"] = [
            visit(child, node_id, root_id, depth + 1)
            for child in node["proposed_children"]
        ]
        return node_id

    log(
        f"{method if isinstance(method, str) else 'callable'} split: "
        f"{len(candidates)} parents  test {test}  alpha {alpha}  "
        f"max depth {max_depth}"
    )
    roots = []
    for index, candidate in enumerate(candidates):
        support = (
            candidate["residues"] if isinstance(candidate, dict) else candidate
        )
        roots.append(visit(support, None, index, 0))
    if scorer is not None and nodes:
        supports = [node["residues"] for node in nodes]
        values = (
            scorer.score_batch(supports)
            if hasattr(scorer, "score_batch")
            else [scorer(support) for support in supports]
        )
        if len(values) != len(nodes):
            raise ValueError("scorer returned wrong number of node scores")
        for node, score in zip(nodes, values):
            node.update(_measurement(score))
    units, seen = [], {}
    for node_id in leaves:
        node = nodes[node_id]
        key = tuple(node["residues"])
        if key not in seen:
            seen[key] = len(units)
            units.append(
                dict(
                    unit=len(units),
                    residues=list(key),
                    size=len(key),
                    node_ids=[],
                    root_ids=[],
                    **node["whole_protein"],
                )
            )
            for field in (
                "nu_in",
                "nu_out",
                "theta_in",
                "theta_out",
                "rho_in",
                "rho_out",
                "keff_in",
                "keff_out",
            ):
                if field in node:
                    units[-1][field] = node[field]
        unit = units[seen[key]]
        unit["node_ids"].append(node_id)
        if node["root"] not in unit["root_ids"]:
            unit["root_ids"].append(node["root"])
    return dict(
        method=method if isinstance(method, str) else "callable",
        markov_time=markov_time,
        test=test,
        n_null=n_null,
        alpha=alpha,
        max_depth=max_depth,
        min_size=min_size,
        seed=seed,
        interface_scores_recomputed=scorer is not None,
        incoming_used_for_split_selection=False,
        bootstrap_samples=len(bootstrap),
        roots=roots,
        nodes=nodes,
        units=units,
        boundaries=boundary_rows,
        inference_scope=(
            "graph-exchangeability null; fixed-cut bootstrap uncertainty"
        ),
    )
