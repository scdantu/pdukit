"""Report global-mode reconstruction from combinations of local PDU bases.

Fits the reversible tICA projection Y(t+lag) = Y(t) K in local coordinates,
using G_reg K = (H + H.T)/2 to match the global tICA reference. Channel
ablation/truncation acts on K, never on a covariance. Spectral mode matching
uses the C00 metric; coverage and one-step prediction are separate readouts.
All results describe the covariance ensemble supplied by the workspace.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def _pinv(matrix, relative_tol=1e-10):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    keep = values > relative_tol * max(float(values.max()), 1e-30)
    inverse = (
        (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
        if keep.any()
        else np.zeros_like(matrix)
    )
    return inverse, int(keep.sum())


def combination_sets(n_units, policy="auto", limit=256, explicit=None):
    """Enumerate bounded combinations, retaining the all-PDU case first."""
    if limit < 1:
        raise ValueError("combination limit must be positive")
    if explicit is not None:
        items = explicit
        requested = len(items)
    elif policy == "all" or (policy == "auto" and n_units <= 8):
        requested = 2**n_units - 1
        items = itertools.chain(
            [tuple(range(n_units))],
            *(
                itertools.combinations(range(n_units), k)
                for k in range(1, n_units)
            ),
        )
    elif policy in {"auto", "singles-pairs"}:
        items = itertools.chain(
            [tuple(range(n_units))],
            ((i,) for i in range(n_units)),
            (
                tuple(j for j in range(n_units) if j != i)
                for i in range(n_units)
            ),
            itertools.combinations(range(n_units), 2),
        )
        requested = 1 + n_units + n_units + n_units * (n_units - 1) // 2
    else:
        raise ValueError("unknown combination policy")
    out, seen = [], set()
    for item in items:
        key = tuple(sorted(set(int(i) for i in item)))
        if not key:
            continue
        if key[0] < 0 or key[-1] >= n_units:
            raise ValueError("mode-recovery PDU index out of range")
        if key in seen:
            continue
        if len(out) >= limit:
            return out, dict(truncated=True, requested_upper_bound=requested)
        seen.add(key)
        out.append(key)
    return out, dict(truncated=False, requested_upper_bound=requested)


def channel_operator(operator, blocks, rank=None):
    """Keep diagonal PDU blocks; truncate each target-vs-rest inflow block.

    rank=None is channels fully on; rank=0 is channels off. A rank-k
    incoming block is fitted jointly against all other selected PDUs,
    rather than truncating each pair separately.
    """
    if rank is None:
        return operator.copy()
    result = np.zeros_like(operator)
    all_indices = np.arange(len(operator))
    for block in blocks:
        block = np.asarray(block, dtype=int)
        result[np.ix_(block, block)] = operator[np.ix_(block, block)]
        others = np.setdiff1d(all_indices, block)
        if rank > 0 and len(block) and len(others):
            cross = operator[np.ix_(others, block)]
            left, values, right = np.linalg.svd(cross, full_matrices=False)
            k = min(int(rank), len(values))
            result[np.ix_(others, block)] = (left[:, :k] * values[:k]) @ right[
                :k
            ]
    return result


def _timescale(value, lag):
    modulus = abs(value)
    if not np.isfinite(modulus) or modulus <= 0 or modulus >= 1:
        return None
    return float(-lag / np.log(modulus))


def recover_global_modes(
    ws,
    units,
    *,
    n_modes=10,
    local_dim=8,
    ranks=(1, 2, 4),
    policy="auto",
    max_combinations=256,
    subsets=None,
    frame_ps=None,
):
    """Compare coverage, mode overlap, prediction and relaxation times.

    The covariance ridge and local basis convention are inherited from ws.
    Overlap and coordinate rank deficiencies are reported per combination;
    channels-off is an intervention on this fitted coordinate model, not a
    claim of physically independent overlapping subsystems.
    """
    if n_modes < 1 or local_dim < 1:
        raise ValueError("mode counts must be positive")
    ranks = sorted(set(int(k) for k in ranks))
    if any(k < 1 for k in ranks):
        raise ValueError("channel ranks must be positive; off is automatic")
    supports = [
        sorted(set(u["residues"] if isinstance(u, dict) else u)) for u in units
    ]
    combinations, enumeration = combination_sets(
        len(supports),
        policy,
        max_combinations,
        explicit=subsets,
    )
    if not supports:
        return dict(status="no_units", rows=[], combinations=[], modes=[])
    c00, c01, c11 = (
        np.asarray(getattr(ws, name), dtype=float)
        for name in ("C00", "C01", "C11")
    )
    dimension = len(c00)
    if (
        not np.isfinite(c00).all()
        or not np.isfinite(c01).all()
        or not np.isfinite(c11).all()
    ):
        raise ValueError("mode recovery requires finite covariance matrices")
    n_modes = min(n_modes, dimension)
    global_ridge = float(ws.ridge) * max(
        float(np.trace(c00)) / dimension, 1e-30
    )
    reference = ws._koopman_basis(
        np.arange(dimension), n_modes, reversible=True
    )
    norm = np.sqrt(np.maximum(np.diag(reference.T @ c00 @ reference), 1e-30))
    reference /= norm
    ref_var0 = np.diag(reference.T @ c00 @ reference)
    ref_var1 = np.diag(reference.T @ c11 @ reference)
    ref_cov = np.diag(reference.T @ c01 @ reference)
    ref_metric = ref_var0 + global_ridge * np.sum(reference**2, axis=0)
    ref_values = ref_cov / np.maximum(ref_metric, 1e-30)
    modes = []
    for i, value in enumerate(ref_values):
        scale = _timescale(value, ws.lag)
        modes.append(
            dict(
                mode=i,
                eigenvalue=float(value),
                timescale_frames=scale,
                timescale_ps=scale * frame_ps
                if scale is not None and frame_ps is not None
                else None,
                tica_prediction_r2=float(
                    (2 * value * ref_cov[i] - value**2 * ref_var0[i])
                    / max(ref_var1[i], 1e-30)
                ),
            )
        )
    bases, owned, cursor = [], [], 0
    for support in supports:
        _, indices, _ = ws._cols(support)
        local = ws._koopman_basis(
            indices, min(local_dim, len(indices)), reversible=True
        )
        basis = np.zeros((dimension, local.shape[1]))
        basis[indices] = local
        variances = np.diag(basis.T @ c00 @ basis)
        keep = variances > 1e-14
        basis = basis[:, keep] / np.sqrt(variances[keep])
        bases.append(basis)
        owned.append(np.arange(cursor, cursor + basis.shape[1]))
        cursor += basis.shape[1]
    basis = np.concatenate(bases, axis=1)
    gram = basis.T @ c00 @ basis
    regularized_gram = gram + global_ridge * (basis.T @ basis)
    lagged = basis.T @ c01 @ basis
    present_targets = basis.T @ c00 @ reference
    future_targets = basis.T @ c01 @ reference
    rows, combination_rows = [], []
    for combination_id, combo in enumerate(combinations):
        columns = np.concatenate([owned[i] for i in combo])
        g = gram[np.ix_(columns, columns)]
        h = lagged[np.ix_(columns, columns)]
        # The assembled operator and reference must use the same reversible
        # convention. Otherwise even a complete basis need not recover tICA.
        h = (h + h.T) / 2
        if not len(g):
            combination_rows.append(
                dict(
                    combination=combination_id,
                    units=list(combo),
                    status="no_nonconstant_coordinates",
                )
            )
            continue
        inverse, effective_rank = _pinv(g)
        regularized_inverse, _ = _pinv(
            regularized_gram[np.ix_(columns, columns)]
        )
        operator = regularized_inverse @ h
        cross = present_targets[columns]
        coefficients = inverse @ cross
        coverage = np.clip(
            np.sum(cross * coefficients, axis=0) / np.maximum(ref_var0, 1e-30),
            0,
            1,
        )
        blocks, start = [], 0
        for unit in combo:
            blocks.append(np.arange(start, start + len(owned[unit])))
            start += len(owned[unit])
        residue_count = sum(len(supports[i]) for i in combo)
        unique_residues = len(set().union(*(set(supports[i]) for i in combo)))
        combination_rows.append(
            dict(
                combination=combination_id,
                units=list(combo),
                coordinates=len(g),
                coordinate_rank=effective_rank,
                redundant_coordinates=len(g) - effective_rank,
                repeated_residue_memberships=residue_count - unique_residues,
                overlapping_supports=residue_count > unique_residues,
                global_feature_ridge=global_ridge,
            )
        )
        settings = [("off", 0), ("on", None)] + [
            (f"rank-{k}", k) for k in ranks
        ]
        for label, rank in settings:
            model = channel_operator(operator, blocks, rank)
            eigenvalues, vectors = np.linalg.eig(model)
            metric_norm = np.real(
                np.sum(vectors.conj() * (g @ vectors), axis=0)
            )
            overlaps = abs(cross.T @ vectors) ** 2
            overlaps /= np.maximum(
                ref_var0[:, None] * metric_norm[None, :], 1e-30
            )
            overlaps = np.clip(overlaps, 0, 1)
            overlaps[:, metric_norm <= 1e-12] = 0
            ii, jj = linear_sum_assignment(-overlaps)
            match = dict(zip(ii.tolist(), jj.tolist()))
            predictions = model @ coefficients
            predicted_var = np.sum(predictions * (g @ predictions), axis=0)
            predicted_cov = np.sum(
                predictions * future_targets[columns], axis=0
            )
            r2 = (2 * predicted_cov - predicted_var) / np.maximum(
                ref_var1, 1e-30
            )
            for mode in range(n_modes):
                selected = match.get(mode)
                value = eigenvalues[selected] if selected is not None else None
                scale = (
                    _timescale(value, ws.lag) if value is not None else None
                )
                reference_scale = modes[mode]["timescale_frames"]
                rows.append(
                    dict(
                        combination=combination_id,
                        units=list(combo),
                        channels=label,
                        mode=mode,
                        coverage_fraction=float(coverage[mode]),
                        mode_overlap_squared=float(overlaps[mode, selected])
                        if selected is not None
                        else 0.0,
                        prediction_r2=float(r2[mode]),
                        matched_eigenvalue_real=float(np.real(value))
                        if value is not None
                        else None,
                        matched_eigenvalue_imag=float(np.imag(value))
                        if value is not None
                        else None,
                        matched_timescale_frames=scale,
                        matched_timescale_ps=scale * frame_ps
                        if scale is not None and frame_ps is not None
                        else None,
                        timescale_relative_error=(
                            float((scale - reference_scale) / reference_scale)
                            if scale is not None and reference_scale
                            else None
                        ),
                        spectral_radius=float(max(abs(eigenvalues))),
                        stable=bool(np.max(abs(eigenvalues)) < 1),
                    )
                )
    off = {
        (r["combination"], r["mode"]): r["prediction_r2"]
        for r in rows
        if r["channels"] == "off"
    }
    for row in rows:
        row["prediction_r2_gain_vs_off"] = (
            row["prediction_r2"] - off[row["combination"], row["mode"]]
        )
    best = {}
    for row in rows:
        key = row["mode"], row["channels"]
        old = best.get(key)
        if old is None or (row["prediction_r2"], -len(row["units"])) > (
            old["prediction_r2"],
            -len(old["units"]),
        ):
            best[key] = row
    return dict(
        status="complete",
        evaluation_ensemble="workspace/discovery, in-sample",
        reference="global reversible tICA on the same features and lag",
        assembly=(
            "[B.T (C00 + global_feature_ridge I) B] K = sym(H); "
            "reversible tICA projection"
        ),
        channels="reduced-propagator target-PDU versus all selected others",
        overlap_interpretation=(
            "coordinate-model ablation; overlapping supports are not "
            "physically independent subsystems"
        ),
        policy=policy,
        enumeration=enumeration,
        local_dim=local_dim,
        n_units=len(supports),
        ranks=ranks,
        lag_frames=int(ws.lag),
        frame_ps=frame_ps,
        modes=modes,
        combinations=combination_rows,
        rows=rows,
        best_combinations=list(best.values()),
    )
