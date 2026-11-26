"""Known-boundary, uncertainty and interface-independence tests."""

import numpy as np
import pytest

from pdukit.cog.units.split import SPLITTERS, parent_cohesion, split_candidates
from pdukit.io.results.stage_files import write_split
from pdukit.tests.cfg import parse_args


def blocks():
    rho = np.full((12, 12), 0.1)
    rho[:6, :6] = 0.8
    rho[6:, 6:] = 0.8
    np.fill_diagonal(rho, 0)
    return rho


def score(support):
    return dict(
        nu_in=20 - len(support),
        nu_out=100 + len(support),
        theta_in=0.1,
        theta_out=0.1,
        rho_in=[0.3],
        rho_out=[0.9],
    )


@pytest.mark.parametrize("method", list(SPLITTERS))
def test_recover_boundary_without_channel_veto(method):
    rho = blocks()
    result = split_candidates(
        rho,
        [list(range(12))],
        score,
        method=method,
        n_null=39,
        bootstrap=[rho, rho],
        seed=1,
    )
    assert sorted(u["residues"] for u in result["units"]) == [
        list(range(6)),
        list(range(6, 12)),
    ]
    assert result["nodes"][0]["split"]
    # Splitting makes incoming count WORSE here; that must not veto cohesion.
    assert all(
        u["nu_in"] > result["nodes"][0]["nu_in"] for u in result["units"]
    )
    assert all(u["nu_out"] > 0 for u in result["units"])
    assert len(result["boundaries"]) == 12


@pytest.mark.parametrize("method", list(SPLITTERS))
def test_no_split_when_parent_is_homogeneous(method):
    rho = np.full((12, 12), 0.4)
    np.fill_diagonal(rho, 0)
    result = split_candidates(
        rho, [list(range(12))], method=method, test="graph-null", n_null=39
    )
    assert len(result["units"]) == 1


def test_whole_protein_cohesion_cannot_replace_within_parent():
    rho = np.full((12, 12), 0.01)
    rho[:6, :6] = 0.5
    np.fill_diagonal(rho, 0)
    assert parent_cohesion(rho, [0, 1, 2])["cohesion"] > 0.5
    assert parent_cohesion(rho, [0, 1, 2], list(range(6)))["cohesion"] == 0.5


def test_uncertain_cut_is_retained_as_unsplit():
    rho = blocks()
    uncertain = np.full_like(rho, 0.4)
    np.fill_diagonal(uncertain, 0)
    result = split_candidates(
        rho,
        [list(range(12))],
        test="bootstrap",
        bootstrap=[uncertain, uncertain],
    )
    assert not result["nodes"][0]["split"]
    assert result["nodes"][0]["bootstrap_cut_lower_bound"] == 0.5


def test_duplicate_leaves_keep_all_parent_provenance(tmp_path):
    rho = blocks()
    result = split_candidates(rho, [list(range(12))] * 2, score, test="none")
    assert len(result["units"]) == 2
    assert all(u["root_ids"] == [0, 1] for u in result["units"])
    write_split(tmp_path, result)
    assert (tmp_path / "incoming_split.json").exists()


def test_bootstrap_is_not_silently_omitted():
    with pytest.raises(ValueError, match="resampled"):
        split_candidates(blocks(), [list(range(12))])


def test_callable_proposal_can_be_plugged_in():
    def propose(rho, min_size, **kwargs):
        half = len(rho) // 2
        return np.arange(half), np.arange(half, len(rho)), 1.0

    result = split_candidates(
        blocks(), [list(range(12))], method=propose, test="none"
    )
    assert len(result["units"]) == 2


def test_cli_defaults_and_parameter_roundtrip(tmp_path):
    args = parse_args(
        ["--planted", "-o", "out", "--grow-objective", "incoming"]
    )
    assert args.incoming_postprocess == "split"
    assert args.split_method == "recursive-spectral"
    assert args.split_test == "both"
    assert args.mode_recovery == "on"
    ppf = tmp_path / "split.ppf"
    ppf.write_text(args.to_ppf())
    again = parse_args(
        [
            "--planted",
            "-o",
            "out",
            "-p",
            str(ppf),
            "--split-method",
            "markov-stability",
            "--split-markov-time",
            "4",
        ]
    )
    assert again.split_method == "markov-stability"
    assert again.split_markov_time == 4
    assert again.recovery_channel_ranks == [1, 2, 4]


def test_readers_use_final_split_supports(tmp_path):
    import json

    from pdukit.analysis.compare import _units
    from pdukit.analysis.summary import _row

    result = dict(
        units_discovery=[dict(cohesion=0.9, residues=list(range(6)))],
        units_final=[dict(residues=[0, 1, 2]), dict(residues=[3, 4, 5])],
        data=dict(n_residues=6),
        incoming_split=dict(
            method="recursive-spectral", nodes=[dict(split=True)]
        ),
    )
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    assert _units(tmp_path)[0] == [{0, 1, 2}, {3, 4, 5}]
    assert _row(path)["n_units"] == 2


def test_merge_overlapping_families_then_split():
    """Versions of the same region merge (incl. nested ones); others do not."""
    from pdukit.cog.units.split import merge_overlapping

    grown = [
        list(range(0, 6)),
        list(range(1, 6)),  # nested version: Jaccard 0.83
        [0, 1, 2],  # small nested: Jaccard 0.5
        list(range(6, 12)),
        list(range(7, 12)),
    ]  # second region
    merge = merge_overlapping(grown, 0.5)
    assert merge["n_families"] == 2
    fam = {tuple(f["residues"]): f for f in merge["families"]}
    assert set(fam) == {tuple(range(6)), tuple(range(6, 12))}
    f0 = fam[tuple(range(6))]
    assert f0["members"] == [0, 1, 2]
    assert f0["core"] == [1, 2]  # found by every member
    assert f0["residue_frequency"][0] == pytest.approx(2 / 3)
    # merged parent spanning two blocks is re-tested and split by the next step
    rho = blocks()
    merge = merge_overlapping([list(range(0, 8)), list(range(4, 12))], 0.5)
    assert merge["n_families"] == 1 and merge["families"][0]["size"] == 12
    result = split_candidates(
        rho,
        [merge["families"][0]["residues"]],
        score,
        n_null=39,
        bootstrap=[rho, rho],
        seed=1,
    )
    assert sorted(u["residues"] for u in result["units"]) == [
        list(range(6)),
        list(range(6, 12)),
    ]


def test_merge_overlap_option_and_separate_regions():
    from pdukit.cog.units.split import merge_overlapping

    merge = merge_overlapping(
        [[0, 1, 2, 3], [3, 4, 5, 6]], 0.5
    )  # share 1 of 4
    assert merge["n_families"] == 2
    with pytest.raises(ValueError):
        merge_overlapping([[0, 1]], 0.0)


def test_merge_does_not_chain_through_loose_versions():
    """Single linkage would join A-B-C; the majority-core rule must not."""
    from pdukit.cog.units.split import merge_overlapping

    A, B, C = list(range(0, 6)), list(range(3, 9)), list(range(6, 12))
    merge = merge_overlapping([A, A, B, C], 0.5)
    assert merge["n_families"] == 2
    first = merge["families"][0]
    assert first["members"] == [0, 1, 2] and first["majority"] == list(
        range(6)
    )
    assert merge["families"][1]["members"] == [3]


def _scorer(nu_union):
    def scorer(support):
        return dict(
            nu_in=nu_union,
            nu_out=1.0,
            theta_in=0.1,
            theta_out=0.1,
            rho_in=[0.3],
            rho_out=[0.3],
        )

    return scorer


def test_union_kept_only_if_incoming_not_worse_than_best_member():
    from pdukit.cog.units.split import merge_overlapping

    X = dict(residues=list(range(0, 6)), nu_in=2.0)
    Y = dict(residues=list(range(2, 8)), nu_in=3.0)
    worse = merge_overlapping([Y, X], 0.5, _scorer(5.0))["families"][0]
    assert worse["decision"] == "best_member"
    assert worse["residues"] == list(range(0, 6)) and worse["union_size"] == 8
    better = merge_overlapping([Y, X], 0.5, _scorer(1.5))["families"][0]
    assert better["decision"] == "union" and better["residues"] == list(
        range(8)
    )
    single = merge_overlapping([X], 0.5, _scorer(9.0))["families"][0]
    assert single["decision"] == "single" and single["residues"] == list(
        range(6)
    )
