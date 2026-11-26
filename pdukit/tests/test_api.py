"""The four-object route: Config, Ensemble, Cog, Result."""

import pytest

pytest.importorskip("MDAnalysis")

from pdukit import Cog, Config, Ensemble, Result  # noqa: E402

PARAMS = __import__("pathlib").Path(__file__).parents[1] / "params"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """Run the demo once through the public objects."""
    root = tmp_path_factory.mktemp("api")
    ens = Ensemble.demo(root / "data")
    cfg = Config.from_ppf(PARAMS / "tutorial.ppf")
    cfg.feature_cache = str(root / "cache")
    return Cog(cfg).run(ens, out=root / "run")


def test_run_gives_units_and_protein(result):
    """The demo's four planted blocks are found and labelled."""
    assert len(result.units) == 4
    prot = result.protein
    assert len(prot) == 40
    assert prot.counts().get("core", 0) > 0
    assert prot.residue(0).resname == "ALA"


def test_tables_figure_pml_and_save(result, tmp_path):
    """Tables are DataFrames; figure and script files are written."""
    pytest.importorskip("pandas")
    assert len(result.table("units")) == 4
    assert result.table("residue_scores").shape[0] > 0
    pytest.importorskip("matplotlib")
    assert result.plot_units().stat().st_size > 0
    assert "pdu_u0" in result.save_pml().read_text()
    again = result.save(tmp_path / "copy")
    assert isinstance(again, Result) and len(again.units) == 4


def test_view_builds_when_py3dmol_present(result):
    """result.view() returns a py3Dmol view."""
    pytest.importorskip("py3Dmol")
    assert result.view() is not None


def test_feature_source_chooses_modes_and_caches_the_report(tmp_path):
    """rse-components = -1 picks K from rse-coverage; the cache keeps it."""
    from pdukit.cog.features import FeatureSource

    ens = Ensemble.demo(tmp_path / "data", frames=400)
    cfg = Config.from_ppf(PARAMS / "tutorial.ppf")
    cfg.update(
        rse_components=-1, rse_coverage=0.9,
        feature_cache=str(tmp_path / "cache"), outdir=str(tmp_path / "o"),
        xtc=[str(p) for p in ens.trajectories],
        xtc_topology=str(ens.topology),
    )
    first = FeatureSource(cfg, {}).get()
    rep = first["rse_report"]
    assert first["feats"].shape[2] == rep["n_components"] >= 2
    assert rep["chosen"] and len(rep["coverage_kept"]) == 40
    again = FeatureSource(cfg, {}).get()
    assert again["meta"]["cache_hit"]
    assert again["rse_report"]["n_components"] == rep["n_components"]
