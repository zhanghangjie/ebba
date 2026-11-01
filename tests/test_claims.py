"""Registry schema, table transcription and exact-tier smoke tests."""

from __future__ import annotations

import pytest

from ebba.claims import load_registry, run
from ebba.claims import tables
from ebba.claims.formulas import a02_clamp_semantics, a05_nte_identity


def test_registry_schema():
    registry = load_registry()
    assert len(registry) == 39
    required = {"tier", "scope", "title_zh", "paper_ref", "notebook", "tolerance"}
    for cid, spec in registry.items():
        assert cid.startswith(("A", "B", "C"))
        assert required <= set(spec), (cid, required - set(spec))
        assert spec["tier"] in {"A", "B", "C"}
        assert spec["scope"] in {"exact", "proxy"}
        assert spec["notebook"].endswith(("overview", "audit", "theory", "detection", "removal", "verdicts"))


def test_tier_sizes():
    registry = load_registry()
    tiers = {"A": 0, "B": 0, "C": 0}
    for spec in registry.values():
        tiers[spec["tier"]] += 1
    assert tiers == {"A": 20, "B": 14, "C": 5}


def test_table_row_counts_and_ranges():
    t3 = tables.table3_rows()
    t4 = tables.table4_rows()
    assert len(t3) == 15 and len(t4) == 15
    for row in t3:
        for method in tables.TABLE3_METHODS:
            for key in (f"{method}_pdr", f"{method}_f1"):
                assert 0.0 <= row[key] <= 1.0
    for row in t4:
        for method in tables.TABLE4_METHODS:
            assert 0.0 <= row[f"{method}_ba"] <= 100.0
            assert 0.0 <= row[f"{method}_asr"] <= 100.0


def test_table5_and_table7_are_identical():
    assert tables.TABLE5 == tables.TABLE7


def test_a02_clamp_is_confirmed():
    result = a02_clamp_semantics()
    assert result.verdict.value == "CONFIRMED"


def test_a05_identity_is_confirmed():
    result = a05_nte_identity()
    assert result.verdict.value == "CONFIRMED"
    assert result.measured["max_identity_error"] < 1e-9


def test_a20_lambda3_hits_the_k10_bound():
    result = run("A20", write=False)
    assert result.verdict.value == "CONTRADICTED"
    assert abs(result.measured["supremum_z"] - 3.0) < 1e-12
    assert result.measured["z_extremal_equal_others"] <= 3.0 + 1e-9
    assert not result.measured["strict_margin_possible_K10"]


def test_a13_gtsrb_typo_is_caught():
    result = run("A13", write=False)
    assert result.verdict.value == "CONTRADICTED"


@pytest.mark.parametrize("cid", ["C01", "C02", "C03", "C04", "C05"])
def test_tier_c_is_recorded_as_not_reproducible(cid):
    result = run(cid, write=False)
    assert result.verdict.value == "NOT_REPRODUCIBLE"
    assert result.measured["needed"]


def test_every_claim_has_an_implementation_route():
    from ebba.claims.model import _module_for

    registry = load_registry()
    for cid in registry:
        assert _module_for(cid) in {"formulas", "theory", "audit", "empirical", "limits"}
