"""Tests for the homoglyph generation logic in dataGeneration.py."""
import random

import pytest

pytest.importorskip("pandas")  # dataGeneration imports pandas at module load

import dataGeneration as dg


def test_glyphs_table_is_populated():
    # Every lowercase ascii letter should have at least one confusable.
    for ch in "abcdefghijklmnopqrstuvwxyz":
        assert ch in dg.glyphs
        assert len(dg.glyphs[ch]) >= 1


def test_homo_gen_1_substitutes_a_confusable():
    random.seed(42)
    domain = "google"
    result = dg.homo_gen_1(domain)
    assert isinstance(result, str)
    # Exactly one homoglyph substitution should change the string.
    assert result != domain
    assert len(result) >= len(domain)  # some glyphs expand (e.g. b -> lb)


def test_homo_gen_2_applies_two_substitutions():
    random.seed(7)
    domain = "facebook"
    result = dg.homo_gen_2(domain)
    assert isinstance(result, str)
    assert result != domain


def test_homo_gen_1_always_modifies_valid_input():
    # Across many seeds/domains the generator must return a modified string
    # (never None, which only happens when the domain has no replaceable chars).
    domains = ["google", "amazon", "facebook", "wikipedia", "github"]
    produced_non_ascii = False
    for seed in range(25):
        random.seed(seed)
        for domain in domains:
            result = dg.homo_gen_1(domain)
            assert result is not None
            assert isinstance(result, str)
            assert result != domain
            if any(ord(c) > 127 for c in result):
                produced_non_ascii = True
    # At least some substitutions should draw non-ASCII Unicode confusables.
    assert produced_non_ascii
