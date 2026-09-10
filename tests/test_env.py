import pytest

from jlens_spec import env


def test_require_env_raises_when_unset(monkeypatch):
    monkeypatch.delenv("JLENS_TEST_VAR_XYZ", raising=False)
    with pytest.raises(RuntimeError):
        env.require_env("JLENS_TEST_VAR_XYZ")


def test_require_env_passes_when_set(monkeypatch):
    monkeypatch.setenv("JLENS_TEST_VAR_XYZ", "value")
    env.require_env("JLENS_TEST_VAR_XYZ")  # must not raise


def test_require_env_never_echoes(monkeypatch, capsys):
    monkeypatch.setenv("JLENS_TEST_VAR_XYZ", "supersecretvalue")
    env.require_env("JLENS_TEST_VAR_XYZ")
    out = capsys.readouterr()
    assert "supersecretvalue" not in out.out
    assert "supersecretvalue" not in out.err


def test_expand_band_single_pair_is_inclusive():
    assert env.expand_band([18, 21]) == [18, 19, 20, 21]


def test_expand_band_multiple_blocks_inclusive():
    assert env.expand_band([[18, 20], [40, 42]]) == [18, 19, 20, 40, 41, 42]


def test_expand_band_overlapping_blocks_dedup_and_sort():
    assert env.expand_band([[12, 14], [10, 13]]) == [10, 11, 12, 13, 14]


def test_expand_band_rejects_reversed_pair():
    with pytest.raises(ValueError):
        env.expand_band([30, 18])


def test_expand_band_raises_on_null():
    with pytest.raises(ValueError):
        env.expand_band(None)


def test_config_hash_deterministic(tmp_path):
    p1 = tmp_path / "a.yaml"
    p1.write_text("a: 1\n")
    p2 = tmp_path / "b.yaml"
    p2.write_text("b: 2\n")
    h1 = env.config_hash(str(p1), str(p2))
    h2 = env.config_hash(str(p1), str(p2))
    assert h1 == h2
    p2.write_text("b: 3\n")
    h3 = env.config_hash(str(p1), str(p2))
    assert h3 != h1
