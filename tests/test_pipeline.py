"""pipeline: M2/M3's gate on the M1 validation run, and the settings files a real run copies."""
import pytest
import yaml

from jlens_spec import io as io_mod
from jlens_spec import pipeline


def _m1_run(tmp_path, name, passed):
    d = tmp_path / "runs" / "M1" / name
    d.mkdir(parents=True)
    (d / "check3_positive_control.yaml").write_text(yaml.safe_dump({"passed": passed, "alphas_run": [1.0, 2.0]}))
    s = tmp_path / "store" / "runs" / "M1" / name
    s.mkdir(parents=True)
    (s / "summary.md").write_text("finished")


def test_a_failed_positive_control_needs_the_humans_acceptance_of_that_exact_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = io_mod.LocalStore(tmp_path / "store")
    _m1_run(tmp_path, "validate_ok", passed=True)
    _m1_run(tmp_path, "validate_a", passed=False)
    acc, exp = tmp_path / "acc.yaml", {"m1_validate_run": "validate_a"}

    # a passed control needs no acceptance file at all
    assert "control passed" in pipeline.check_m1_passed({"m1_validate_run": "validate_ok"}, store, str(acc))
    for refused in ({"validate_run": None, "reason": "x"},            # not filled
                    {"validate_run": "validate_b", "reason": "x"},    # another run
                    {"validate_run": "validate_a", "reason": "  "}):  # no reason
        acc.write_text(yaml.safe_dump(refused))
        with pytest.raises(SystemExit, match="did NOT pass"):
            pipeline.check_m1_passed(exp, store, str(acc))
    acc.write_text(yaml.safe_dump({"validate_run": "validate_a", "reason": "an effect, smaller than the paper's"}))
    line = pipeline.check_m1_passed(exp, store, str(acc))
    assert "NOT passed" in line and "ACCEPTED" in line and "an effect, smaller than the paper's" in line


def test_store_dir_sends_a_real_runs_uploads_to_a_folder_instead_of_the_hub(tmp_path):
    dry_exp = {"dryrun": {"store": str(tmp_path / "dry")}}
    assert isinstance(pipeline.store_for(dry_exp), io_mod.LocalStore)
    assert isinstance(pipeline.store_for({}), io_mod.HFStore)
    store = pipeline.store_for({}, store_dir=str(tmp_path / "backup"))
    assert isinstance(store, io_mod.NoUploadStore)
    assert store.local.root == tmp_path / "backup" and str(tmp_path / "backup") in store.name


def test_real_runs_copy_the_acceptance_file_and_keep_the_bands_file_last():
    real = pipeline.settings_files({})
    assert real[-1] == "configs/bands.yaml" and pipeline.ACCEPTANCE_FILE in real
    dry = pipeline.settings_files({"dryrun": {"bands_file": "configs/experiments/dryrun/bands.yaml"}})
    assert dry[-1] == "configs/experiments/dryrun/bands.yaml" and pipeline.ACCEPTANCE_FILE not in dry
