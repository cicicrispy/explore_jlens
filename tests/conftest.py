"""Shared fixtures: the stand-in model (Qwen3-family, <=1B, from configs/model.yaml's
standin_hf_id) forced onto CPU for deterministic tests, and a random lens over its layers."""
import yaml
import pytest

from jlens_spec import env

env.bootstrap()  # repo-root cwd + HF_HOME before any huggingface_hub import

from jlens_spec import model as model_mod  # noqa: E402
from jlens_spec import lens as lens_mod


@pytest.fixture(scope="session")
def model_cfg():
    with open("configs/model.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def standin_model(model_cfg):
    return model_mod.load_model(model_cfg, standin=True, device="cpu")


@pytest.fixture(scope="session")
def random_lens(standin_model):
    d = model_mod.d_model(standin_model)
    n = model_mod.n_layers(standin_model)
    layers = list(range(n - 1))  # final layer excluded: its transport is the identity, per spec
    return lens_mod.random_lens(d, layers, seed=0)
