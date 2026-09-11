import pandas as pd
import torch
import torch.nn.functional as F

from jlens_spec import lens as lens_mod
from jlens_spec import loading as loading_mod


def test_pair_score_hand_computed():
    df = pd.DataFrame(
        [
            {"stimulus_id": "sp_01", "class": "matrix", "layer": 5, "token": "Spanish", "cos": 0.8},
            {"stimulus_id": "sp_01", "class": "matrix", "layer": 5, "token": "Spanish", "cos": 0.6},
            {"stimulus_id": "fr_01", "class": "matrix", "layer": 5, "token": "French", "cos": 0.9},
            {"stimulus_id": "fr_01", "class": "intrusion", "layer": 5, "token": "Spanish", "cos": 0.5},
            {"stimulus_id": "sp_01", "class": "intrusion", "layer": 5, "token": "French", "cos": 0.7},
        ]
    )
    score = loading_mod.pair_score(df, ("Spanish", "French"), band=[5])
    # es-language positions: sp_01/matrix (Spanish tokens) + fr_01/intrusion (Spanish token)
    expected_es = (0.8 + 0.6 + 0.5) / 3
    # fr-language positions: fr_01/matrix (French token) + sp_01/intrusion (French token)
    expected_fr = (0.9 + 0.7) / 2
    assert abs(score - min(expected_es, expected_fr)) < 1e-9


def test_matrix_lang_inference_from_stimulus_id():
    assert loading_mod._matrix_lang_of("sp_03") == "es"
    assert loading_mod._matrix_lang_of("fr_07") == "fr"
    assert loading_mod._matrix_lang_of("other") is None


def test_cos_in_range_and_rank_at_least_1(standin_model, random_lens):
    from jlens_spec import model as model_mod

    h = torch.randn(4, random_lens.d_model, device=model_mod.device_of(standin_model))
    l = random_lens.layers[0]
    v = lens_mod.lens_vectors(standin_model, random_lens, ["the"], l)[0]
    cos = F.cosine_similarity(h, v.unsqueeze(0), dim=-1)
    assert torch.all(cos >= -1 - 1e-5) and torch.all(cos <= 1 + 1e-5)

    tok_id = standin_model.tokenizer.encode("the", add_special_tokens=False)[0]
    rank = lens_mod.token_rank(standin_model, random_lens, h, l, tok_id)
    assert torch.all(rank >= 1)


def test_prompt_pass_saves_every_position_and_layer_in_one_pass(standin_model):
    """M2's one forward pass per prompt: loadings for every position x lens layer x language token,
    the top-k readout at every position x layer (identical to lens.readout), and the model's own
    answer logits."""
    import json

    import numpy as np
    import yaml

    from jlens_spec import model as model_mod
    from jlens_spec import prompts as prompts_mod

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    with open("configs/prompt_format.yaml") as f:
        fmt = prompts_mod.make_fmt(standin_model.tokenizer, stim, yaml.safe_load(f))
    p = prompts_mod.build_prompt(stim["passages"][0], "report", fmt)
    lens = lens_mod.random_lens(model_mod.d_model(standin_model), [0, 4, 9], seed=3)
    tok = standin_model.tokenizer
    ids = [tok.encode(t, add_special_tokens=False)[0] for t in (" Spanish", " French")]
    loadings, topk, answer = loading_mod.prompt_pass(standin_model, lens, p, ids, save_k=7)

    n = len(p.input_ids)
    assert len(loadings) == n * 3 * 2 and topk.num_rows == n * 3
    assert loadings["cos"].between(-1 - 1e-5, 1 + 1e-5).all() and (loadings["rank"] >= 1).all()
    first = loadings[(loadings["layer"] == 4) & (loadings["token_id"] == ids[0])]
    assert list(first["class"]) == p.classes and list(first["pos"]) == list(range(n))
    with standin_model.trace(p.input_ids):
        h = model_mod.layer_output(standin_model, 4).float()[0].save()
        clean = standin_model.output.logits[0, p.metric_pos].float().save()
    want_ids, _ = lens_mod.readout(standin_model, lens, h, 4, k=7)
    t = topk.to_pandas()
    got = np.stack(t[t["layer"] == 4].sort_values("pos")["topk_ids"].to_numpy())
    assert np.array_equal(got, want_ids.cpu().numpy())
    assert torch.allclose(answer.cpu(), clean.cpu(), atol=1e-4)
