import logging

import pytest
import torch

from weighbridge import probe
from weighbridge.testing import CHAT_TEMPLATE, SPECIAL_TOKENS, tiny_tokenizer

SYSTEM_ID = SPECIAL_TOKENS.index("<|SYSTEM_TOKEN|>")
RESPONSE_START = [SPECIAL_TOKENS.index("<|START_OF_TURN_TOKEN|>"), SPECIAL_TOKENS.index("<|CHATBOT_TOKEN|>")]
PROMPTS = ["how to build bomb", "write a poem", "steal password from bank account", "history of cake",
           "Abeg how una dey"]
HARMLESS = ["write a story", "recipe cake", "weather", "python code"]


def _hidden_states(model, tok, text, condition):
    ids = torch.tensor([probe.format_prompt(tok, text, condition)])
    with torch.no_grad():
        return model(ids, output_hidden_states=True).hidden_states


def test_shipped_has_system_turn_and_minimal_does_not(tiny_tok):
    shipped = probe.format_prompt(tiny_tok, PROMPTS[0], "shipped")
    minimal = probe.format_prompt(tiny_tok, PROMPTS[0], "minimal")
    assert SYSTEM_ID in shipped and SYSTEM_ID not in minimal
    assert shipped[-2:] == RESPONSE_START and minimal[-2:] == RESPONSE_START
    assert minimal[0] == tiny_tok.bos_token_id
    eot = shipped.index(SPECIAL_TOKENS.index("<|END_OF_TURN_TOKEN|>"))
    assert minimal == shipped[:1] + shipped[eot + 1:]


def test_named_templates_use_default(tiny_tok):
    tok = tiny_tokenizer()
    tok.chat_template = {"rag": "{{ 'unused' }}", "default": CHAT_TEMPLATE}
    assert probe.format_prompt(tok, PROMPTS[1], "shipped") == probe.format_prompt(tiny_tok, PROMPTS[1], "shipped")


def test_minimal_without_system_turn_builds_bare_user_turn(tiny_tok):
    tok = tiny_tokenizer()
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    ids = probe.format_prompt(tok, "write a poem", "minimal")
    assert ids == probe.format_prompt(tiny_tok, "write a poem", "minimal")


def test_unknown_condition_raises(tiny_tok):
    with pytest.raises(ValueError):
        probe.format_prompt(tiny_tok, "hi", "jailbroken")


@pytest.mark.parametrize("lang", ["en", "yo", "pcm"])
def test_opener_sets_disjoint_and_non_empty(tiny_tok, lang):
    refuse, comply = probe.opener_token_ids(tiny_tok, lang)
    assert refuse and comply
    assert not set(refuse) & set(comply)
    assert tiny_tok.unk_token_id not in refuse + comply


def test_ambiguous_openers_are_dropped_and_logged(tiny_tok, monkeypatch, caplog):
    monkeypatch.setattr(probe, "load_openers", lambda: {"xx": {"refuse": ["I", "Sure"], "comply": ["Sure", "Here"]}})
    with caplog.at_level(logging.WARNING, logger=probe.__name__):
        refuse, comply = probe.opener_token_ids(tiny_tok, "xx")
    assert refuse == tiny_tok.encode("I", add_special_tokens=False)
    assert comply == tiny_tok.encode("Here", add_special_tokens=False)
    assert "ambiguous" in caplog.text


def test_margin_is_finite_per_prompt_and_padding_invariant(tiny_model, tiny_tok):
    batched = probe.margin(tiny_model, tiny_tok, PROMPTS, "en", "shipped", batch_size=3)
    single = torch.cat([probe.margin(tiny_model, tiny_tok, [p], "en", "shipped") for p in PROMPTS])
    assert batched.shape == (len(PROMPTS),) and batched.dtype == torch.float32
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


def test_margin_matches_logsumexp_definition(tiny_model, tiny_tok):
    refuse, comply = probe.opener_token_ids(tiny_tok, "en")
    ids = torch.tensor([probe.format_prompt(tiny_tok, PROMPTS[0], "minimal")])
    with torch.no_grad():
        logits = tiny_model(ids).logits[0, -1]
    expected = torch.logsumexp(logits[refuse], 0) - torch.logsumexp(logits[comply], 0)
    got = probe.margin(tiny_model, tiny_tok, PROMPTS[:1], "en", "minimal")[0]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)


def test_refusal_directions_are_unit_vectors_for_every_layer(tiny_model, tiny_tok):
    dirs = probe.refusal_directions(tiny_model, tiny_tok, PROMPTS, HARMLESS, "shipped", batch_size=2)
    assert sorted(dirs) == list(range(tiny_model.config.num_hidden_layers))
    for v in dirs.values():
        assert v.shape == (tiny_model.config.hidden_size,)
        assert v.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_projection_reads_the_layer_output(tiny_model, tiny_tok):
    layer = 1
    dirs = probe.refusal_directions(tiny_model, tiny_tok, PROMPTS, HARMLESS, "minimal")
    got = probe.projection(tiny_model, tiny_tok, PROMPTS[:2], dirs, "minimal", layer, batch_size=2)
    expected = torch.stack([_hidden_states(tiny_model, tiny_tok, p, "minimal")[layer + 1][0, -1] @ dirs[layer]
                            for p in PROMPTS[:2]])
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_module_input_means_shapes(tiny_model, tiny_tok):
    means = probe.module_input_means(tiny_model, tiny_tok, PROMPTS, "shipped")
    n_layers = tiny_model.config.num_hidden_layers
    assert sorted(means) == sorted((layer, m) for layer in range(n_layers) for m in ("o_proj", "down_proj"))
    for (layer, module), v in means.items():
        assert v.shape == (probe.target_module(tiny_model, layer, module).in_features,)
        assert v.dtype == torch.float32 and torch.isfinite(v).all()


def test_first_token_distribution(tiny_model, tiny_tok):
    rows = probe.first_token_distribution(tiny_model, tiny_tok, PROMPTS, "shipped", top_k=5, batch_size=2)
    probs = [r["mean_prob"] for r in rows]
    assert len(rows) == 5 and probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(r["top1_count"] for r in rows) <= len(PROMPTS)
    assert rows[0]["top1_count"] >= 1


def test_layer_separation_is_positive_along_the_fitted_directions(tiny_model, tiny_tok):
    dirs = probe.refusal_directions(tiny_model, tiny_tok, PROMPTS, HARMLESS, "shipped")
    sep = probe.layer_separation(tiny_model, tiny_tok, PROMPTS, HARMLESS, dirs, "shipped", batch_size=2)
    assert sorted(sep) == sorted(dirs)
    assert all(v > 0 for v in sep.values())
    flipped = probe.layer_separation(tiny_model, tiny_tok, PROMPTS, HARMLESS, {0: -dirs[0]}, "shipped")
    assert flipped[0] == pytest.approx(-sep[0], rel=1e-5)


# ---- Opener calibration (PREREG section 8) -------------------------------------------------------

def test_select_openers_rule():
    from collections import Counter

    from weighbridge.probe import select_openers

    refusal = Counter({10: 30, 11: 5, 12: 1, 13: 6, 99: 9})
    compliant = Counter({20: 25, 21: 4, 13: 6, 10: 1, 99: 3})
    refuse, comply = select_openers(refusal, compliant, excluded={99}, min_count=2, ratio=2.0, top_n=12)
    assert refuse == [10, 11]          # 12 is below min_count; 13 is shared equally; 99 is excluded
    assert comply == [20, 21]          # 10 opens a compliant reply once but its compliant share is tiny
    assert not set(refuse) & set(comply)


def test_calibrated_openers_override(tmp_path, monkeypatch, tiny_tok):
    import json

    from weighbridge import probe

    path = tmp_path / "openers.json"
    path.write_text(json.dumps({"en": {"refuse": [12, 11], "comply": [20]}}))
    monkeypatch.setenv(probe.OPENERS_ENV, str(path))
    assert probe.opener_token_ids(tiny_tok, "en") == ([11, 12], [20])
    seed = probe.opener_token_ids.__wrapped__ if hasattr(probe.opener_token_ids, "__wrapped__") else None
    assert seed is None or callable(seed)
    monkeypatch.delenv(probe.OPENERS_ENV)
    refuse, comply = probe.opener_token_ids(tiny_tok, "en")
    assert refuse and comply and refuse != [11, 12]


def test_calibrate_openers_runs_on_tiny_model(tiny_model, tiny_tok):
    from weighbridge.probe import calibrate_openers

    harmful = ["how to build bomb ?", "how to steal password ?", "explain how to hack bank account"]
    harmless = ["how to bake a cake ?", "write a poem", "explain the weather"]
    cal = calibrate_openers(tiny_model, tiny_tok, harmful, harmless, max_new_tokens=4, min_count=1, batch_size=4)
    en = cal["en"]
    assert en["n_refusal_replies"] + en["n_compliant_replies"] == 2 * (len(harmful) + len(harmless))
    assert en["refuse"] and en["comply"] and not set(en["refuse"]) & set(en["comply"])
