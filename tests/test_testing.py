import torch

from weighbridge.common import STOP_TOKEN_IDS
from weighbridge.testing import read_adapter_config


def test_tiny_model_runs(tiny_model):
    out = tiny_model(input_ids=torch.randint(9, 200, (2, 8)))
    assert out.logits.shape == (2, 8, 256)


def test_tokenizer_template_and_stop_tokens(tiny_tok):
    ids = tiny_tok.apply_chat_template([{"role": "user", "content": "how to bake a cake ?"}],
                                       add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    text = tiny_tok.decode(ids)
    assert "<|SYSTEM_TOKEN|>" in text and text.endswith("<|CHATBOT_TOKEN|>")
    assert tiny_tok.convert_tokens_to_ids("<EOS_TOKEN>") == STOP_TOKEN_IDS[0]
    assert tiny_tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>") == STOP_TOKEN_IDS[1]


def test_random_lora_changes_model(tiny_model, random_lora):
    from peft import PeftModel
    import copy
    cfg = read_adapter_config(random_lora)
    assert cfg["r"] == 4
    x = torch.randint(9, 200, (1, 6))
    base = tiny_model(input_ids=x).logits
    pm = PeftModel.from_pretrained(copy.deepcopy(tiny_model), str(random_lora)).eval()
    assert not torch.allclose(pm(input_ids=x).logits, base)
