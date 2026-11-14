"""Tiny stand-ins for unit tests: a random Cohere2 model, a word-level tokenizer that mimics Tiny Aya's
chat markers, and a factory for random PEFT LoRA adapters. Everything runs on CPU in well under a second.

The real model is gated and 6.7 GB; these fixtures let every module be tested offline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from .common import STOP_TOKEN_IDS, TARGET_MODULES

# Special tokens mirror Tiny Aya's Command-style turn markers. Ids 3 and 6 are the two stop tokens,
# exactly as in the real model (common.STOP_TOKEN_IDS == (3, 6)).
SPECIAL_TOKENS = [
    "<PAD>",                      # 0
    "<UNK>",                      # 1
    "<BOS_TOKEN>",                # 2
    "<EOS_TOKEN>",                # 3  (stop)
    "<|START_OF_TURN_TOKEN|>",    # 4
    "<|SYSTEM_TOKEN|>",           # 5
    "<|END_OF_TURN_TOKEN|>",      # 6  (stop)
    "<|USER_TOKEN|>",             # 7
    "<|CHATBOT_TOKEN|>",          # 8
]
assert SPECIAL_TOKENS.index("<EOS_TOKEN>") == STOP_TOKEN_IDS[0]
assert SPECIAL_TOKENS.index("<|END_OF_TURN_TOKEN|>") == STOP_TOKEN_IDS[1]

# A small vocabulary: English, Yoruba and Pidgin words, including refusal and compliance openers.
WORDS = (
    "I Sorry sorry cannot can't help with that . , ! ? Sure Here is how to the a an and of to you "
    "Certainly Step 1 2 3 First make write explain how build bomb steal password hack bank account "
    "recipe cake history poem story weather friend kill time process python code "
    "Mi ò lè ràn ọ́ lọ́wọ́ Ẹ jọ̀wọ́ Mo kò bẹ́ẹ̀ ni Báyìí ni o ṣe Dájúdájú "
    "no fit go help you with am Abeg wetin una dey talk Na so e be make I show you how "
    "You are in contextual safety mode will not provide harmful content"
).split()

PREAMBLE = "You are in contextual safety mode . You will not provide harmful content ."

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ messages[0]['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" + PREAMBLE + "<|END_OF_TURN_TOKEN|>"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for m in loop_messages %}"
    "{% if m['role'] == 'user' %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% elif m['role'] == 'assistant' %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"
)


def tiny_tokenizer():
    """A PreTrainedTokenizerFast with a word-level vocabulary and a Tiny-Aya-like chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {}
    for tok in SPECIAL_TOKENS + WORDS:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<BOS_TOKEN>", eos_token="<EOS_TOKEN>", pad_token="<PAD>", unk_token="<UNK>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "left"
    return fast


def tiny_cohere2(seed: int = 0, n_layers: int = 4, hidden: int = 64, heads: int = 4, kv_heads: int = 2,
                 intermediate: int = 128, vocab: int = 256, dtype: torch.dtype = torch.float32):
    """A random Cohere2ForCausalLM with Tiny Aya's layout (parallel block, GQA, 3 sliding : 1 full attention)."""
    from transformers import Cohere2Config, Cohere2ForCausalLM

    cfg = Cohere2Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=n_layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, max_position_embeddings=512, sliding_window=16,
        pad_token_id=0, bos_token_id=2, eos_token_id=STOP_TOKEN_IDS[0],
    )
    torch.manual_seed(seed)
    model = Cohere2ForCausalLM(cfg).to(dtype).eval()
    return model


def make_random_lora(model, out_dir: str | Path | None = None, rank: int = 4, alpha: float = 8.0, seed: int = 0,
                     target_modules: tuple[str, ...] = TARGET_MODULES, scale: float = 0.05,
                     use_rslora: bool = False, layers_to_transform: list[int] | None = None) -> Path:
    """Create a PEFT LoRA adapter on `model` with random non-zero A and B, save it, and return the folder.

    PEFT initialises B to zero; we overwrite both factors so the adapter actually changes the model.
    The base model passed in is left unmodified (we wrap a deep copy).
    """
    import copy

    from peft import LoraConfig, get_peft_model

    out = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="wb_lora_"))
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(target_modules), lora_dropout=0.0,
                     bias="none", use_rslora=use_rslora, layers_to_transform=layers_to_transform,
                     task_type="CAUSAL_LM")
    peft_model = get_peft_model(copy.deepcopy(model), cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * scale)
    peft_model.save_pretrained(str(out))
    return out


def read_adapter_config(path: str | Path) -> dict:
    return json.loads((Path(path) / "adapter_config.json").read_text())
