import pytest
import torch

from weighbridge.testing import make_random_lora, tiny_cohere2, tiny_tokenizer


@pytest.fixture(scope="session")
def tiny_model():
    return tiny_cohere2(seed=0)


@pytest.fixture(scope="session")
def tiny_tok():
    return tiny_tokenizer()


@pytest.fixture()
def random_lora(tiny_model, tmp_path):
    return make_random_lora(tiny_model, tmp_path / "lora", rank=4, alpha=8.0, seed=1)
