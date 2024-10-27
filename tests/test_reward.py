#!/usr/bin/env python3
from tiny_stories_rl.train import get_reward
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)


def test_reward():
    assert get_reward("") == 0
    assert get_reward("cow") == 0
    assert get_reward("cow cat") == 1
    assert get_reward("dog horse") == 0
    assert get_reward("duck goose duck") == 0
    assert get_reward("duck duck goose") == 1
    assert get_reward("goose duck duck goose") == 1
    assert get_reward("goose goose duck duck") == 2
    assert get_reward("goose gopher giraffe duck") == 2
    assert get_reward("goose gopher giraffe gorilla") == 3
    assert get_reward("Goose gopher Giraffe gorilla") == 3


def test_dim():
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    text = torch.tensor([[1, 2, 3, 4, 5]])
    llm_logits = llm(input_ids=text, labels=text).logits
    assert llm_logits.shape == torch.Size([1, 5, 50257])
