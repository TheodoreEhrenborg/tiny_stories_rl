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


def reference_kl_loss(inputs, targets):
    # Based on https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    loss_pointwise = targets.exp() * (targets - inputs)
    return loss_pointwise.sum() / inputs.size(0)


def test_kl_loss():
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    text1 = torch.tensor([[1, 2, 3, 4, 5]])
    text2 = torch.tensor([[6, 7, 8, 9, 10]])
    logits1 = llm(input_ids=text1, labels=text1).logits
    logits2 = llm(input_ids=text2, labels=text2).logits
    torch_kl_loss = kl_loss_fn(logits1, logits2)
    ref_kl_loss = reference_kl_loss(logits1, logits2)
    assert ref_kl_loss.shape == torch.Size([])
    assert torch.allclose(torch_kl_loss, ref_kl_loss)
