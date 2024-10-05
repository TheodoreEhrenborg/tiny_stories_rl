#!/usr/bin/env python3
import torch
from beartype import beartype
from jaxtyping import Int, jaxtyped
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)


@jaxtyped(typechecker=beartype)
def generate(
    llm: GPTNeoForCausalLM, prompt_tokens: Int[torch.Tensor, "1 input_seq_len"]
) -> Int[torch.Tensor, "1 output_seq_len"]:
    return llm.generate(
        prompt_tokens,
        max_length=1000,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )


@beartype
def setup(cuda: bool) -> tuple[GPTNeoForCausalLM, GPT2TokenizerFast]:
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    if cuda:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer
