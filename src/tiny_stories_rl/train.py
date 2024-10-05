#!/usr/bin/env python3
from beartype import beartype
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)


@beartype
def setup(cuda: bool) -> tuple[GPTNeoForCausalLM, GPT2TokenizerFast]:
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    if cuda:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer
