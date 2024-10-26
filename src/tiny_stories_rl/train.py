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
    with torch.no_grad():
        return llm.generate(
            prompt_tokens,
            max_length=100,
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


def main():
    llm, tokenizer = setup(False)
    input_tokens = torch.tensor(tokenizer("Once upon a time")["input_ids"]).unsqueeze(0)
    print(input_tokens.shape)
    output_tokens = generate(llm, input_tokens)
    print(output_tokens.shape)
    entire_sequence = torch.cat((input_tokens, output_tokens), dim=1)
    foo = llm(input_ids=entire_sequence, labels=entire_sequence)
    print(foo)


if __name__ == "__main__":
    main()
