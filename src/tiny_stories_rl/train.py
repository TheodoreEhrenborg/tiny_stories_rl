#!/usr/bin/env python3
import torch
from beartype import beartype
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from coolname import generate_slug
from jaxtyping import Int, jaxtyped
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)

# TODO Can I silence the attention error by passing the attention mask around?


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
    output_dir = f"/results/{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)
    llm, tokenizer = setup(True)
    # To increase the probability of a sequence, we take
    # a step to minimize the loss, since loss measures how far we're missing perfect prediction.
    # To decrease the probability, first multiply by a negative
    # reward---minimizing the product will have the effect of maximizing the loss
    optimizer = SGD(llm.parameters(), lr=0.0001)
    step = 0
    while True:
        optimizer.zero_grad()
        input_tokens = (
            torch.tensor(tokenizer("Once upon a time")["input_ids"]).unsqueeze(0).cuda()
        )
        output_tokens = generate(llm, input_tokens)
        output_text = tokenizer.decode(output_tokens[0])
        reward = get_reward(output_text)
        writer.add_scalar("Reward", reward, step)
        print(output_text)
        loss = llm(input_ids=output_tokens, labels=output_tokens).loss
        (loss * reward).backward()
        optimizer.step()
        step += 1


@beartype
def get_reward(text: str) -> int:
    return len([word for word in text.split() if word[0].lower() == "a"])


if __name__ == "__main__":
    main()
