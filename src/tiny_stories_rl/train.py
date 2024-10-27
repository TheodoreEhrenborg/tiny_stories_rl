#!/usr/bin/env python3
import torch
import copy
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
def generate(llm: GPTNeoForCausalLM) -> Int[torch.Tensor, "1 output_seq_len"]:
    with torch.no_grad():
        return llm.generate(
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
    unmodified_llm, _ = setup(True)
    rloo_group = 10
    # To increase the probability of a sequence, we take
    # a step to minimize the loss, since loss measures how far we're missing perfect prediction.
    # To decrease the probability, first multiply by a negative
    # reward---minimizing the product will have the effect of maximizing the loss
    optimizer = SGD(llm.parameters(), lr=0.0001)
    step = 0
    while True:
        optimizer.zero_grad()
        sequences = []
        rewards = []
        print("Starting group")
        for _ in range(rloo_group):
            output_tokens = generate(llm)
            output_text = tokenizer.decode(output_tokens[0])
            reward = get_reward(output_text)
            writer.add_scalar("Reward", reward, step)
            writer.add_text("Generation", output_text, step)
            print(output_text)
            print(f"has reward {reward}")
            sequences.append(output_tokens)
            rewards.append(reward)
            step += 1
        for i in range(rloo_group):
            other_rewards = copy.deepcopy(rewards)
            this_reward = other_rewards.pop(i)
            mean_other_rewards = sum(other_rewards) / len(other_rewards)
            these_tokens = sequences[i]
            cross_entropy_loss = llm(input_ids=these_tokens, labels=these_tokens).loss
            scaled_cross_entropy_loss = cross_entropy_loss * (
                this_reward - mean_other_rewards
            )
            kl_loss = 0
            kl_coeff = 0
            loss = scaled_cross_entropy_loss + kl_loss * kl_coeff
            loss.backward()
        optimizer.step()


@beartype
def get_reward(text: str) -> int:
    words = text.split()
    return sum(
        1
        for word, next_word in zip(words, words[1:])
        if word[0].lower() == next_word[0].lower()
    )


if __name__ == "__main__":
    main()
