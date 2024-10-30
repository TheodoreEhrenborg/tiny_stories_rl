#!/usr/bin/env python3
import time
import torch
import copy
from beartype import beartype
from argparse import ArgumentParser, Namespace
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


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--kl-coefficient", type=float, default=0.0)
    parser.add_argument("--max-generations", type=int, default=2000)
    return parser


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


@beartype
def main(user_args: Namespace):
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
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
    writer.add_scalar("KL coefficent", user_args.kl_coefficient, step)
    input_tokens = (
        torch.tensor(tokenizer("Once upon a time")["input_ids"]).unsqueeze(0).cuda()
    )
    while step <= user_args.max_generations:
        optimizer.zero_grad()
        sequences = []
        rewards = []
        print("Starting group")
        for i in range(rloo_group):
            output_tokens = generate(llm, input_tokens)
            output_text = tokenizer.decode(output_tokens[0])
            raw_reward = get_raw_reward(output_text)
            writer.add_scalar("Raw reward", raw_reward, step + i)
            writer.add_text("Generation", output_text, step + i)
            print(output_text)
            print(f"has raw reward {raw_reward}")

            with torch.no_grad():
                modified_llm_loss = llm(
                    input_ids=output_tokens, labels=output_tokens
                ).loss
                unmodified_llm_loss = unmodified_llm(
                    input_ids=output_tokens, labels=output_tokens
                ).loss

            # There should be a penalty (kl_loss_term<0) if
            # the unmodified LLM thinks the sequence is much less likely than
            # the modified LLM thinks, since then the modified LLM has strayed.
            # That happens when the unmodified LLM has a high cross entropy loss.
            kl_penalty_term = modified_llm_loss - unmodified_llm_loss
            writer.add_scalar("Unscaled KL penalty", kl_penalty_term, step + i)
            scaled_kl_penalty = kl_penalty_term * user_args.kl_coefficient
            writer.add_scalar("Scaled KL penalty", scaled_kl_penalty, step + i)

            reward = float(raw_reward + scaled_kl_penalty)
            writer.add_scalar("Reward", reward, step + i)

            sequences.append(output_tokens)
            rewards.append(reward)
        for i in range(rloo_group):
            other_rewards = copy.deepcopy(rewards)
            this_reward = other_rewards.pop(i)
            mean_other_rewards = sum(other_rewards) / len(other_rewards)
            these_tokens = sequences[i]
            cross_entropy_loss = llm(input_ids=these_tokens, labels=these_tokens).loss
            writer.add_scalar("Cross entropy loss", cross_entropy_loss, step + i)
            normalized_reward = this_reward - mean_other_rewards
            writer.add_scalar("Normalized reward", normalized_reward, step + i)
            loss = cross_entropy_loss * normalized_reward
            loss.backward()
        optimizer.step()
        step += rloo_group
    writer.close()


@beartype
def get_raw_reward(text: str) -> int:
    words = text.split()
    return sum(
        1
        for word, next_word in zip(words, words[1:])
        if word[0].lower() == next_word[0].lower()
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
