#!/usr/bin/env python3
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
    parser.add_argument("--kl_coefficient", type=float, default=0.0)
    return parser


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


@beartype
def main(user_args: Namespace):
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
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    writer.add_scalar("KL coefficent", user_args.kl_coefficient, step)
    while True:
        optimizer.zero_grad()
        sequences = []
        rewards = []
        print("Starting group")
        for i in range(rloo_group):
            output_tokens = generate(llm)
            output_text = tokenizer.decode(output_tokens[0])
            reward = get_reward(output_text)
            writer.add_scalar("Reward", reward, step + i)
            writer.add_text("Generation", output_text, step + i)
            print(output_text)
            print(f"has reward {reward}")
            sequences.append(output_tokens)
            rewards.append(reward)
        for i in range(rloo_group):
            other_rewards = copy.deepcopy(rewards)
            this_reward = other_rewards.pop(i)
            mean_other_rewards = sum(other_rewards) / len(other_rewards)
            these_tokens = sequences[i]
            llm_output = llm(input_ids=these_tokens, labels=these_tokens)
            cross_entropy_loss = llm_output.loss
            writer.add_scalar("Cross entropy loss", cross_entropy_loss, step + i)
            scaled_cross_entropy_loss = cross_entropy_loss * (
                this_reward - mean_other_rewards
            )
            writer.add_scalar(
                "Scaled cross entropy loss", scaled_cross_entropy_loss, step + i
            )
            llm_logits = llm_output.logits
            with torch.no_grad():
                unmodified_llm_logits = unmodified_llm(
                    input_ids=these_tokens, labels=these_tokens
                ).logits
            kl_loss = kl_loss_fn(llm_logits, unmodified_llm_logits)
            writer.add_scalar("Unscaled KL loss", kl_loss, step + i)
            loss = scaled_cross_entropy_loss + kl_loss * user_args.kl_coefficient
            loss.backward()
        optimizer.step()
        step += rloo_group


@beartype
def get_reward(text: str) -> int:
    words = text.split()
    return sum(
        1
        for word, next_word in zip(words, words[1:])
        if word[0].lower() == next_word[0].lower()
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
