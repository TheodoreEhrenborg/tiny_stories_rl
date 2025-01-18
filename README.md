# tiny_stories_rl

Uses reinforcement learning
to encourage `roneneldan/TinyStories-33M`
to generate stories with alliteration

Docs are [here](https://rl.ehrenborg.dev)

## Installation

If you install
[uv](https://docs.astral.sh/uv/getting-started/installation/),
it'll get the dependencies.

Backup plan: `./build.sh` and `./run.sh` will build and run a Docker container
that has `uv`, in case your system is weird (like my NixOS laptop)
and doesn't work with `uv`.
Once you're in the container,
you can run the commands in the following sections.

## Usage

```bash
uv run src/tiny_stories_rl/train.py
```

The KL penalty coefficient is configurable via `--kl-coefficient`;
see [here](https://rl.ehrenborg.dev/what_is_the_kl_penalty.html) for more.

## Running tests

```bash
uv run pytest tests
```
