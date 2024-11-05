# tiny_stories_rl

Work in progress


Reinforcement learning

Docs 
## Installtion 

If you install 
[uv](https://docs.astral.sh/uv/getting-started/installation/),
it'll handle getting the dependencies.

Backup plan: `./build.sh` and `./run.sh` will build and run a Docker container
that has `uv`, in case your system is weird (like my NixOS laptop) 
and doesn't work with `uv`.
Once you're in the container, 
you can run the commands in the following sections.

## Usage

``` bash
uv run src/tiny_stories_rl/train.py
```


## Tests

``` bash
uv run pytest tests
```
