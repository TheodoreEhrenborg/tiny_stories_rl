#!/usr/bin/env bash
command="/root/.cargo/bin/uv run tensorboard --logdir=/results/ --port 6010 --host 0.0.0.0"
command=$command ./run.sh -v "$(realpath ~/projects/results/rl_results)":/results -p 6010:6010
