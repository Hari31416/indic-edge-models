---
name: kagglex
description: Install and use the kagglex module to run Python scripts, modules, or experiments on remote Kaggle GPUs and TPUs, stream remote execution logs, pull output artifacts, manage Kaggle datasets, or run interactive REPL commands on active Kaggle notebooks. Use whenever the user asks to run code on Kaggle, train models on cloud GPUs or TPUs, offload heavy compute, dispatch remote training jobs, run torchrun on Kaggle, or interact with Kaggle notebook sessions.
---

# kagglex

Execute local Python code, modules, and experiments on Kaggle GPUs and TPUs.

## Overview

Use this skill when you need cloud compute (Nvidia T4 x2, P100, or TPU v3-8) to train models, evaluate checkpoints, run batch inference, or test code interactively on Kaggle hardware.

The module packages your local workspace, injects dependencies and secrets, uploads the payload to Kaggle, triggers execution, streams remote logs, and downloads resulting artifacts.

## Installation and setup

Install the CLI using `uv`. Pick the installation method that matches your environment:

Install globally as a CLI tool:

```bash
uv tool install kagglex
```

Install into the active Python virtual environment:

```bash
uv pip install kagglex
```

When working inside the `kagglex` repository itself, install in editable mode:

```bash
uv pip install -e .
```

Run ephemerally without permanent installation:

```bash
uv run --with kagglex kagglex --help
```

## Pre-flight verification

Before dispatching jobs, verify credentials and network access:

```bash
python -c "from kagglex.client import check_kaggle_health; ok, user = check_kaggle_health(); print(f'Authenticated as {user}' if ok else f'Auth failed: {user}')"
```

Authentication requires either:

- A valid credential file at `~/.kaggle/kaggle.json` with your username and API key.
- The environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Ensure your Kaggle account has accepted phone verification if you plan to use GPU or TPU accelerators.

## Batch job execution

Run local code remotely using `kagglex run`.

### Standalone script execution

Run a single script on a dual T4 GPU accelerator:

```bash
kagglex run --file train.py --gpu t4-2x --title "Baseline Training"
```

### Module and package execution

For projects with package layouts or multiple files, pass the command string directly:

```bash
kagglex run "python -m mypkg.train --epochs 10 --batch-size 32" --gpu t4-2x
```

### Multi-GPU acceleration

When running multi-GPU jobs on `t4-2x`, use the `--multi-gpu` flag. The runner wraps the command in `torchrun --nproc_per_node=2`:

```bash
kagglex run "python -m mypkg.train --batch-size 64" --gpu t4-2x --multi-gpu
```

### Remote dependencies and secrets

Inject external pip packages, environment variables, or Kaggle secrets into the remote environment:

```bash
kagglex run --file train.py \
  --extra-deps "transformers>=4.40.0" "accelerate" "wandb" \
  --kaggle-secrets WANDB_API_KEY HF_TOKEN \
  --env WANDB_PROJECT=image-classifier
```

### Mounting datasets and local data

Attach existing Kaggle datasets or bundle local data folders:

```bash
kagglex run --file train.py \
  --datasets "zillow/zecon" \
  --include-data ./data/splits
```

Attached datasets mount under `/kaggle/input/<dataset-name>/`.

## Monitoring and artifact sync

### Real-time streaming

Stream logs directly to stdout as the kernel runs:

```bash
kagglex run "python train.py" --stream
```

### Non-blocking execution

Submit a job without waiting, and retrieve logs or status later:

```bash
kagglex run "python train.py" --no-wait --slug my-run-01
```

Inspect run status:

```bash
kagglex status my-run-01
```

Stream logs for a running or completed kernel:

```bash
kagglex logs my-run-01
```

List recent runs:

```bash
kagglex list
```

Cancel an ongoing run:

```bash
kagglex cancel my-run-01
```

### Downloading outputs

Download produced model checkpoints, logs, and artifacts:

```bash
kagglex pull my-run-01 --output-dir ./results --include-outputs "*.json" "checkpoints/*"
```

Default output exclusion filters skip temporary zip files, `.pyc` files, and raw `.pt` weights unless explicitly matched by `--include-outputs`.

## Interactive REPL execution

When a Kaggle notebook session is active, connect directly to its Jupyter proxy server for low-latency command execution and debugging.

Set the proxy URL via environment variable:

```bash
export KAGGLE_JUPYTER_URL="https://kkb-production.jupyter-proxy.kaggle.net?token=YOUR_TOKEN"
```

Or pass `--url` with each command.

Test connection:

```bash
kagglex exec --test
```

Inspect available GPUs and memory:

```bash
kagglex exec --gpu-info
```

Execute a Python code string:

```bash
kagglex exec "import torch; print(torch.cuda.get_device_name(0))"
```

Execute a local script on the remote kernel:

```bash
kagglex exec --file debug_step.py
```

List remote files in `/kaggle/working/`:

```bash
kagglex exec --list-files
```

Transfer files between local and remote environments:

```bash
kagglex exec --upload ./local_weights.pt --remote-name weights.pt
kagglex exec --download output_metrics.json -o ./metrics.json
```

## Kaggle dataset management

Push local directories to Kaggle as versioned datasets:

```bash
kagglex dataset push --data-dir ./data/embeddings --title "Text Embeddings Dataset"
```

## Python SDK usage

Integrate `kagglex` programmatically in Python workflows:

```python
from kagglex import KaggleRunner, RunConfig

runner = KaggleRunner()
config = RunConfig(
    command="python -m mypkg.train --epochs 5",
    gpu="t4-2x",
    multi_gpu=True,
    extra_deps=["wandb"],
    kaggle_secrets=["WANDB_API_KEY"],
)

job = runner.run(config=config, wait=True)
print(f"Final status: {job.status}")

job.pull_outputs(destination_dir="./results", include_patterns=["*.json"])
```

## References

For full parameter listings, configuration options, and debugging workflows, consult the reference documents:

- [CLI Reference](references/cli-reference.md)
- [SDK Reference](references/sdk-reference.md)
- [Interactive REPL Reference](references/interactive-repl.md)
- [Troubleshooting Guide](references/troubleshooting.md)
