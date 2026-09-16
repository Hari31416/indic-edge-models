# SDK Reference

Programmatic Python interface for `kagglex`.

## Importing the module

```python
from kagglex import (
    AcceleratorType,
    InteractiveClient,
    Job,
    KaggleRunner,
    RunConfig,
)
```

## KaggleRunner

The primary class for dispatching and managing remote Kaggle jobs.

```python
runner = KaggleRunner(staging_dir=None)
```

### Methods

#### run

Stage, submit, and optionally wait for a job:

```python
job = runner.run(
    command="python -m train --epochs 5",
    title="Fine Tuning",
    gpu="t4-2x",
    multi_gpu=True,
    wait=True,
    output_dir="./results",
)
```

Parameters:

- `command`: Remote shell command to run.
- `target_file`: Path to a standalone Python script to execute.
- `config`: Optional `RunConfig` instance providing full configuration.
- `wait`: When True, blocks until the job finishes. Defaults to True.
- `title`: Human-readable kernel title.
- `gpu`: Accelerator string (`t4-2x`, `p100`, `v3-8`, or `none`).
- `multi_gpu`: When True, runs multi-GPU wrapping with torchrun.
- `datasets`: List of Kaggle dataset slugs to attach.
- `extra_deps`: List of pip dependencies to install remotely.
- `output_dir`: Path to folder for downloading artifacts.

Returns a `Job` handle.

#### push

Submit a job to Kaggle without blocking:

```python
job = runner.push(command="python train.py", title="Async Training")
```

#### stage

Package project files locally without uploading to Kaggle. Useful for dry runs and inspection:

```python
staged = runner.stage(command="python train.py")
print(f"Payload created at: {staged.metadata_dir}")
```

## Job

Handle returned by `runner.run()` and `runner.push()`.

### Properties

- `job.kernel_id`: Kaggle kernel identifier in `username/kernel-slug` format.
- `job.url`: Web URL to the kernel on Kaggle.
- `job.status`: Current execution state string (`queued`, `running`, `complete`, `error`).
- `job.last_status_info`: Dictionary containing full response from Kaggle status check.

### Methods

#### wait

Block until execution completes, fails, or is cancelled:

```python
info = job.wait(poll_interval_sec=20, timeout_sec=3600)
print(f"Finished with status: {info['status']}")
```

#### stream_logs

Stream live console output to stdout or a custom handler:

```python
job.stream_logs(on_line=lambda line: print(f"[REMOTE] {line}"))
```

#### pull_outputs

Download generated artifacts from the remote kernel:

```python
downloaded_files = job.pull_outputs(
    destination_dir="./results",
    include_patterns=["*.json", "*.pt"],
    exclude_patterns=["cache/*"],
)
```

#### cancel

Request immediate job termination:

```python
success = job.cancel()
```

## RunConfig

Dataclass specifying run options:

```python
config = RunConfig(
    command="python -m mypkg.train",
    gpu="t4-2x",
    multi_gpu=True,
    extra_deps=["torch>=2.2.0", "wandb"],
    env_vars={"WANDB_PROJECT": "experiment-1"},
    kaggle_secrets=["WANDB_API_KEY"],
    datasets=["username/my-dataset"],
    poll_interval=15,
    timeout_sec=7200,
)
```

## InteractiveClient

Client for communicating with active Kaggle Jupyter sessions:

```python
from kagglex import InteractiveClient

client = InteractiveClient(
    base_url="https://kkb-production.jupyter-proxy.kaggle.net?token=TOKEN",
    timeout=120,
)

# Test connectivity
is_alive = client.test_connection()

# Query hardware
gpu_stats = client.get_gpu_info()

# Execute code snippet
result = client.execute_code("import torch; print(torch.cuda.is_available())")
print(result["stdout"])

# Transfer files
client.upload_file(local_path=Path("./model.py"), remote_filename="model.py")
client.download_file(remote_filename="metrics.json", local_path=Path("./metrics.json"))
```
