# CLI Reference

Reference of all commands, arguments, and options for the `kagglex` CLI tool.

## General options

Available on all invocations:

- `-h`, `--help`: Show usage help and exit.
- `-v`, `--verbose`: Enable verbose debug logging output.

## kagglex run

Stage, package, dispatch, monitor, and sync outputs from a remote Kaggle execution.

```bash
kagglex run [COMMAND] [OPTIONS]
```

### Execution target

- `COMMAND`: Positional shell command string to run inside the remote kernel (for example: `python -m train --lr 1e-4`).
- `--command COMMAND`: Explicit remote command string.
- `--file FILE`: Path to a standalone Python script to execute remotely.
- `--dir DIR`: Local directory to package as the project root. Defaults to current working directory.

### Hardware and acceleration

- `--gpu {t4-2x,p100,v3-8,none}`: Accelerator type. Defaults to `t4-2x`.
- `--multi-gpu`: Automatically wrap execution with `torchrun --nproc_per_node=2` on dual GPU instances.

### Environment and dependencies

- `--extra-deps [EXTRA_DEPS ...]`: Additional pip packages to install before running the target command.
- `--env [ENV ...]`: Environment variables to pass to the remote process in `KEY=VAL` or `KEY` format.
- `--env-file ENV_FILE`: Path to a local `.env` file containing variables to inject.
- `--kaggle-secrets [KAGGLE_SECRETS ...]`: Kaggle user secrets to fetch and inject into the environment.
- `--no-internet`: Disable internet access inside the remote Kaggle kernel.

### Data and datasets

- `--datasets [DATASETS ...]`: Kaggle dataset slugs to mount (for example: `username/dataset-slug`).
- `--include-data [INCLUDE_DATA ...]`: Local data files or folders to package into the upload bundle.
- `--parent-kernels [PARENT_KERNELS ...]`: Slugs of parent kernels whose outputs should be attached.

### Job identification and lifecycle

- `--title TITLE`: Human-readable kernel title.
- `--slug SLUG`: Custom slug identifier for the kernel. Defaults to a slug derived from title or script name.
- `--poll-interval POLL_INTERVAL`: Seconds between status polling queries. Defaults to 20 seconds.
- `--stream`: Stream execution logs in real time.
- `--no-wait`: Submit job and exit immediately without polling.
- `--dry-run`: Stage all files and metadata locally without uploading to Kaggle.

### Output synchronization

- `--output-dir OUTPUT_DIR`: Local folder to download artifacts to upon completion.
- `--record-file RECORD_FILE`: Markdown file (such as `README.md`) to append run metadata to.
- `--include-outputs [INCLUDE_OUTPUTS ...]`: Glob patterns for files to download from remote outputs.
- `--exclude-outputs [EXCLUDE_OUTPUTS ...]`: Glob patterns for files to ignore during artifact download.

## kagglex exec

Execute code and commands interactively on an active Kaggle Jupyter session.

```bash
kagglex exec [CODE] [OPTIONS]
```

### Options

- `CODE`: Python code string to execute remotely.
- `--url URL`: Kaggle Jupyter proxy URL including access token. Defaults to `KAGGLE_JUPYTER_URL` environment variable.
- `--file FILE`: Path to a local Python script to execute on the remote kernel.
- `--timeout TIMEOUT`: Execution timeout in seconds. Defaults to 120 seconds.
- `--test`: Test HTTP connectivity to the Jupyter proxy server.
- `--gpu-info`: Print remote GPU device names, driver versions, and VRAM utilization.
- `--list-files [SUBPATH]`: List files and directories within `/kaggle/working/`.
- `--upload LOCAL_PATH`: Upload a local file into `/kaggle/working/`.
- `--remote-name REMOTE_NAME`: Custom destination name when uploading.
- `--download REMOTE_NAME`: Download a file from `/kaggle/working/`.
- `-o OUTPUT`, `--output OUTPUT`: Local path to save downloaded file to.

## kagglex status

Check the status of a Kaggle kernel.

```bash
kagglex status KERNEL_SLUG
```

Outputs include state (`queued`, `running`, `complete`, `error`, `cancelAcknowledged`), execution duration, and error messages if failed.

## kagglex logs

Fetch stdout and stderr logs for a kernel.

```bash
kagglex logs KERNEL_SLUG
```

## kagglex cancel

Request cancellation of a running or queued Kaggle kernel.

```bash
kagglex cancel KERNEL_SLUG
```

## kagglex list

Display recent Kaggle runs tracked locally and remotely.

```bash
kagglex list
```

## kagglex pull

Download outputs from an existing completed kernel.

```bash
kagglex pull KERNEL_SLUG --output-dir ./results [OPTIONS]
```

### Options

- `--output-dir OUTPUT_DIR`: Destination folder for downloaded files.
- `--include-outputs [PATTERNS ...]`: Glob patterns to download.
- `--exclude-outputs [PATTERNS ...]`: Glob patterns to skip.

## kagglex dataset push

Upload or update a Kaggle dataset from a local folder.

```bash
kagglex dataset push --data-dir ./data/processed --title "Processed Corpus"
```

### Options

- `--data-dir DATA_DIR`: Local folder path containing data files to package.
- `--title TITLE`: Dataset title.
- `--slug SLUG`: Optional custom dataset slug.
- `--public`: Set dataset visibility to public. Defaults to private.
