# Interactive REPL Reference

Guide for running code interactively on active Kaggle notebooks through the Jupyter proxy interface.

## Purpose

Batch jobs take time to queue and initialize. When you need rapid iterations, debugging, or exploratory analysis on Kaggle GPUs, start an interactive session directly inside a running Kaggle notebook.

## Obtaining the proxy URL

1. Open or start a notebook in the Kaggle web interface.
2. Ensure accelerator hardware (GPU or TPU) is enabled under notebook settings.
3. In the top navigation bar, open the `Run` menu.
4. Select `Kaggle Jupyter Server` and click `Copy URL`.
5. The copied URL follows this format:

```text
https://kkb-production.jupyter-proxy.kaggle.net?token=<AUTH_TOKEN>
```

## Configuring authentication

Export the URL in your local shell to avoid typing `--url` on every invocation:

```bash
export KAGGLE_JUPYTER_URL="https://kkb-production.jupyter-proxy.kaggle.net?token=YOUR_TOKEN"
```

You can also store this variable in a local `.env` file.

## Verifying connectivity

Confirm network reachability and valid credentials:

```bash
kagglex exec --test
```

A healthy response returns `Connection successful`.

## Inspecting remote hardware

Query GPU hardware, available VRAM, and CUDA status:

```bash
kagglex exec --gpu-info
```

Output displays CUDA availability, driver version, device count, and memory metrics for each accelerator.

## Executing code snippets

Run arbitrary Python statements on the remote kernel:

```bash
kagglex exec "import torch; x = torch.randn(1000, 1000, device='cuda'); print(x.mean().item())"
```

Standard output and standard error from the remote process stream back to your local terminal.

## Executing local scripts remotely

Run a full local Python script on the remote kernel:

```bash
kagglex exec --file ./eval_script.py
```

Execution runs with the remote Python runtime and GPU drivers while reading your local script contents.

## Inspecting remote directory contents

Check files generated in `/kaggle/working/`:

```bash
kagglex exec --list-files
```

To view a subdirectory, pass the relative path:

```bash
kagglex exec --list-files checkpoints
```

## Uploading and downloading files

Send local datasets or script files to `/kaggle/working/`:

```bash
kagglex exec --upload ./configs/experiment.yaml --remote-name experiment.yaml
```

Retrieve generated outputs or model weights from the remote session:

```bash
kagglex exec --download results.json -o ./local_results.json
```

## Timeouts and session lifecycles

The default command timeout is 120 seconds. For longer operations, increase `--timeout`:

```bash
kagglex exec --file heavy_benchmark.py --timeout 600
```

Interactive sessions remain active as long as the Kaggle notebook tab remains open and connected.
