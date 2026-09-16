# Troubleshooting Guide

Common issues, diagnostic checks, and fixes when using `kagglex`.

## Authentication errors

### Problem

Commands fail with:

```text
Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json exists or KAGGLE_USERNAME and KAGGLE_KEY environment variables are set.
```

### Solutions

Verify that `~/.kaggle/kaggle.json` exists with strict file permissions:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Or set the credentials in your shell:

```bash
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
```

Verify status using Python:

```bash
python -c "from kagglex.client import check_kaggle_health; print(check_kaggle_health())"
```

## Accelerator availability and quotas

### Problem

The job fails to launch with an accelerator error or falls back to CPU.

### Causes and solutions

- **Phone verification**: Kaggle requires account phone verification before enabling GPU or TPU accelerators. Log in to Kaggle settings and verify your phone number.
- **Weekly quota**: Kaggle provides 30 hours of free GPU per week. If your weekly quota is exhausted, dispatch jobs with `--gpu none` until the quota resets.
- **Accelerator slug matching**: Accepted accelerators are `t4-2x`, `p100`, `v3-8`, and `none`. Dual T4 GPUs are specified as `t4-2x`.

## Payload size limit exceeded

### Problem

Packaging errors indicate the project archive exceeds 80MB.

### Causes and solutions

Kaggle enforces strict limits on kernel code uploads. Never bundle raw datasets or virtual environments directly in the code payload.

Add an ignore file named `.kaggleignore` to your project root:

```text
.venv/
data/
*.pt
*.bin
*.tar.gz
*.csv
```

To use datasets larger than a few megabytes, push them separately as a Kaggle dataset:

```bash
kagglex dataset push --data-dir ./data --title "My Large Dataset"
```

Then attach the dataset when running your job:

```bash
kagglex run --file train.py --datasets "username/my-large-dataset"
```

Attached datasets mount at `/kaggle/input/my-large-dataset/` without counting toward the upload limit.

## Syntax validation failure before dispatch

### Problem

Execution halts locally before anything is sent to Kaggle:

```text
SyntaxError: invalid syntax
```

### Cause

`kagglex` performs pre-flight AST parsing on target Python scripts to protect your weekly GPU quota from being wasted on syntax typos.

Fix the syntax error identified by line number before dispatching.

## Missing Kaggle secrets

### Problem

Remote logs show missing environment variables or API keys during execution.

### Solutions

Secrets specified with `--kaggle-secrets` must exist in your Kaggle account.

1. Open any notebook in Kaggle.
2. Click `Add-ons` -> `Secrets`.
3. Add the secret label (for example: `WANDB_API_KEY`) and save.
4. Re-run your command with `--kaggle-secrets WANDB_API_KEY`.

## Interactive Jupyter proxy connection dropped

### Problem

`kagglex exec` returns connection timeout or HTTP 403 / 401.

### Solutions

- Check that the Kaggle notebook is actively running. If the session went idle, Kaggle shuts down the server.
- Verify the token has not expired by copying a fresh URL from `Run -> Kaggle Jupyter Server -> Copy URL`.
- Test the endpoint:

```bash
kagglex exec --url "https://kkb-production.jupyter-proxy.kaggle.net?token=TOKEN" --test
```
