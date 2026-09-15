"""Generator for the remote Kaggle bootstrap orchestrator script."""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from scripts.kaggle_runner.config import KernelConfig

logger = logging.getLogger(__name__)

BOOTSTRAP_TEMPLATE = '''"""Bootstrap orchestrator executed remotely inside Kaggle kernel."""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Configuration injected by runner
TARGET_COMMAND = __TARGET_COMMAND_JSON__
MULTI_GPU = __MULTI_GPU_JSON__
EXTRA_PIP_DEPS = __EXTRA_PIP_DEPS_JSON__
ENV_VARS = __ENV_VARS_JSON__
PKG_PAYLOAD_B64 = __PKG_PAYLOAD_B64_JSON__
DATA_PAYLOAD_B64 = __DATA_PAYLOAD_B64_JSON__


def log(msg: str) -> None:
    """Print timestamped log message."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [KAGGLE-BOOTSTRAP] {msg}", flush=True)


def print_environment_info() -> None:
    """Print hardware and python environment diagnostic info."""
    log("=" * 60)
    log("Diagnostic & System Information:")
    log(f"Python: {sys.version.split()[0]} ({sys.executable})")
    log(f"Working Directory: {os.getcwd()}")

    try:
        import torch
        log(f"PyTorch Version: {torch.__version__}")
        log(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            log(f"CUDA Device Count: {device_count}")
            for i in range(device_count):
                prop = torch.cuda.get_device_properties(i)
                vram_gb = prop.total_memory / (1024 ** 3)
                log(f"  GPU {i}: {prop.name} ({vram_gb:.2f} GB VRAM)")
        else:
            log("Running on CPU.")
    except Exception as e:
        log(f"PyTorch check failed: {e}")

    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            log("nvidia-smi output:")
            for line in res.stdout.strip().splitlines():
                log(f"  {line}")
    except Exception:
        pass
    log("=" * 60)


def extract_package_and_install() -> None:
    """Extract local package source and install in editable mode."""
    if not PKG_PAYLOAD_B64:
        log("No package payload provided.")
        return

    extract_dir = Path("/kaggle/working/pkg_src")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    pkg_zip = Path("/kaggle/working/pkg_payload.zip")
    pkg_zip.write_bytes(base64.b64decode(PKG_PAYLOAD_B64.encode("ascii")))

    log(f"Extracting package payload to {extract_dir}...")
    with zipfile.ZipFile(pkg_zip, "r") as zf:
        zf.extractall(extract_dir)

    # Add src to sys.path immediately
    src_dir = extract_dir / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir.resolve()))
        log(f"Added {src_dir} to sys.path.")

    # Pip install in editable mode to ensure all project requirements are met
    log(f"Installing subproject dependencies from {extract_dir} in editable mode...")
    cmd = [sys.executable, "-m", "pip", "install", "-e", str(extract_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Pip install output: {res.stdout}")
        log(f"Pip install stderr: {res.stderr}")
    else:
        log("Successfully installed subproject package.")


def extract_data_payload() -> None:
    """Extract bundled data payload if present."""
    if not DATA_PAYLOAD_B64:
        return

    data_zip = Path("/kaggle/working/data_payload.zip")
    data_zip.write_bytes(base64.b64decode(DATA_PAYLOAD_B64.encode("ascii")))

    log("Extracting data payload to /kaggle/working...")
    with zipfile.ZipFile(data_zip, "r") as zf:
        zf.extractall("/kaggle/working")
    log("Extracted data payload.")


def install_extra_deps() -> None:
    """Install extra pip dependencies if requested."""
    if not EXTRA_PIP_DEPS:
        return
    log(f"Installing extra dependencies: {EXTRA_PIP_DEPS}")
    cmd = [sys.executable, "-m", "pip", "install"] + EXTRA_PIP_DEPS
    subprocess.run(cmd, check=True)


def build_execution_command() -> str:
    """Wrap command with torchrun if multi-GPU is enabled."""
    cmd = TARGET_COMMAND.strip()

    if MULTI_GPU:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1

        if num_gpus > 1:
            log(f"Multi-GPU enabled: wrapping command with torchrun for {num_gpus} GPUs.")
            if cmd.startswith("python -m "):
                module_part = cmd[len("python -m "):]
                cmd = f"torchrun --nproc_per_node={num_gpus} -m {module_part}"
            elif cmd.startswith("python "):
                script_part = cmd[len("python "):]
                cmd = f"torchrun --nproc_per_node={num_gpus} {script_part}"

    return cmd


def main() -> int:
    """Main execution sequence."""
    start_time = time.time()
    log("Starting Kaggle remote runner bootstrap...")

    output_dir = Path("/kaggle/working/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set custom environment variables
    for k, v in ENV_VARS.items():
        os.environ[k] = str(v)

    # Default caching paths
    os.environ.setdefault("HF_HOME", "/kaggle/working/cache/huggingface")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print_environment_info()
    extract_package_and_install()
    extract_data_payload()
    install_extra_deps()

    exec_cmd = build_execution_command()
    log(f"Executing workload command: {exec_cmd}")
    log("=" * 60)

    # Execute and stream output
    proc = subprocess.Popen(
        exec_cmd,
        shell=True,
        cwd="/kaggle/working",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_file_path = output_dir / "execution.log"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)

    proc.wait()
    return_code = proc.returncode
    duration_sec = time.time() - start_time

    log("=" * 60)
    log(f"Command finished with exit code {return_code} in {duration_sec:.1f}s.")

    # Write summary json for automated tracking
    summary = {
        "command": TARGET_COMMAND,
        "executed_command": exec_cmd,
        "exit_code": return_code,
        "duration_seconds": duration_sec,
        "status": "SUCCESS" if return_code == 0 else "FAILED",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
'''


def generate_bootstrap_script(
    config: KernelConfig,
    pkg_zip_path: Optional[Path],
    data_zip_path: Optional[Path],
    output_path: Path,
) -> Path:
    """Generate and write kaggle_bootstrap.py with injected parameters and embedded payloads.

    Args:
        config: Kernel configuration.
        pkg_zip_path: Path to package zip payload.
        data_zip_path: Path to optional data zip payload.
        output_path: Target path for kaggle_bootstrap.py.

    Returns:
        Path to the generated bootstrap script.
    """
    pkg_b64 = ""
    if pkg_zip_path and pkg_zip_path.exists():
        pkg_b64 = base64.b64encode(pkg_zip_path.read_bytes()).decode("ascii")

    data_b64 = ""
    if data_zip_path and data_zip_path.exists():
        data_b64 = base64.b64encode(data_zip_path.read_bytes()).decode("ascii")

    script_content = (
        BOOTSTRAP_TEMPLATE.replace(
            "__TARGET_COMMAND_JSON__", json.dumps(config.command)
        )
        .replace("__MULTI_GPU_JSON__", repr(config.multi_gpu))
        .replace("__EXTRA_PIP_DEPS_JSON__", json.dumps(config.extra_pip_deps))
        .replace("__ENV_VARS_JSON__", json.dumps(config.env_vars))
        .replace("__PKG_PAYLOAD_B64_JSON__", json.dumps(pkg_b64))
        .replace("__DATA_PAYLOAD_B64_JSON__", json.dumps(data_b64))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    logger.debug("Generated bootstrap script at %s", output_path)
    return output_path
