"""Kaggle API client wrapper for kernel management, polling, and output retrieval."""

import fnmatch
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_kaggle_api() -> Any:
    """Instantiate and authenticate KaggleApi client."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.error("Failed to authenticate Kaggle API: %s", e)
        raise RuntimeError(
            "Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json exists or "
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables are set."
        ) from e


def get_authenticated_username() -> str:
    """Retrieve the current authenticated Kaggle username."""
    api = get_kaggle_api()
    username = api.get_config_value("username")
    if not username:
        # Fallback to checking config
        if hasattr(api, "config_values") and "username" in api.config_values:
            username = api.config_values["username"]
    if not username:
        raise ValueError("Could not determine authenticated Kaggle username.")
    return str(username)


def push_kernel(staging_dir: Path) -> Tuple[str, str]:
    """Push staged kernel directory to Kaggle.

    Args:
        staging_dir: Path containing kernel-metadata.json and code.

    Returns:
        Tuple of (actual_kernel_id, kernel_url).
    """
    metadata_file = staging_dir / "kernel-metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"kernel-metadata.json not found in {staging_dir}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fallback_id = meta.get("id", "")

    logger.info("Pushing kernel '%s' to Kaggle...", fallback_id)
    api = get_kaggle_api()
    response = api.kernels_push(str(staging_dir))

    actual_id = fallback_id
    url = f"https://www.kaggle.com/code/{fallback_id}"

    if hasattr(response, "url") and response.url:
        url = response.url
        # Extract username/slug from url: https://www.kaggle.com/code/user/slug
        if "/code/" in url:
            actual_id = url.split("/code/")[-1].strip("/")

    logger.info("Kernel pushed successfully: %s (id: %s)", url, actual_id)
    if hasattr(response, "error") and response.error:
        logger.warning("Push response error note: %s", response.error)
    return actual_id, url


def get_kernel_status(kernel_id: str) -> Dict[str, Any]:
    """Fetch status for a Kaggle kernel.

    Args:
        kernel_id: Kernel identifier (e.g. 'username/slug').

    Returns:
        Dictionary containing status and message info.
    """
    api = get_kaggle_api()
    status_obj = api.kernels_status(kernel_id)

    status_str = "unknown"
    failure_msg = None

    if hasattr(status_obj, "status"):
        val = getattr(status_obj, "status")
        if hasattr(val, "name"):
            status_str = str(val.name).lower()
        else:
            status_str = str(val).lower()
        failure_msg = getattr(status_obj, "failure_message", None)
    elif isinstance(status_obj, dict):
        status_str = str(status_obj.get("status", "unknown")).lower()
        failure_msg = status_obj.get("failureMessage")
    else:
        status_str = str(status_obj).lower()

    if "kernelworkerstatus." in status_str:
        status_str = status_str.split("kernelworkerstatus.")[-1]

    return {
        "id": kernel_id,
        "status": status_str,
        "failure_message": failure_msg,
    }


def poll_kernel(
    kernel_id: str,
    poll_interval_sec: int = 20,
    timeout_sec: int = 43200,
) -> Dict[str, Any]:
    """Poll kernel status until completion, failure, or timeout.

    Args:
        kernel_id: Identifier of the kernel to monitor.
        poll_interval_sec: Seconds between status polls.
        timeout_sec: Maximum time to wait before timing out (default 12 hours).

    Returns:
        Final status dictionary.
    """
    logger.info("Monitoring Kaggle kernel '%s'...", kernel_id)
    start_time = time.time()
    last_status = ""

    terminal_statuses = {"complete", "error", "cancelacknowledged", "failed"}

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            raise TimeoutError(
                f"Kernel {kernel_id} timed out after {elapsed:.0f} seconds."
            )

        try:
            status_info = get_kernel_status(kernel_id)
            current_status = status_info["status"]
        except Exception as e:
            logger.warning(
                "Transient error checking kernel status (%s). Retrying in %ds...",
                e,
                poll_interval_sec,
            )
            time.sleep(poll_interval_sec)
            continue

        if current_status != last_status:
            logger.info(
                "Kernel status changed: [%s] (elapsed: %.0fs)",
                current_status.upper(),
                elapsed,
            )
            last_status = current_status
        else:
            logger.debug("Kernel status: %s (elapsed: %.0fs)", current_status, elapsed)

        if current_status in terminal_statuses:
            logger.info("Kernel reached terminal status: %s", current_status.upper())
            return status_info

        time.sleep(poll_interval_sec)


def stream_kernel_logs(kernel_id: str) -> None:
    """Stream live logs from a running Kaggle kernel in real-time.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').
    """
    api = get_kaggle_api()
    logger.info("Connecting to live log stream for '%s'...", kernel_id)
    try:
        for chunk in api.kernels_logs_stream(kernel_id):
            data = chunk.get("data", "")
            sys.stdout.write(data)
            sys.stdout.flush()
    except Exception as e:
        logger.debug("Live log streaming ended: %s", e)


DEFAULT_OUTPUT_EXCLUDES = [
    "cache/**",
    "cache",
    "pkg_src/**",
    "pkg_src",
    "pkg_payload.zip",
    "data_payload.zip",
    "__results__.html",
    "*.pyc",
]


def _match_pattern(rel_path_str: str, patterns: List[str]) -> bool:
    """Check if relative path matches any glob pattern."""
    for pat in patterns:
        if fnmatch.fnmatch(rel_path_str, pat) or fnmatch.fnmatch(
            Path(rel_path_str).name, pat
        ):
            return True
        if pat.endswith("/**") and rel_path_str.startswith(pat[:-3]):
            return True
    return False


def pull_kernel_output(
    kernel_id: str,
    destination_dir: Path,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """Download output artifacts from a completed Kaggle kernel with selective filtering.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').
        destination_dir: Directory to save filtered downloaded files.
        include_patterns: Optional glob patterns to include (e.g. ['outputs/**', '*.log']).
        exclude_patterns: Optional glob patterns to exclude (e.g. ['cache/**', 'pkg_src/**']).

    Returns:
        List of saved file paths in destination_dir.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading outputs for '%s' to %s...", kernel_id, destination_dir)

    effective_excludes = list(exclude_patterns or DEFAULT_OUTPUT_EXCLUDES)
    api = get_kaggle_api()

    # If user specified a single include prefix (e.g. 'outputs/**' or 'outputs/*'), pass server-side pattern
    server_pattern = None
    if include_patterns and len(include_patterns) == 1:
        pat = include_patterns[0]
        if pat.endswith("/**"):
            server_pattern = pat[:-3] + "/*"
        elif not pat.startswith("*"):
            server_pattern = pat

    saved_files: List[Path] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        api.kernels_output(
            kernel_id,
            path=str(tmp_path),
            file_pattern=server_pattern,
            force=True,
            quiet=False,
        )

        for file_path in tmp_path.rglob("*"):
            if not file_path.is_file():
                continue

            rel_str = str(file_path.relative_to(tmp_path))

            # Check explicit exclusions
            if _match_pattern(rel_str, effective_excludes):
                logger.debug("Excluding file from output download: %s", rel_str)
                continue

            # Check explicit inclusions if specified
            if include_patterns and not _match_pattern(rel_str, include_patterns):
                logger.debug("Skipping non-included output file: %s", rel_str)
                continue

            dest_file = destination_dir / rel_str
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_file)
            saved_files.append(dest_file)

    logger.info("Saved %d output files to %s.", len(saved_files), destination_dir)
    return saved_files


def append_experiment_record(
    record_file: Path,
    kernel_id: str,
    status_info: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> None:
    """Update or append Kaggle execution details to an experiment record file.

    Args:
        record_file: Path to experiment README.md.
        kernel_id: Kaggle kernel identifier.
        status_info: Final status information.
        output_dir: Local path where artifacts were pulled.
    """
    if not record_file.exists():
        logger.warning(
            "Experiment record file %s does not exist. Skipping update.", record_file
        )
        return

    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
    status_str = status_info.get("status", "unknown").upper()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    log_entry = [
        "",
        "### Kaggle Remote Execution",
        "",
        f"- **Kernel ID:** `{kernel_id}`",
        f"- **Kernel URL:** {kernel_url}",
        f"- **Completed At:** {timestamp}",
        f"- **Status:** `{status_str}`",
    ]

    if status_info.get("failure_message"):
        log_entry.append(f"- **Error:** {status_info['failure_message']}")

    if output_dir:
        try:
            rel_output = output_dir.relative_to(record_file.parent)
        except ValueError:
            rel_output = output_dir
        log_entry.append(f"- **Downloaded Outputs:** `{rel_output}`")

    log_entry.append("")

    with open(record_file, "a", encoding="utf-8") as f:
        f.write("\n".join(log_entry))

    logger.info("Updated experiment record: %s", record_file)
