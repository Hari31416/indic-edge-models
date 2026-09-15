"""Command-line interface for Kaggle remote experiment runner."""

import argparse
import logging
from pathlib import Path
import sys
import tempfile
from typing import List, Optional

from scripts.kaggle_runner.bootstrap import generate_bootstrap_script
from scripts.kaggle_runner.client import (
    append_experiment_record,
    get_authenticated_username,
    get_kernel_status,
    poll_kernel,
    pull_kernel_output,
    push_kernel,
)
from scripts.kaggle_runner.config import (
    DatasetConfig,
    KernelConfig,
    find_repo_root,
    resolve_subproject_path,
)
from scripts.kaggle_runner.dataset import create_dataset_metadata, push_dataset
from scripts.kaggle_runner.packager import (
    create_kernel_metadata,
    package_local_data,
    package_subproject,
)

logger = logging.getLogger("kaggle_runner")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def stage_kernel_bundle(
    config: KernelConfig,
    staging_dir: Path,
    repo_root: Optional[Path] = None,
    username: Optional[str] = None,
) -> Path:
    """Prepare the complete staging directory for Kaggle kernel submission.

    Args:
        config: Kernel configuration.
        staging_dir: Target directory to stage kernel files.
        repo_root: Base repository root directory.
        username: Kaggle username.

    Returns:
        Path to prepared staging directory.
    """
    root = repo_root or find_repo_root()
    subproject_dir = resolve_subproject_path(config.subproject, root)
    user = username or get_authenticated_username()

    staging_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Staging kernel bundle in %s...", staging_dir)

    # 1. Package Python subproject into pkg_payload.zip
    pkg_zip = staging_dir / "pkg_payload.zip"
    package_subproject(subproject_dir, pkg_zip)

    # 2. Package optional local data
    has_data_payload = False
    if config.include_data:
        data_paths = [Path(p) for p in config.include_data]
        data_zip = staging_dir / "data_payload.zip"
        res = package_local_data(
            data_paths, base_dir=subproject_dir, output_zip=data_zip
        )
        has_data_payload = res is not None

    # 3. Generate remote kaggle_bootstrap.py
    bootstrap_file = staging_dir / "kaggle_bootstrap.py"
    generate_bootstrap_script(
        config, has_data_payload=has_data_payload, output_path=bootstrap_file
    )

    # 4. Generate kernel-metadata.json
    create_kernel_metadata(config, kaggle_username=user, staging_dir=staging_dir)

    return staging_dir


def handle_run(args: argparse.Namespace) -> int:
    """Handle the 'run' subcommand: Stage -> Push -> Poll -> Pull -> Record."""
    config = KernelConfig(
        title=args.title,
        slug=args.slug,
        subproject=args.subproject,
        command=args.command,
        gpu_type=args.gpu,
        multi_gpu=args.multi_gpu,
        enable_internet=not args.no_internet,
        dataset_slugs=args.datasets or [],
        include_data=args.include_data or [],
        extra_pip_deps=args.extra_deps or [],
        output_dir=Path(args.output_dir) if args.output_dir else None,
        record_file=Path(args.record_file) if args.record_file else None,
    )

    repo_root = find_repo_root()
    staging_base = repo_root / "artifacts" / "kaggle_staging" / config.slug

    if args.dry_run:
        logger.info("Dry run requested. Staging files locally without pushing...")
        stage_kernel_bundle(config, staging_base, repo_root=repo_root)
        logger.info("Staged kernel successfully at: %s", staging_base)
        return 0

    user = get_authenticated_username()
    stage_kernel_bundle(config, staging_base, repo_root=repo_root, username=user)

    kernel_id = f"{user}/{config.slug}"
    push_kernel(staging_base)

    if args.no_wait:
        logger.info(
            "Kernel submitted. To monitor later, run: python scripts/run_kaggle.py status --kernel %s",
            kernel_id,
        )
        return 0

    # Poll status until complete
    status_info = poll_kernel(kernel_id, poll_interval_sec=args.poll_interval)
    terminal_status = status_info.get("status", "")

    # Pull outputs if completed and output_dir is specified
    destination_dir = config.output_dir
    if not destination_dir and config.record_file:
        destination_dir = config.record_file.parent / "outputs"

    if destination_dir:
        try:
            pull_kernel_output(kernel_id, destination_dir)
        except Exception as e:
            logger.error("Failed to pull outputs: %s", e)

    # Append to experiment record if requested
    if config.record_file:
        append_experiment_record(
            record_file=config.record_file,
            kernel_id=kernel_id,
            status_info=status_info,
            output_dir=destination_dir,
        )

    return 0 if terminal_status == "complete" else 1


def handle_push(args: argparse.Namespace) -> int:
    """Handle the 'push' subcommand: Stage and submit kernel without waiting."""
    config = KernelConfig(
        title=args.title,
        slug=args.slug,
        subproject=args.subproject,
        command=args.command,
        gpu_type=args.gpu,
        multi_gpu=args.multi_gpu,
        enable_internet=not args.no_internet,
        dataset_slugs=args.datasets or [],
        include_data=args.include_data or [],
        extra_pip_deps=args.extra_deps or [],
    )

    repo_root = find_repo_root()
    staging_base = repo_root / "artifacts" / "kaggle_staging" / config.slug

    if args.dry_run:
        stage_kernel_bundle(config, staging_base, repo_root=repo_root)
        logger.info("Dry run: staged kernel at %s", staging_base)
        return 0

    user = get_authenticated_username()
    stage_kernel_bundle(config, staging_base, repo_root=repo_root, username=user)
    push_kernel(staging_base)
    return 0


def handle_status(args: argparse.Namespace) -> int:
    """Handle the 'status' subcommand."""
    kernel_id = args.kernel
    if "/" not in kernel_id:
        user = get_authenticated_username()
        kernel_id = f"{user}/{kernel_id}"

    if args.stream:
        status_info = poll_kernel(kernel_id, poll_interval_sec=args.poll_interval)
        return 0 if status_info.get("status") == "complete" else 1

    status_info = get_kernel_status(kernel_id)
    logger.info("Kernel '%s' status: %s", kernel_id, status_info["status"].upper())
    if status_info.get("failure_message"):
        logger.warning("Failure message: %s", status_info["failure_message"])
    return 0


def handle_pull(args: argparse.Namespace) -> int:
    """Handle the 'pull' subcommand."""
    kernel_id = args.kernel
    if "/" not in kernel_id:
        user = get_authenticated_username()
        kernel_id = f"{user}/{kernel_id}"

    output_dir = Path(args.output_dir)
    pull_kernel_output(kernel_id, destination_dir=output_dir)
    return 0


def handle_dataset_push(args: argparse.Namespace) -> int:
    """Handle the 'dataset push' subcommand."""
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        logger.error("Dataset directory does not exist: %s", data_dir)
        return 1

    config = DatasetConfig(
        title=args.title or data_dir.name,
        slug=args.slug or data_dir.name,
        data_dir=data_dir,
        is_public=args.public,
        license_name=args.license,
    )

    if args.dry_run:
        user = get_authenticated_username()
        meta_file = create_dataset_metadata(config, user, data_dir)
        logger.info("Dry run: created dataset metadata at %s", meta_file)
        return 0

    url = push_dataset(config, version_notes=args.notes)
    logger.info("Dataset available at: %s", url)
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_kaggle",
        description="Run experiments, evaluate models, and manage datasets on Kaggle GPU instances.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. 'run' subcommand
    run_p = subparsers.add_parser(
        "run",
        help="Package, dispatch, monitor, and pull outputs from a Kaggle job.",
    )
    run_p.add_argument(
        "--command",
        type=str,
        required=True,
        help="Command to execute inside Kaggle (e.g. 'python -m indic_edge_embeddings.train.pilot_train')",
    )
    run_p.add_argument(
        "--title",
        type=str,
        required=True,
        help="Kernel title (e.g. 'indic-edge-pilot-01')",
    )
    run_p.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Optional custom kernel slug (auto-generated from title if omitted)",
    )
    run_p.add_argument(
        "--subproject",
        type=str,
        default="embeddings",
        help="Subproject directory name containing src/ and pyproject.toml (default: 'embeddings')",
    )
    run_p.add_argument(
        "--gpu",
        type=str,
        choices=["t4-2x", "p100", "none"],
        default="t4-2x",
        help="Kaggle GPU accelerator type (default: 't4-2x')",
    )
    run_p.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Automatically wrap command in torchrun / accelerate across multiple GPUs",
    )
    run_p.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Kaggle dataset slugs to attach (e.g. 'user/my-dataset')",
    )
    run_p.add_argument(
        "--include-data",
        nargs="*",
        default=[],
        help="Local data paths or manifests to package and bundle into kernel payload",
    )
    run_p.add_argument(
        "--extra-deps",
        nargs="*",
        default=[],
        help="Additional pip packages to install remotely",
    )
    run_p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to download artifacts to after run finishes",
    )
    run_p.add_argument(
        "--record-file",
        type=str,
        default=None,
        help="Path to experiment README.md to automatically log run metadata",
    )
    run_p.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="Seconds between status checks when polling (default: 20s)",
    )
    run_p.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and exit immediately without waiting for completion",
    )
    run_p.add_argument(
        "--no-internet",
        action="store_true",
        help="Disable internet access in Kaggle kernel",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage all files locally without submitting to Kaggle",
    )

    # 2. 'push' subcommand
    push_p = subparsers.add_parser(
        "push",
        help="Package and push a kernel to Kaggle without waiting for results.",
    )
    push_p.add_argument("--command", type=str, required=True, help="Command to execute")
    push_p.add_argument("--title", type=str, required=True, help="Kernel title")
    push_p.add_argument("--slug", type=str, default=None, help="Custom slug")
    push_p.add_argument(
        "--subproject", type=str, default="embeddings", help="Subproject"
    )
    push_p.add_argument(
        "--gpu", type=str, choices=["t4-2x", "p100", "none"], default="t4-2x"
    )
    push_p.add_argument(
        "--multi-gpu", action="store_true", help="Wrap in multi-GPU launcher"
    )
    push_p.add_argument("--datasets", nargs="*", default=[], help="Kaggle datasets")
    push_p.add_argument(
        "--include-data", nargs="*", default=[], help="Local data paths"
    )
    push_p.add_argument(
        "--extra-deps", nargs="*", default=[], help="Extra pip packages"
    )
    push_p.add_argument("--no-internet", action="store_true", help="Disable internet")
    push_p.add_argument(
        "--dry-run", action="store_true", help="Stage files locally only"
    )

    # 3. 'status' subcommand
    status_p = subparsers.add_parser("status", help="Check status of a Kaggle kernel.")
    status_p.add_argument("--kernel", type=str, required=True, help="Kernel ID or slug")
    status_p.add_argument(
        "--stream", action="store_true", help="Poll until terminal status"
    )
    status_p.add_argument(
        "--poll-interval", type=int, default=20, help="Seconds between polls"
    )

    # 4. 'pull' subcommand
    pull_p = subparsers.add_parser(
        "pull", help="Download output files from a completed kernel."
    )
    pull_p.add_argument("--kernel", type=str, required=True, help="Kernel ID or slug")
    pull_p.add_argument(
        "--output-dir", type=str, required=True, help="Local directory to save files"
    )

    # 5. 'dataset' subcommand with sub-subcommands
    ds_p = subparsers.add_parser("dataset", help="Manage Kaggle datasets.")
    ds_sub = ds_p.add_subparsers(dest="dataset_action", required=True)

    ds_push = ds_sub.add_parser(
        "push", help="Create or update a Kaggle dataset from local directory."
    )
    ds_push.add_argument(
        "--data-dir", type=str, required=True, help="Local data directory to upload"
    )
    ds_push.add_argument(
        "--title", type=str, default=None, help="Dataset title (default: folder name)"
    )
    ds_push.add_argument(
        "--slug", type=str, default=None, help="Dataset slug (default: folder name)"
    )
    ds_push.add_argument(
        "--public", action="store_true", help="Make dataset public (default: private)"
    )
    ds_push.add_argument(
        "--license", type=str, default="CC0-1.0", help="Dataset license"
    )
    ds_push.add_argument(
        "--notes", type=str, default="Upload dataset", help="Version description notes"
    )
    ds_push.add_argument(
        "--dry-run", action="store_true", help="Generate metadata only"
    )

    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(args_list)
    setup_logging(args.verbose)

    try:
        if args.subcommand == "run":
            return handle_run(args)
        elif args.subcommand == "push":
            return handle_push(args)
        elif args.subcommand == "status":
            return handle_status(args)
        elif args.subcommand == "pull":
            return handle_pull(args)
        elif args.subcommand == "dataset":
            if args.dataset_action == "push":
                return handle_dataset_push(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.exception("Operation failed: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
