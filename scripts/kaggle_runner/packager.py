"""Packaging and staging utilities for local code and data."""

import json
import logging
from pathlib import Path
from typing import List, Optional
import zipfile

from scripts.kaggle_runner.config import KernelConfig

logger = logging.getLogger(__name__)

# Root directories to ignore at the subproject root level
ROOT_EXCLUDES = {
    ".venv",
    "venv",
    "artifacts",
    "experiments",
    "data",
    ".pytest_cache",
    ".mypy_cache",
    ".git",
}

# Patterns to ignore anywhere in the file tree
GLOBAL_IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".DS_Store",
}

MAX_KERNEL_UPLOAD_BYTES = 80 * 1024 * 1024  # 80 MB safe margin under Kaggle 100MB limit


def _should_exclude(path: Path, root: Path) -> bool:
    """Check if a given path should be excluded from package bundle."""
    rel = path.relative_to(root)
    parts = rel.parts

    # If file is directly under a root excluded folder (not under src/)
    if len(parts) > 1 and parts[0] != "src" and parts[0] in ROOT_EXCLUDES:
        return True

    # Check global ignores and extensions
    for part in parts:
        if (
            part in GLOBAL_IGNORE_NAMES
            or part.endswith(".egg-info")
            or part.endswith(".pyc")
        ):
            return True
        if part.startswith(".") and part not in {".", ".."}:
            return True

    return False


def package_subproject(subproject_dir: Path, output_zip: Path) -> Path:
    """Package subproject source and configuration into a zip archive.

    Args:
        subproject_dir: Directory containing subproject (with pyproject.toml and src/).
        output_zip: Destination path for the zip payload.

    Returns:
        Path to the generated zip archive.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    src_dir = subproject_dir / "src"
    pyproject_file = subproject_dir / "pyproject.toml"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not pyproject_file.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_file}")

    file_count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add pyproject.toml
        zf.write(pyproject_file, arcname="pyproject.toml")
        file_count += 1

        # Add optional README if present
        readme_file = subproject_dir / "README.md"
        if readme_file.exists():
            zf.write(readme_file, arcname="README.md")
            file_count += 1

        # Add all source files recursively
        for file_path in src_dir.rglob("*"):
            if file_path.is_file() and not _should_exclude(file_path, subproject_dir):
                arcname = file_path.relative_to(subproject_dir)
                zf.write(file_path, arcname=str(arcname))
                file_count += 1

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    logger.info(
        "Packaged subproject '%s' (%d files, %.2f MB) into %s",
        subproject_dir.name,
        file_count,
        size_mb,
        output_zip.name,
    )
    return output_zip


def package_local_data(
    data_paths: List[Path],
    base_dir: Path,
    output_zip: Path,
) -> Optional[Path]:
    """Package small local data files or folders into a data zip archive.

    Args:
        data_paths: List of file or directory paths to bundle.
        base_dir: Base directory for relative path preservation.
        output_zip: Target zip archive path.

    Returns:
        Path to zip archive if files were added, None otherwise.
    """
    if not data_paths:
        return None

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in data_paths:
            resolved_p = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
            if not resolved_p.exists():
                logger.warning("Specified data path does not exist: %s", resolved_p)
                continue

            if resolved_p.is_file():
                try:
                    rel_name = resolved_p.relative_to(base_dir.resolve())
                except ValueError:
                    rel_name = resolved_p.name
                zf.write(resolved_p, arcname=str(rel_name))
                file_count += 1
            elif resolved_p.is_dir():
                for f in resolved_p.rglob("*"):
                    if f.is_file() and not _should_exclude(f, resolved_p):
                        try:
                            rel_name = f.relative_to(base_dir.resolve())
                        except ValueError:
                            rel_name = f.relative_to(resolved_p.parent)
                        zf.write(f, arcname=str(rel_name))
                        file_count += 1

    if file_count == 0:
        if output_zip.exists():
            output_zip.unlink()
        return None

    size_bytes = output_zip.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes > MAX_KERNEL_UPLOAD_BYTES:
        raise ValueError(
            f"Bundled data size ({size_mb:.1f} MB) exceeds safe upload limit (80 MB). "
            "Please upload large corpora as a Kaggle Dataset using `run_kaggle.py dataset push`."
        )

    logger.info(
        "Packaged %d local data files (%.2f MB) into %s",
        file_count,
        size_mb,
        output_zip.name,
    )
    return output_zip


def create_kernel_metadata(
    config: KernelConfig,
    kaggle_username: str,
    staging_dir: Path,
    code_file: str = "kaggle_bootstrap.py",
) -> Path:
    """Generate Kaggle kernel-metadata.json file in staging directory.

    Args:
        config: Kernel configuration instance.
        kaggle_username: Authenticated Kaggle username.
        staging_dir: Directory where kernel files are staged.
        code_file: Name of the bootstrap code file.

    Returns:
        Path to the written kernel-metadata.json file.
    """
    enable_gpu = config.gpu_type in {"t4-2x", "p100"}

    # Format dataset sources to ensure correct username prefix if needed
    dataset_sources = []
    for ds in config.dataset_slugs:
        ds = ds.strip()
        if "/" not in ds:
            ds = f"{kaggle_username}/{ds}"
        dataset_sources.append(ds)

    metadata = {
        "id": f"{kaggle_username}/{config.slug}",
        "title": config.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if config.enable_internet else "false",
        "dataset_sources": dataset_sources,
        "kernel_sources": [],
        "competition_sources": [],
    }

    metadata_path = staging_dir / "kernel-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.debug("Generated kernel metadata: %s", metadata)
    return metadata_path
