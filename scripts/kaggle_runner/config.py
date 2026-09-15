"""Configuration models and path resolution for Kaggle runner."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Convert arbitrary string into a valid Kaggle slug.

    Kaggle slugs must be lowercase, alphanumeric with hyphens, and 5-50 chars.
    """
    slug = re.sub(r"[^\w\s-]", "", value.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    if len(slug) < 5:
        slug = f"{slug}-exp"
    return slug[:50]


def find_repo_root(start_dir: Optional[Path] = None) -> Path:
    """Find repository root by looking for .git or context.md."""
    current = start_dir or Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "context.md").exists():
            return parent.resolve()
    return current.resolve()


def resolve_subproject_path(
    subproject_name: str, repo_root: Optional[Path] = None
) -> Path:
    """Resolve subproject directory path and verify pyproject.toml exists."""
    root = repo_root or find_repo_root()
    subproject_dir = root / subproject_name
    if not subproject_dir.exists():
        raise FileNotFoundError(
            f"Subproject directory '{subproject_name}' not found under {root}"
        )
    if not (subproject_dir / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"No pyproject.toml found in subproject '{subproject_name}' at {subproject_dir}"
        )
    return subproject_dir.resolve()


@dataclass
class KernelConfig:
    """Configuration for a Kaggle kernel execution."""

    title: str
    command: str
    slug: Optional[str] = None
    subproject: str = "embeddings"
    gpu_type: str = "t4-2x"
    multi_gpu: bool = False
    enable_internet: bool = True
    dataset_slugs: List[str] = field(default_factory=list)
    include_data: List[str] = field(default_factory=list)
    extra_pip_deps: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    output_dir: Optional[Path] = None
    record_file: Optional[Path] = None
    include_outputs: List[str] = field(default_factory=list)
    exclude_outputs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and normalize configuration values."""
        if not self.slug:
            self.slug = slugify(self.title)
        else:
            self.slug = slugify(self.slug)

        valid_gpus = {"t4-2x", "p100", "none"}
        if self.gpu_type not in valid_gpus:
            raise ValueError(
                f"Invalid gpu_type '{self.gpu_type}'. Must be one of {valid_gpus}"
            )

        if not self.command or not self.command.strip():
            raise ValueError("Command cannot be empty.")

        if self.output_dir and isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if self.record_file and isinstance(self.record_file, str):
            self.record_file = Path(self.record_file)


@dataclass
class DatasetConfig:
    """Configuration for a Kaggle dataset upload."""

    title: str
    data_dir: Path
    slug: Optional[str] = None
    is_public: bool = False
    license_name: str = "CC0-1.0"

    def __post_init__(self) -> None:
        """Validate and normalize dataset configuration."""
        if not self.slug:
            self.slug = slugify(self.title)
        else:
            self.slug = slugify(self.slug)

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.data_dir}"
            )
