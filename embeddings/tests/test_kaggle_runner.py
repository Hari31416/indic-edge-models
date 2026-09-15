"""Unit tests for Kaggle remote experiment runner."""

import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from scripts.kaggle_runner.bootstrap import generate_bootstrap_script
from scripts.kaggle_runner.cli import create_parser, stage_kernel_bundle
from scripts.kaggle_runner.config import (
    DatasetConfig,
    KernelConfig,
    find_repo_root,
    resolve_subproject_path,
    slugify,
)
from scripts.kaggle_runner.dataset import create_dataset_metadata
from scripts.kaggle_runner.packager import (
    create_kernel_metadata,
    package_local_data,
    package_subproject,
)


def test_slugify() -> None:
    """Test Kaggle slug generation."""
    assert slugify("My Experiment #1") == "my-experiment-1"
    assert slugify("Indic_Edge_Pilot_01") == "indic_edge_pilot_01"
    assert slugify("a") == "a-exp"
    assert len(slugify("a" * 100)) <= 50


def test_kernel_config_validation() -> None:
    """Test KernelConfig validation rules."""
    cfg = KernelConfig(title="Test Kernel", command="python -m test")
    assert cfg.slug == "test-kernel"
    assert cfg.gpu_type == "t4-2x"
    assert cfg.enable_internet is True

    with pytest.raises(ValueError, match="Invalid gpu_type"):
        KernelConfig(title="Test", command="test", gpu_type="invalid_gpu")

    with pytest.raises(ValueError, match="Command cannot be empty"):
        KernelConfig(title="Test", command="   ")


def test_dataset_config_validation(tmp_path: Path) -> None:
    """Test DatasetConfig validation rules."""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()

    cfg = DatasetConfig(title="Sample Dataset", data_dir=data_dir)
    assert cfg.slug == "sample-dataset"
    assert cfg.license_name == "CC0-1.0"

    with pytest.raises(FileNotFoundError):
        DatasetConfig(title="Test", data_dir=tmp_path / "non_existent")


def test_resolve_subproject_path() -> None:
    """Test resolving subproject path."""
    repo_root = find_repo_root()
    subproject_dir = resolve_subproject_path("embeddings", repo_root)
    assert subproject_dir.exists()
    assert (subproject_dir / "pyproject.toml").exists()

    with pytest.raises(FileNotFoundError):
        resolve_subproject_path("non_existent_subproject", repo_root)


def test_package_subproject(tmp_path: Path) -> None:
    """Test subproject source packaging and exclusions."""
    subproject_dir = tmp_path / "test_subproject"
    subproject_dir.mkdir()
    (subproject_dir / "pyproject.toml").write_text(
        "[project]\nname='test'", encoding="utf-8"
    )
    src_dir = subproject_dir / "src" / "test_pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("# init", encoding="utf-8")
    (src_dir / "module.py").write_text("def foo(): pass", encoding="utf-8")

    # Add ignored items
    venv_dir = subproject_dir / ".venv" / "bin"
    venv_dir.mkdir(parents=True)
    (venv_dir / "python").write_text("bin", encoding="utf-8")
    pycache = src_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython-310.pyc").write_text("cache", encoding="utf-8")

    output_zip = tmp_path / "pkg_payload.zip"
    package_subproject(subproject_dir, output_zip)

    assert output_zip.exists()
    with zipfile.ZipFile(output_zip, "r") as zf:
        namelist = zf.namelist()
        assert "pyproject.toml" in namelist
        assert "src/test_pkg/__init__.py" in namelist
        assert "src/test_pkg/module.py" in namelist
        # Excluded items should not be in zip
        assert not any(".venv" in name for name in namelist)
        assert not any("__pycache__" in name for name in namelist)


def test_package_local_data(tmp_path: Path) -> None:
    """Test packaging local data manifests and files."""
    data_dir = tmp_path / "data" / "manifest"
    data_dir.mkdir(parents=True)
    (data_dir / "eval.jsonl").write_text('{"id": 1}', encoding="utf-8")

    output_zip = tmp_path / "data_payload.zip"
    res = package_local_data([data_dir], base_dir=tmp_path, output_zip=output_zip)

    assert res is not None
    assert output_zip.exists()
    with zipfile.ZipFile(output_zip, "r") as zf:
        namelist = zf.namelist()
        assert any("eval.jsonl" in name for name in namelist)


def test_create_kernel_metadata(tmp_path: Path) -> None:
    """Test generating kernel-metadata.json."""
    cfg = KernelConfig(
        title="Indic Experiment 1",
        command="python -m test",
        slug="indic-exp-1",
        gpu_type="t4-2x",
        dataset_slugs=["user/my-data", "external-dataset"],
    )

    meta_file = create_kernel_metadata(
        config=cfg,
        kaggle_username="testuser",
        staging_dir=tmp_path,
        code_file="kaggle_bootstrap.py",
    )

    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["id"] == "testuser/indic-exp-1"
    assert meta["title"] == "Indic Experiment 1"
    assert meta["enable_gpu"] == "true"
    assert meta["enable_internet"] == "true"
    assert meta["dataset_sources"] == ["user/my-data", "testuser/external-dataset"]


def test_generate_bootstrap_script(tmp_path: Path) -> None:
    """Test generating remote bootstrap orchestrator."""
    cfg = KernelConfig(
        title="Test",
        command="python -m indic_edge_embeddings.train --epochs 5",
        multi_gpu=True,
        extra_pip_deps=["scikit-learn>=1.4.0"],
        env_vars={"WANDB_MODE": "disabled"},
    )
    bootstrap_file = tmp_path / "kaggle_bootstrap.py"
    generate_bootstrap_script(cfg, has_data_payload=True, output_path=bootstrap_file)

    assert bootstrap_file.exists()
    content = bootstrap_file.read_text(encoding="utf-8")
    assert "indic_edge_embeddings.train" in content
    assert "torchrun --nproc_per_node=" in content
    assert "scikit-learn>=1.4.0" in content
    assert "WANDB_MODE" in content
    assert "pkg_payload.zip" in content
    # Verify the generated file is syntactically valid Python
    compile(content, "kaggle_bootstrap.py", "exec")


def test_create_dataset_metadata(tmp_path: Path) -> None:
    """Test generating dataset-metadata.json."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    cfg = DatasetConfig(title="Test Dataset", slug="test-dataset", data_dir=data_dir)

    meta_path = create_dataset_metadata(
        cfg, kaggle_username="testuser", target_dir=data_dir
    )
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == "testuser/test-dataset"
    assert meta["title"] == "Test Dataset"


def test_cli_parser() -> None:
    """Test CLI argument parsing."""
    parser = create_parser()

    # Test 'run' parser
    args = parser.parse_args(
        [
            "run",
            "--command",
            "python -m test",
            "--title",
            "Test Run",
            "--gpu",
            "t4-2x",
            "--multi-gpu",
            "--datasets",
            "user/data1",
            "user/data2",
        ]
    )
    assert args.subcommand == "run"
    assert args.command == "python -m test"
    assert args.title == "Test Run"
    assert args.gpu == "t4-2x"
    assert args.multi_gpu is True
    assert args.datasets == ["user/data1", "user/data2"]

    # Test 'dataset push' parser
    ds_args = parser.parse_args(
        ["dataset", "push", "--data-dir", "embeddings/data", "--public"]
    )
    assert ds_args.subcommand == "dataset"
    assert ds_args.dataset_action == "push"
    assert ds_args.data_dir == "embeddings/data"
    assert ds_args.public is True


def test_stage_kernel_bundle_dry_run(tmp_path: Path) -> None:
    """Test staging a complete bundle locally."""
    repo_root = find_repo_root()
    cfg = KernelConfig(
        title="Staging Test",
        command="python -m indic_edge_embeddings.eval.benchmark_cli",
        slug="staging-test",
        subproject="embeddings",
    )

    staging_dir = tmp_path / "staging"
    stage_kernel_bundle(
        config=cfg,
        staging_dir=staging_dir,
        repo_root=repo_root,
        username="testuser",
    )

    assert (staging_dir / "pkg_payload.zip").exists()
    assert (staging_dir / "kaggle_bootstrap.py").exists()
    assert (staging_dir / "kernel-metadata.json").exists()
