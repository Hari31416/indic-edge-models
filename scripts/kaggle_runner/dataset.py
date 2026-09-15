"""Kaggle dataset creation and version management."""

import json
import logging
from pathlib import Path
from typing import Optional

from scripts.kaggle_runner.client import get_authenticated_username, get_kaggle_api
from scripts.kaggle_runner.config import DatasetConfig

logger = logging.getLogger(__name__)


def create_dataset_metadata(
    config: DatasetConfig,
    kaggle_username: str,
    target_dir: Path,
) -> Path:
    """Generate dataset-metadata.json in target data directory.

    Args:
        config: Dataset configuration.
        kaggle_username: Authenticated Kaggle username.
        target_dir: Directory containing data files.

    Returns:
        Path to generated dataset-metadata.json.
    """
    dataset_id = f"{kaggle_username}/{config.slug}"
    metadata = {
        "title": config.title,
        "id": dataset_id,
        "licenses": [{"name": config.license_name}],
    }

    metadata_path = target_dir / "dataset-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.debug("Generated dataset metadata: %s", metadata)
    return metadata_path


def push_dataset(
    config: DatasetConfig,
    version_notes: str = "Update dataset files",
    username: Optional[str] = None,
) -> str:
    """Create or update a dataset on Kaggle.

    Args:
        config: Dataset configuration instance.
        version_notes: Description of changes for new versions.
        username: Optional Kaggle username (auto-detected if None).

    Returns:
        URL of the Kaggle dataset.
    """
    user = username or get_authenticated_username()
    data_dir = config.data_dir.resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Generate metadata in directory
    create_dataset_metadata(config, user, data_dir)
    dataset_id = f"{user}/{config.slug}"

    api = get_kaggle_api()
    logger.info("Checking if dataset '%s' already exists on Kaggle...", dataset_id)

    dataset_exists = False
    try:
        status = api.dataset_status(dataset_id)
        if status:
            dataset_exists = True
    except Exception:
        dataset_exists = False

    if dataset_exists:
        logger.info("Dataset '%s' exists. Creating new version...", dataset_id)
        api.dataset_create_version(
            str(data_dir),
            version_notes=version_notes,
            quiet=False,
            dir_mode="zip",
        )
    else:
        logger.info("Creating new dataset '%s' on Kaggle...", dataset_id)
        api.dataset_create_new(
            str(data_dir),
            public=config.is_public,
            quiet=False,
            dir_mode="zip",
        )

    dataset_url = f"https://www.kaggle.com/datasets/{dataset_id}"
    logger.info("Dataset pushed successfully: %s", dataset_url)
    return dataset_url
