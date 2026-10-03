from pathlib import Path
from src.hamvcc.paths import PROJECT_ROOT, resolve_project_path


def test_project_root_is_package_root():
    assert (PROJECT_ROOT / "configs" / "real_archive.json").exists()


def test_relative_data_path_is_root_anchored():
    p = resolve_project_path("DATA/archive_a_features.npz")
    assert p == PROJECT_ROOT / "DATA" / "archive_a_features.npz"
