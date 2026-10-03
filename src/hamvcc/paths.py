from pathlib import Path

# Repository/package root: <package>/src/hamvcc/paths.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path):
    """Resolve a config path relative to the HAMV-CC package root.

    Absolute paths are preserved. Relative paths are anchored to PROJECT_ROOT
    rather than the process working directory, making Windows/PowerShell runs
    deterministic.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
