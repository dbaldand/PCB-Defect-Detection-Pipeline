"""Path helpers for scripts run from the src/ package layout."""

from pathlib import Path


def repo_root(caller_file: str | Path) -> Path:
    """Return the repository root (parent directory of src/)."""
    resolved = Path(caller_file).resolve()
    if resolved.parent.name == "src":
        return resolved.parent.parent
    # src/<module>/script.py
    if len(resolved.parents) >= 2 and resolved.parents[1].name == "src":
        return resolved.parents[2]
    return resolved.parent


def src_dir(caller_file: str | Path) -> Path:
    """Return the src/ directory."""
    return repo_root(caller_file) / "src"
