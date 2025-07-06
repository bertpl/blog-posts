"""
Helper functions for project paths.
"""

from pathlib import Path


# -------------------------------------------------------------------------
#  Helper functions
# -------------------------------------------------------------------------
def get_project_root() -> Path:
    """
    Get the root directory of the project.
    """
    p = Path(__file__).resolve()
    while not (p / "pyproject.toml").exists() and (p.parent != p):
        p = p.parent

    if p.parent == p:
        raise FileNotFoundError(
            "Project root not found. Ensure you are inside a project subdirectory"
            + " and the root folder contains pyproject.toml."
        )
    else:
        return p


def get_data_folder(post_nr: int) -> Path:
    """
    Get the data folder path.
    """
    return _ensure_exists(get_project_root() / "data" / f"post_{post_nr}")


def get_figures_folder(post_nr: int) -> Path:
    """
    Get the figures folder path.
    """
    return _ensure_exists(get_project_root() / "figures" / f"post_{post_nr:0>4}")


# -------------------------------------------------------------------------
#  Internal
# -------------------------------------------------------------------------
def _ensure_exists(path: Path) -> Path:
    """
    Ensure that the given path exists.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
