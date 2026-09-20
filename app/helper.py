import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_app_version() -> str:
    # 1. Try the standard way (works when the package is installed)
    try:
        return version("nescio-memory")
    except PackageNotFoundError:
        pass

    # 2. Fallback: Read pyproject.toml directly (works when running from source)
    try:
        # Adjust this path based on where this file is relative to pyproject.toml
        # If this file is in `app/main.py`, parent.parent is the project root.
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"

        if not pyproject_path.exists():
            return "unknown"

        # Use built-in tomllib (Python 3.11+) or fallback to tomli
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        return data["project"]["version"]
    except Exception:
        return "unknown"