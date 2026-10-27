"""A wiki read from a folder on disk: development and tests."""
from __future__ import annotations

from pathlib import Path


class LocalWikiSource:
    """Every regular `.md` file under `root`, named by its path without `.md`.

    Symlinks are skipped (rglob doesn't descend into linked folders either), so
    nothing outside the folder is read.
    """

    label = "a local folder"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()

    def load(self) -> dict[str, str]:
        if not self._root.is_dir():
            raise FileNotFoundError(f"wiki folder {self._root} does not exist")
        pages: dict[str, str] = {}
        for path in sorted(self._root.rglob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            name = path.relative_to(self._root).with_suffix("").as_posix()
            try:
                pages[name] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"wiki page {name!r} is not UTF-8") from exc
        return pages
